# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
"""
run infer and eval batching.
"""

from collections import Counter
import dataclasses
import json
from dotenv import load_dotenv
import os
import sys
import time
import traceback
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import numpy as np
from loguru import logger

from eval.evaluation.data_loader import (
    WideSearchDataLoaderHF,
    WideSearchQuery,
    WideSearchResponse,
    WideSearchResponseLoader,
)
from eval.evaluation.evaluation import EvaluationResult, evaluate_single_query


logger.remove()
logger.add(sys.stderr, level="INFO")


class SingleTask:
    def __init__(
        self,
        query: WideSearchQuery,
        model_config_name: str,
        response_path: str,
        result_save_path: str,
        trial_idx: int = 1,
        use_cache: bool = False,
        eval_model_config_name: str = "default_eval_config"
    ):
        self.query = query
        self.response_path = response_path
        self.result_save_path = result_save_path
        self.trial_idx = trial_idx
        self.model_config_name = model_config_name
        self.use_cache = use_cache
        self.eval_model_config_name = eval_model_config_name
        self.eval_result_path = self.result_save_path.replace(".csv", ".json")

    def load_response(self) -> list[WideSearchResponse]:
        if not os.path.exists(self.response_path):
            raise FileNotFoundError(f"response_path {self.response_path} not found")
        return WideSearchResponseLoader.load_response(self.response_path)

    def eval(self):
        start_time = time.time()
        if os.path.exists(self.eval_result_path) and self.use_cache:
            with open(self.eval_result_path, "r") as f:
                eval_result = json.load(f)
            eval_result = EvaluationResult(**eval_result)
        else:
            if not os.path.exists(self.response_path):
                logger.error(f"response_path {self.response_path} not found, skip")
                response_list = [None]
            else:
                response_list = self.load_response()
            assert (
                response_list
            ), f"response is None, response_path: {self.response_path}"

            eval_result = evaluate_single_query(
                self.query,
                response_list[0],
                self.result_save_path,
                self.eval_model_config_name,
            )
            # eval_result is a dataclass, convert it to dict and then write to a json file
            eval_result_dict = dataclasses.asdict(eval_result)
            with open(self.eval_result_path, "w") as f:
                json.dump(eval_result_dict, f, ensure_ascii=False, indent=4)
        end_time = time.time()
        logger.info(
            f"eval end, instance_id: {self.query.instance_id}, cost(s): {end_time - start_time:.2f}"
        )
        return eval_result


def calc_summary_results(tasks: list[SingleTask], summary_result_path: str):
    metrics = [
        "score",
        "entity_acc",
        "search_tool_num",
        "visit_tool_num",
        "precision_by_row",
        "recall_by_row",
        "f1_by_row",
        "precision_by_item",
        "recall_by_item",
        "f1_by_item",
        "column_precision",
        "column_recall", 
        "column_f1"
    ]

    all_results = {m: [] for m in metrics}
    id_to_task = {}
    for task in tasks:
        if task.query.instance_id not in id_to_task:
            id_to_task[task.query.instance_id] = []
        id_to_task[task.query.instance_id].append(task)

    for iid, task_list in id_to_task.items():
        trial_metrics = {m: [] for m in metrics}
        for task in task_list:
            eval_result_path = task.eval_result_path
            if not os.path.exists(eval_result_path):
                continue
            with open(eval_result_path, "r") as f:
                result = json.load(f)
            for m in metrics:
                if m in result:
                    trial_metrics[m].append(result[m])
        # For each metric, compute avg_n, best_of_n, all_pass_n
        for m in metrics:
            values = trial_metrics[m]
            if not values or len(values) < trial_num:
                # If not enough trials, skip this instance for this metric
                logger.info(f"Skipping {m} for instance {iid}, not enough trials")
                raise ValueError(
                    f"Not enough trials for metric {m} on instance {iid}. "
                    f"Expected {trial_num}, got {len(values)}."
                )
            avg_n = float(np.mean(values))
            max_n = float(np.max(values))
            min_n = float(np.min(values))
            all_results[m].append({"avg_n": avg_n, "max_n": max_n, "min_n": min_n})

    # Aggregate over all instances
    summary = {}
    for m in metrics:
        vals = all_results[m]
        if not vals:
            continue
        summary[m] = {
            "avg_n": float(np.mean([v["avg_n"] for v in vals])),
            "max_n": float(np.mean([v["max_n"] for v in vals])),
            "min_n": float(np.mean([v["min_n"] for v in vals])),
        }
    logger.info(json.dumps(summary, indent=2, ensure_ascii=False))

    with open(summary_result_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model_config_name", type=str, default="doubao-1.6", help="model config name"
    )
    parser.add_argument(
        "--query_path", type=str, default="", help="the query file path"
    )
    parser.add_argument(
        "--answer_root_path", type=str, default="", help="the answer root path saves several csv files"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="eval",
        choices=["eval", "infer", "both"],
        help="stage to run",
    )

    parser.add_argument(
        "--response_root", type=str, default="data/output", help="response root"
    )
    parser.add_argument(
        "--result_save_root", type=str, default="data/output", help="result save root"
    )
    parser.add_argument(
        "--eval_model_config_name",
        type=str,
        default="default_eval_config",
        help="eval model config name",
    )
    parser.add_argument("--trial_num", type=int, default=4, help="trial num to run")
    parser.add_argument(
        "--instance_id", type=str, default="", help="instance id to run"
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="use cache to save and load the response files",
    )
    parser.add_argument(
        "--thread_num", type=int, default=4, help="thread num to run infer and eval"
    )

    args = parser.parse_args()

    trial_num = args.trial_num
    model_config_name = args.model_config_name
    response_root = args.response_root
    result_save_root = args.result_save_root
    #import ipdb; ipdb.set_trace()

    data_loader = WideSearchDataLoaderHF(
        query_path=args.query_path,
        answer_root=args.answer_root_path
    )

    instance_id_list = data_loader.get_instance_id_list()

    tasks = []

    for instance_id in instance_id_list:

        if args.instance_id and instance_id not in args.instance_id.split(","):
            continue
        query = data_loader.load_query_by_instance_id(instance_id)
        for trial_idx in range(trial_num):
            trial_idx += 1
            response_path = f"{response_root}/{model_config_name}/{instance_id}_{trial_idx}_response.jsonl"
            result_save_path = f"{result_save_root}/{model_config_name}/{instance_id}_{trial_idx}_eval_result.csv"
            if not os.path.exists(result_save_root):
                os.makedirs(result_save_root, exist_ok=True)
            if not os.path.exists(f'{result_save_root}/{model_config_name}'):
                os.makedirs(f'{result_save_root}/{model_config_name}', exist_ok=True)
            tasks.append(
                SingleTask(
                    query=deepcopy(query),
                    response_path=response_path,
                    result_save_path=result_save_path,
                    trial_idx=trial_idx,
                    model_config_name=model_config_name,
                    use_cache=args.use_cache,
                    eval_model_config_name=args.eval_model_config_name
                )
            )
    logger.info(f"total task num: {len(tasks)}")
    # multi threading
    with ThreadPoolExecutor(max_workers=args.thread_num) as executor:
        results = executor.map(lambda task: task.eval(), tasks)
        try:
            for result in results:
                logger.info(f"eval success, instance_id: {result.instance_id}")
        except Exception as e:
            logger.error(f"eval error: {e}")

    summary_result_path = (
        f"{result_save_root}/{model_config_name}_trial_num_{trial_num}_summary.json"
    )
    calc_summary_results(tasks=tasks, summary_result_path=summary_result_path)
