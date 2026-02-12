# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
This module provides evaluation functions for the WideSearch dataset.

It includes functions to evaluate single queries and calculate consistency between automatic and human evaluations.
"""

import traceback
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from pandarallel import pandarallel

from eval.evaluation.data_loader import (
    WideSearchQuery,
    WideSearchResponse,
)
from eval.evaluation.metric_utils import (
    llm_judge_column,
    metric_function_registry,
    preprocess_function_registry,
    primary_key_preprocess,
    get_entity_acc_llm_as_a_judge_template
)
from eval.utils.utils import norm_column
from eval.utils.llm import llm_completion
from my_utils import request_api

pandarallel.initialize(nb_workers=8)


@dataclass
class EvaluationResult:
    instance_id: str
    score: float = 0.0
    entity_acc: float = 0.0
    search_tool_num: int = 0
    visit_tool_num: int = 0
    precision_by_row: float = 0.0
    recall_by_row: float = 0.0
    f1_by_row: float = 0.0
    precision_by_item: float = 0.0
    recall_by_item: float = 0.0
    f1_by_item: float = 0.0
    column_precision: float = 0.0
    column_recall: float = 0.0
    column_f1: float = 0.0
    msg: str = ""


def preprocess_call(content, preprocess_func_name):
    assert (
        preprocess_func_name in preprocess_function_registry
    ), f"preprocess_func_name {preprocess_func_name} not in preprocess_function_registry"

    preprocess_func = preprocess_function_registry[preprocess_func_name]
    return preprocess_func(content)


def metric_call(response, target, criterion, metric_func_name):
    assert (
        metric_func_name in metric_function_registry
    ), f"metric_func_name {metric_func_name} not in metric_function_registry"

    metric_func = metric_function_registry[metric_func_name]
    if metric_func_name == "llm_judge" or metric_func_name == "number_near":
        score, msg = metric_func(response, target, criterion)
    else:
        score, msg = metric_func(response, target)

    logger.debug(f"metric_func_name {metric_func_name} score {score} msg {msg}")
    return score, msg


def evaluate_single_query(
    query: WideSearchQuery,
    response: Optional[WideSearchResponse],
    result_save_path: Optional[str] = None,
    eval_model_config_name: str = "default_eval_config",
) -> EvaluationResult:
    if response is None:
        logger.info(f"response is None, instance_id: {query.instance_id}")
        return EvaluationResult(
            instance_id=query.instance_id,
            msg="response is None",
        )

    assert (
        query.instance_id == response.instance_id
    ), f"query.instance_id {query.instance_id} != response.instance_id {response.instance_id}"

    entity_acc = 0.0
    search_tool_num = 0.0 # use response.messages
    visit_tool_num = 0.0
    score = 0.0
    precision_by_row = 0.0
    recall_by_row = 0.0
    f1_by_row = 0.0
    precision_by_item = 0.0
    recall_by_item = 0.0
    f1_by_item = 0.0
    precision_by_unique_col = 0.0
    recall_by_unique_col = 0.0
    f1_by_unique_col = 0.0
    msg = ""

    try:
        # tool calling
        for response_item in response.messages:
            content = response_item['content'].replace(' ', '')
            if '{\"name\":\"search\",' in content:
                search_tool_num += content.count('{\"name\":\"search\",')
            if '{\"name\":\"visit\",' in content:
                visit_tool_num += content.count('{\"name\":\"visit\",')

        # LLM-as-a-Judge on the entity acc.
        entity = query.entity
        response_ = response.response
        question = query.query
        entity_acc_check_prompt = get_entity_acc_llm_as_a_judge_template(question, response_, entity)
        response = llm_completion(
            messages=[{'role': 'user', 'content': entity_acc_check_prompt}],
            model_config_name=eval_model_config_name
        )
        if "yes" in response.content.lower():
            entity_acc = 1.0
        else:
            # entity is wrong, the answer is wrong
            return EvaluationResult(
                entity_acc=0.0,
                search_tool_num=search_tool_num,
                visit_tool_num=visit_tool_num,
                instance_id=query.instance_id,
                score=0.0,
                precision_by_row=0.0,
                recall_by_row=0.0,
                f1_by_row=0.0,
                precision_by_item=0.0,
                recall_by_item=0.0,
                f1_by_item=0.0,
                msg="the entity is wrong, all the results are wrong, set as 0",
            )
        
        # pre check
        required_columns = query.evaluation["required"]
        unique_columns = query.evaluation["unique_columns"]
        answer_df = query.answer
        answer_df.columns = [norm_column(col) for col in answer_df.columns]
        response_df = response.extract_dataframe()
        if response_df is None:
            msg = "response_df is None"
            logger.info(f"instance_id: {query.instance_id}, msg: {msg}")
            return EvaluationResult(
                instance_id=query.instance_id,
                entity_acc=entity_acc,
                search_tool_num=search_tool_num,
                visit_tool_num=visit_tool_num,
                msg=msg,
            )

        response_df.columns = [norm_column(col) for col in response_df.columns]
        if set(required_columns) != set(response_df.columns):
            column_map = primary_key_preprocess(
                response_df.columns.tolist(), required_columns, eval_model_config_name
            )
            logger.info(f"column map: {column_map}")
            logger.info(f"befor mapping: {response_df.columns}")
            response_df.rename(columns=column_map, inplace=True)
            logger.info(f"after mapping: {response_df.columns}")

        if set(required_columns) != set(response_df.columns):
            msg = f"required_columns {required_columns} != response_df {response_df.columns}"
            logger.info(f"instance_id: {query.instance_id}, msg: {msg}")
            return EvaluationResult(
                instance_id=query.instance_id,
                msg=msg,
                entity_acc=entity_acc,
                search_tool_num=search_tool_num,
                visit_tool_num=visit_tool_num,
            )
        # evaluation
        else:
            # preprocess
            for col in required_columns:
                try:
                    answer_type = answer_df[col].dtype
                    response_type = response_df[col].dtype
                except Exception:
                    answer_type = None
                    response_type = None
                answer_df[col] = answer_df[col].astype(str)
                if (response_type is float and answer_type is int) or (
                    response_type is int and answer_type is float
                ):
                    if response_type is int:
                        response_df[col] = response_df[col].astype(float)
                    elif answer_type is int:
                        answer_df[col] = answer_df[col].astype(float)
                response_df[col] = response_df[col].astype(str)
            response_df.drop_duplicates(subset=unique_columns, inplace=True)
            answer_df.drop_duplicates(subset=unique_columns, inplace=True)

            for col in unique_columns:
                item = query.evaluation["eval_pipeline"].get(col, None)
                if item is None:
                    continue
                metric_func_name_list = item.get("metric", [])
                if (
                    "llm_judge" in metric_func_name_list
                    or "exact_match" in metric_func_name_list
                ):
                    primary_key_map = primary_key_preprocess(
                        response_df[col].tolist(),
                        answer_df[col].tolist(),
                        eval_model_config_name,
                    )
                    logger.info(f"col: {col}, primary_key_map {primary_key_map}")
                    response_df[col + "_before_map"] = response_df[col]
                    response_df[col] = response_df[col].apply(
                        lambda x: primary_key_map.get(x, x)
                    )

            for col, item in query.evaluation["eval_pipeline"].items():
                preprocess_func_name_list = item.get("preprocess", [])
                for preprocess_func_name in preprocess_func_name_list:
                    response_df[col] = response_df[col].apply(
                        lambda x: preprocess_call(x, preprocess_func_name)
                    )
                    answer_df[col] = answer_df[col].apply(
                        lambda x: preprocess_call(x, preprocess_func_name)
                    )

            temp_score = 0.0
            if answer_df.shape == response_df.shape:
                gt_sorted = answer_df.sort_values(by=required_columns).reset_index(
                    drop=True
                )
                pred_sorted = response_df.sort_values(by=required_columns).reset_index(
                    drop=True
                )
                if gt_sorted.equals(pred_sorted):
                    temp_score = 1.0
            score = temp_score
            df_inner = pd.merge(
                answer_df,
                response_df,
                on=unique_columns,
                how="inner",
                suffixes=("_query", "_response"),
            )

            answer_df_outer = deepcopy(answer_df)
            answer_df_outer["exist_flag_gt"] = 1
            response_df_outer = deepcopy(response_df)
            response_df_outer["exist_flag_response"] = 1

            df_outer = pd.merge(
                answer_df_outer,
                response_df_outer,
                on=unique_columns,
                how="outer",
                suffixes=("_query", "_response"),
            )
            df_outer_wo_inner = df_outer[
                df_outer["exist_flag_gt"].isna()
                | df_outer["exist_flag_response"].isna()
            ]

            logger.info(
                f"{query.instance_id}, df_inner shape: {df_inner.shape}, "
                f"answer_df shape: {answer_df.shape}, response_df shape: {response_df.shape}"
            )

            # metric
            df_inner_score = pd.DataFrame(index=df_inner.index)
            df_inner_msg = pd.DataFrame(index=df_inner.index)

            for col in required_columns:
                if col in unique_columns:
                    df_inner_score[f"{col}_exact_match"] = 1.0
                    df_inner_msg[f"{col}_exact_match_eval_msg"] = "key_match"
                    continue

                item = query.evaluation["eval_pipeline"][col]
                metric_func_name_list = item.get("metric", [])
                criterion = item.get("criterion")
                for metric_func_name in metric_func_name_list:
                    if metric_func_name != "llm_judge":
                        metric_info_series = df_inner.apply(
                            lambda x: metric_call(
                                x[col + "_response"],
                                x[col + "_query"],
                                criterion,
                                metric_func_name,
                            ),
                            axis=1,
                        )
                    else:
                        score_list, msg_list = llm_judge_column(
                            df_inner[col + "_response"].tolist(),
                            df_inner[col + "_query"].tolist(),
                            criterion,
                            eval_model_config_name,
                        )
                        metric_info_series = pd.Series(
                            zip(score_list, msg_list),
                            index=df_inner.index,
                        )
                    df_inner_score[f"{col}_{metric_func_name}"] = (
                        metric_info_series.apply(lambda x: x[0])
                    )
                    df_inner_msg[f"{col}_{metric_func_name}_eval_msg"] = (
                        metric_info_series.apply(lambda x: x[1])
                    )
                    logger.debug(
                        f"{query.instance_id}, col {col}, metric_func_name {metric_func_name}, score {df_inner_score[f'{col}_{metric_func_name}'].tolist()}"
                    )

            if result_save_path is not None:
                result_df = pd.concat([df_inner, df_inner_score, df_inner_msg], axis=1)
                result_df = pd.concat([result_df, df_outer_wo_inner])
                result_columns = result_df.columns.tolist()
                key_cols = (
                    unique_columns
                    + [col + "_before_map" for col in unique_columns]
                    + ["exist_flag_gt", "exist_flag_response"]
                )

                cols1 = sorted([col for col in result_columns if col in key_cols])
                cols2 = sorted([col for col in result_columns if col not in key_cols])
                result_df = result_df[cols1 + cols2]
                result_df.to_csv(result_save_path, index=False)  # pyright: ignore[reportAttributeAccessIssue]

            row_scores = df_inner_score.min(axis=1)
            tp_by_row = row_scores.sum()
            tp_by_item = df_inner_score.sum().sum()

            num_pred_rows = len(response_df)
            num_gt_rows = len(answer_df)
            num_pred_items = num_pred_rows * len(required_columns)
            num_gt_items = num_gt_rows * len(required_columns)

            precision_by_row = tp_by_row / num_pred_rows if num_pred_rows > 0 else 0.0
            recall_by_row = tp_by_row / num_gt_rows if num_gt_rows > 0 else 0.0

            precision_by_item = (
                tp_by_item / num_pred_items if num_pred_items > 0 else 0.0
            )
            recall_by_item = tp_by_item / num_gt_items if num_gt_items > 0 else 0.0

            def calc_f1(precision, recall):
                epsilon = 1e-9
                return (
                    (2 * precision * recall / (precision + recall))
                    if (precision + recall > epsilon)
                    else 0.0
                )

            f1_by_row = calc_f1(precision_by_row, recall_by_row)
            f1_by_item = calc_f1(precision_by_item, recall_by_item)

            # === 新增 unique_columns 的 column-f1 计算 ===
            # 只有当所有 unique_columns 都正确才算 TP
            unique_col_score_cols = [f"{col}_exact_match" for col in unique_columns]
            # 取每行 unique_columns 的最小值，只有全对才为1
            unique_col_row_scores = df_inner_score[unique_col_score_cols].min(axis=1)
            tp_by_unique_col = unique_col_row_scores.sum()
            num_pred_unique_col = len(response_df)
            num_gt_unique_col = len(answer_df)
            precision_by_unique_col = tp_by_unique_col / num_pred_unique_col if num_pred_unique_col > 0 else 0.0
            recall_by_unique_col = tp_by_unique_col / num_gt_unique_col if num_gt_unique_col > 0 else 0.0
            f1_by_unique_col = calc_f1(precision_by_unique_col, recall_by_unique_col)

            logger.info(
                f"{query.instance_id}, P/R/F1 by unique_columns: {precision_by_unique_col:.4f}/{recall_by_unique_col:.4f}/{f1_by_unique_col:.4f}"
            )
            logger.info(
                f"{query.instance_id}, P/R/F1 by row: {precision_by_row:.4f}/{recall_by_row:.4f}/{f1_by_row:.4f}"
            )
            logger.info(
                f"{query.instance_id}, P/R/F1 by item: {precision_by_item:.4f}/{recall_by_item:.4f}/{f1_by_item:.4f}"
            )

            msg = df_inner_score.to_string()
            if (
                precision_by_item == recall_by_item == f1_by_item == 1.0
                and precision_by_row == recall_by_row == f1_by_row == 1.0
            ):
                msg += "\nAll items match perfectly."
                score = 1

            logger.info(f"{query.instance_id}, final table score: {score}")

    except Exception:
        logger.error(f"evaluator error: \n{traceback.format_exc()}")
        return EvaluationResult(
            instance_id=query.instance_id,
            entity_acc=entity_acc,
            search_tool_num=search_tool_num,
            visit_tool_num=visit_tool_num,
            msg=f"evaluator error: \n{traceback.format_exc()}",
        )

    return EvaluationResult(
        entity_acc=entity_acc,
        instance_id=query.instance_id,
        score=score,
        precision_by_row=precision_by_row,
        recall_by_row=recall_by_row,
        f1_by_row=f1_by_row,
        precision_by_item=precision_by_item,
        recall_by_item=recall_by_item,
        f1_by_item=f1_by_item,
        column_precision=precision_by_unique_col,
        column_recall=recall_by_unique_col,
        column_f1=f1_by_unique_col,
        msg=msg,
    )


def evaluatation_consistency(
    query: WideSearchQuery, auto_result_path: str, human_result_path: str
):
    unique_columns = query.evaluation["unique_columns"]
    assert unique_columns, "unique_columns must be set in evaluation"
    df_auto = pd.read_csv(auto_result_path)
    df_human = pd.read_csv(human_result_path)
    if set(df_auto.columns) != set(df_human.columns):
        logger.warning(
            f"auto: {set(df_auto.columns) - set(df_human.columns)}, human: {set(df_human.columns) - set(df_auto.columns)}"
        )

    score_column_suffix = [
        "_exact_match",
        "_number_near",
        "_date_near",
        "_llm_judge",
        "_near_match",
    ]

    score_columns_trans = []
    for col in df_auto.columns:
        for suffix in score_column_suffix:
            if col.endswith(suffix):
                df_auto[col.replace(suffix, "_eval_score")] = (
                    df_auto[col].fillna(0).astype(int)
                )
                score_columns_trans.append(col.replace(suffix, "_eval_score"))
                break
    for col in df_human.columns:
        for suffix in score_column_suffix:
            if col.endswith(suffix):
                df_human[col.replace(suffix, "_eval_score")] = (
                    df_human[col].fillna(0).astype(int)
                )
                break

    df_auto.drop_duplicates(subset=unique_columns, inplace=True)
    df_auto["auto_result_flag"] = 1
    df_human.drop_duplicates(subset=unique_columns, inplace=True)
    df_human["human_result_flag"] = 1

    df = pd.merge(
        df_auto, df_human, on=unique_columns, how="outer", suffixes=("_auto", "_human")
    )
    logger.info(
        df[df["human_result_flag"].isnull() | df["auto_result_flag"].isnull()][
            unique_columns
        ]
    )
    logger.info(
        f"instance_id: {query.instance_id}, len_df_auto: {len(df_auto)}, len_df_human: {len(df_human)}, len_df: {len(df)}"
    )
    consistency_map = {}
    for col in score_columns_trans:
        df[col + "_auto"] = df[col + "_auto"].fillna(0)
        df[col + "_human"] = df[col + "_human"].fillna(0)
        df_consistency = df[col + "_auto"] == df[col + "_human"]
        consistency_map[col] = float(df_consistency.mean())
        logger.info(
            f"{query.instance_id}, {col}, {df[unique_columns][~df_consistency].to_dict(orient='records')}"  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue]
        )  # pyright: ignore[reportCallIssue]
    consistency_map["mean"] = float(np.mean(list(consistency_map.values())))
    return consistency_map
