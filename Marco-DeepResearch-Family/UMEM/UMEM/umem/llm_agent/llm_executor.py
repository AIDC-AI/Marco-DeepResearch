import asyncio
import numpy as np
import torch
import re
import time
from typing import List, Dict, Any
from collections import defaultdict

from tensordict import TensorDict
from verl import DataProto
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from umem.extractor.utils import is_prediction_correct

from .async_llm_client import AsyncLLMClient
from .executor_prompt import build_executor_prompt
from .extractor_prompt import build_extractor_prompt

class LLMExecutor:
    """Run retrieval, executor generation, and extractor-prompt construction."""

    def __init__(self, tokenizer, retriever_group, config):
        self.tokenizer = tokenizer
        self.retriever = retriever_group
        self.cfg = config

        self.repeat_n = config.executor.repeat_n
        self.executor_temperature = config.executor.temperature
        self.client = None

        self.metric = {}

        if config.executor.api_code in ["qwen3-235b-a22b"] :
            executor_kwags = {
                "temperature": config.executor.temperature,
                "max_tokens": config.executor.max_tokens,
                "max_input_tokens": config.executor.max_input_tokens,
                "enable_thinking": config.executor.enable_thinking,
            }
        elif config.executor.api_code in ["executor_llm"] :
            executor_kwags = {
                "temperature": config.executor.temperature,
                "top_p": config.executor.top_p,
                "top_k": config.executor.top_k,
                "MinP": config.executor.MinP,
                "max_tokens": config.executor.max_tokens,
                "max_model_len": config.executor.max_input_tokens,
                "enable_thinking": config.executor.enable_thinking,
            }
        else:
            executor_kwags = {
                "temperature": config.executor.temperature,
                "max_tokens": config.executor.max_tokens,
                "max_input_tokens": config.executor.max_input_tokens,
            }

        self.client = AsyncLLMClient(
            url=config.executor.api_url,
            api_code=config.executor.api_code,
            ak=config.executor.ak,
            max_concurrency=config.executor.max_concurrency,
            timeout=config.executor.timeout,
            max_retries=config.executor.max_retries,
            **executor_kwags
        )

    def _expand_batch(self, batch: DataProto, is_validate=False):

        extra_infos = batch.non_tensor_batch["extra_info"]
        questions = [item["question"] for item in extra_infos]
        B = len(questions)

        executor_ids = np.arange(B, dtype=np.int64)

        for i, item in enumerate(extra_infos):
            item["executor_id"] = executor_ids[i]

        if is_validate:
            expanded_batch = batch.repeat(1, interleave=True)
        else:
            expanded_batch = batch.repeat(self.repeat_n, interleave=True)
        return expanded_batch

    def _search_memories(self, batch):

        extra_infos = batch.non_tensor_batch["extra_info"]
        questions = [item["question"] for item in extra_infos]

        dp = DataProto.from_dict(non_tensors={"queries": questions})

        start_time = time.time()
        out = self.retriever.search(dp)
        end_time = time.time()
        self.metric["retriever/time"] = end_time - start_time

        if hasattr(out, "meta_info") and out.meta_info:

            for k, v in out.meta_info.items():

                self.metric[k] = v
        topk_indices_tensor = out.batch["topk_indices"]

        retrieved_memory_indices = topk_indices_tensor.detach().cpu().numpy().tolist()

        retrieved_memory_items = out.non_tensor_batch["topk_items"]

        for i, item in enumerate(extra_infos):
            item["retrieved_memory_indices"] = retrieved_memory_indices[i]
            item["retrieved_memory_items"] = retrieved_memory_items[i]

        return batch

    def _build_executor_prompts(self, questions, choices, retrieved_memory_items):
        return [
            build_executor_prompt(q, c, m)
            for q, c, m in zip(questions, choices, retrieved_memory_items)
        ]

    async def _call_executor_llm_batch(self, prompts):
        tasks = [self.client.call(system_prompt="", query=p, idx=i)
                 for i, p in enumerate(prompts)]
        return await asyncio.gather(*tasks)


    def _filter_by_executor_id(
        self,
        questions,
        choices,
        trajectories,
        ground_truth,
        executor_ids,
        retrieved_memory_indices,
        retrieved_memory_items,
        trace_ids,
        reward_models,
        extra_infos,
    ):
        """Keep samples that should be sent to the extractor stage."""
        n_total = len(questions)

        if reward_models is not None:
            if hasattr(reward_models, 'tolist'):
                reward_models_list = reward_models.tolist()
            else:
                reward_models_list = list(reward_models)
        else:
            reward_models_list = None

        if trace_ids is not None:
            if hasattr(trace_ids, 'tolist'):
                trace_ids_list = trace_ids.tolist()
            else:
                trace_ids_list = list(trace_ids)
        else:
            trace_ids_list = None

        safe_trace_ids = trace_ids_list if trace_ids_list is not None else [None] * n_total
        safe_reward_models = reward_models_list if reward_models_list is not None else [{"style": "rule"}] * n_total

        first_executor_score = []
        parse_failures = 0

        for traj, gt, c in zip(trajectories, ground_truth, choices):
            score, parsed = is_prediction_correct(traj, gt, c)
            first_executor_score.append(score)
            if not parsed:
                parse_failures += 1

        if self.repeat_n == 1:

            valid_indices = []
            for i, traj in enumerate(trajectories):

                t_str = str(traj).strip() if traj is not None else ""

                if t_str and "<answer>" in t_str and "</answer>" in t_str:
                    valid_indices.append(i)

            return (
                [questions[i] for i in valid_indices],
                [choices[i] for i in valid_indices],
                [trajectories[i] for i in valid_indices],
                [ground_truth[i] for i in valid_indices],
                [executor_ids[i] for i in valid_indices],
                [first_executor_score[i] for i in valid_indices],
                [retrieved_memory_indices[i] for i in valid_indices],
                [retrieved_memory_items[i] for i in valid_indices],
                [safe_trace_ids[i] for i in valid_indices],
                [safe_reward_models[i] for i in valid_indices],
                [first_executor_score[i] for i in valid_indices],
                first_executor_score,
                [extra_infos[i] for i in valid_indices]
            )

        if n_total == 0:
            return (questions, choices, trajectories, ground_truth, executor_ids, [],
                    retrieved_memory_indices, retrieved_memory_items, safe_trace_ids, safe_reward_models, [], [])

        group_all_correct = {}
        group_correct_count = defaultdict(int)
        group_total_count = defaultdict(int)

        for i, (eid, score) in enumerate(zip(executor_ids, first_executor_score)):

            group_total_count[eid] += 1
            if score > 0.999:
                group_correct_count[eid] += 1

            if eid not in group_all_correct:
                group_all_correct[eid] = True

            if score < 0.999:
                group_all_correct[eid] = False

        new_data = {k: [] for k in ["q", "c", "traj", "gt", "eid", "fc", "mem_idx", "mem_item", "tid", "rm", "pr", "raw_extra"]}
        kept_groups = set()
        total_groups = len(group_all_correct)

        all_pass_rates_stats = []
        for eid in group_total_count:
            pr = group_correct_count[eid] / group_total_count[eid] if group_total_count[eid] > 0 else 0.0
            all_pass_rates_stats.append(pr)

        for i in range(n_total):
            eid = executor_ids[i]

            if group_all_correct[eid]:
                continue

            kept_groups.add(eid)
            current_pass_rate = group_correct_count[eid] / group_total_count[eid]

            new_data["q"].append(questions[i])
            new_data["c"].append(choices[i])
            new_data["traj"].append(trajectories[i])
            new_data["gt"].append(ground_truth[i])
            new_data["eid"].append(eid)
            new_data["fc"].append(first_executor_score[i])
            new_data["mem_idx"].append(retrieved_memory_indices[i])
            new_data["mem_item"].append(retrieved_memory_items[i])
            new_data["tid"].append(safe_trace_ids[i])
            new_data["rm"].append(safe_reward_models[i])
            new_data["pr"].append(current_pass_rate)
            new_data["raw_extra"].append(extra_infos[i])

        n_kept = len(kept_groups)
        filtration_rate = (total_groups - n_kept) / total_groups if total_groups > 0 else 0.0
        avg_score = sum(first_executor_score) / len(first_executor_score) if len(first_executor_score) > 0 else 0.0
        parse_fail_rate = parse_failures / n_total if n_total > 0 else 0.0
        avg_pass_rate = sum(all_pass_rates_stats) / len(all_pass_rates_stats) if len(all_pass_rates_stats) > 0 else 0.0

        self.metric["executor/num_total_queries"] = total_groups
        self.metric["executor/num_valid_queries"] = n_kept
        self.metric["executor/filter_out_rate"] = filtration_rate
        self.metric["executor/format_error_rate"] = parse_fail_rate
        self.metric["executor/current_rollout_acc"] = avg_score
        self.metric["executor/avg_pass_rate"] = avg_pass_rate

        return (
            new_data["q"],
            new_data["c"],
            new_data["traj"],
            new_data["gt"],
            new_data["eid"],
            new_data["fc"],
            new_data["mem_idx"],
            new_data["mem_item"],
            new_data["tid"],
            new_data["rm"],
            new_data["pr"],
            first_executor_score,
            new_data["raw_extra"]
        )

    def _build_extractor_prompts(self, questions, choices, trajectories, gt,retrieved_memory_items):
        return [
            build_extractor_prompt(
                question=q,
                choices=ch,
                trajectory=traj,
                ground_truth_answer=g,
                tokenizer=self.tokenizer,
                retrieved_memory_item=rm,
            )
            for q, ch, traj, g, rm in zip(questions, choices, trajectories, gt,retrieved_memory_items)
        ]

    async def _async_tokenize_prompts_batch(self, prompts):
        """Tokenize extractor prompts concurrently in the default executor."""
        loop = asyncio.get_event_loop()

        tasks = [
            loop.run_in_executor(None, self._tokenize_prompt, p)
            for p in prompts
        ]

        return await asyncio.gather(*tasks)

    def _tokenize_prompt(self, prompt: str):

        max_prompt_length = self.cfg.data.get("max_prompt_length", 1024)
        truncation_strategy = self.cfg.data.get("truncation", "error")
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        initial_input_ids = encoded["input_ids"]
        initial_attention_mask = encoded["attention_mask"]

        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > max_prompt_length:
            if truncation_strategy == "left":
                raw_prompt_ids = raw_prompt_ids[-max_prompt_length:]
            elif truncation_strategy == "right":
                raw_prompt_ids = raw_prompt_ids[:max_prompt_length]
            elif truncation_strategy == "middle":
                left_half = max_prompt_length // 2
                right_half = max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif truncation_strategy == "error":
                pass

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=initial_input_ids,
            attention_mask=initial_attention_mask,
            max_length=max_prompt_length,
            pad_token_id=pad_token_id,
            left_pad=True,
            truncation=truncation_strategy,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        return raw_prompt_ids, input_ids, attention_mask, position_ids

    async def close(self):
        """Close the executor LLM HTTP client."""
        if self.client:
            await self.client.close()
            self.client = None

    async def run(self, batch: DataProto):
        start_time = time.time()

        batch = self._search_memories(batch)

        is_validate = batch.meta_info.get("validate", False)

        if is_validate:
            val_cfg = {
                "temperature": 0.0,
            }
            self.client.generation_kwargs.update(val_cfg)
            batch = self._expand_batch(batch, is_validate)
        else:
            cfg = {
                "temperature": self.executor_temperature,
            }
            self.client.generation_kwargs.update(cfg)

            batch = self._expand_batch(batch)

        extra_infos = batch.non_tensor_batch["extra_info"]
        reward_models_raw = batch.non_tensor_batch["reward_model"]
        trace_ids = batch.non_tensor_batch.get("trace_id", None)

        questions_raw = [item["question"] for item in extra_infos]

        choices = [item.get("choices", []) for item in extra_infos]

        ground_truth_raw = [item["ground_truth"] for item in reward_models_raw]
        executor_ids = [item["executor_id"] for item in extra_infos]
        retrieved_memory_indices = [item["retrieved_memory_indices"] for item in extra_infos]
        retrieved_memory_items = [item["retrieved_memory_items"] for item in extra_infos]

        exe_prompts = self._build_executor_prompts(questions_raw, choices, retrieved_memory_items)

        trajectories_raw = await self._call_executor_llm_batch(exe_prompts)

        if is_validate:
            raw_list, id_list, mask_list, pos_list = [], [], [], []
            for p in exe_prompts:
                raw_ids, in_ids, attn, pos = self._tokenize_prompt(p)

                id_list.append(in_ids)
                mask_list.append(attn)

            device = batch.batch["input_ids"].device

            resp_out = self.tokenizer(trajectories_raw, return_tensors="pt", padding=True, add_special_tokens=False)

            response_ids = resp_out.input_ids.to(device)
            response_mask = resp_out.attention_mask.to(device)

            prompt_ids = torch.cat(id_list, dim=0)
            prompt_mask = torch.cat(mask_list, dim=0)

            input_ids = torch.cat([prompt_ids, response_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, response_mask], dim=1)

            seq_len = input_ids.shape[1]
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(input_ids.shape)

            non_tensor_batch = batch.non_tensor_batch
            non_tensor_batch["trajectories"] = np.array(trajectories_raw, dtype=object)

            batch_size = input_ids.shape[0]
            new_batch = TensorDict(
                {
                    "prompts": prompt_ids,
                    "responses": response_ids,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                },
                batch_size=batch_size,
            )

            return DataProto(batch=new_batch, non_tensor_batch=non_tensor_batch)

        (questions, choices, trajectories, ground_truth, executor_ids, first_executor_score,
        retrieved_memory_indices, retrieved_memory_items, final_trace_ids, reward_models, pass_rates, first_executor_score_raw, filtered_extra_infos) =\
            self._filter_by_executor_id(
                questions_raw, choices, trajectories_raw, ground_truth_raw, executor_ids,
                retrieved_memory_indices, retrieved_memory_items, trace_ids, reward_models_raw, extra_infos
            )

        first_executor_lens = [
        len(self.tokenizer.encode(str(t), add_special_tokens=False))
        for t in trajectories
        ]

        if len(questions) == 0:

            p_len = self.cfg.data.get("max_prompt_length", 1024)
            r_len = self.cfg.data.get("max_response_length", 1024)
            seq_len = p_len + r_len

            empty_prompts = torch.empty((0, p_len), dtype=torch.long)
            empty_responses = torch.empty((0, r_len), dtype=torch.long)
            empty_ids = torch.empty((0, seq_len), dtype=torch.long)
            empty_mask = torch.empty((0, seq_len), dtype=torch.long)
            empty_pos = torch.empty((0, seq_len), dtype=torch.long)

            batch = TensorDict(
                {
                    "prompts": empty_prompts,
                    "responses": empty_responses,
                    "input_ids": empty_ids,
                    "attention_mask": empty_mask,
                    "position_ids": empty_pos,
                },
                batch_size=0,
            )

            non_tensor_batch = {
                "uid": np.empty((0,), dtype=object),
                "extra_info": np.empty((0,), dtype=object),
                "trace_id": np.empty((0,), dtype=object),
                "reward_model": np.empty((0,), dtype=object),
            }

            return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

        extractor_prompts = self._build_extractor_prompts(
            questions, choices, trajectories, ground_truth, retrieved_memory_items
        )

        tokenize_results = await self._async_tokenize_prompts_batch(extractor_prompts)

        raw_list, id_list, mask_list, pos_list = zip(*tokenize_results)

        id_tensor = torch.cat(id_list, dim=0)
        mask_tensor = torch.cat(mask_list, dim=0)
        pos_tensor = torch.cat(pos_list, dim=0)

        raw_prompt_ids = np.array(raw_list, dtype=object)
        first_executor_score = np.array(first_executor_score, dtype=object)
        executor_ids = np.array(executor_ids, dtype=object)

        extra_info_list = []
        reward_model_list = []

        for i in range(len(executor_ids)):

            base_info = filtered_extra_infos[i]

            new_fields = {
                "first_executor_score": first_executor_score[i],
                "executor_ids": executor_ids[i],
                "retrieved_memory_indices": retrieved_memory_indices[i],
                "retrieved_memory_items": retrieved_memory_items[i],
                "question": questions[i],
                "choices": choices[i],
                "pass_rate": pass_rates[i],
                "first_executor_len": first_executor_lens[i]
            }

            merged_info = {**base_info, **new_fields}

            extra_info_list.append(merged_info)

            r_model = reward_models[i] if reward_models else {"style": "rule", "ground_truth": ground_truth[i]}

            r_model["ground_truth"] = ground_truth[i]
            reward_model_list.append(r_model)

        extra_info_arr = np.array(extra_info_list, dtype=object)
        reward_model_arr = np.array(reward_model_list, dtype=object)
        trace_id_arr = np.array(final_trace_ids, dtype=object)

        exec_time = time.time() - start_time
        batch_size = mask_tensor.size(0)

        if batch_size > 0:

            seq_lengths = mask_tensor.sum(dim=1).float()

            total_valid_tokens = seq_lengths.sum().item()
            max_len = seq_lengths.max().item()
            avg_len = total_valid_tokens / batch_size
        else:
            total_valid_tokens = 0
            max_len = 0.0
            avg_len = 0.0

        self.metric["executor/num_gen_samples_for_extractor"] = len(questions)
        self.metric["executor/avg_prompt_length"] = avg_len
        self.metric["executor/max_prompt_length"] = max_len
        self.metric["executor/exec_time"] = exec_time

        gen_batch = DataProto.from_dict(
            tensors={
                "input_ids": id_tensor.long(),
                "attention_mask": mask_tensor.long(),
                "position_ids": pos_tensor.long(),
            },
            non_tensors={
                "raw_prompt_ids": raw_prompt_ids,
                "trace_id": trace_id_arr,
                "extra_info": extra_info_arr,
                "reward_model": reward_model_arr,
            }
        )

        return gen_batch
