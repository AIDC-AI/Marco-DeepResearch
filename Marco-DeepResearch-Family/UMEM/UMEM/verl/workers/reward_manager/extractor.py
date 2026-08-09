from collections import defaultdict
import asyncio
import re

import torch
from verl import DataProto
from verl.workers.reward_manager import register
from umem.extractor.utils import is_prediction_correct
from umem.llm_agent.async_llm_client import AsyncLLMClient
from umem.llm_agent.executor_prompt import build_executor_prompt

@register("extractor")
class ExecutorRewardManager:
    """Reward manager for extractor outputs and memory updates."""

    def __init__(
        self,
        tokenizer,
        num_examine=0,
        compute_score=None,
        reward_fn_key="data_source",
        config=None,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.cfg = config
        reward_cfg = getattr(config, "extractor", None) or {}

        self.w_format = reward_cfg.get("w_format", 1.0)
        self.w_quality = reward_cfg.get("w_quality", 1.0)

        if config.executor.api_code in ["qwen3-235b-a22b"]:
            executor_kwargs = {
                "temperature": config.executor.temperature,
                "max_tokens": config.executor.max_tokens,
                "max_input_tokens": config.executor.max_input_tokens,
                "enable_thinking": config.executor.enable_thinking,
            }
        elif config.executor.api_code in ["executor_llm"]:
            executor_kwargs = {
                "temperature": config.executor.temperature,
                "top_p": config.executor.top_p,
                "top_k": config.executor.top_k,
                "MinP": config.executor.MinP,
                "max_tokens": config.executor.max_tokens,
                "max_model_len": config.executor.max_input_tokens,
                "enable_thinking": config.executor.enable_thinking,
            }
        else:
            executor_kwargs = {
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
            **executor_kwargs,
        )
        self.retriever_wg = None
        self.metric = {}

    def bind_worker_group(self, worker_group):
        self.retriever_wg = worker_group

    def __call__(self, data: DataProto, return_dict=False):
        valid_resp_len_list = [0] * len(data)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra = defaultdict(list)
        items_for_quality = []
        value_lists_for_quality = []
        key_lists_for_quality = []
        quality_item_indices = []

        for i in range(len(data)):
            item = data[i]
            prompt_len = item.batch["prompts"].shape[-1]
            attn_mask = item.batch["attention_mask"]
            response_ids = item.batch["responses"]
            valid_resp_len = int(attn_mask[prompt_len:].sum().item()) if torch.is_tensor(attn_mask) else int(attn_mask[prompt_len:].sum())
            valid_resp_ids = response_ids[:valid_resp_len]
            response_str = self.tokenizer.decode(valid_resp_ids, skip_special_tokens=False)
            valid_resp_len_list[i] = valid_resp_len

            q_text = item.non_tensor_batch["extra_info"]["question"]
            key_str, value_list, operation_str, is_valid = self.extract_xml_experience(response_str, question_str=q_text)
            r_format = 1.0 if is_valid else 0.0

            reward_extra["format_reward"].append(r_format)
            reward_extra["uid"].append(item.non_tensor_batch["uid"])
            reward_extra["value_list"].append([])
            reward_extra["key_str"].append("")
            reward_extra["operation_list"].append("ADD")
            reward_extra["retrieval_reward"].append(0.0)
            reward_extra["quality_reward"].append(0.0)
            reward_extra["final_reward"].append(0.0)
            reward_extra["valid_mask"].append(True)

            if not is_valid:
                if valid_resp_len > 0:
                    reward_tensor[i, valid_resp_len - 1] = 0.0
                continue

            reward_extra["value_list"][i] = value_list
            reward_extra["key_str"][i] = key_str
            reward_extra["operation_list"][i] = operation_str
            reward_extra["final_reward"][i] = self.w_format * r_format
            reward_tensor[i, valid_resp_len - 1] += self.w_format * r_format

            items_for_quality.append(item)
            value_lists_for_quality.append(value_list)
            key_lists_for_quality.append(key_str)
            quality_item_indices.append(i)

        quality_scores = []
        second_scores_list = []
        executor_parse_success_count = 0
        valid_mask_list = []
        if items_for_quality:
            quality_scores, second_scores_list, executor_parse_success_count, valid_mask_list = self._quality_reward(
                items_for_quality,
                value_lists_for_quality,
                key_lists_for_quality,
            )

        for quality_ptr, i in enumerate(quality_item_indices):
            is_valid_api = valid_mask_list[quality_ptr]
            reward_extra["valid_mask"][i] = is_valid_api

            if not is_valid_api:
                continue

            r_quality = quality_scores[quality_ptr]
            reward_extra["quality_reward"][i] = r_quality

            valid_resp_len = valid_resp_len_list[i]
            if r_quality != 0.0 and valid_resp_len > 0:
                reward_tensor[i, valid_resp_len - 1] += r_quality * self.w_quality

            final_reward = self.w_format * reward_extra["format_reward"][i] + self.w_quality * r_quality
            reward_extra["final_reward"][i] = final_reward

        format_rewards = reward_extra["format_reward"]
        n_total = len(format_rewards)
        n_valid_format = sum(format_rewards)
        quality_rewards = reward_extra["quality_reward"]

        self.metric["extractor/format_rate"] = sum(format_rewards) / n_total if n_total > 0 else 0.0
        self.metric["extractor/quality_reward_mean"] = sum(quality_rewards) / n_total if n_total > 0 else 0.0
        self.metric["extractor/quality_improvement_rate"] = sum(1 for q in quality_rewards if q > 0.99) / n_total if n_total > 0 else 0.0
        self.metric["extractor/executor_accuracy"] = sum(second_scores_list) / n_valid_format if n_valid_format > 0 else 0.0
        self.metric["extractor/executor_format_error_rate"] = (n_valid_format - executor_parse_success_count) / n_valid_format if n_valid_format > 0 else 0.0
        self.metric["extractor/final_reward_mean"] = sum(reward_extra["final_reward"]) / n_total if n_total > 0 else 0.0

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra,
            }
        return reward_tensor

    def extract_xml_experience(self, response: str, question_str: str = ""):
        """Parse an extractor response into memory key, values, and operation."""
        if not response:
            return None, [], None, False

        exp_matches = list(re.finditer(r"<experience>(.*?)</experience>", response, re.DOTALL | re.IGNORECASE))
        if not exp_matches:
            return None, [], None, False

        exp_match = exp_matches[-1]
        content = exp_match.group(1)
        content_start = exp_match.start(1)
        content_end = exp_match.end(1)

        value_open_count = content.count("<value>")
        value_close_count = content.count("</value>")
        if value_open_count == 0 or value_open_count != value_close_count:
            return None, [], None, False
        if content.count("<operation>") != 1 or content.count("</operation>") != 1:
            return None, [], None, False

        op_match = re.search(r"<operation>(.*?)</operation>", response[content_start:content_end], re.DOTALL | re.IGNORECASE)
        if not op_match:
            return None, [], None, False

        op_str = op_match.group(1).strip()
        if not op_str:
            return None, [], None, False

        try:
            top_k = self.cfg.retriever.topk
        except AttributeError:
            top_k = 3

        if "UPDATE" in op_str.upper():
            match = re.search(r"(\d+)", op_str)
            if not match or int(match.group(1)) >= top_k:
                return None, [], None, False

        value_matches = list(re.finditer(r"<value>(.*?)</value>", response[content_start:content_end], re.DOTALL | re.IGNORECASE))
        if len(value_matches) != value_open_count:
            return None, [], None, False

        value_list = [match.group(1).strip() for match in value_matches if match.group(1).strip()]
        return question_str.strip(), value_list, op_str, True

    def _quality_reward(self, items, value_lists, key_lists):
        """Score extracted memory values with the fixed value-reward objective."""
        prompts = []
        questions = []
        meta_list = []
        batch_counts = []

        for item, vlist, kstr in zip(items, value_lists, key_lists):
            extra = item.non_tensor_batch["extra_info"]
            sim_qs = extra.get("similar_queries", [])
            sim_cs = extra.get("similar_choices", [])
            sim_gts = extra.get("similar_gt", [])
            targets = []

            min_len = min(len(sim_qs), len(sim_cs), len(sim_gts))
            if min_len > 0:
                for idx in range(min_len):
                    targets.append((sim_qs[idx], sim_cs[idx], sim_gts[idx]))
            else:
                targets.append((
                    extra["question"],
                    extra.get("choices", []),
                    item.non_tensor_batch["reward_model"]["ground_truth"],
                ))

            item_meta = []
            for target_question, target_choices, target_gt in targets:
                prompts.append(build_executor_prompt(target_question, target_choices, []))
                questions.append(f"[Base] {target_question}")

                memory = [{"key": kstr, "value": vlist}]
                prompts.append(build_executor_prompt(target_question, target_choices, memory))
                questions.append(f"[Test] {target_question}")
                item_meta.append((target_gt, target_choices))

            meta_list.append(item_meta)
            batch_counts.append(len(item_meta))

        async def _run_batch():
            tasks = [self.client.call(system_prompt="", query=prompt, idx=idx) for idx, prompt in enumerate(prompts)]
            return await asyncio.gather(*tasks, return_exceptions=True)

        outputs = asyncio.run(_run_batch())
        second_scores = []
        quality_rewards = []
        valid_mask_list = []
        len_diffs = []
        parse_success_count = 0
        output_ptr = 0

        for item_metas, count in zip(meta_list, batch_counts):
            total_gain = 0.0
            valid_comparisons = 0

            for local_idx in range(count):
                out_base = outputs[output_ptr]
                out_test = outputs[output_ptr + 1]
                output_ptr += 2

                ground_truth, choices = item_metas[local_idx]
                if not (out_base and isinstance(out_base, str) and out_test and isinstance(out_test, str)):
                    second_scores.extend([0.0, 0.0])
                    continue

                base_score, _ = is_prediction_correct(out_base, ground_truth, choices)
                test_score, is_parsed = is_prediction_correct(out_test, ground_truth, choices)
                if is_parsed:
                    parse_success_count += 1

                second_scores.extend([base_score, test_score])
                base_len = len(self.tokenizer.encode(out_base, add_special_tokens=False))
                test_len = len(self.tokenizer.encode(out_test, add_special_tokens=False))
                len_diffs.append(base_len - test_len)

                if test_score < 0.99:
                    gain = 0.0
                elif base_score < 0.01:
                    gain = 1.0
                elif test_len >= base_len:
                    gain = 0.0
                else:
                    gain = (base_len - test_len) / (base_len + 1e-6)
                    gain = max(0.0, min(1.0, gain))

                total_gain += gain
                valid_comparisons += 1

            valid_mask_list.append(valid_comparisons > 0)
            quality_rewards.append(total_gain / valid_comparisons if valid_comparisons > 0 else 0.0)

        if len_diffs:
            self.metric["extractor/length_difference"] = sum(len_diffs) / len(len_diffs)

        return quality_rewards, second_scores[1::2], parse_success_count, valid_mask_list

@register("val_extractor")
class ExecutorValidManager:
    """The executor validation reward manager."""
    def __init__(
        self,
        tokenizer,
        num_examine=1,
        compute_score=None,
        reward_fn_key="data_source",
        config=None,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.cfg = config
        self.metric = {}

        if config.executor.api_code in ["qwen3-235b-a22b"]:
            executor_kwags = {
                "temperature": 0.0,
                "do_sample": False,
                "max_tokens": config.executor.max_tokens,
                "max_input_tokens": config.executor.max_input_tokens,
                "enable_thinking": config.executor.enable_thinking,
            }
        elif config.executor.api_code in ["executor_llm"]:
            executor_kwags = {
                "temperature": 0.0,
                "do_sample": False,
                "max_tokens": config.executor.max_tokens,
                "max_model_len": config.executor.max_input_tokens,
                "enable_thinking": config.executor.enable_thinking,
            }
        else:
            executor_kwags = {
                "temperature": 0.0,
                "do_sample": False,
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

    def __call__(self, data: DataProto, return_dict=False):
        """Compute reward using executor LLM validation."""
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        valid_response_lengths = []

        total_score = 0.0
        parse_success_count = 0

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            attn_mask = data_item.batch['attention_mask']

            if torch.is_tensor(attn_mask):
                valid_len = int(attn_mask[prompt_length:].sum().item())
            else:
                valid_len = int(attn_mask[prompt_length:].sum())
            valid_response_lengths.append(valid_len)

            extra = data_item.non_tensor_batch["extra_info"]
            choices = extra.get("choices", [])
            gt = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            traj = data_item.non_tensor_batch["trajectories"]

            score, is_parsed = is_prediction_correct(traj, gt, choices)
            total_score += score
            if is_parsed:
                parse_success_count += 1
            valid_len = valid_response_lengths[i]
            if valid_len > 0:
                reward_tensor[i, valid_len - 1] = score

        n_total = len(data)
        avg_acc = total_score / n_total if n_total > 0 else 0.0
        parse_rate = parse_success_count / n_total if n_total > 0 else 0.0
        self.metric["val-core/executor_accuracy"] = avg_acc
        self.metric["val-core/format_parse_rate"] = parse_rate

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {},
            }

        return reward_tensor