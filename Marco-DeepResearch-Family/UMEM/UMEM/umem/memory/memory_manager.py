import os
import json
import uuid
import re
from typing import List, Dict, Any
from collections import defaultdict
import threading
import time
import numpy as np
import torch
from verl import DataProto

from .build_index import build_faiss_index

class MemoryManager:

    def __init__(self, retriever_group, cfg):
        self.retriever = retriever_group
        self.cfg = cfg
        self.memory_max_size = cfg.get("memory_max_size", 10000)
        self.merge_freq = cfg.get("memory_merge_freq", 1)
        self.jsonl_path = cfg.get("memory_jsonl_path","")
        self.index_path = cfg.get("memory_index_path","")
        jsonl_partner_path = os.path.dirname(self.jsonl_path)
        os.makedirs(jsonl_partner_path, exist_ok=True)
        index_path = os.path.dirname(self.index_path)
        os.makedirs(index_path, exist_ok=True)
        self.cur_step = 0
        self.memories = []
        self.id_to_idx = {}

        self.build_index_thread = None
        self.is_building = True
        self.lock = threading.Lock()

        self.load()

    def update_step(self, step: int):
        self.cur_step = step

    def load(self):
        if not os.path.exists(self.jsonl_path):
            self.memories = []
            self.id_to_idx = {}
            return

        mems = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                obj.setdefault("reward_score", 0.0)
                obj.setdefault("usage_count", 0)
                mems.append(obj)

        self.memories = mems
        self.id_to_idx = {m["id"]: i for i, m in enumerate(self.memories)}

    def _create_new_memory(self, uid, key, value, total_reward):
        memory_id = str(uuid.uuid4())
        while memory_id in self.id_to_idx:
            memory_id = str(uuid.uuid4())

        new_memory = {
            "id": memory_id,
            "key": key,
            "value": value,
            "usage_count": 0,
            "reward_score": total_reward,
        }
        self.memories.append(new_memory)
        self.id_to_idx[memory_id] = len(self.memories) - 1

        return {
            "action": "ADD",
            "memory_id": memory_id,
            "uid": uid
        }

    def save(self):
        with self.lock:
            ids = [m["id"] for m in self.memories]
            if len(ids) != len(set(ids)):
                from collections import Counter
                duplicates = [id for id, count in Counter(ids).items() if count > 1]
                raise ValueError(f"Memory IDs are not unique: {duplicates}")

            with open(self.jsonl_path, "w", encoding="utf-8") as f:
                for m in self.memories:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")

    def process(self, batch, reward_extra_infos_dict):
        batch_uids = batch.non_tensor_batch["uid"]
        reward_uids = reward_extra_infos_dict["uid"]
        assert len(batch_uids) == len(reward_uids),\
            f"Batch size mismatch: {len(batch_uids)} vs {len(reward_uids)}"

        uid_groups = defaultdict(list)
        extra_infos = batch.non_tensor_batch["extra_info"]

        for idx in range(len(batch_uids)):
            uid = batch_uids[idx]
            assert uid == reward_uids[idx]

            retrieved_items = extra_infos[idx]["retrieved_memory_items"]

            if retrieved_items:
                seen_ids_in_this_step = set()
                for item in retrieved_items:
                    if isinstance(item, dict) and "id" in item:
                        mem_id = item["id"]
                        if mem_id in self.id_to_idx and mem_id not in seen_ids_in_this_step:
                            mem_idx = self.id_to_idx[mem_id]
                            self.memories[mem_idx]["usage_count"] += 1
                            seen_ids_in_this_step.add(mem_id)

            if retrieved_items is not None and len(retrieved_items) > 0:
                memory_ids = [
                    item["id"]
                    for item in retrieved_items
                    if isinstance(item, dict) and "id" in item
                ]
            else:
                memory_ids = []

            operation_list = reward_extra_infos_dict.get("operation_list", [])
            operation_str = operation_list[idx] if idx < len(operation_list) else "ADD"

            uid_groups[uid].append({
                "batch_idx": idx,
                "retrieval_reward": reward_extra_infos_dict["retrieval_reward"][idx],
                "quality_reward": reward_extra_infos_dict["quality_reward"][idx],
                "key_str": reward_extra_infos_dict["key_str"][idx],
                "value_list": reward_extra_infos_dict["value_list"][idx],
                "operation_str": operation_str,
                "retrieved_memory_ids": memory_ids
            })

        results = []
        for uid, group_items in uid_groups.items():
            result = self._process_group(uid, group_items)
            if result:
                results.append(result)

        self._commit()

        return results

    def _process_group(self, uid: str, group_items: List[Dict[str, Any]]):

        valid_items = []
        for item in group_items:
            has_key = item["key_str"] and str(item["key_str"]).strip()
            has_value = item["value_list"] and len(item["value_list"]) > 0

            is_positive_reward = item["quality_reward"] > 1e-6

            if has_key and has_value and is_positive_reward:
                item["reward_score"] = item["quality_reward"]
                valid_items.append(item)

        if not valid_items:
            return None

        best_new_item = max(valid_items, key=lambda x: x["reward_score"])
        new_memory_key = best_new_item["key_str"]
        new_memory_value = best_new_item["value_list"]
        new_reward_score = best_new_item["reward_score"]
        operation_str = best_new_item["operation_str"]

        retrieved_memory_ids = group_items[0]["retrieved_memory_ids"]

        if "UPDATE" in operation_str.upper():

            match = re.search(r"(\d+)", operation_str)
            if match and retrieved_memory_ids:
                target_idx = int(match.group(1))

                if 0 <= target_idx < len(retrieved_memory_ids):
                    target_memory_id = retrieved_memory_ids[target_idx]

                    if target_memory_id in self.id_to_idx:

                        mem_idx = self.id_to_idx[target_memory_id]
                        updated_memory = {
                            "id": target_memory_id,
                            "key": new_memory_key,
                            "value": new_memory_value,
                            "usage_count": 0,
                            "reward_score": new_reward_score,
                        }
                        self.memories[mem_idx] = updated_memory

                        return {
                            "action": "UPDATE",
                            "memory_id": target_memory_id,
                            "uid": uid
                        }
                    else:
                        pass
                else:
                    pass
            else:
                pass

        return self._create_new_memory(
            uid, new_memory_key, new_memory_value, new_reward_score
        )

    def _prune_memories(self):
        if len(self.memories) <= self.memory_max_size:
            return
        self.memories.sort(key=lambda x: x["usage_count"], reverse=True)
        self.memories = self.memories[:self.memory_max_size]
        self.id_to_idx = {m["id"]: i for i, m in enumerate(self.memories)}

    def _merge_duplicates(self):
        if not self.memories:
            return

        key_map = {}
        merged_memories = []
        merge_count = 0

        for mem in self.memories:
            k = mem["key"]
            if k in key_map:
                existing = key_map[k]
                existing["value"].extend(mem["value"])
                existing["usage_count"] += mem["usage_count"]
                merge_count += 1
            else:
                key_map[k] = mem
                merged_memories.append(mem)

        for mem in merged_memories:
            unique_values = list(set(mem["value"]))
            mem["value"] = unique_values

        self.memories = merged_memories
        self.id_to_idx = {m["id"]: i for i, m in enumerate(self.memories)}

    def _commit(self):
        if self.cur_step > 0 and self.cur_step % self.merge_freq == 0:
            self._merge_duplicates()

        self._prune_memories()

        self.save()
        self.id_to_idx = {m["id"]: i for i, m in enumerate(self.memories)}

        build_faiss_index(self.jsonl_path, self.index_path, retriever=self.retriever)

    def _async_commit(self):
        self.save()
        if self.is_building:
            return
        self.build_index_thread = threading.Thread(target=self._build_index_worker)
        self.build_index_thread.start()

    def _build_index_worker(self):
        self.is_building = True
        try:
            build_faiss_index(self.jsonl_path, self.index_path, retriever=self.retriever)
        except Exception:
            pass
        finally:
            self.is_building = False

    def flush(self):
        self._commit()