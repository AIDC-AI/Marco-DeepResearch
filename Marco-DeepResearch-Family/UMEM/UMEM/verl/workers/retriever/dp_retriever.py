

"""Data-parallel FAISS retriever worker."""

from typing import List, Dict, Any, Optional, Tuple
import os
import json
import logging

import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel

from verl import DataProto
from verl.workers.retriever.base import BaseRetrieverModel

__all__ = ["DataParallelRetriever"]

logger = logging.getLogger(__name__)

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read memory items from a JSONL file."""
    items = []
    if not os.path.exists(path):
        logger.warning(f"[Retriever] memory_jsonl_path not found: {path}")
        return items

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize vectors along the last dimension."""
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norm + eps)

class DataParallelRetriever(BaseRetrieverModel):
    """Embed queries and search a FAISS memory index."""

    def __init__(self, config):
        super().__init__(config=config)

        self.model_path = self.config.get("model_path", None)
        self.use_fp16 = bool(self.config.get("use_fp16", True))
        self.max_length = int(self.config.get("max_length", 512))
        self.batch_size = int(self.config.get("batch_size", 128))
        self.topk = int(self.config.get("topk", 3))
        self.retrieval_threshold = self.config.get("threshold", 0.5)

        self.memory_jsonl_path = self.config.get("memory_jsonl_path", None)
        self.memory_index_path = self.config.get("memory_index_path", None)

        if not self.model_path:
            raise ValueError("retriever.model_path is required.")

        logger.info(f"[Retriever] Loading embedding model from: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)

        self.model.eval().cuda()
        if self.use_fp16:
            self.model = self.model.half()

        self.memory_items: List[Dict[str, Any]] = []
        self.index = None
        self.index_dim = None

        self.last_index_mtime = 0.0

        self.load_index()

    def load_index(self):
        """Load memory items and the FAISS index from disk."""

        if self.memory_jsonl_path and os.path.exists(self.memory_jsonl_path):
            try:
                self.memory_items = _read_jsonl(self.memory_jsonl_path)
                logger.info(f"[Retriever] Loaded {len(self.memory_items)} memory items.")
            except Exception as e:
                logger.error(f"[Retriever] Failed to read jsonl: {e}")
        else:
            self.memory_items = []

        if self.memory_index_path and os.path.exists(self.memory_index_path):

            current_mtime = os.path.getmtime(self.memory_index_path)

            cpu_index = faiss.read_index(self.memory_index_path)

            if bool(self.config.get("faiss_gpu", True)):
                co = faiss.GpuMultipleClonerOptions()
                co.useFloat16 = True
                co.shard = True
                self.index = faiss.index_cpu_to_all_gpus(cpu_index, co=co)
            else:
                self.index = cpu_index

            self.index_dim = self.index.d
            self.last_index_mtime = current_mtime
            logger.info(f"[Retriever] Successfully loaded FAISS index (mtime={current_mtime})")

        else:
            logger.warning("[Retriever] Index file not found.")

        self.index_dim = self.index.d if self.index is not None else None

    def _check_and_refresh_index(self):
        """Reload the FAISS index when the index file changes."""
        if not self.memory_index_path or not os.path.exists(self.memory_index_path):
            return

        try:

            disk_mtime = os.path.getmtime(self.memory_index_path)

            if disk_mtime > self.last_index_mtime + 1.0:
                logger.info(f"[Retriever] Detect index update on disk. Reloading...")
                self.load_index()
        except OSError:

            pass

    @torch.no_grad()
    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode text queries as normalized dense vectors."""
        if isinstance(queries, str):
            queries = [queries]
        elif isinstance(queries, np.ndarray):
            queries = queries.tolist()

        if len(queries) == 0:
            return np.zeros((0, self.index_dim or 1), dtype=np.float32)

        inputs = self.tokenizer(
            queries,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        outputs = self.model(**inputs, return_dict=True)
        emb = outputs.last_hidden_state[:, 0]

        emb = torch.nn.functional.normalize(emb, dim=-1)
        emb = emb.detach().cpu().numpy().astype(np.float32)

        del inputs, outputs
        torch.cuda.empty_cache()
        return emb

    def search_topk(self, query_emb: np.ndarray, topk: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Search the FAISS index and return scores, indices, and metrics."""
        if topk is None:
            topk = self.topk

        if self.index is None or query_emb.shape[0] == 0:

            scores = np.zeros((query_emb.shape[0], topk), dtype=np.float32)
            idxs = -np.ones((query_emb.shape[0], topk), dtype=np.int64)
            return scores, idxs, {}

        query_emb = np.ascontiguousarray(query_emb.astype(np.float32))
        scores, idxs = self.index.search(query_emb, k=topk)

        stats = {}
        if scores.size > 0:

            top1_scores = scores[:, 0]
            stats["retriever/raw_top1_max"] = float(np.max(top1_scores))
            stats["retriever/raw_top1_min"] = float(np.min(top1_scores))
            stats["retriever/raw_top1_mean"] = float(np.mean(top1_scores))

            stats["retriever/raw_topk_max"] = float(np.max(scores))
            stats["retriever/raw_topk_min"] = float(np.min(scores))
            stats["retriever/raw_topk_mean"] = float(np.mean(scores))

            total_candidates = scores.size
            valid_candidates = total_candidates
        else:
            total_candidates = 0
            valid_candidates = 0

        if self.retrieval_threshold is not None and scores.size > 0:

            mask = scores < self.retrieval_threshold

            filtered_count = np.sum(mask)
            valid_candidates = total_candidates - filtered_count

            idxs[mask] = -1

            scores[mask] = 0.0

        if total_candidates > 0:
            stats["retriever/total_candidates"] = int(total_candidates)
            stats["retriever/valid_candidates"] = int(valid_candidates)
            stats["retriever/valid_ratio"] = float(valid_candidates / total_candidates)

            stats["retriever/avg_valid_per_query"] = float(valid_candidates / scores.shape[0])
        return scores, idxs, stats

    def compute_encode(self, data: DataProto) -> DataProto:
        if "queries" not in data.non_tensor_batch:
            raise ValueError("non_tensor_batch must contain 'queries'.")

        queries = data.non_tensor_batch["queries"]
        query_emb = self.encode_queries(queries)

        return DataProto.from_dict(
            tensors={"query_emb": torch.from_numpy(query_emb)}
        )

    def compute_search(self, data: DataProto) -> DataProto:

        self._check_and_refresh_index()

        if data.batch is not None and "query_emb" in data.batch:
            query_emb = data.batch["query_emb"].detach().cpu().numpy().astype(np.float32)
            source = "External"
        else:
            if "queries" not in data.non_tensor_batch:
                raise ValueError("non_tensor_batch must contain 'queries'.")
            queries = data.non_tensor_batch["queries"]
            query_emb = self.encode_queries(queries)
            source = "Internal"

        scores_np, idxs_np, stats = self.search_topk(query_emb, topk=self.topk)

        topk_items: List[List[Dict[str, Any]]] = []
        if self.memory_items and idxs_np.size > 0:
            for row in idxs_np:
                row_items = []
                for idx in row:
                    if 0 <= idx < len(self.memory_items):
                        row_items.append(self.memory_items[int(idx)])
                topk_items.append(row_items)
        else:
            topk_items = [[] for _ in range(scores_np.shape[0])]

        topk_items_arr = np.empty(len(topk_items), dtype=object)
        for i, item in enumerate(topk_items):
            topk_items_arr[i] = item

        from tensordict import TensorDict

        batch_size = scores_np.shape[0]
        ret_dp = DataProto(
            batch=TensorDict(
                {
                    "topk_scores": torch.from_numpy(scores_np),
                    "topk_indices": torch.from_numpy(idxs_np),
                },
                batch_size=batch_size
            ),
            non_tensor_batch={
                "topk_items": topk_items_arr
            },
            meta_info=stats
        )

        return ret_dp

    def compute_retrieval(self, data: DataProto) -> DataProto:
        if "queries" not in data.non_tensor_batch:
            raise ValueError("non_tensor_batch must contain 'queries'.")
        queries = data.non_tensor_batch["queries"]

        query_emb = self.encode_queries(queries)
        scores_np, idxs_np, _ = self.search_topk(query_emb, topk=self.topk)

        topk_items: List[List[Dict[str, Any]]] = []
        if self.memory_items and idxs_np.size > 0:
            for row in idxs_np:
                row_items = []
                for idx in row:
                    if 0 <= idx < len(self.memory_items):
                        row_items.append(self.memory_items[int(idx)])
                topk_items.append(row_items)
        else:
            topk_items = [[] for _ in range(len(queries))]

        return DataProto.from_dict(
            tensors={
                "query_emb": torch.from_numpy(query_emb),
                "topk_scores": torch.from_numpy(scores_np),
                "topk_indices": torch.from_numpy(idxs_np),
            },
            non_tensors={"topk_items": topk_items},
        )