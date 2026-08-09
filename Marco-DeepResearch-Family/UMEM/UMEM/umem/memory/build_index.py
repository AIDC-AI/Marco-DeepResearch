
import json
import numpy as np
import faiss
import os
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

def build_faiss_index(jsonl_path: str, index_path: str, retriever=None):
    """Build a FAISS inner-product index over memory keys."""
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Memory jsonl not found: {jsonl_path}")

    if retriever is None:
        raise ValueError("retriever is required for building index")

    key_texts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if "key" not in obj:
                    continue
                key_texts.append(obj["key"])
            except json.JSONDecodeError:
                continue

    if len(key_texts) == 0:
        dim = 1024
        index = faiss.IndexFlatIP(dim)
        faiss.write_index(index, index_path)
        return

    batch_size = 256
    all_embs = []

    for i in range(0, len(key_texts), batch_size):
        batch_keys = key_texts[i : i + batch_size]

        dp = DataProto.from_dict(non_tensors={"queries": batch_keys})

        dp_padded, pad_size = pad_dataproto_to_divisor(dp, retriever.world_size)

        out = retriever.encode(dp_padded)

        out = unpad_dataproto(out, pad_size)

        emb = out.batch["query_emb"].cpu().numpy().astype(np.float32)
        all_embs.append(emb)

    if not all_embs:
        return

    key_embs = np.concatenate(all_embs, axis=0)
    dim = key_embs.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(key_embs)

    faiss.write_index(index, index_path)
