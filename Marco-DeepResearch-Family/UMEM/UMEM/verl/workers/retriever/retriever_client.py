from verl import DataProto

class RetrieverClient:
    """Thin local or worker-group client for retriever calls."""

    def __init__(self, retriever=None, retriever_group=None):
        self.retriever = retriever
        self.retriever_group = retriever_group
        assert (retriever is not None) ^ (retriever_group is not None),\
            "Pass exactly one of retriever or retriever_group."

    def encode(self, queries):
        data = DataProto.from_dict(non_tensors={"queries": queries})
        if self.retriever:
            return self.retriever.compute_encode(data)
        return self.retriever_group.encode(data)

    def search(self, queries=None, query_emb=None):
        if query_emb is not None:
            data = DataProto.from_dict(tensors={"query_emb": query_emb})
        else:
            data = DataProto.from_dict(non_tensors={"queries": queries})

        if self.retriever:
            return self.retriever.compute_search(data)
        return self.retriever_group.search(data)

    def retrieval(self, queries):
        data = DataProto.from_dict(non_tensors={"queries": queries})
        if self.retriever:
            return self.retriever.compute_retrieval(data)
        return self.retriever_group.retrieval(data)
