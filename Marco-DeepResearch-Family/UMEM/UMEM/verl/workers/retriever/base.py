

from abc import ABC, abstractmethod
from verl import DataProto

class BaseRetrieverModel(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def compute_retrieval(self, data: DataProto) -> DataProto:
        """Return encoded queries, retrieved indices, scores, and memory items."""
        pass
