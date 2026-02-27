from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class EngineResult:
    id: str
    score: float
    metadata: Dict[str, Any]


@dataclass
class FilteredEngineResult(EngineResult):
    filter_applied: Optional[Dict[str, Any]] = None


class BaseEngine:
    name: str

    def __init__(self, host: str, port: int, **kwargs):
        """
        Initializes the base engine configuration. 
        Note: Subclasses should initialize their persistent client/connection pool 
        here to prevent [WinError 10048] socket exhaustion.
        """
        self.host = host
        self.port = port
        self.kwargs = kwargs
        # self.client = None  # Subclasses should set this up as a persistent client

    def index(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        """Uploads a batch of vectors and metadata to the engine."""
        raise NotImplementedError

    def search(self, query: List[float], k: int = 10) -> List[EngineResult]:
        """Performs a vector search. Subclasses must use a persistent client here."""
        raise NotImplementedError

    def search_with_filter(self, query, k=10, filter_query=None):
        """Standard search wrapped with filter metadata."""
        base = self.search(query, k=k)
        return [
            FilteredEngineResult(
                id=r.id, 
                score=r.score, 
                metadata=r.metadata, 
                filter_applied=filter_query
            ) for r in base
        ]

    def insert(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        """Inserts new data into the index."""
        raise NotImplementedError

    def delete(self, ids: List[str]) -> None:
        """Deletes specific records by ID."""
        raise NotImplementedError

    def flush(self) -> None:
        """Ensures all data is committed/indexed."""
        pass

    def close(self) -> None:
        """
        Closes the persistent connection pool. 
        Should be called at the end of the benchmark to free resources.
        """
        pass