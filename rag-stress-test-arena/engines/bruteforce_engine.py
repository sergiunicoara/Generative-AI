import numpy as np
from typing import List, Dict, Any
from .base import BaseEngine

class BruteForceEngine(BaseEngine):
    def __init__(self, config: Dict[str, Any]):
        self.vectors = None
        self.metadata = None

    def index(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]):
        raw = np.array(vectors, dtype='float32')
        # Pre-normalize corpus vectors once so search() can use fast dot-product
        # as a proxy for cosine similarity (matching all engines' cosine distance).
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        self.vectors = raw / norms
        self.metadata = metadata

    def search(self, query_vector: List[float], k: int = 10) -> List[str]:
        q = np.array(query_vector, dtype='float32')
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm
        # Dot product of two unit vectors == cosine similarity
        sims = self.vectors @ q
        idx = np.argsort(sims)[-k:][::-1]
        return [self.metadata[i]['id'] for i in idx]
