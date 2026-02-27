import random
import time
from statistics import mean
from typing import Any, Dict, List, Tuple
import numpy as np


def make_random_vectors(n: int, dim: int = 768) -> List[List[float]]:
    return np.random.randn(n, dim).astype("float32").tolist()


def make_metadata(n: int, num_tags: int = 10) -> List[Dict[str, Any]]:
    return [{"id": f"doc-{i}", "tag": f"tag-{random.randint(0, num_tags-1)}", "group": f"group-{i % 5}"} for i in range(n)]


def timed(func, *args, **kwargs) -> Tuple[Any, float]:
    start = time.time()
    res = func(*args, **kwargs)
    return res, (time.time() - start) * 1000.0


def p95(values: List[float]) -> float:
    return float(np.percentile(values, 95)) if values else 0.0


def p99(values: List[float]) -> float:
    return float(np.percentile(values, 99)) if values else 0.0


def avg(values: List[float]) -> float:
    return float(mean(values)) if values else 0.0


def recall_at_k(gold_ids, retrieved_ids, k):
    gold_set = set(gold_ids[:k])
    retrieved_set = set(retrieved_ids[:k])
    return len(gold_set & retrieved_set) / len(gold_set) if gold_set else 0.0