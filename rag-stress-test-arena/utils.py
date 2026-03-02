import random
import time
import os
import yaml
from statistics import mean, stdev
from typing import Any, Dict, List, Tuple
import numpy as np

def load_config(path="config.yml"):
    """Loads the configuration from the specified YAML file."""
    if os.path.exists(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    # Fallback for the old path if needed
    alt_path = "config/scenario_config.yml"
    if os.path.exists(alt_path):
        with open(alt_path, 'r') as f:
            return yaml.safe_load(f)
    print(f"Warning: Configuration file not found at {path} or {alt_path}")
    return {}

def _load_npy(path: str, n: int, label: str) -> List[List[float]]:
    """Load the first n rows from a .npy file.  Fails loudly if missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nCorpus file not found: {path}\n"
            f"Run:  py -3.11 generate_distribution.py\n"
            f"This encodes {label} from Wikipedia into 768-d vectors.\n"
            f"Running the benchmark without real embeddings would produce\n"
            f"meaningless results — no silent fallback is provided.\n"
        )
    data = np.load(path)
    if len(data) < n:
        raise ValueError(
            f"{path} has only {len(data)} rows but {n} were requested.\n"
            f"Re-run generate_distribution.py to regenerate the file."
        )
    return data[:n].tolist()


def make_random_vectors(n: int, dim: int = 768) -> List[List[float]]:
    """Load corpus vectors from the pre-generated Wikipedia embedding file."""
    cfg = load_config()
    path = cfg.get("distribution", {}).get("file", "corpus_768d.npy")
    return _load_npy(path, n, "20 000 Wikipedia corpus articles")


def load_query_vectors(n: int) -> List[List[float]]:
    """Load held-out query vectors from the pre-generated Wikipedia embedding file."""
    cfg = load_config()
    path = cfg.get("distribution", {}).get("query_file", "queries_768d.npy")
    return _load_npy(path, n, "3 000 held-out Wikipedia query articles")

def make_metadata(n: int, num_tags: int = 10) -> List[Dict[str, Any]]:
    return [{"id": f"doc-{i}", "tag": f"tag-{random.randint(0, num_tags-1)}", "group": f"group-{i % 5}"} for i in range(n)]


def timed(func, *args, **kwargs) -> Tuple[Any, float]:
    start = time.time()
    res = func(*args, **kwargs)
    return res, (time.time() - start) * 1000.0


def calculate_recall(gold_ids: List[str], test_ids: List[str]) -> float:
    if not gold_ids: return 0.0
    intersection = set(gold_ids) & set(test_ids)
    return len(intersection) / len(gold_ids)

def p95(values: List[float]) -> float:
    return float(np.percentile(values, 95)) if values else 0.0

def p99(values: List[float]) -> float:
    return float(np.percentile(values, 99)) if values else 0.0

def avg(values: List[float]) -> float:
    return float(mean(values)) if values else 0.0

def confidence_interval(values: List[float]):
    if len(values) < 2: return 0.0
    return 1.96 * (stdev(values) / (len(values)**0.5))