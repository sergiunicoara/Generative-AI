import uuid, json
from typing import Any, Dict, List, Optional
import requests
from .base import BaseEngine, EngineResult, FilteredEngineResult

_TIMEOUT = (5.0, 30.0)  # (connect_timeout, read_timeout) in seconds

class ElasticsearchEngine(BaseEngine):
    name = "elasticsearch"

    def __init__(self, host="localhost", port=9201, index="stress_test", **kwargs):
        super().__init__(host, port, **kwargs)
        self.index_name = index
        self._base_url = f"http://{self.host}:{self.port}"
        self._session = requests.Session()
        self._ensure_index()

    def _ensure_index(self):
        # Always drop and recreate to guarantee a clean index per benchmark run
        r = self._session.get(f"{self._base_url}/{self.index_name}", timeout=_TIMEOUT)
        if r.status_code == 200:
            self._session.delete(f"{self._base_url}/{self.index_name}", timeout=_TIMEOUT)
        m = self.kwargs.get('m', 16)
        ef_construction = self.kwargs.get('ef_construction', 100)
        mapping = {
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "dense_vector",
                        "dims": 768,
                        "index": True,
                        "similarity": "cosine",
                        "index_options": {
                            "type": "hnsw",
                            "m": m,
                            "ef_construction": ef_construction,
                        },
                    }
                }
            }
        }
        self._session.put(f"{self._base_url}/{self.index_name}", json=mapping, timeout=_TIMEOUT)

    def index(self, vectors, metadata, batch_size=500):
        pairs = list(zip(vectors, metadata))
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            lines = []
            for vec, meta in batch:
                doc_id = meta.get("id", str(uuid.uuid4()))
                m = {k: v for k, v in meta.items() if k != "id"}
                lines.append(json.dumps({"index": {"_index": self.index_name, "_id": doc_id}}))
                lines.append(json.dumps({"vector": vec, **m}))
            body = "\n".join(lines) + "\n"
            r = self._session.post(f"{self._base_url}/_bulk", data=body, headers={"Content-Type": "application/x-ndjson"}, timeout=_TIMEOUT)
            r.raise_for_status()
        self.flush()

    def insert(self, vectors, metadata):
        self.index(vectors, metadata)

    def delete(self, ids):
        lines = [json.dumps({"delete": {"_index": self.index_name, "_id": i}}) for i in ids]
        body = "\n".join(lines) + "\n"
        self._session.post(f"{self._base_url}/_bulk", data=body, headers={"Content-Type": "application/x-ndjson"}, timeout=_TIMEOUT)
        self.flush()

    def flush(self):
        self._session.post(f"{self._base_url}/{self.index_name}/_refresh", timeout=_TIMEOUT)

    def search(self, query, k=10):
        # FIX: Use ef_search from config for num_candidates
        ef_search = self.kwargs.get('ef_search', 100)
        body = {"size": k, "knn": {"field": "vector", "query_vector": query, "k": k, "num_candidates": ef_search}}
        r = self._session.post(f"{self._base_url}/{self.index_name}/_search", json=body, timeout=_TIMEOUT)
        r.raise_for_status()
        return [EngineResult(id=h["_id"], score=h["_score"], metadata=h.get("_source", {})) for h in r.json()["hits"]["hits"]]

    def search_with_filter(self, query, k=10, filter_query=None):
        ef_search = self.kwargs.get('ef_search', 100)
        knn_body = {"field": "vector", "query_vector": query, "k": k, "num_candidates": ef_search}
        if filter_query and isinstance(filter_query, dict) and "field" in filter_query:
            # Translate generic {"field": "tag", "value": "tag-0"} → ES term filter.
            # Dynamically-mapped string fields get a .keyword sub-field for exact matching.
            knn_body["filter"] = {"term": {filter_query["field"] + ".keyword": filter_query["value"]}}
        elif filter_query:
            knn_body["filter"] = filter_query
        body = {"size": k, "knn": knn_body}
        r = self._session.post(f"{self._base_url}/{self.index_name}/_search", json=body, timeout=_TIMEOUT)
        r.raise_for_status()
        return [FilteredEngineResult(id=h["_id"], score=h["_score"], metadata=h.get("_source", {}), filter_applied=filter_query) for h in r.json()["hits"]["hits"]]
