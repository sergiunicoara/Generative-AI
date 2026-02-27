import uuid, json, time
import requests
from .base import BaseEngine, EngineResult

class ElasticsearchEngine(BaseEngine):
    name = "elasticsearch"

    def __init__(
        self,
        host="localhost",
        port=9201,
        index="stress_test",
        connect_timeout_s: float = 2.0,
        op_timeout_s: float = 60.0,
        **kwargs,
    ):
        super().__init__(host, port, **kwargs)
        self.index_name = index
        self._base_url = f"http://{self.host}:{self.port}"
        self._connect_timeout_s = connect_timeout_s
        self._op_timeout_s = op_timeout_s
        # This session object is what prevents the socket error
        self.session = requests.Session()
        self._wait_for_connection()
        self._ensure_index()

    def _wait_for_connection(self):
        for i in range(10):
            try:
                r = self.session.get(self._base_url, timeout=self._connect_timeout_s)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(3)
        raise ConnectionError(f"Elasticsearch not reachable at {self._base_url}")

    def _ensure_index(self):
        r = self.session.get(f"{self._base_url}/{self.index_name}", timeout=self._connect_timeout_s)
        if r.status_code == 200: return
        mapping = {
            "mappings": {
                "properties": {
                    "vector": {"type": "dense_vector", "dims": 768, "index": True, "similarity": "cosine"}
                }
            }
        }
        self.session.put(f"{self._base_url}/{self.index_name}", json=mapping, timeout=self._connect_timeout_s)

    def _bulk(self, lines):
        if not lines:
            return
        body = "\n".join(lines) + "\n"
        r = self.session.post(
            f"{self._base_url}/_bulk",
            data=body,
            headers={"Content-Type": "application/x-ndjson"},
            timeout=self._op_timeout_s,
        )
        r.raise_for_status()
        payload = r.json()
        if payload.get("errors"):
            # Keep the error short but actionable.
            items = payload.get("items", [])
            first_error = None
            for it in items:
                action = it.get("index") or it.get("delete") or it.get("create") or it.get("update") or {}
                err = action.get("error")
                if err:
                    first_error = err
                    break
            raise RuntimeError(f"Elasticsearch bulk request reported errors: {first_error or 'unknown error'}")

    def index(self, vectors, metadata):
        batch_docs = 500
        lines = []
        docs_in_batch = 0
        for vec, meta in zip(vectors, metadata):
            doc_id = meta.get("id", str(uuid.uuid4()))
            m = {k: v for k, v in meta.items() if k != "id"}
            lines.append(json.dumps({"index": {"_index": self.index_name, "_id": doc_id}}))
            lines.append(json.dumps({"vector": vec, **m}))
            docs_in_batch += 1
            if docs_in_batch >= batch_docs:
                self._bulk(lines)
                lines = []
                docs_in_batch = 0
        self._bulk(lines)
        self.flush()

    def insert(self, vectors, metadata):
        # Elasticsearch "index" is already an upsert.
        return self.index(vectors, metadata)

    def delete(self, ids):
        batch_docs = 1000
        lines = []
        docs_in_batch = 0
        for doc_id in ids:
            lines.append(json.dumps({"delete": {"_index": self.index_name, "_id": doc_id}}))
            docs_in_batch += 1
            if docs_in_batch >= batch_docs:
                self._bulk(lines)
                lines = []
                docs_in_batch = 0
        self._bulk(lines)
        self.flush()

    def flush(self):
        self.session.post(f"{self._base_url}/{self.index_name}/_refresh", timeout=self._op_timeout_s)

    def search(self, query, k=10):
        body = {"size": k, "knn": {"field": "vector", "query_vector": query, "k": k, "num_candidates": 50}}
        # FIXED: Using self.session instead of requests directly
        r = self.session.post(f"{self._base_url}/{self.index_name}/_search", json=body, timeout=self._op_timeout_s)
        r.raise_for_status()
        hits = r.json().get("hits", {}).get("hits", [])
        return [EngineResult(id=h["_id"], score=h["_score"], metadata=h.get("_source", {})) for h in hits]