import uuid
import json
import numpy as np
import redis as redis_lib
from redis.commands.search.field import VectorField, TagField, TextField
from redis.commands.search.query import Query

try:
    from redis.commands.search.index_definition import IndexDefinition, IndexType
except ImportError:
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType

from .base import BaseEngine, EngineResult

class RedisEngine(BaseEngine):
    name = "redis"

    def __init__(
        self,
        host="localhost",
        port=6379,
        index="stress_test",
        connect_timeout_s: float = 2.0,
        op_timeout_s: float = 30.0,
        index_batch_size: int = 500,
        **kwargs,
    ):
        super().__init__(host, port, **kwargs)
        self.index_name = index
        self._index_batch_size = index_batch_size
        self.client = redis_lib.Redis(
            host=self.host,
            port=self.port,
            socket_connect_timeout=connect_timeout_s,
            socket_timeout=op_timeout_s,
            retry_on_timeout=True,
        )
        # Fail fast if Redis isn't reachable.
        self.client.ping()
        self._ensure_index()

    def _ensure_index(self):
        # Always drop and recreate to guarantee a clean index per benchmark run
        try:
            self.client.ft(self.index_name).dropindex(delete_documents=True)
        except Exception:
            pass  # Index didn't exist yet — that's fine
        m = self.kwargs.get('m', 16)
        ef_construction = self.kwargs.get('ef_construction', 100)
        schema = (
            VectorField(
                "vector", "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": 768,
                    "DISTANCE_METRIC": "COSINE",
                    "M": m,
                    "EF_CONSTRUCTION": ef_construction,
                    "INITIAL_CAP": 25000,
                },
            ),
            TagField("id"),
            TextField("metadata"),
        )
        self.client.ft(self.index_name).create_index(schema, definition=IndexDefinition(prefix=["doc:"], index_type=IndexType.HASH))

    def index(self, vectors, metadata):
        pipe = self.client.pipeline(transaction=False)
        pending = 0
        for vec, meta in zip(vectors, metadata):
            did = meta.get("id", str(uuid.uuid4()))
            pipe.hset(
                f"doc:{did}",
                mapping={
                    "vector": np.array(vec, dtype=np.float32).tobytes(),
                    "id": did,
                    "metadata": json.dumps(meta),
                },
            )
            pending += 1
            if pending >= self._index_batch_size:
                pipe.execute()
                pipe = self.client.pipeline(transaction=False)
                pending = 0
        if pending:
            pipe.execute()

    def insert(self, vectors, metadata):
        # Redis hashes are overwritten on HSET; this behaves like an upsert.
        return self.index(vectors, metadata)

    def delete(self, ids):
        if not ids:
            return
        pipe = self.client.pipeline(transaction=False)
        pending = 0
        for did in ids:
            pipe.delete(f"doc:{did}")
            pending += 1
            if pending >= 1000:
                pipe.execute()
                pipe = self.client.pipeline(transaction=False)
                pending = 0
        if pending:
            pipe.execute()

    def search(self, query, k=10):
        ef_search = self.kwargs.get('ef_search', 100)
        q = Query(f"*=>[KNN {k} @vector $vec EF_RUNTIME {ef_search} AS score]").sort_by("score").return_fields("id", "score", "metadata").dialect(2)
        params = {"vec": np.array(query, dtype=np.float32).tobytes()}
        res = self.client.ft(self.index_name).search(q, params)
        return [
            EngineResult(
                id=r.id.replace("doc:", ""), 
                score=1.0 - float(r.score), 
                metadata=json.loads(r.metadata) if hasattr(r, 'metadata') else {}
            ) for r in res.docs
        ]

    def flush(self):
        pass