import uuid
import threading
from qdrant_client import QdrantClient
from qdrant_client.http import models
from .base import BaseEngine, EngineResult

class QdrantEngine(BaseEngine):
    name = "qdrant"

    def __init__(self, host="localhost", port=6333, index="stress_test", timeout_s: float = 2.0, **kwargs):
        super().__init__(host, port, **kwargs)
        self.index_name = index
        self._timeout_s = timeout_s
        self._tls = threading.local()
        # Namespace for deterministic UUIDs from string IDs like "doc-0"
        self._namespace_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")
        # Use a persistent client for connection pooling
        self.client = QdrantClient(host=self.host, port=self.port, prefer_grpc=False, timeout=timeout_s)
        self._ensure_collection()

    def _thread_client(self) -> QdrantClient:
        c = getattr(self._tls, "client", None)
        if c is None:
            c = QdrantClient(host=self.host, port=self.port, prefer_grpc=False, timeout=self._timeout_s)
            self._tls.client = c
        return c

    def _ensure_collection(self):
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.index_name for c in collections)
            if not exists:
                self.client.create_collection(
                    collection_name=self.index_name,
                    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
                )
        except Exception as e:
            print(f"Error checking/creating Qdrant collection: {e}")

    def index(self, vectors, metadata):
        # Chunking to stay under the 32MB limit
        batch_size = 100 

        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i : i + batch_size]
            batch_meta = metadata[i : i + batch_size]
            
            points = []
            for v, m in zip(batch_vectors, batch_meta):
                raw_id = str(m.get("id", uuid.uuid4()))
                
                # Convert string IDs like "doc-0" to valid UUIDs
                try:
                    point_id = str(uuid.UUID(raw_id))
                except ValueError:
                    # If it's a string like "doc-0", hash it into a UUID
                    point_id = str(uuid.uuid5(self._namespace_uuid, raw_id))

                points.append(models.PointStruct(
                    id=point_id,
                    vector=v,
                    payload=m
                ))
            
            self.client.upsert(
                collection_name=self.index_name, 
                points=points,
                wait=True
            )

    def insert(self, vectors, metadata):
        return self.index(vectors, metadata)

    def delete(self, ids):
        if not ids:
            return
        # Qdrant only accepts int or UUID point IDs; our metadata uses "doc-*" strings.
        point_ids = []
        for raw_id in ids:
            s = str(raw_id)
            try:
                point_ids.append(uuid.UUID(s))
            except ValueError:
                point_ids.append(uuid.uuid5(self._namespace_uuid, s))

        # Pass a plain list; the client will wrap it as a PointsSelector.
        self._thread_client().delete(collection_name=self.index_name, points_selector=point_ids, wait=True)

    def search(self, query, k=10):
        # qdrant-client API has changed over time:
        # - older: client.search(collection_name=..., query_vector=..., limit=...)
        # - newer: client.query_points(collection_name=..., query=..., limit=...)
        client = self._thread_client()
        if hasattr(client, "search"):
            results = client.search(
                collection_name=self.index_name,
                query_vector=query,
                limit=k,
            )
            points = results
        elif hasattr(client, "query_points"):
            resp = client.query_points(
                collection_name=self.index_name,
                query=query,
                limit=k,
                with_payload=True,
            )
            points = getattr(resp, "points", resp)
        else:
            raise RuntimeError(
                "Installed qdrant-client is missing both 'search' and 'query_points' APIs. "
                "Please upgrade qdrant-client."
            )

        return [EngineResult(id=str(p.id), score=float(p.score), metadata=(p.payload or {})) for p in points]

    def flush(self):
        pass