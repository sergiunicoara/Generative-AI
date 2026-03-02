
import uuid
import hashlib
from typing import Any, Dict, List, Optional
from .base import BaseEngine, EngineResult, FilteredEngineResult
from qdrant_client import QdrantClient, models
# Import the new Query models
from qdrant_client.http.models import VectorParams, Distance, QueryRequest, Filter, HnswConfigDiff, FieldCondition, MatchValue

class QdrantEngine(BaseEngine):
    name = "qdrant"

    def __init__(self, host="localhost", port=6333, collection="stress_test", distance_metric="cosine", **kwargs):
        super().__init__(host, port, **kwargs)
        self.collection = collection
        self.distance_metric = distance_metric
        self.client = QdrantClient(host=host, port=port)
        # skip_index=True → worker-thread mode: connect only, don't touch the collection
        if not kwargs.get('skip_index', False):
            self._ensure_collection()

    def _ensure_collection(self):
        try:
            self.client.get_collection(self.collection)
            return
        except:
            pass

        m = self.kwargs.get('m', 32)
        ef_construct = self.kwargs.get('ef_construct', 256)

        # Map string to Distance enum
        distance_enum = {
            "euclid": Distance.EUCLID,
            "cosine": Distance.COSINE,
            "dot": Distance.DOT
        }.get(self.distance_metric.lower(), Distance.EUCLID)

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=768, distance=distance_enum),
            hnsw_config=HnswConfigDiff(
                m=m, 
                ef_construct=ef_construct, 
                full_scan_threshold=10000
            ),
        )

    @staticmethod
    def _to_point_id(raw_id: str):
        """Convert an arbitrary string ID to a valid Qdrant point ID (uint or UUID)."""
        # Fast path: bare integer string → int
        try:
            return int(raw_id)
        except (ValueError, TypeError):
            pass
        # Common pattern "doc-42", "chunk-7", etc. → extract trailing integer
        parts = raw_id.rsplit("-", 1)
        if len(parts) == 2:
            try:
                return int(parts[1])
            except ValueError:
                pass
        # Fallback: deterministic UUID from the string
        return str(uuid.UUID(hashlib.md5(raw_id.encode()).hexdigest()))

    def _reset_collection(self):
        """Drop and recreate the collection for a clean HNSW index."""
        try:
            self.client.delete_collection(self.collection)
        except Exception:
            pass
        self._ensure_collection()

    def index(self, vectors, metadata):
        self._reset_collection()
        points = []
        for vec, meta in zip(vectors, metadata):
            raw_id = meta.get("id", str(uuid.uuid4()))
            pid = self._to_point_id(raw_id)
            # Keep original string ID in payload so search can return it
            payload = {k: v for k, v in meta.items() if k != "id"}
            payload["_id"] = raw_id
            points.append(models.PointStruct(id=pid, vector=vec, payload=payload))

        for i in range(0, len(points), 500):
            batch = points[i:i+500]
            self.client.upsert(collection_name=self.collection, points=batch)

    def search(self, query, k=10):
        ef_search = self.kwargs.get('ef_search', 512)
        query_vec = query.tolist() if hasattr(query, 'tolist') else list(query)

        query_request = QueryRequest(
            query=query_vec,
            limit=k,
            params=models.SearchParams(hnsw_ef=ef_search),
            with_payload=True,
        )

        results = self.client.query_batch_points(
            collection_name=self.collection,
            requests=[query_request],
        )

        scored_points = results[0].points
        return [
            EngineResult(
                id=p.payload.get("_id", str(p.id)) if p.payload else str(p.id),
                score=p.score,
                metadata={k: v for k, v in (p.payload or {}).items() if k != "_id"},
            ) for p in scored_points
        ]

    def insert(self, vectors, metadata):
        """Upsert new points without dropping the collection (incremental add)."""
        points = []
        for vec, meta in zip(vectors, metadata):
            raw_id = meta.get("id", str(uuid.uuid4()))
            pid = self._to_point_id(raw_id)
            payload = {k: v for k, v in meta.items() if k != "id"}
            payload["_id"] = raw_id
            points.append(models.PointStruct(id=pid, vector=vec, payload=payload))
        for i in range(0, len(points), 500):
            self.client.upsert(collection_name=self.collection, points=points[i:i+500])

    def delete(self, ids):
        """Delete points by their original string IDs."""
        point_ids = [self._to_point_id(i) for i in ids]
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.PointIdsList(points=point_ids),
        )

    def search_with_filter(self, query, k=10, filter_query=None):
        ef_search = self.kwargs.get('ef_search', 512)
        query_vec = query.tolist() if hasattr(query, 'tolist') else list(query)

        if filter_query and isinstance(filter_query, dict) and "field" in filter_query:
            # Translate generic {"field": "tag", "value": "tag-0"} → Qdrant FieldCondition
            qdrant_filter = Filter(
                must=[FieldCondition(key=filter_query["field"], match=MatchValue(value=filter_query["value"]))]
            )
        elif isinstance(filter_query, dict):
            qdrant_filter = Filter(**filter_query)
        else:
            qdrant_filter = filter_query

        query_request = QueryRequest(
            query=query_vec,
            limit=k,
            params=models.SearchParams(hnsw_ef=ef_search),
            with_payload=True,
            filter=qdrant_filter,
        )

        results = self.client.query_batch_points(
            collection_name=self.collection,
            requests=[query_request],
        )

        scored_points = results[0].points
        return [
            FilteredEngineResult(
                id=p.payload.get("_id", str(p.id)) if p.payload else str(p.id),
                score=p.score,
                metadata={k: v for k, v in (p.payload or {}).items() if k != "_id"},
                filter_applied=filter_query,
            ) for p in scored_points
        ]
