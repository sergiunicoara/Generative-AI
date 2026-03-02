import uuid
import psycopg2
from psycopg2.extras import execute_values, Json
import numpy as np
from .base import BaseEngine, EngineResult

class PgVectorEngine(BaseEngine):
    name = "pgvector"

    def __init__(
        self,
        host="localhost",
        port=5432,
        dbname="postgres",
        user="postgres",
        password="postgres",
        table="stress_test",
        connect_timeout_s: int = 3,
        **kwargs,
    ):
        super().__init__(host, port, **kwargs)
        self.table_name = table
        self.params = {
            "host": host,
            "port": port,
            "dbname": dbname,
            "user": user,
            "password": password,
            "connect_timeout": connect_timeout_s,
        }
        # skip_index=True → worker-thread mode: open one persistent connection,
        # skip all schema setup so the already-indexed data is preserved.
        if kwargs.get('skip_index', False):
            self._conn = psycopg2.connect(**self.params)
        else:
            self._conn = None
            self._ensure_table()

    def _get_connection(self):
        return psycopg2.connect(**self.params)

    def _ensure_table(self):
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                # Always drop and recreate to guarantee a clean table per benchmark run
                cur.execute(f"DROP TABLE IF EXISTS {self.table_name}")
                cur.execute(f"CREATE TABLE {self.table_name} (id TEXT PRIMARY KEY, vector vector(768), metadata JSONB)")
                conn.commit()

    def _rebuild_hnsw_index(self, conn):
        """Drop and rebuild the HNSW index after data is fully loaded for best graph quality."""
        m = self.kwargs.get('m', 16)
        ef_construction = self.kwargs.get('ef_construction', 100)
        idx_name = f"{self.table_name}_hnsw_idx"
        with conn.cursor() as cur:
            cur.execute(f"DROP INDEX IF EXISTS {idx_name}")
            cur.execute(
                f"CREATE INDEX {idx_name} ON {self.table_name} "
                f"USING hnsw (vector vector_cosine_ops) "
                f"WITH (m={m}, ef_construction={ef_construction})"
            )
            conn.commit()

    def index(self, vectors, metadata):
        data = [(str(m.get("id", uuid.uuid4())), v, Json(m)) for v, m in zip(vectors, metadata)]
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    f"INSERT INTO {self.table_name} (id, vector, metadata) VALUES %s "
                    f"ON CONFLICT (id) DO UPDATE SET vector=EXCLUDED.vector, metadata=EXCLUDED.metadata",
                    data,
                )
                conn.commit()
            # Rebuild HNSW index after every full load for a clean graph
            self._rebuild_hnsw_index(conn)

    def insert(self, vectors, metadata):
        return self.index(vectors, metadata)

    def delete(self, ids):
        if not ids:
            return
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {self.table_name} WHERE id = ANY(%s)", (list(ids),))
                conn.commit()

    def search(self, query, k=10):
        ef_search = self.kwargs.get('ef_search', 100)
        if self._conn is not None:
            # Worker mode: reuse the persistent connection (no per-query connect overhead)
            cur = self._conn.cursor()
            try:
                cur.execute(f"SET hnsw.ef_search = {ef_search}")
                cur.execute(
                    f"SELECT id, metadata, (vector <=> %s::vector) as dist "
                    f"FROM {self.table_name} ORDER BY dist LIMIT %s",
                    (query, k),
                )
                return [EngineResult(id=r[0], score=1 - float(r[2]), metadata=r[1]) for r in cur.fetchall()]
            finally:
                cur.close()
        else:
            # Standard mode: new connection per query (original behaviour)
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SET hnsw.ef_search = {ef_search}")
                    cur.execute(
                        f"SELECT id, metadata, (vector <=> %s::vector) as dist "
                        f"FROM {self.table_name} ORDER BY dist LIMIT %s",
                        (query, k),
                    )
                    return [EngineResult(id=r[0], score=1 - float(r[2]), metadata=r[1]) for r in cur.fetchall()]

    def flush(self):
        pass

    def close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
