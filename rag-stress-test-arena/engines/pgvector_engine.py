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
        self._ensure_table()

    def _get_connection(self):
        return psycopg2.connect(**self.params)

    def _ensure_table(self):
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(f"CREATE TABLE IF NOT EXISTS {self.table_name} (id TEXT PRIMARY KEY, vector vector(768), metadata JSONB)")
                conn.commit()

    def index(self, vectors, metadata):
        data = [(str(m.get("id", uuid.uuid4())), v, Json(m)) for v, m in zip(vectors, metadata)]
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                execute_values(cur, f"INSERT INTO {self.table_name} (id, vector, metadata) VALUES %s ON CONFLICT (id) DO UPDATE SET vector=EXCLUDED.vector, metadata=EXCLUDED.metadata", data)
                conn.commit()

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
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT id, metadata, (vector <=> %s::vector) as dist FROM {self.table_name} ORDER BY dist LIMIT %s", (query, k))
                return [EngineResult(id=r[0], score=1-float(r[2]), metadata=r[1]) for r in cur.fetchall()]

    def flush(self):
        pass