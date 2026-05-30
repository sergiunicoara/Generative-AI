"""Exchange, queue, and routing key constants."""

INGEST_EXCHANGE = "graphrag.ingest"
QUERY_EXCHANGE = "graphrag.query"
EVAL_EXCHANGE = "graphrag.eval"

INGEST_QUEUE = "graphrag.ingest.queue"
QUERY_QUEUE = "graphrag.query.queue"
EVAL_QUEUE = "graphrag.eval.queue"

INGEST_ROUTING_KEY = "doc.#"
QUERY_ROUTING_KEY = "query.#"
EVAL_ROUTING_KEY = "eval.#"
