# Lazy package — consumers import directly from submodules.
# EvaluationAgent eagerly imports ragas+datasets; removed to keep workers image lean.
__all__ = ["IngestionAgent", "QueryAgent", "EvaluationAgent"]
