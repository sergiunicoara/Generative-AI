# Lazy package — consumers import directly from submodules.
# Eager imports removed to avoid pulling sentence_transformers into the API image.
__all__ = ["LocalSearch", "GlobalSearch", "HybridRetriever", "ContextBuilder"]
