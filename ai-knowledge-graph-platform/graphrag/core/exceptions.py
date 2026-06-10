class GraphRAGError(Exception):
    """Base exception for all GraphRAG errors."""


class IngestionError(GraphRAGError):
    """Raised when document ingestion fails."""


class ExtractionError(GraphRAGError):
    """Raised when entity/relation extraction fails."""


class RetrievalError(GraphRAGError):
    """Raised when graph retrieval fails."""


class EvaluationError(GraphRAGError):
    """Raised when RAGAS evaluation fails."""


class MessagingError(GraphRAGError):
    """Raised when RabbitMQ publish/consume fails."""
