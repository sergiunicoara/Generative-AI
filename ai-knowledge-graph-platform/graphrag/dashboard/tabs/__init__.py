# Tab modules — each imported here triggers their @callback registrations.
from graphrag.dashboard.tabs import (
    calibration, communities, conflicts, gdpr, health, review_queue,
)

__all__ = ["health", "conflicts", "communities", "gdpr", "calibration", "review_queue"]
