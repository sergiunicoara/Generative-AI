# Tab modules — each imported here triggers their @callback registrations.
from graphrag.dashboard.tabs import health, conflicts, communities, gdpr, calibration

__all__ = ["health", "conflicts", "communities", "gdpr", "calibration"]
