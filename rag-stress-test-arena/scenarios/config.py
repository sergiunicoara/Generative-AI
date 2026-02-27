import os
from dataclasses import dataclass
from typing import Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class ScenarioConfig:
    mode: str

    @property
    def is_quick(self) -> bool:
        return self.mode in {"quick", "dev", "development", "1", "true", "yes", "y"}

    def pick(self, full: T, quick: T) -> T:
        return quick if self.is_quick else full


cfg = ScenarioConfig(mode="full")
