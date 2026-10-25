from abc import ABC, abstractmethod
from typing import Mapping


class IRunnable(ABC):
    @abstractmethod
    def run(self, context: Mapping[str, object]) -> None:
        raise NotImplementedError
