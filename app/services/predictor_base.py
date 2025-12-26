from abc import ABC, abstractmethod
from typing import Dict, Any

class PredictorBase(ABC):
    @abstractmethod
    def predict(self, text: str) -> Dict[str, Any]:
        ...
