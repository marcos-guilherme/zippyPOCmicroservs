from abc import ABC, abstractmethod


class VLMModel(ABC):
    @abstractmethod
    def load(self) -> None:
        """Load model resources. Should be idempotent."""
        raise NotImplementedError

    @abstractmethod
    def infer(self, image_path: str, prompt: str) -> dict:
        """Run inference and return a dict with keys: 'text' and 'device'."""
        raise NotImplementedError


