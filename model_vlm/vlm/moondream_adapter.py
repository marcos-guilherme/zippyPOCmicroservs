from PIL import Image
from transformers import AutoModelForCausalLM
import torch

from .base import VLMModel


class MoondreamAdapter(VLMModel):
    def __init__(self, model_id: str = "vikhyatk/moondream2", revision: str = "2025-06-21", device: str = None):
        self.model_id = model_id
        self.revision = revision
        self.device = device  # None means autodetect
        self._model = None
        self._resolved_device = None

    def load(self) -> None:
        if self._model is None:
            resolved = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                revision=self.revision,
                trust_remote_code=True,
                device_map={"": resolved},
            )
            self._resolved_device = resolved

    def infer(self, image_path: str, prompt: str) -> dict:
        self.load()
        image = Image.open(image_path).convert("RGB")
        # Use inference_mode only to avoid dtype mismatches (BF16 vs FP16)
        with torch.inference_mode():
            answer = self._model.query(image, prompt)["answer"]
        return {"text": answer, "device": self._resolved_device or self.device or "cpu"}


