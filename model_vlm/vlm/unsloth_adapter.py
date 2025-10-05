from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

from .base import VLMModel


class UnslothLlamaVisionAdapter(VLMModel):
    def __init__(self, model_id: str = "unsloth/Llama-3.2-11B-Vision-Instruct", device: str = None):
        self.model_id = model_id
        self.device = device  # None => autodetect
        self._processor = None
        self._model = None
        self._resolved_device = None

    def load(self) -> None:
        if self._model is None or self._processor is None:
            resolved = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForVision2Seq.from_pretrained(self.model_id)
            if resolved == "cuda":
                self._model = self._model.to("cuda")
            self._resolved_device = resolved

    def infer(self, image_path: str, prompt: str) -> dict:
        self.load()
        image = Image.open(image_path).convert("RGB")

        text = prompt or ""
        with torch.inference_mode():
            inputs = self._processor(images=image, text=text, return_tensors="pt")
            if self._resolved_device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            outputs = self._model.generate(**inputs)
            decoded = self._processor.batch_decode(outputs, skip_special_tokens=True)
            caption = decoded[0] if decoded else ""

        return {"text": caption, "device": self._resolved_device or self.device or "cpu"}


