from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

from .base import VLMModel


class BLIPAdapter(VLMModel):
    def __init__(self, model_id: str = "Salesforce/blip-image-captioning-large", device: str = None):
        self.model_id = model_id
        self.device = device  # None => autodetect
        self._processor = None
        self._model = None
        self._resolved_device = None

    def load(self) -> None:
        if self._model is None or self._processor is None:
            resolved = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._processor = BlipProcessor.from_pretrained(self.model_id)
            self._model = BlipForConditionalGeneration.from_pretrained(self.model_id)
            if resolved == "cuda":
                self._model = self._model.to("cuda")
            self._resolved_device = resolved

    def infer(self, image_path: str, prompt: str) -> dict:
        self.load()
        image = Image.open(image_path).convert("RGB")

        # BLIP: if prompt provided, run conditional captioning, else unconditional
        with torch.inference_mode():
            if prompt and prompt.strip():
                inputs = self._processor(image, prompt, return_tensors="pt")
            else:
                inputs = self._processor(image, return_tensors="pt")
            if self._resolved_device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            out = self._model.generate(**inputs)
            text = self._processor.decode(out[0], skip_special_tokens=True)

        return {"text": text, "device": self._resolved_device or self.device or "cpu"}


