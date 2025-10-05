from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

from .base import VLMModel


class PixtralAdapter(VLMModel):
    def __init__(self, model_id: str = "mgoin/pixtral-12b", device: str = None):
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

        user_text = prompt or "Describe the image."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_text},
                ],
            }
        ]

        with torch.inference_mode():
            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            # Move tensors to device if needed
            if self._resolved_device == "cuda":
                inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}

            outputs = self._model.generate(**inputs, max_new_tokens=64)
            # Decode only the generated continuation like reference snippet
            input_len = inputs["input_ids"].shape[-1]
            decoded = self._processor.decode(outputs[0][input_len:])

        return {"text": decoded, "device": self._resolved_device or self.device or "cpu"}


