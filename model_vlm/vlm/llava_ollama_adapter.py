import base64
import os
import ollama

from .base import VLMModel


class LLaVAOllamaAdapter(VLMModel):
    def __init__(self, model_name: str = "llava-llama3:8b"):
        self.model_name = model_name
        self._device_label = f"ollama:{os.getenv('OLLAMA_HOST', 'ollama:11434')}"

    def load(self) -> None:
        # Ollama client is stateless; server handles model caching
        return None

    def infer(self, image_path: str, prompt: str) -> dict:
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode("utf-8")

        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt, "images": [base64_image]}],
        )

        return {"text": response["message"]["content"], "device": self._device_label}


