from typing import Dict, Callable

from .moondream_adapter import MoondreamAdapter
from .llava_ollama_adapter import LLaVAOllamaAdapter
from .blip_adapter import BLIPAdapter
from .unsloth_adapter import UnslothLlamaVisionAdapter
from .pixtral_adapter import PixtralAdapter


_registry: Dict[str, Callable[[], object]] = {
    "moondream": lambda: MoondreamAdapter(),
    "ollama_llava": lambda: LLaVAOllamaAdapter(),
    "blip": lambda: BLIPAdapter(),
    "unsloth_vision": lambda: UnslothLlamaVisionAdapter(),
    "pixtral": lambda: PixtralAdapter(),
}

_instances: Dict[str, object] = {}


def get_model(name: str):
    if name not in _registry:
        raise ValueError(f"Unknown model: {name}")
    if name not in _instances:
        instance = _registry[name]()
        instance.load()
        _instances[name] = instance
    return _instances[name]


def list_models():
    return list(_registry.keys())


