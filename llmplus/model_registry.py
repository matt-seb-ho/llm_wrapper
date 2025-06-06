import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from functools import cache

from dotenv import load_dotenv

_NO_KEY = object()  # sentinel → “no key required”


@dataclass(frozen=True, slots=True)
class ModelMeta:
    name: str
    param_renaming: dict[str, str] | None = None
    unsupported_kw: tuple[str, ...] = ("seed",)


@dataclass(frozen=True, slots=True)
class ProviderMeta:
    env_key: str | object
    base_url: str
    supports_multi: bool
    models: dict[str, ModelMeta]  # model‑name → meta

    # -- helpers -------------------------------------------------------------
    def api_key(self, dotenv_path: str | Path | None = None) -> str | None:
        if self.env_key is _NO_KEY:
            return None
        if dotenv_path:
            load_dotenv(dotenv_path)
        return os.getenv(self.env_key)


class Provider(Enum):
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    VLLM = "vllm"
    SGLANG = "sglang"


MODEL_REGISTRY: dict[Provider, ProviderMeta] = {
    Provider.OPENAI: ProviderMeta(
        env_key="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        supports_multi=True,
        models={
            "gpt-4o": ModelMeta("gpt-4o", unsupported_kw=()),
            "gpt-4.1-2025-04-14": ModelMeta(
                "gpt-4.1-2025-04-14",
                param_renaming={"max_tokens": "max_completion_tokens"},
                unsupported_kw=(),
            ),
            "o3-mini-2025-01-31": ModelMeta(
                "o3-mini-2025-01-31",
                param_renaming={"max_tokens": "max_completion_tokens"},
                unsupported_kw=("temperature", "top_p"),
            ),
            "o4-mini-2025-04-16": ModelMeta(
                "o4-mini-2025-04-16",
                param_renaming={"max_tokens": "max_completion_tokens"},
                unsupported_kw=("temperature", "top_p"),
            ),
        },
    ),
    Provider.DEEPSEEK: ProviderMeta(
        env_key="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com/v1",
        supports_multi=False,
        models={
            "deepseek-chat": ModelMeta("deepseek-chat"),
            "deepseek-reasoner": ModelMeta("deepseek-reasoner"),
        },
    ),
    Provider.VLLM: ProviderMeta(
        env_key=_NO_KEY,  # local endpoint
        base_url="http://localhost:8100/v1",
        supports_multi=True,
        models={
            "meta-llama/Meta-Llama-3-70B-Instruct": ModelMeta(
                "meta-llama/Meta-Llama-3-70B-Instruct"
            ),
        },
    ),
    Provider.SGLANG: ProviderMeta(
        env_key=_NO_KEY,
        base_url="http://127.0.0.1:30000/v1",
        supports_multi=True,
        models={
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": ModelMeta(
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
            ),
        },
    ),
}


def register_model(provider: Provider, name: str, **kwargs):
    MODEL_REGISTRY[provider].models[name] = ModelMeta(name, **kwargs)
