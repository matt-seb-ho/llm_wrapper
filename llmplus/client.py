import asyncio
import dataclasses
import logging
import random
import typing as t
from datetime import UTC, datetime
from pathlib import Path

import diskcache as dc
import orjson
from openai import AsyncOpenAI
from openai.types import CompletionUsage
from tqdm import tqdm

from .providers import PROVIDERS, ModelMeta, Provider
from .utils import stable_hash, transient_retry

# Libraries should ship no handlers to avoid double logging or “No handlers…” warnings.
# - if the host application calls logging.basicConfig() (or sets handlers/levels explicitly),
#   this library’s logs will propagate and honour that configuration.
# - if users do nothing, this library stays silent
# - users have full control by touching the root (or a parent) logger.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclasses.dataclass
class GenerationConfig:
    n: int = 1
    temperature: float = 0.3
    max_tokens: int = 1024
    top_p: float = 1.0
    seed: int | None = None
    extra_kwargs: dict[str, t.Any] | None = None

    def override(self, **kwargs) -> "GenerationConfig":
        """Create a new instance with updated values."""
        arg_dict = dataclasses.asdict(self)
        extra_kwargs = arg_dict.pop("extra_kwargs", {})
        for k, v in kwargs.items():
            if k in arg_dict:
                arg_dict[k] = v
            else:
                extra_kwargs[k] = v
        return GenerationConfig(**arg_dict, extra_kwargs=extra_kwargs)

    # expose kwargs for OpenAI client (drop `n`)
    def to_kwargs(self, model_meta: ModelMeta) -> dict[str, t.Any]:
        param_dict = dataclasses.asdict(self)
        extra_kwargs = param_dict.pop("extra_kwargs", None) or {}
        # apply renaming
        param_renaming = model_meta.param_renaming or {}
        for default_name, custom_name in param_renaming.items():
            if default_name in param_dict:
                param_dict[custom_name] = param_dict.pop(default_name)
            if default_name in extra_kwargs:
                extra_kwargs[custom_name] = extra_kwargs.pop(default_name)
        # apply extra kwargs
        param_dict.update(extra_kwargs)
        # remove unsupported args
        for unsupported_arg in model_meta.unsupported_kw:
            param_dict.pop(unsupported_arg, None)
        return param_dict


@dataclasses.dataclass
class RetryConfig:
    attempts: int = 5
    wait_min: int = 1
    wait_max: int = 120


class LLMClient:
    def __init__(
        self,
        provider: Provider = Provider.OPENAI,
        cache_dir: str = ".llm_cache",
        system_prompt: str | None = None,
        default_max_concurrency: int = 32,
        retry_cfg: RetryConfig | None = None,
        dotenv_path: str | Path | None = None,
    ):
        self.provider_meta = PROVIDERS[provider]
        self._client_async = AsyncOpenAI(
            api_key=self.provider_meta.api_key(dotenv_path=dotenv_path),
            base_url=self.provider_meta.base_url,
        )

        # default semaphore (used when caller does *not* supply one)
        self._default_sem = asyncio.Semaphore(default_max_concurrency)

        # caches
        self._resp_cache = dc.Cache(cache_dir)
        self.session_stats: dict[str, dict[str, int]] = {}

        # timestamps
        self._session_start = datetime.now(tz=UTC).isoformat()
        self._last_request: str | None = None

        # retry decorator
        self.retry_cfg = retry_cfg or RetryConfig()
        self._retry_deco = None
        self._configure_retry(self.retry_cfg)

        self.system_prompt = system_prompt

    def generate(
        self,
        prompt: str | list[dict],
        *,
        model: str | None = None,
        gen_cfg: GenerationConfig | None = None,
        ignore_cache: bool = False,
        expand_multi: bool | None = None,
        **gen_kwargs,
    ) -> list[str]:
        return asyncio.run(
            self.async_generate(
                prompt,
                model=model,
                gen_cfg=gen_cfg,
                ignore_cache=ignore_cache,
                expand_multi=expand_multi,
                **gen_kwargs,
            )
        )

    # ------------------------------------------------------------------
    # core async generation api
    # ------------------------------------------------------------------
    async def async_generate(
        self,
        prompt: str | list[dict],
        *,
        model: str | None = None,
        gen_cfg: GenerationConfig | None = None,
        ignore_cache: bool = False,
        expand_multi: bool | None = None,
        request_sem: asyncio.Semaphore | None = None,
        **gen_kwargs,
    ) -> list[str]:
        # update generation config with kwarg overrides
        gen_cfg = gen_cfg or GenerationConfig()
        if gen_kwargs:
            gen_cfg = gen_cfg.override(**gen_kwargs)
        n = gen_cfg.n

        model = model or next(iter(self.provider_meta.models))
        assert model in self.provider_meta.models, f"Unknown model {model}"
        mmeta = self.provider_meta.models[model]

        expand_multi = (
            (not self.provider_meta.supports_multi)
            if expand_multi is None
            else expand_multi
        )

        cache_key = self._make_cache_key(prompt, model, mmeta, gen_cfg)
        cached = self._resp_cache.get(cache_key) or []
        usable_cached = [] if ignore_cache else cached
        if usable_cached and len(usable_cached) >= n:
            return random.sample(usable_cached, n)

        # only fetch missing samples
        gen_cfg.n = n - len(usable_cached)
        fetched = await self._request_completions(
            prompt=prompt,
            model=model,
            model_meta=mmeta,
            gen_cfg=gen_cfg,
            expand_multi=expand_multi,
            request_sem=request_sem,
        )
        self._resp_cache[cache_key] = cached + fetched
        return usable_cached + fetched

    # ------------------------------------------------------------------
    # batch helper – single semaphore governs *HTTP* concurrency
    # ------------------------------------------------------------------
    async def async_batch_generate(
        self,
        prompts: list[str | list[dict]],
        *,
        model: str | None = None,
        args: GenerationConfig | None = None,
        batch_size: int = 16,
        ignore_cache: bool = False,
        progress_file: str | Path = "batch_progress.json",
        show_progress: bool = True,
        **gen_kwargs,
    ) -> list[list[str]]:
        request_sem = asyncio.Semaphore(batch_size)
        results: list[list[str]] = [[] for _ in prompts]
        pbar = tqdm(total=len(prompts), disable=not show_progress)
        file_path = Path(progress_file).expanduser() if progress_file else None
        file_lock = asyncio.Lock()

        async def _job(idx: int, prm: str | list[dict]):
            try:
                res = await self.async_generate(
                    prm,
                    model=model,
                    gen_cfg=args,
                    ignore_cache=ignore_cache,
                    request_sem=request_sem,
                    **gen_kwargs,
                )
                results[idx] = res
            except Exception as e:
                logger.error("Prompt %s failed: %s", idx, e, exc_info=False)
                results[idx] = [None] * (args.n if args else gen_kwargs.get("n", 1))
            finally:
                pbar.update()
                if file_path:
                    async with file_lock:
                        tmp = file_path.with_suffix(".tmp")
                        tmp.write_bytes(orjson.dumps(results))
                        tmp.replace(file_path)

        await asyncio.gather(*(_job(i, p) for i, p in enumerate(prompts)))
        pbar.close()
        return results

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save_session_usage(self, path: str | Path = "session_usage.json") -> None:
        record = {
            "session_start": self._session_start,
            "session_end": self._last_request,
            "stats": self.session_stats,
        }
        Path(path).write_bytes(orjson.dumps(record))

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    async def _request_completions(
        self,
        prompt: str | list[dict],
        model: str,
        model_meta: ModelMeta,
        gen_cfg: GenerationConfig,
        expand_multi: bool,
        request_sem: asyncio.Semaphore | None = None,
    ) -> list[str]:
        n = gen_cfg.n
        if n <= 0:
            return []
        request_sem = request_sem or self._default_sem
        if expand_multi:
            expand_multi_cfg = dataclasses.replace(gen_cfg, n=1)
            tasks = [
                self._async_request(
                    prompt=prompt,
                    model=model,
                    model_meta=model_meta,
                    gen_cfg=expand_multi_cfg,
                    request_sem=request_sem,
                )
                for _ in range(n)
            ]
            results: list[str] = []
            for coro in asyncio.as_completed(tasks):
                try:
                    results.extend(await coro)
                except Exception as e:
                    logger.error("one sub‑request failed: %s", e, exc_info=False)
            return results
        else:
            try:
                return await self._async_request(
                    prompt=prompt,
                    model=model,
                    model_meta=model_meta,
                    gen_cfg=gen_cfg,
                    request_sem=request_sem,
                )
            except Exception as e:
                logger.error("one request failed: %s", e, exc_info=False)
                return []

    @transient_retry()
    async def _async_request(
        self,
        prompt: str | list[dict],
        model: str,
        model_meta: ModelMeta,
        gen_cfg: GenerationConfig,
        request_sem: asyncio.Semaphore,
    ) -> list[str]:
        @self._retry_deco
        async def _send():
            async with request_sem:
                msgs = self._format_chat(prompt)
                gen_kwargs = gen_cfg.to_kwargs(model_meta)
                resp = await self._client_async.chat.completions.create(
                    model=model,
                    messages=msgs,
                    **gen_kwargs,
                )
                return resp

        resp = await _send()
        self._acc_usage(model, resp.usage, len(resp.choices))
        return [c.message.content for c in resp.choices]

    def _format_chat(self, inp: str | list[dict]):
        if isinstance(inp, str):
            msgs = [{"role": "user", "content": inp}]
        else:
            msgs = inp
        if self.system_prompt:
            msgs = [{"role": "system", "content": self.system_prompt}, *msgs]
        return msgs

    def _acc_usage(self, model: str, usage: CompletionUsage, num_completions: int):
        stats = self.session_stats.setdefault(
            model,
            {"input_tokens": 0, "output_tokens": 0, "requests": 0, "completions": 0},
        )
        stats["input_tokens"] += usage.prompt_tokens
        stats["output_tokens"] += usage.completion_tokens
        stats["requests"] += 1
        stats["completions"] += num_completions
        self._last_request = datetime.now(tz=UTC).isoformat()

    def _make_cache_key(
        self,
        prompt: str | list[dict],
        model: str,
        model_meta: ModelMeta,
        args: GenerationConfig,
    ):
        key_attributes = args.to_kwargs(model_meta)
        key_attributes.pop("n", None)
        key_attributes["model"] = model
        key_attributes["prompt"] = self._format_chat(prompt)
        return stable_hash(key_attributes)

    def _configure_retry(self, retry_cfg: RetryConfig | None = None):
        self._retry_deco = transient_retry(
            **(dataclasses.asdict(retry_cfg or self.retry_cfg))
        )
