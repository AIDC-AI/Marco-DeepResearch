import asyncio
import itertools
import random
from typing import Iterable, Union

import aiohttp
from aiohttp import ClientTimeout

class AsyncLLMClient:
    """Minimal OpenAI-compatible async chat-completions client."""

    def __init__(
        self,
        url: Union[str, Iterable[str]],
        api_code: str,
        ak: str = "EMPTY",
        max_concurrency: int = 16,
        timeout: int = 120,
        max_retries: int = 3,
        **generation_kwargs,
    ):
        urls = list(url) if isinstance(url, (list, tuple)) else [url]
        self.urls = [self._normalize_url(item) for item in urls]
        self.url_iterator = itertools.cycle(self.urls)
        self.api_key = ak
        self.model = api_code
        self.timeout = timeout
        self.max_retries = max_retries
        self.generation_kwargs = generation_kwargs
        self.max_concurrency = max_concurrency
        self._loop_resources = {}

    @staticmethod
    def _normalize_url(url: str) -> str:
        url = url.strip()
        if not url:
            raise ValueError("LLM API URL must not be empty.")
        if "chat/completions" not in url:
            return f"{url.rstrip('/')}/chat/completions"
        return url

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _build_payload(self, system_prompt: str, query: str) -> dict:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        passthrough_keys = (
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
        )
        for key in passthrough_keys:
            value = self.generation_kwargs.get(key)
            if value is not None:
                payload[key] = value

        min_p = self.generation_kwargs.get("min_p", self.generation_kwargs.get("MinP"))
        if min_p is not None:
            payload["min_p"] = min_p

        if "enable_thinking" in self.generation_kwargs:
            payload["chat_template_kwargs"] = {
                "enable_thinking": self._as_bool(self.generation_kwargs.get("enable_thinking"))
            }

        return payload

    async def _post_once(self, session: aiohttp.ClientSession, payload: dict) -> str:
        target_url = next(self.url_iterator)
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with session.post(target_url, headers=headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"LLM request failed with HTTP {response.status}: {error_text}")

            response_json = await response.json()
            choices = response_json.get("choices") or []
            if not choices:
                raise RuntimeError(f"LLM response did not contain choices: {response_json}")

            message = choices[0].get("message") or {}
            return (message.get("content") or "").strip()

    async def call(self, system_prompt: str, query: str, idx: int = 0) -> str:
        current_loop = asyncio.get_running_loop()
        if current_loop not in self._loop_resources:
            self._loop_resources[current_loop] = asyncio.Semaphore(self.max_concurrency)

        semaphore = self._loop_resources[current_loop]
        payload = self._build_payload(system_prompt, query)
        timeout = ClientTimeout(total=self.timeout)

        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            async with semaphore:
                for attempt in range(1, self.max_retries + 1):
                    try:
                        return await self._post_once(session, payload)
                    except Exception:
                        if attempt == self.max_retries:
                            return ""
                        await asyncio.sleep(min(30.0, attempt + random.random()))

        return ""
