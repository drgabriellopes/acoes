"""
Provedores de LLM para o app de debates.
Cada provedor implementa a interface call(messages) -> dict
"""

import os
import json
import asyncio
import aiohttp
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMModel:
    id: str
    name: str
    provider: str


# ============================================================
# Preços por modelo (USD por milhão de tokens)
# ============================================================
MODEL_PRICING = {
    # Google Gemini
    "gemini-3.1-pro-preview": {"input": 2.0, "output": 12.0, "context_window": 1048576},
    "gemini-3-flash-preview": {"input": 0.5, "output": 3.0, "context_window": 1048576},
    "gemini-3.1-flash-lite-preview": {"input": 0.25, "output": 1.5, "context_window": 1048576},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.0, "context_window": 1048576},
    "gemini-2.5-flash": {"input": 0.3, "output": 2.5, "context_window": 1048576},
    "gemini-2.5-flash-lite": {"input": 0.075, "output": 0.3, "context_window": 1048576},
    # Anthropic
    "claude-opus-4-6": {"input": 5.0, "output": 25.0, "context_window": 1000000},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "context_window": 1000000},
    "claude-opus-4-5": {"input": 5.0, "output": 25.0, "context_window": 1000000},
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0, "context_window": 1000000},
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0, "context_window": 1000000},
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0, "context_window": 200000},
    # OpenAI
    "gpt-5.4": {"input": 2.5, "output": 15.0, "context_window": 1048576},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.5, "context_window": 1048576},
    "gpt-5.4-nano": {"input": 0.05, "output": 0.4, "context_window": 1048576},
    "gpt-5.2": {"input": 1.25, "output": 10.0, "context_window": 1048576},
    "gpt-4.1": {"input": 2.0, "output": 8.0, "context_window": 1048576},
    "gpt-4.1-mini": {"input": 0.4, "output": 1.6, "context_window": 1048576},
    "gpt-4.1-nano": {"input": 0.1, "output": 0.4, "context_window": 1048576},
    "gpt-4o": {"input": 2.5, "output": 10.0, "context_window": 128000},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6, "context_window": 128000},
    "o3": {"input": 2.0, "output": 8.0, "context_window": 200000},
    "o3-mini": {"input": 1.1, "output": 4.4, "context_window": 200000},
    "o4-mini": {"input": 1.1, "output": 4.4, "context_window": 200000},
    # DeepSeek
    "deepseek-chat": {"input": 0.32, "output": 0.89, "context_window": 164000},
    "deepseek-reasoner": {"input": 0.70, "output": 2.50, "context_window": 164000},
    # Kimi
    "kimi-k2.5": {"input": 0.5, "output": 2.0, "context_window": 256000},
    "kimi-k2-0905-preview": {"input": 0.5, "output": 2.0, "context_window": 256000},
    "kimi-k2-turbo-preview": {"input": 0.3, "output": 1.0, "context_window": 256000},
    "kimi-k2-thinking": {"input": 0.5, "output": 2.0, "context_window": 256000},
    "kimi-k2-thinking-turbo": {"input": 0.3, "output": 1.0, "context_window": 256000},
    "moonshot-v1-128k": {"input": 0.8, "output": 0.8, "context_window": 128000},
    "moonshot-v1-32k": {"input": 0.3, "output": 0.3, "context_window": 32000},
    "moonshot-v1-8k": {"input": 0.15, "output": 0.15, "context_window": 8000},
    # Z.ai / GLM
    "glm-5": {"input": 1.0, "output": 4.0, "context_window": 200000},
    "glm-5-turbo": {"input": 0.5, "output": 2.0, "context_window": 128000},
    "glm-4.7": {"input": 0.5, "output": 2.0, "context_window": 128000},
    "glm-4.7-flash": {"input": 0.0, "output": 0.0, "context_window": 128000},
    "glm-4.6": {"input": 0.5, "output": 2.0, "context_window": 128000},
    "glm-4.5": {"input": 0.3, "output": 1.0, "context_window": 128000},
    "glm-4.5-air": {"input": 0.1, "output": 0.3, "context_window": 128000},
    "glm-4.5-airx": {"input": 0.1, "output": 0.3, "context_window": 128000},
    "glm-4.5-flash": {"input": 0.0, "output": 0.0, "context_window": 128000},
    "glm-4-plus": {"input": 0.5, "output": 2.0, "context_window": 128000},
    # Grok
    "grok-4.20-0309-reasoning": {"input": 2.0, "output": 6.0, "context_window": 2000000},
    "grok-4.20-0309-non-reasoning": {"input": 2.0, "output": 6.0, "context_window": 2000000},
    "grok-4-1-fast-reasoning": {"input": 0.2, "output": 0.5, "context_window": 131000},
    "grok-4-1-fast-non-reasoning": {"input": 0.2, "output": 0.5, "context_window": 131000},
    "grok-3": {"input": 3.0, "output": 15.0, "context_window": 131000},
    "grok-3-mini": {"input": 0.3, "output": 0.5, "context_window": 131000},
    # Perplexity
    "sonar-pro": {"input": 3.0, "output": 15.0, "context_window": 200000},
    "sonar": {"input": 1.0, "output": 1.0, "context_window": 127000},
    "sonar-reasoning-pro": {"input": 3.0, "output": 15.0, "context_window": 200000},
    "sonar-deep-research": {"input": 3.0, "output": 15.0, "context_window": 200000},
    # Qwen
    "qwen3-max": {"input": 1.0, "output": 4.0, "context_window": 131000},
    "qwen3.5-plus": {"input": 0.26, "output": 1.56, "context_window": 128000},
    "qwen3.5-flash": {"input": 0.1, "output": 0.3, "context_window": 128000},
    "qwen-plus": {"input": 0.5, "output": 2.0, "context_window": 128000},
    "qwen-turbo": {"input": 0.1, "output": 0.3, "context_window": 128000},
    "qwq-plus": {"input": 0.5, "output": 2.0, "context_window": 128000},
    "qwen3-235b-a22b": {"input": 0.5, "output": 2.0, "context_window": 131000},
    "qwen3-32b": {"input": 0.2, "output": 0.6, "context_window": 128000},
    "qwen3-14b": {"input": 0.1, "output": 0.3, "context_window": 128000},
    "qwen3-8b": {"input": 0.05, "output": 0.15, "context_window": 128000},
    "qwen3-coder-plus": {"input": 0.5, "output": 2.0, "context_window": 128000},
    # Mistral
    "mistral-large-latest": {"input": 2.0, "output": 6.0, "context_window": 128000},
    "mistral-medium-latest": {"input": 1.0, "output": 3.0, "context_window": 128000},
    "mistral-small-latest": {"input": 0.15, "output": 0.6, "context_window": 128000},
    "magistral-medium-latest": {"input": 1.0, "output": 4.0, "context_window": 128000},
    "magistral-small-latest": {"input": 0.5, "output": 2.0, "context_window": 128000},
    "codestral-latest": {"input": 0.3, "output": 0.9, "context_window": 128000},
    "ministral-8b-latest": {"input": 0.05, "output": 0.15, "context_window": 128000},
    # OpenRouter
    "meta-llama/llama-4-maverick": {"input": 0.15, "output": 0.6, "context_window": 128000},
    "meta-llama/llama-4-scout": {"input": 0.08, "output": 0.3, "context_window": 128000},
    "meta-llama/llama-3.3-70b-instruct": {"input": 0.1, "output": 0.3, "context_window": 128000},
    "nvidia/llama-3.1-nemotron-ultra-253b-v1": {"input": 0.6, "output": 1.8, "context_window": 128000},
    "deepseek/deepseek-r1-0528": {"input": 0.45, "output": 2.15, "context_window": 164000},
    "minimax/minimax-m2.5": {"input": 0.2, "output": 1.17, "context_window": 128000},
    "cohere/command-a": {"input": 2.5, "output": 10.0, "context_window": 128000},
    "xiaomi/mimo-v2-pro": {"input": 1.0, "output": 3.0, "context_window": 1000000},
    "xiaomi/mimo-v2-flash": {"input": 0.15, "output": 0.6, "context_window": 1000000},
}

# Models to hide from selection UI (broken, incompatible, or not chat models)
_HIDDEN_MODELS = {
    # Safety/classification models (not for chat)
    "meta-llama/llama-guard-3-8b", "meta-llama/llama-guard-4-12b",
    # Router models (not actual models)
    "openrouter/auto", "openrouter/free", "openrouter/bodybuilder",
    "recursal/switchpoint-router",
    # Audio/image models (not text chat)
    "openai/gpt-4o-audio", "openai/gpt-4o-audio-mini", "openai/gpt-audio",
    "openai/gpt-5-image", "openai/gpt-5-image-mini",
    # Models that consistently produce gibberish
    "bytedance/ui-tars-7b", "essentialai/rnj-1-instruct",
    "alfredpros/codellama-7b-instruct-solidity",
    "eleutherai/llemma-7b",
    # Models that refuse tasks or are off-topic
    "aionlabs/aion-rp-1.0-8b", "mancer/weaver",
    "meta-llama/llama-3.2-1b-instruct", "meta-llama/llama-3.1-8b-instruct",
    # Models that consistently return empty/0 words
    "openai/o3-deep-research", "openai/gpt-5-pro",
    "prime-intellect/intellect-3",
    "google/nano-banana-pro",
    # Models with 404 (no endpoints)
    "allenai/olmo-2-32b-instruct",
    "openai/gpt-oss-120b:free", "openai/gpt-oss-20b:free",
    "minimax/minimax-m2.5:free",
    # Consistently degenerate
    "thedrummer/rocinante-12b", "thedrummer/skyfall-36b-v2",
    "sao10k/llama-3-euryale-70b-v2.1",
    "liquid/lfm-2.5-1.2b-thinking:free", "liquid/lfm-2.5-1.2b-instruct:free",
    "liquid/lfm2-2.6b", "liquid/lfm2-8b-a1b",
    "reka/reka-edge",
    # Arcee models returning 400
    "arcee-ai/spotlight", "arcee-ai/maestro-reasoning",
    "arcee-ai/coder-large", "arcee-ai/virtuoso-large",
    # Morph models returning 400
    "morph/morph-v3-large", "morph/morph-v3-fast",
    "relace/relace-apply-3",
    # Gemma models returning 400 via OpenRouter (system prompts not supported)
    "google/gemma-3-12b-it:free", "google/gemma-3-4b-it:free",
    "google/gemma-3n-2b-it:free", "google/gemma-3n-e2b-it:free", "google/gemma-3n-e4b-it:free",
    # AllenAI OLMo with 404 (no endpoints matching data policy)
    "allenai/olmo-2-0325-32b-instruct",
}


class BaseLLMProvider(ABC):
    name: str
    api_key_env: str
    models: list[LLMModel]
    base_url: str

    def is_configured(self) -> bool:
        return bool(os.getenv(self.api_key_env, ""))

    def get_api_key(self) -> str:
        return os.getenv(self.api_key_env, "")

    @abstractmethod
    async def call(self, model_id: str, messages: list[dict], max_tokens: int = 1024) -> dict:
        """Retorna {content, usage, finish_reason, cost}"""
        pass

    async def stream(self, model_id: str, messages: list[dict], max_tokens: int = 1024):
        """Async generator. Override nos providers que suportam streaming real."""
        result = await self.call(model_id, messages, max_tokens)
        yield {"type": "token", "content": result["content"]}
        yield {"type": "done", "content": result["content"], "usage": result.get("usage", {}), "finish_reason": result.get("finish_reason", "stop"), "cost": result.get("cost", 0)}

    def _calc_cost(self, model_id: str, usage: dict) -> float:
        p = MODEL_PRICING.get(model_id, {"input": 0, "output": 0})
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        return (pt * p["input"] + ct * p["output"]) / 1_000_000

    async def _post_json(self, url: str, headers: dict, payload: dict) -> dict:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise Exception(f"[{self.name}] HTTP {resp.status}: {text[:300]}")
                return await resp.json()

    async def _call_via_openrouter(self, or_model_id: str, messages: list[dict], max_tokens: int, display_model_id: str) -> dict:
        """Primário: chama via OpenRouter. Fallback é a API nativa."""
        or_key = os.getenv("OPENROUTER_API_KEY", "")
        if not or_key:
            raise Exception(f"[{self.name}] OpenRouter não configurado")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {or_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://synapse-colosseum.app",
                    "X-Title": "Synapse Colosseum",
                },
                json={"model": or_model_id, "messages": messages, "max_tokens": max_tokens},
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise Exception(f"[{self.name} via OR] HTTP {resp.status}: {text[:300]}")
                data = await resp.json()
        return _openai_result(data, display_model_id, self._calc_cost)


# ============================================================
# Helper: resposta padrão para provedores OpenAI-compatible
# ============================================================
async def _stream_openai_compat(url, headers, payload, model_id, calc_cost, provider_name):
    """Async generator: yield dicts com type='token' ou type='done'."""
    payload = {**payload, "stream": True}
    full_content = ""
    usage = {}
    finish_reason = "stop"

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=180)) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"[{provider_name}] HTTP {resp.status}: {text[:300]}")

            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = data.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                token = delta.get("content", "")
                # Fallback: reasoning models may use reasoning_content
                if not token:
                    token = delta.get("reasoning_content", "") or delta.get("reasoning", "")
                # Handle content as list (Mistral Magistral sends [{type:"text",text:"..."}])
                if isinstance(token, list):
                    parts = []
                    for item in token:
                        if isinstance(item, dict):
                            # Ensure the extracted value is a string (guard against nested lists)
                            val = item.get("text") or item.get("content") or ""
                            parts.append(val if isinstance(val, str) else str(val))
                        elif isinstance(item, str):
                            parts.append(item)
                        else:
                            parts.append(str(item))
                    token = "".join(parts)
                # Final guard: ensure token is always a string before concatenation
                if token is not None and not isinstance(token, str):
                    token = str(token)
                if token and isinstance(token, str):
                    full_content += token
                    yield {"type": "token", "content": token}

                fr = choices[0].get("finish_reason")
                if fr:
                    finish_reason = fr

                if data.get("usage"):
                    usage = {
                        "prompt_tokens": data["usage"].get("prompt_tokens", 0),
                        "completion_tokens": data["usage"].get("completion_tokens", 0),
                    }

    if not usage:
        usage = {"prompt_tokens": 0, "completion_tokens": max(1, len(full_content) // 4)}

    yield {
        "type": "done",
        "content": full_content,
        "usage": usage,
        "finish_reason": finish_reason,
        "cost": calc_cost(model_id, usage),
    }


def _openai_result(data: dict, model_id: str, calc_cost) -> dict:
    # Check for error in response
    if "error" in data:
        raise Exception(f"API error: {data['error'].get('message', str(data['error']))[:200]}")
    choices = data.get("choices") or []
    if not choices:
        raise Exception(f"Resposta vazia do modelo (sem choices)")
    choice = choices[0]
    usage = data.get("usage", {})
    std_usage = {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }
    # Handle content that might be string, list, or None
    raw_content = (choice.get("message") or {}).get("content") or ""
    if isinstance(raw_content, list):
        parts = []
        for item in raw_content:
            if isinstance(item, dict):
                parts.append(item.get("text", "") or item.get("content", "") or str(item))
            else:
                parts.append(str(item))
        raw_content = " ".join(parts)
    # If content is empty, try reasoning_content (some thinking models)
    if not raw_content:
        raw_content = (choice.get("message") or {}).get("reasoning_content") or ""
        if raw_content:
            raw_content = f"[Raciocínio]\n{raw_content}"
    return {
        "content": raw_content,
        "usage": std_usage,
        "finish_reason": choice.get("finish_reason", "stop"),
        "cost": calc_cost(model_id, std_usage),
    }


# ============================================================
# 1. Google Gemini
# ============================================================
class GeminiProvider(BaseLLMProvider):
    name = "Google Gemini"
    api_key_env = "GEMINI_API_KEY"
    base_url = "https://generativelanguage.googleapis.com/v1beta"
    models = [
        LLMModel("gemini-3.1-pro-preview", "Gemini 3.1 Pro", "Google Gemini"),
        LLMModel("gemini-3-flash-preview", "Gemini 3.0 Flash", "Google Gemini"),
        LLMModel("gemini-3.1-flash-lite-preview", "Gemini 3.1 Flash Lite", "Google Gemini"),
        LLMModel("gemini-2.5-pro", "Gemini 2.5 Pro", "Google Gemini"),
        LLMModel("gemini-2.5-flash", "Gemini 2.5 Flash", "Google Gemini"),
        LLMModel("gemini-2.5-flash-lite", "Gemini 2.5 Flash Lite", "Google Gemini"),
    ]

    async def call(self, model_id, messages, max_tokens=1024):
        system_text = ""
        contents = []
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            else:
                role = "user" if m["role"] == "user" else "model"
                contents.append({"role": role, "parts": [{"text": m["content"]}]})

        gen_config = {"maxOutputTokens": max_tokens, "temperature": 0.8}
        # Desabilitar thinking em modelos 2.5+/3.x para não consumir maxOutputTokens
        # Modelos thinking: gemini-2.5-*, gemini-3-*, gemini-3.1-*
        is_thinking = any(tag in model_id for tag in ["2.5", "3-", "3."])
        if is_thinking:
            gen_config.pop("temperature", None)  # Thinking models don't support temperature
            gen_config["thinkingConfig"] = {"thinkingBudget": 8192}
        payload = {"contents": contents, "generationConfig": gen_config}
        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": system_text}]}

        url = f"{self.base_url}/models/{model_id}:generateContent"
        data = await self._post_json(url, {"Content-Type": "application/json", "x-goog-api-key": self.get_api_key()}, payload)

        candidates = data.get("candidates") or []
        if not candidates:
            feedback = data.get("promptFeedback", {})
            reason = feedback.get("blockReason", "Unknown")
            raise Exception(f"[Google Gemini] Resposta bloqueada: {reason}")
        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        text_parts = [p["text"] for p in parts if "text" in p and not p.get("thought")]
        if not text_parts:
            text_parts = [p["text"] for p in parts if "text" in p]
        text = "\n".join(text_parts) if text_parts else ""

        um = data.get("usageMetadata", {})
        usage = {
            "prompt_tokens": um.get("promptTokenCount", 0),
            "completion_tokens": um.get("candidatesTokenCount", 0),
        }
        fr = candidate.get("finishReason", "STOP")
        fr_map = {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter"}

        return {
            "content": text,
            "usage": usage,
            "finish_reason": fr_map.get(fr, fr.lower()),
            "cost": self._calc_cost(model_id, usage),
        }

    async def stream(self, model_id, messages, max_tokens=1024):
        """Real streaming via Gemini streamGenerateContent endpoint."""
        system_text = ""
        contents = []
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            else:
                role = "user" if m["role"] == "user" else "model"
                contents.append({"role": role, "parts": [{"text": m["content"]}]})

        gen_config = {"maxOutputTokens": max_tokens, "temperature": 0.8}
        is_thinking = any(tag in model_id for tag in ["2.5", "3-", "3."])
        if is_thinking:
            gen_config.pop("temperature", None)  # Thinking models don't support temperature
            gen_config["thinkingConfig"] = {"thinkingBudget": 8192}
        payload = {"contents": contents, "generationConfig": gen_config}
        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": system_text}]}

        url = f"{self.base_url}/models/{model_id}:streamGenerateContent?alt=sse"

        full_content = ""
        usage = {}
        finish_reason = "stop"

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers={"Content-Type": "application/json", "x-goog-api-key": self.get_api_key()}, json=payload, timeout=aiohttp.ClientTimeout(total=180)) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise Exception(f"[Google Gemini] HTTP {resp.status}: {text[:300]}")

                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    candidates = data.get("candidates", [])
                    if not candidates:
                        continue
                    candidate = candidates[0]
                    content = candidate.get("content", {})
                    parts = content.get("parts", [])

                    for part in parts:
                        if "text" in part and not part.get("thought"):
                            token = part["text"]
                            full_content += token
                            yield {"type": "token", "content": token}

                    fr = candidate.get("finishReason")
                    if fr:
                        fr_map = {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter"}
                        finish_reason = fr_map.get(fr, fr.lower())

                    um = data.get("usageMetadata", {})
                    if um:
                        usage = {
                            "prompt_tokens": um.get("promptTokenCount", 0),
                            "completion_tokens": um.get("candidatesTokenCount", 0),
                        }

        if not usage:
            usage = {"prompt_tokens": 0, "completion_tokens": max(1, len(full_content) // 4)}

        yield {
            "type": "done",
            "content": full_content,
            "usage": usage,
            "finish_reason": finish_reason,
            "cost": self._calc_cost(model_id, usage),
        }


# ============================================================
# 2. Anthropic (Claude)
# ============================================================
class AnthropicProvider(BaseLLMProvider):
    name = "Anthropic"
    api_key_env = "ANTHROPIC_API_KEY"
    base_url = "https://api.anthropic.com/v1"
    models = [
        LLMModel("claude-opus-4-6", "Claude Opus 4.6", "Anthropic"),
        LLMModel("claude-sonnet-4-6", "Claude Sonnet 4.6", "Anthropic"),
        LLMModel("claude-opus-4-5", "Claude Opus 4.5", "Anthropic"),
        LLMModel("claude-sonnet-4-5", "Claude Sonnet 4.5", "Anthropic"),
        LLMModel("claude-sonnet-4-20250514", "Claude Sonnet 4.0", "Anthropic"),
        LLMModel("claude-haiku-4-5-20251001", "Claude Haiku 4.5", "Anthropic"),
    ]

    async def call(self, model_id, messages, max_tokens=1024):
        system_msg = ""
        filtered = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                filtered.append(m)
        payload = {"model": model_id, "max_tokens": max_tokens, "messages": filtered}
        if system_msg:
            payload["system"] = system_msg
        data = await self._post_json(
            f"{self.base_url}/messages",
            {
                "x-api-key": self.get_api_key(),
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            payload,
        )
        usage = {
            "prompt_tokens": data.get("usage", {}).get("input_tokens", 0),
            "completion_tokens": data.get("usage", {}).get("output_tokens", 0),
        }
        content_blocks = data.get("content") or []
        text_blocks = [b.get("text", "") for b in content_blocks if b.get("type") == "text"]
        text = "\n".join(text_blocks) if text_blocks else ""
        sr = data.get("stop_reason", "end_turn")
        fr_map = {"end_turn": "stop", "max_tokens": "length"}
        return {
            "content": text,
            "usage": usage,
            "finish_reason": fr_map.get(sr, sr),
            "cost": self._calc_cost(model_id, usage),
        }

    async def stream(self, model_id, messages, max_tokens=1024):
        """Real streaming via Anthropic Messages API with stream=true."""
        system_msg = ""
        filtered = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                filtered.append(m)

        payload = {"model": model_id, "max_tokens": max_tokens, "messages": filtered, "stream": True}
        if system_msg:
            payload["system"] = system_msg

        headers = {
            "x-api-key": self.get_api_key(),
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        full_content = ""
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        finish_reason = "stop"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=180),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise Exception(f"[Anthropic] HTTP {resp.status}: {text[:300]}")

                event_type = None
                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", errors="replace").strip()

                    if not line:
                        continue

                    if line.startswith("event: "):
                        event_type = line[7:].strip()
                        continue

                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if event_type == "content_block_delta":
                        delta = data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            token = delta.get("text", "")
                            if token:
                                full_content += token
                                yield {"type": "token", "content": token}

                    elif event_type == "message_start":
                        msg_usage = data.get("message", {}).get("usage", {})
                        usage["prompt_tokens"] = msg_usage.get("input_tokens", 0)

                    elif event_type == "message_delta":
                        delta = data.get("delta", {})
                        stop = delta.get("stop_reason")
                        if stop:
                            fr_map = {"end_turn": "stop", "max_tokens": "length"}
                            finish_reason = fr_map.get(stop, stop)
                        delta_usage = data.get("usage", {})
                        usage["completion_tokens"] = delta_usage.get("output_tokens", usage["completion_tokens"])

                    elif event_type == "message_stop":
                        break

        if not usage["completion_tokens"]:
            usage["completion_tokens"] = max(1, len(full_content) // 4)

        yield {
            "type": "done",
            "content": full_content,
            "usage": usage,
            "finish_reason": finish_reason,
            "cost": self._calc_cost(model_id, usage),
        }


# ============================================================
# 3. OpenAI (ChatGPT)
# ============================================================
class OpenAIProvider(BaseLLMProvider):
    name = "ChatGPT"
    api_key_env = "OPENAI_API_KEY"
    base_url = "https://api.openai.com/v1"
    models = [
        LLMModel("gpt-5.4", "GPT-5.4", "ChatGPT"),
        LLMModel("gpt-5.4-mini", "GPT-5.4 Mini", "ChatGPT"),
        LLMModel("gpt-5.4-nano", "GPT-5.4 Nano", "ChatGPT"),
        LLMModel("gpt-5.2", "GPT-5.2", "ChatGPT"),
        LLMModel("gpt-4.1", "GPT-4.1", "ChatGPT"),
        LLMModel("gpt-4.1-mini", "GPT-4.1 Mini", "ChatGPT"),
        LLMModel("gpt-4.1-nano", "GPT-4.1 Nano", "ChatGPT"),
        LLMModel("gpt-4o", "GPT-4o", "ChatGPT"),
        LLMModel("gpt-4o-mini", "GPT-4o Mini", "ChatGPT"),
        LLMModel("o3", "o3 (Reasoning)", "ChatGPT"),
        LLMModel("o3-mini", "o3 Mini (Reasoning)", "ChatGPT"),
        LLMModel("o4-mini", "o4 Mini (Reasoning)", "ChatGPT"),
    ]

    def _headers(self):
        return {"Authorization": f"Bearer {self.get_api_key()}", "Content-Type": "application/json"}

    async def call(self, model_id, messages, max_tokens=1024):
        data = await self._post_json(
            f"{self.base_url}/chat/completions", self._headers(),
            {"model": model_id, "messages": messages, "max_completion_tokens": max_tokens},
        )
        return _openai_result(data, model_id, self._calc_cost)

    async def stream(self, model_id, messages, max_tokens=1024):
        async for chunk in _stream_openai_compat(
            f"{self.base_url}/chat/completions", self._headers(),
            {"model": model_id, "messages": messages, "max_completion_tokens": max_tokens},
            model_id, self._calc_cost, self.name,
        ):
            yield chunk


# ============================================================
# 4. DeepSeek
# ============================================================
class DeepSeekProvider(BaseLLMProvider):
    name = "DeepSeek"
    api_key_env = "DEEPSEEK_API_KEY"
    base_url = "https://api.deepseek.com/v1"
    _or_map = {"deepseek-chat": "deepseek/deepseek-chat", "deepseek-reasoner": "deepseek/deepseek-r1"}
    models = [
        LLMModel("deepseek-chat", "DeepSeek Chat (V3)", "DeepSeek"),
        LLMModel("deepseek-reasoner", "DeepSeek Reasoner (R1)", "DeepSeek"),
    ]

    async def call(self, model_id, messages, max_tokens=1024):
        or_id = self._or_map.get(model_id)
        if or_id:
            try:
                return await self._call_via_openrouter(or_id, messages, max_tokens, model_id)
            except Exception as e:
                import logging
                logging.getLogger("debate").debug(f"[{self.name}] OR fallback failed: {e}")
        data = await self._post_json(
            f"{self.base_url}/chat/completions",
            {"Authorization": f"Bearer {self.get_api_key()}", "Content-Type": "application/json"},
            {"model": model_id, "messages": messages, "max_tokens": max_tokens},
        )
        return _openai_result(data, model_id, self._calc_cost)

    async def stream(self, model_id, messages, max_tokens=1024):
        or_id = self._or_map.get(model_id)
        or_key = os.getenv("OPENROUTER_API_KEY", "")
        if or_id and or_key:
            try:
                async for chunk in _stream_openai_compat(
                    "https://openrouter.ai/api/v1/chat/completions",
                    {"Authorization": f"Bearer {or_key}", "Content-Type": "application/json", "HTTP-Referer": "https://synapse-colosseum.app", "X-Title": "Synapse Colosseum"},
                    {"model": or_id, "messages": messages, "max_tokens": max_tokens},
                    model_id, self._calc_cost, f"{self.name} via OR",
                ):
                    yield chunk
                return
            except Exception as e:
                import logging
                logging.getLogger("debate").debug(f"[{self.name}] OR fallback failed: {e}")
        # Fallback: non-streaming via call()
        async for chunk in super().stream(model_id, messages, max_tokens):
            yield chunk


# ============================================================
# 5. Kimi K2 (Moonshot AI)
# ============================================================
class KimiProvider(BaseLLMProvider):
    name = "Kimi"
    api_key_env = "MOONSHOT_API_KEY"
    base_url = "https://api.moonshot.ai/v1"
    # K2 thinking models require temperature=1 (API rejects any other value)
    _temp1_models = {"kimi-k2.5", "kimi-k2-0905-preview", "kimi-k2-turbo-preview", "kimi-k2-thinking", "kimi-k2-thinking-turbo"}
    _or_map = {
        "kimi-k2.5": "moonshotai/kimi-k2.5",
        "kimi-k2-0905-preview": "moonshotai/kimi-k2",
        "kimi-k2-turbo-preview": "moonshotai/kimi-k2",
        "kimi-k2-thinking": "moonshotai/kimi-k2",
        "kimi-k2-thinking-turbo": "moonshotai/kimi-k2",
    }
    models = [
        LLMModel("kimi-k2.5", "Kimi K2.5", "Kimi"),
        LLMModel("kimi-k2-0905-preview", "Kimi K2", "Kimi"),
        LLMModel("kimi-k2-turbo-preview", "Kimi K2 Turbo", "Kimi"),
        LLMModel("kimi-k2-thinking", "Kimi K2 Thinking", "Kimi"),
        LLMModel("kimi-k2-thinking-turbo", "Kimi K2 Thinking Turbo", "Kimi"),
        LLMModel("moonshot-v1-128k", "Moonshot V1 128K", "Kimi"),
        LLMModel("moonshot-v1-32k", "Moonshot V1 32K", "Kimi"),
        LLMModel("moonshot-v1-8k", "Moonshot V1 8K", "Kimi"),
    ]

    async def call(self, model_id, messages, max_tokens=1024):
        or_id = self._or_map.get(model_id)
        if or_id:
            try:
                return await self._call_via_openrouter(or_id, messages, max_tokens, model_id)
            except Exception as e:
                import logging
                logging.getLogger("debate").debug(f"[{self.name}] OR fallback failed: {e}")
        payload = {"model": model_id, "messages": messages, "max_tokens": max_tokens}
        if model_id in self._temp1_models:
            payload["temperature"] = 1
        data = await self._post_json(
            f"{self.base_url}/chat/completions",
            {"Authorization": f"Bearer {self.get_api_key()}", "Content-Type": "application/json"},
            payload,
        )
        return _openai_result(data, model_id, self._calc_cost)

    async def stream(self, model_id, messages, max_tokens=1024):
        or_id = self._or_map.get(model_id)
        or_key = os.getenv("OPENROUTER_API_KEY", "")
        if or_id and or_key:
            try:
                async for chunk in _stream_openai_compat(
                    "https://openrouter.ai/api/v1/chat/completions",
                    {"Authorization": f"Bearer {or_key}", "Content-Type": "application/json", "HTTP-Referer": "https://synapse-colosseum.app", "X-Title": "Synapse Colosseum"},
                    {"model": or_id, "messages": messages, "max_tokens": max_tokens},
                    model_id, self._calc_cost, f"{self.name} via OR",
                ):
                    yield chunk
                return
            except Exception as e:
                import logging
                logging.getLogger("debate").debug(f"[{self.name}] OR fallback failed: {e}")
        # Fallback: native streaming with temperature=1 for K2 thinking models
        payload = {"model": model_id, "messages": messages, "max_tokens": max_tokens}
        if model_id in self._temp1_models:
            payload["temperature"] = 1
        async for chunk in _stream_openai_compat(
            f"{self.base_url}/chat/completions",
            {"Authorization": f"Bearer {self.get_api_key()}", "Content-Type": "application/json"},
            payload, model_id, self._calc_cost, self.name,
        ):
            yield chunk


# ============================================================
# 6. Z.ai / GLM (Zhipu AI)
# ============================================================
class ZaiProvider(BaseLLMProvider):
    name = "Z.ai (GLM)"
    api_key_env = "ZAI_API_KEY"
    base_url = "https://api.z.ai/api/paas/v4"
    _or_map = {
        "glm-5": "z-ai/glm-5",
        "glm-5-turbo": "z-ai/glm-5-turbo",
        "glm-4.7": "z-ai/glm-4.7",
        # glm-4.7-flash and glm-4.5-flash :free IDs don't exist on OpenRouter — removed
    }
    models = [
        # Só modelos com OR fallback (conta Z.ai sem saldo)
        LLMModel("glm-5", "GLM-5", "Z.ai (GLM)"),
        LLMModel("glm-5-turbo", "GLM-5 Turbo", "Z.ai (GLM)"),
        LLMModel("glm-4.7", "GLM-4.7", "Z.ai (GLM)"),
        LLMModel("glm-4.7-flash", "GLM-4.7 Flash", "Z.ai (GLM)"),
        LLMModel("glm-4.5-flash", "GLM-4.5 Flash", "Z.ai (GLM)"),
    ]

    async def call(self, model_id, messages, max_tokens=1024):
        or_id = self._or_map.get(model_id)
        if or_id:
            try:
                return await self._call_via_openrouter(or_id, messages, max_tokens, model_id)
            except Exception as e:
                import logging
                logging.getLogger("debate").debug(f"[{self.name}] OR fallback failed: {e}")
        data = await self._post_json(
            f"{self.base_url}/chat/completions",
            {"Authorization": f"Bearer {self.get_api_key()}", "Content-Type": "application/json"},
            {"model": model_id, "messages": messages, "max_tokens": max_tokens},
        )
        return _openai_result(data, model_id, self._calc_cost)

    async def stream(self, model_id, messages, max_tokens=1024):
        or_id = self._or_map.get(model_id)
        or_key = os.getenv("OPENROUTER_API_KEY", "")
        if or_id and or_key:
            try:
                async for chunk in _stream_openai_compat(
                    "https://openrouter.ai/api/v1/chat/completions",
                    {"Authorization": f"Bearer {or_key}", "Content-Type": "application/json", "HTTP-Referer": "https://synapse-colosseum.app", "X-Title": "Synapse Colosseum"},
                    {"model": or_id, "messages": messages, "max_tokens": max_tokens},
                    model_id, self._calc_cost, f"{self.name} via OR",
                ):
                    yield chunk
                return
            except Exception as e:
                import logging
                logging.getLogger("debate").debug(f"[{self.name}] OR fallback failed: {e}")
        # Fallback: non-streaming via call()
        async for chunk in super().stream(model_id, messages, max_tokens):
            yield chunk


# ============================================================
# 7. Grok (xAI)
# ============================================================
class GrokProvider(BaseLLMProvider):
    name = "Grok"
    api_key_env = "XAI_API_KEY"
    base_url = "https://api.x.ai/v1"
    models = [
        LLMModel("grok-4.20-0309-reasoning", "Grok 4.20 Reasoning", "Grok"),
        LLMModel("grok-4.20-0309-non-reasoning", "Grok 4.20", "Grok"),
        LLMModel("grok-4-1-fast-reasoning", "Grok 4.1 Fast Reasoning", "Grok"),
        LLMModel("grok-4-1-fast-non-reasoning", "Grok 4.1 Fast", "Grok"),
        LLMModel("grok-3", "Grok 3", "Grok"),
        LLMModel("grok-3-mini", "Grok 3 Mini", "Grok"),
    ]

    def _headers(self):
        return {"Authorization": f"Bearer {self.get_api_key()}", "Content-Type": "application/json"}

    async def call(self, model_id, messages, max_tokens=1024):
        data = await self._post_json(f"{self.base_url}/chat/completions", self._headers(), {"model": model_id, "messages": messages, "max_tokens": max_tokens})
        return _openai_result(data, model_id, self._calc_cost)

    async def stream(self, model_id, messages, max_tokens=1024):
        async for chunk in _stream_openai_compat(f"{self.base_url}/chat/completions", self._headers(), {"model": model_id, "messages": messages, "max_tokens": max_tokens}, model_id, self._calc_cost, self.name):
            yield chunk


# ============================================================
# 8. Perplexity
# ============================================================
class PerplexityProvider(BaseLLMProvider):
    name = "Perplexity"
    api_key_env = "PERPLEXITY_API_KEY"
    base_url = "https://api.perplexity.ai"
    models = [
        LLMModel("sonar-pro", "Sonar Pro", "Perplexity"),
        LLMModel("sonar", "Sonar", "Perplexity"),
        LLMModel("sonar-reasoning-pro", "Sonar Reasoning Pro", "Perplexity"),
        LLMModel("sonar-deep-research", "Sonar Deep Research", "Perplexity"),
    ]

    def _headers(self):
        return {"Authorization": f"Bearer {self.get_api_key()}", "Content-Type": "application/json"}

    async def call(self, model_id, messages, max_tokens=1024):
        data = await self._post_json(f"{self.base_url}/chat/completions", self._headers(), {"model": model_id, "messages": messages, "max_tokens": max_tokens})
        return _openai_result(data, model_id, self._calc_cost)

    async def stream(self, model_id, messages, max_tokens=1024):
        async for chunk in _stream_openai_compat(f"{self.base_url}/chat/completions", self._headers(), {"model": model_id, "messages": messages, "max_tokens": max_tokens}, model_id, self._calc_cost, self.name):
            yield chunk


# ============================================================
# 10. Qwen Chat (Alibaba DashScope)
# ============================================================
class QwenProvider(BaseLLMProvider):
    name = "Qwen Chat"
    api_key_env = "DASHSCOPE_API_KEY"
    base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    _or_map = {
        "qwen-turbo": "qwen/qwen-turbo",
        "qwen-plus": "qwen/qwen-plus",
        "qwen3-8b": "qwen/qwen3-8b",
        "qwen3-14b": "qwen/qwen3-14b",
        "qwen3-32b": "qwen/qwen3-32b",
        "qwq-plus": "qwen/qwq-32b",
        "qwen3.5-flash": "qwen/qwen3.5-flash",
        "qwen3.5-plus": "qwen/qwen3.5-plus-02-15",
        "qwen3-max": "qwen/qwen3-max",
    }
    models = [
        LLMModel("qwen-turbo", "Qwen Turbo", "Qwen Chat"),
        LLMModel("qwen-plus", "Qwen Plus", "Qwen Chat"),
        LLMModel("qwen3-8b", "Qwen3 8B", "Qwen Chat"),
        LLMModel("qwen3-14b", "Qwen3 14B", "Qwen Chat"),
        LLMModel("qwen3-32b", "Qwen3 32B", "Qwen Chat"),
        LLMModel("qwq-plus", "QwQ Plus (Reasoning)", "Qwen Chat"),
        LLMModel("qwen3.5-flash", "Qwen3.5 Flash", "Qwen Chat"),
        LLMModel("qwen3.5-plus", "Qwen3.5 Plus", "Qwen Chat"),
        LLMModel("qwen3-max", "Qwen3 Max", "Qwen Chat"),
        # qwen3-235b e qwen3-coder-plus removidos — dão 403 sem ativação no DashScope
    ]

    async def call(self, model_id, messages, max_tokens=1024):
        or_id = self._or_map.get(model_id)
        if or_id:
            try:
                return await self._call_via_openrouter(or_id, messages, max_tokens, model_id)
            except Exception as e:
                import logging
                logging.getLogger("debate").debug(f"[{self.name}] OR fallback failed: {e}")
        data = await self._post_json(
            f"{self.base_url}/chat/completions",
            {"Authorization": f"Bearer {self.get_api_key()}", "Content-Type": "application/json"},
            {"model": model_id, "messages": messages, "max_tokens": max_tokens},
        )
        return _openai_result(data, model_id, self._calc_cost)

    async def stream(self, model_id, messages, max_tokens=1024):
        or_id = self._or_map.get(model_id)
        or_key = os.getenv("OPENROUTER_API_KEY", "")
        if or_id and or_key:
            try:
                async for chunk in _stream_openai_compat(
                    "https://openrouter.ai/api/v1/chat/completions",
                    {"Authorization": f"Bearer {or_key}", "Content-Type": "application/json", "HTTP-Referer": "https://synapse-colosseum.app", "X-Title": "Synapse Colosseum"},
                    {"model": or_id, "messages": messages, "max_tokens": max_tokens},
                    model_id, self._calc_cost, f"{self.name} via OR",
                ):
                    yield chunk
                return
            except Exception as e:
                import logging
                logging.getLogger("debate").debug(f"[{self.name}] OR fallback failed: {e}")
        # Fallback: non-streaming via call()
        async for chunk in super().stream(model_id, messages, max_tokens):
            yield chunk


# ============================================================
# 11. Mistral
# ============================================================
class MistralProvider(BaseLLMProvider):
    name = "Mistral"
    api_key_env = "MISTRAL_API_KEY"
    base_url = "https://api.mistral.ai/v1"
    models = [
        LLMModel("mistral-large-latest", "Mistral Large", "Mistral"),
        LLMModel("mistral-medium-latest", "Mistral Medium", "Mistral"),
        LLMModel("mistral-small-latest", "Mistral Small", "Mistral"),
        LLMModel("magistral-medium-latest", "Magistral Medium", "Mistral"),
        LLMModel("magistral-small-latest", "Magistral Small", "Mistral"),
        LLMModel("codestral-latest", "Codestral", "Mistral"),
        LLMModel("ministral-8b-latest", "Ministral 8B", "Mistral"),
    ]

    def _headers(self):
        return {"Authorization": f"Bearer {self.get_api_key()}", "Content-Type": "application/json"}

    async def call(self, model_id, messages, max_tokens=1024):
        data = await self._post_json(f"{self.base_url}/chat/completions", self._headers(), {"model": model_id, "messages": messages, "max_tokens": max_tokens})
        return _openai_result(data, model_id, self._calc_cost)

    async def stream(self, model_id, messages, max_tokens=1024):
        async for chunk in _stream_openai_compat(f"{self.base_url}/chat/completions", self._headers(), {"model": model_id, "messages": messages, "max_tokens": max_tokens}, model_id, self._calc_cost, self.name):
            yield chunk


# ============================================================
# 12. OpenRouter (gateway — modelos extras não cobertos acima)
# ============================================================
def _fetch_openrouter_models() -> list[LLMModel]:
    """Fetch all text models from OpenRouter API. Called once at startup."""
    import urllib.request
    import json as _json

    try:
        req = urllib.request.Request("https://openrouter.ai/api/v1/models")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = _json.loads(resp.read().decode())

        models = []
        for m in data.get("data", []):
            out_mod = m.get("architecture", {}).get("output_modalities", [])
            if "text" not in out_mod:
                continue

            mid = m["id"]
            name = m.get("name", mid)
            pricing = m.get("pricing", {})
            inp = float(pricing.get("prompt", "0")) * 1e6
            outp = float(pricing.get("completion", "0")) * 1e6

            # Skip router/auto models with negative pricing
            if inp < 0 or outp < 0:
                continue

            # Skip modelos que sabemos que não funcionam para debate de texto
            _BLACKLIST_PATTERNS = [
                "audio", "image", "vision", "tts", "whisper", "dall-e",  # multimodais sem texto
                "guard", "shield", "safety",  # segurança (não geram debate)
                "embed", "embedding",  # embeddings
                "router", "switchpoint",  # routers
                "gpt-4-0314", "gpt-4-turbo-preview", "gpt-4-32k", "gpt-3.5",  # OpenAI legacy
                "gpt-oss",  # modelos OSS sem endpoints
                "olmo-2", "olmo 2",  # endpoints indisponíveis
                "aion-rp", "weaver", "rocinante", "mancer",  # RP/ficção
                "llemma", "codesolidity", "codellama",  # ultra-específicos
                "ui-tars",  # modelo de UI
                "morph-v3", "relace-apply",  # instáveis
                "nano-banana",  # experimental
                "gemma-3n-e2b", "gemma-3n-2b",  # Gemma sem system prompt
                "llama-3.2-1b", "llama-3.1-8b-instruct:free",  # recusam tarefas
                "qwen2.5-coder-7b", "qwen2.5-7b-instruct",  # gibberish
                "rnj-1-instruct",  # gibberish
                "lfm-2.5-1.2b", "lfm2-2b",  # LiquidAI instável
                "reka-edge",  # gibberish
                "spotlight", "maestro-reasoning", "coder-large", "virtuoso-large",  # Arcee instável
            ]
            mid_lower = mid.lower()
            name_lower = name.lower()
            skip = False
            for pattern in _BLACKLIST_PATTERNS:
                if pattern in mid_lower or pattern in name_lower:
                    skip = True
                    break
            if skip:
                continue

            # Skip modelos com contexto muito pequeno (<4K) — inúteis para debate
            ctx = m.get("context_length", 0)
            if 0 < ctx < 4096:
                continue

            # Add to MODEL_PRICING dynamically
            if mid not in MODEL_PRICING:
                MODEL_PRICING[mid] = {"input": round(inp, 4), "output": round(outp, 4), "context_window": m.get("context_length", 128000)}

            models.append(LLMModel(mid, name, "OpenRouter"))

        print(f"[OpenRouter] {len(models)} text models loaded from API")
        return models
    except Exception as e:
        print(f"[OpenRouter] Falha ao carregar modelos: {e}")
        return []


class OpenRouterProvider(BaseLLMProvider):
    name = "OpenRouter"
    api_key_env = "OPENROUTER_API_KEY"
    base_url = "https://openrouter.ai/api/v1"

    # Hardcoded fallback models (used if API fetch fails)
    _fallback_models = [
        LLMModel("meta-llama/llama-4-maverick", "Llama 4 Maverick", "OpenRouter"),
        LLMModel("meta-llama/llama-4-scout", "Llama 4 Scout", "OpenRouter"),
        LLMModel("meta-llama/llama-3.3-70b-instruct", "Llama 3.3 70B", "OpenRouter"),
        LLMModel("nvidia/llama-3.1-nemotron-ultra-253b-v1", "Nemotron Ultra 253B", "OpenRouter"),
        LLMModel("deepseek/deepseek-r1-0528", "DeepSeek R1 0528", "OpenRouter"),
        LLMModel("minimax/minimax-m2.5", "MiniMax M2.5", "OpenRouter"),
        LLMModel("cohere/command-a", "Cohere Command A", "OpenRouter"),
        LLMModel("xiaomi/mimo-v2-pro", "MiMo-V2 Pro", "OpenRouter"),
        LLMModel("xiaomi/mimo-v2-flash", "MiMo-V2 Flash", "OpenRouter"),
    ]

    models = []  # Populated at module load time by _init_models()

    @classmethod
    def _init_models(cls):
        """Called once at module load to populate models."""
        fetched = _fetch_openrouter_models()
        if fetched:
            cls.models = fetched
        else:
            cls.models = cls._fallback_models[:]

    def _headers(self):
        return {"Authorization": f"Bearer {self.get_api_key()}", "Content-Type": "application/json", "HTTP-Referer": "https://synapse-colosseum.app", "X-Title": "Synapse Colosseum"}

    async def call(self, model_id, messages, max_tokens=1024):
        data = await self._post_json(f"{self.base_url}/chat/completions", self._headers(), {"model": model_id, "messages": messages, "max_tokens": max_tokens})
        return _openai_result(data, model_id, self._calc_cost)

    async def stream(self, model_id, messages, max_tokens=1024):
        async for chunk in _stream_openai_compat(f"{self.base_url}/chat/completions", self._headers(), {"model": model_id, "messages": messages, "max_tokens": max_tokens}, model_id, self._calc_cost, self.name):
            yield chunk


# ============================================================
# Registry
# ============================================================

# Load OpenRouter models dynamically at startup
OpenRouterProvider._init_models()

ALL_PROVIDERS: list[BaseLLMProvider] = [
    GeminiProvider(),       # 1
    AnthropicProvider(),    # 2
    OpenAIProvider(),       # 3
    DeepSeekProvider(),     # 4
    KimiProvider(),         # 5
    ZaiProvider(),          # 6
    GrokProvider(),         # 7
    PerplexityProvider(),   # 8
    QwenProvider(),         # 9
    MistralProvider(),      # 10
    OpenRouterProvider(),   # 11
]


def get_available_models() -> list[dict]:
    """Retorna modelos disponíveis, deduplicados. Nativos têm prioridade sobre OpenRouter."""
    # Fase 1: coletar todos os modelos nativos (não-OpenRouter)
    native_models = []
    or_models = []

    for provider in ALL_PROVIDERS:
        configured = provider.is_configured()
        for model in provider.models:
            pricing = MODEL_PRICING.get(model.id, {"input": 0, "output": 0})
            entry = {
                "id": model.id,
                "name": model.name,
                "provider": model.provider,
                "configured": configured,
                "pricing": pricing,
                "context_window": MODEL_PRICING.get(model.id, {}).get("context_window", 128000),
            }
            if provider.name == "OpenRouter":
                or_models.append(entry)
            else:
                native_models.append(entry)

    # Fase 2: construir conjunto de slugs nativos para dedup
    # Ex: "deepseek-chat" (nativo) deve bloquear "deepseek/deepseek-chat" (OR)
    native_slugs = set()
    for m in native_models:
        mid = m["id"].lower()
        native_slugs.add(mid)
        # Variações comuns
        if "/" in mid:
            native_slugs.add(mid.split("/")[-1])

    # Mapear nomes nativos normalizados para dedup por nome
    native_names = set()
    for m in native_models:
        norm = m["name"].lower().replace("(", "").replace(")", "").strip()
        native_names.add(norm)

    # Fase 3: filtrar OpenRouter — remover duplicatas de nativos
    filtered_or = []
    for m in or_models:
        mid = m["id"].lower()
        slug = mid.split("/")[-1] if "/" in mid else mid
        # Remover sufixo :free para comparação
        slug_base = slug.replace(":free", "").replace(":floor", "").replace(":extended", "")
        name_norm = m["name"].lower().replace("(or)", "").replace("(free)", "").replace("(", "").replace(")", "").strip()

        # Checar se equivalente nativo existe
        is_dup = False

        # Match por slug exato
        if slug_base in native_slugs or slug in native_slugs:
            is_dup = True

        # Match por nome do modelo dentro do ID do OpenRouter
        # Ex: "anthropic/claude-sonnet-4.6" → "claude-sonnet-4-6" existe como nativo
        if not is_dup:
            for ns in native_slugs:
                # Normalizar hífens e pontos para comparar
                ns_clean = ns.replace("-", "").replace(".", "").replace("_", "")
                slug_clean = slug_base.replace("-", "").replace(".", "").replace("_", "")
                if ns_clean == slug_clean:
                    is_dup = True
                    break

        # Match por fabricante + nome parcial
        # Ex: "google/gemini-2.5-flash" deve matchear "gemini-2.5-flash" nativo
        if not is_dup:
            for ns in native_slugs:
                if len(ns) > 5 and (ns in slug_base or slug_base in ns):
                    is_dup = True
                    break

        if not is_dup:
            filtered_or.append(m)

    # Fase 4: dedup dentro do OpenRouter — :free vs pago do mesmo modelo
    # Manter o :free quando existir (custo $0), senão manter o pago
    or_by_base = {}
    for m in filtered_or:
        base_slug = m["id"].replace(":free", "").replace(":floor", "").replace(":extended", "")
        existing = or_by_base.get(base_slug)
        if not existing:
            or_by_base[base_slug] = m
        else:
            # Preferir :free (custo 0)
            m_cost = m["pricing"].get("input", 0) + m["pricing"].get("output", 0)
            e_cost = existing["pricing"].get("input", 0) + existing["pricing"].get("output", 0)
            if m_cost < e_cost:
                or_by_base[base_slug] = m

    # Fase 5: juntar nativos + OpenRouter deduplicados
    result = native_models + list(or_by_base.values())
    # Fase 6: filtrar modelos escondidos (broken/incompatíveis)
    result = [m for m in result if m["id"] not in _HIDDEN_MODELS]
    return result


def get_provider_for_model(model_id: str) -> BaseLLMProvider | None:
    for provider in ALL_PROVIDERS:
        for model in provider.models:
            if model.id == model_id:
                return provider
    return None


def get_model_display_name(model_id: str) -> str:
    for provider in ALL_PROVIDERS:
        for model in provider.models:
            if model.id == model_id:
                return model.name
    return model_id


def get_model_pricing(model_id: str) -> dict:
    return MODEL_PRICING.get(model_id, {"input": 0, "output": 0})


_compression_blacklist: set[str] = set()


# ============================================================
# Seleção dinâmica de modelo de compressão
# Qualidade > Velocidade. Se qualidade igual, preferir velocidade.
# Cada perfil de situação tem um modelo ideal baseado no perfil cognitivo.
# ============================================================

# Perfis de compressão por tipo de demanda
_COMP_PROFILES = {
    # Debate profundo (rodadas R6+): evolução temporal, refutações, trajetórias
    # Opus é insubstituível aqui — único que mapeia como posições evoluíram ao longo de 6+ rodadas
    "debate_deep": [
        "claude-opus-4-6",          # #1: Melhor absoluto em evolução temporal e refutações reais
        "claude-sonnet-4-6",        # #2: 80% do Opus, 2x mais rápido
        "gemini-3.1-pro-preview",   # #3: Ótimo em categorização, fallback sólido
        "gpt-5.4",                  # #4: Neutro, bom JSON, 1M ctx
        "gemini-2.5-pro",           # #5: Boa síntese longa
    ],
    # Debate com agrupamento (8+ modelos, rodadas R2-R4): coalizões, scoring, divergências cruzadas
    # Gemini 3.1 Pro é o melhor em "A, D, K concordam; B e F discordam por razões diferentes"
    "debate_group": [
        "gemini-3.1-pro-preview",   # #1: Melhor categorização adversarial de muitas vozes
        "claude-sonnet-4-6",        # #2: Bom em divergências reais
        "gpt-5.4",                  # #3: JSON perfeito, neutro
        "claude-opus-4-6",          # #4: Excelente mas mais lento que necessário aqui
        "gpt-4.1",                  # #5: Forte em structured output
    ],
    # Debate transição (score 6-7): complexo mas não máximo — Sonnet resolve sem precisar de Opus
    "debate_transition": [
        "claude-sonnet-4-6",        # #1: 80% do Opus, 2x mais rápido — ideal para zona cinzenta
        "gemini-3.1-pro-preview",   # #2: Boa categorização
        "claude-opus-4-6",          # #3: Se Sonnet falhar, escala para Opus
        "gpt-5.4",                  # #4: Neutro, JSON
        "gpt-4.1",                  # #5: Structured output
    ],
    # Debate simples (poucos modelos, rodadas iniciais):
    # Preservar tudo, resumir posições, JSON confiável
    "debate_simple": [
        "gpt-4o",                   # JSON perfeito + rápido (85tk/s) — ideal para Raio-X simples
        "gemini-3-flash-preview",   # Denso, rápido, neutro
        "gemini-2.5-flash",         # Muito rápido
        "claude-sonnet-4-6",        # Se os acima falharem
        "gpt-4o-mini",              # Fallback rápido
    ],
    # Brainstorm profundo (R6+ ou muitos modelos + respostas longas):
    # Rastrear genealogia de ideias, detectar redundância, identificar combinações não-óbvias
    # Opus é melhor aqui que em debate_deep — brainstorm longo precisa rastrear fusões, não refutações
    "brainstorm_deep": [
        "claude-opus-4-6",          # #1: Mapeia genealogia de ideias e detecta redundância vs. novidade real
        "gemini-3.1-pro-preview",   # #2: Excelente em categorizar convergências emergentes
        "claude-sonnet-4-6",        # #3: 80% do Opus, mais rápido
        "gpt-5.4",                  # #4: Neutro, bom fallback
        "gemini-2.5-pro",           # #5: Contexto grande
    ],
    # Brainstorm transição (score 5-6): complexo mas não máximo — Sonnet resolve
    "brainstorm_transition": [
        "claude-sonnet-4-6",        # #1: Boa síntese + rastreia fusões parciais
        "gpt-5.4",                  # #2: Neutro, 1M ctx
        "gemini-3.1-pro-preview",   # #3: Categorização
        "claude-opus-4-6",          # #4: Se Sonnet falhar
        "gemini-2.5-pro",           # #5: Contexto grande
    ],
    # Brainstorm com muitas ideias (8+ modelos, rodadas iniciais/médias):
    # Agrupar ideias em temas, eliminar repetições — neutralidade importa
    "brainstorm_many": [
        "gpt-5.4",                  # #1: Mais neutro, melhor síntese cooperativa, JSON excelente, 1M ctx
        "gemini-3.1-pro-preview",   # #2: Forte em categorização
        "claude-sonnet-4-6",        # #3: Bom agrupamento
        "gemini-2.5-pro",           # #4: Contexto grande
        "gpt-4.1",                  # #5: Structured output
    ],
    # Brainstorm simples (poucos modelos, rodadas iniciais):
    # Listar ideias sem perder nenhuma, JSON confiável
    "brainstorm_simple": [
        "gpt-4o",                   # #1: JSON perfeito + rápido — ideal para brainstorms leves
        "gemini-3-flash-preview",   # #2: Máxima densidade info/token
        "gemini-2.5-flash",         # #3: Rápido
        "gpt-4o-mini",              # #4: Rápido
        "gpt-5.4-nano",             # #5: Ultrarrápido
    ],
}

# Models com context window < 200K (excluídos para cenários grandes)
_SMALL_CTX = {"gpt-4o", "gpt-4o-mini", "gpt-4.1-mini", "mistral-large-latest", "gpt-5.4-nano"}


def _calc_complexity(mode: str, num_models: int, round_num: int, max_tokens: int) -> int:
    """Calcula score de complexidade (0-12+). Usado para classificar a situação."""
    num_models = max(0, num_models)
    round_num = max(0, round_num)
    max_tokens = max(0, max_tokens)
    score = 0

    # Modo
    if mode == "debate":
        score += 2  # adversarial é estruturalmente mais complexo

    # Número de modelos
    if num_models >= 20:
        score += 4
    elif num_models >= 15:
        score += 3
    elif num_models >= 8:
        score += 2
    elif num_models >= 5:
        score += 1

    # Rodada (evolução temporal)
    if round_num >= 8:
        score += 3
    elif round_num >= 6:
        score += 2
    elif round_num >= 4:
        score += 1

    # Tokens por resposta
    if max_tokens >= 3000:
        score += 2
    elif max_tokens >= 2000:
        score += 1

    return score


def _classify_situation(mode: str, num_models: int, round_num: int, max_tokens: int) -> str:
    """Classifica a situação usando score de complexidade contínuo.

    Score 0-2:  simple  (GPT-4o, Flash — rápido, JSON confiável)
    Score 3-5:  many/group (GPT-5.4 para BS, 3.1 Pro para DB — agrupamento)
    Score 6-7:  transição (Sonnet — 80% do Opus, 2x mais rápido)
    Score 8+:   deep    (Opus — evolução temporal, genealogia, poda profunda)
    """
    score = _calc_complexity(mode, num_models, round_num, max_tokens)

    if mode == "debate":
        if score >= 8:
            return "debate_deep"        # Opus: evolução temporal complexa
        if score >= 6:
            return "debate_transition"  # Sonnet: complexo mas não máximo
        if score >= 3:
            return "debate_group"       # 3.1 Pro: categorização
        return "debate_simple"          # GPT-4o: rápido, JSON
    else:
        if score >= 7:
            return "brainstorm_deep"        # Opus: genealogia de ideias
        if score >= 5:
            return "brainstorm_transition"  # Sonnet: complexo mas não máximo
        if score >= 3:
            return "brainstorm_many"        # GPT-5.4: síntese neutra
        return "brainstorm_simple"          # GPT-4o: rápido


def get_best_compression_model(
    mode: str = "debate",
    num_models: int = 5,
    round_num: int = 2,
    max_tokens: int = 1000,
) -> str | None:
    """Seleciona o modelo ideal de compressão baseado no perfil cognitivo necessário.

    Qualidade > Velocidade. Se qualidade equivalente, prefere o mais rápido.
    Exclui automaticamente modelos com context pequeno para cenários grandes.
    """
    situation = _classify_situation(mode, num_models, round_num, max_tokens)
    priority_list = _COMP_PROFILES.get(situation, _COMP_PROFILES["debate_simple"])

    # Estimar volume de input
    estimated_input = num_models * max_tokens * 2

    # Tentar lista do perfil ideal
    for model_id in priority_list:
        if model_id in _compression_blacklist:
            continue
        if estimated_input > 100_000 and model_id in _SMALL_CTX:
            continue
        provider = get_provider_for_model(model_id)
        if provider and provider.is_configured():
            return model_id

    # Fallback: tentar TODOS os perfis em ordem de qualidade
    tried = set(priority_list)
    fallback_order = ["debate_deep", "debate_group", "brainstorm_many", "debate_simple", "brainstorm_simple"]
    for profile_name in fallback_order:
        for model_id in _COMP_PROFILES[profile_name]:
            if model_id in tried or model_id in _compression_blacklist:
                continue
            tried.add(model_id)
            if estimated_input > 100_000 and model_id in _SMALL_CTX:
                continue
            provider = get_provider_for_model(model_id)
            if provider and provider.is_configured():
                return model_id

    # Último fallback: qualquer modelo configurado
    best_id = None
    best_cost = float("inf")
    for provider in ALL_PROVIDERS:
        if not provider.is_configured():
            continue
        for model in provider.models:
            if model.id in _compression_blacklist:
                continue
            p = MODEL_PRICING.get(model.id, {"input": 999, "output": 999})
            cost = p["input"] + p["output"]
            if cost < best_cost and cost >= 0:
                best_cost = cost
                best_id = model.id
    return best_id


# Alias para compatibilidade
def get_cheapest_configured_model() -> str | None:
    return get_best_compression_model()


def blacklist_compression_model(model_id: str):
    _compression_blacklist.add(model_id)


def reset_compression_blacklist():
    _compression_blacklist.clear()
