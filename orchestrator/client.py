"""orchestrator/client.py -- minimal Nebius Token Factory chat client (stdlib urllib).

POST {base}/chat/completions, OpenAI-compatible. The key comes ONLY from the environment
variable NEBIUS_TOKEN_FACTORY_KEY; it is sent in the Authorization header and nowhere
else, and every string this module hands back (errors, response bodies) goes through
``redact`` first. Base URL: NEBIUS_TOKEN_FACTORY_BASE_URL, default
https://api.tokenfactory.nebius.com/v1/ . Model: SOLAR_TF_MODEL, default
deepseek-ai/DeepSeek-V4.1-Flash (nvidia/Nemotron-3_5-Lightning is the submission model).

Structured output: ``response_format`` json_schema (strict) first; if the endpoint rejects
it (4xx mentioning schema / response_format) the client switches PERMANENTLY to
``{"type": "json_object"}`` and relies on schema.validate_plan. Nemotron models get
``chat_template_kwargs: {"enable_thinking": false}``, DeepSeek models ``{"thinking": false}``
(without it DeepSeek-V4.1-Flash spent all 1500 tokens reasoning: finish=length, no content).
8 s timeout, one retry on 429 / 5xx /
timeout. Cost is an ESTIMATE from the built-in price table (USD per 1M tokens in/out).
"""

from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from .schema import response_format

KEY_ENV = "NEBIUS_TOKEN_FACTORY_KEY"
BASE_ENV = "NEBIUS_TOKEN_FACTORY_BASE_URL"
MODEL_ENV = "SOLAR_TF_MODEL"
DEFAULT_BASE = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
SUBMISSION_MODEL = "nvidia/Nemotron-3_5-Lightning"

PRICES = {                     # USD per 1M tokens (input, output) -- estimates only
    "deepseek-ai/DeepSeek-V4.1-Flash": (0.30, 1.20),
    "nvidia/Nemotron-3_5-Lightning": (0.06, 0.24),
    "nvidia/nemotron-3-super-120b-a12b": (0.30, 0.90),
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B": (0.06, 0.24),
}
UNKNOWN_PRICE = (1.00, 3.00)   # conservative

_TOKEN_RE = re.compile(r"v1\.[A-Za-z0-9._-]{20,}")
Transport = Callable[[str, dict, bytes, float], Tuple[int, str]]


class MissingKey(RuntimeError):
    pass


def redact(text, key: Optional[str] = None) -> str:
    """Remove the key (raw and JSON-escaped) and anything that looks like a v1. token."""
    s = text if isinstance(text, str) else str(text)
    if key:
        s = s.replace(key, "[REDACTED]")
        esc = json.dumps(key)[1:-1]
        if esc != key:
            s = s.replace(esc, "[REDACTED]")
    return _TOKEN_RE.sub("v1.[REDACTED]", s)


def thinking_off(model: str) -> dict:
    """Template kwargs that turn reasoning off, so the token budget goes to the JSON."""
    m = model.lower()
    if "nemotron" in m:
        return {"enable_thinking": False}
    if "deepseek" in m:                    # DeepSeek V3.1+ chat template flag
        return {"thinking": False}
    return {}


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    pin, pout = PRICES.get(model, UNKNOWN_PRICE)
    return (tokens_in * pin + tokens_out * pout) / 1e6


def urllib_transport(url: str, headers: dict, body: bytes, timeout: float) -> Tuple[int, str]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except urllib.error.URLError as e:
        if isinstance(e.reason, (socket.timeout, TimeoutError)):
            raise TimeoutError("timeout") from None
        raise ConnectionError(f"network error ({type(e.reason).__name__})") from None
    except (socket.timeout, TimeoutError):
        raise TimeoutError("timeout") from None


@dataclass
class ChatResult:
    ok: bool
    text: str = ""                 # model message content (redacted)
    status: int = 0
    latency_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    schema_mode: str = ""
    error: str = ""                # short, redacted
    finish_reason: str = ""
    reasoning_chars: int = 0       # size of any reasoning trace the server returned
    attempts: List[str] = field(default_factory=list)


def _mentions_schema(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in ("schema", "response_format", "json_schema", "guided"))


class TokenFactoryClient:
    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None,
                 timeout_s: float = 8.0, transport: Optional[Transport] = None,
                 temperature: float = 0.1, max_tokens: int = 1500):
        self._key = os.environ.get(KEY_ENV, "").strip()
        if not self._key:
            raise MissingKey(f"{KEY_ENV} is not set")
        self.model = model or os.environ.get(MODEL_ENV, "").strip() or DEFAULT_MODEL
        base = base_url or os.environ.get(BASE_ENV, "").strip() or DEFAULT_BASE
        self.url = base.rstrip("/") + "/chat/completions"
        self.timeout_s = timeout_s
        self.transport = transport or urllib_transport
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.schema_mode = "json_schema"

    def redact(self, text) -> str:
        return redact(text, self._key)

    def _body(self, messages: Sequence[dict], rover_ids: Sequence[str]) -> dict:
        body = {"model": self.model, "messages": list(messages),
                "temperature": self.temperature, "max_tokens": self.max_tokens,
                "response_format": response_format(rover_ids, self.schema_mode)}
        kw = thinking_off(self.model)
        if kw:
            body["chat_template_kwargs"] = kw
        return body

    def chat(self, messages: Sequence[dict], rover_ids: Sequence[str]) -> ChatResult:
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json",
                   "Accept": "application/json", "User-Agent": "solar-sheep-orchestrator/1"}
        res = ChatResult(ok=False)
        t0 = time.monotonic()
        retried = False
        for _ in range(4):                       # schema switch + one retry, bounded
            res.schema_mode = self.schema_mode
            data = json.dumps(self._body(messages, rover_ids)).encode("utf-8")
            try:
                status, text = self.transport(self.url, headers, data, self.timeout_s)
            except TimeoutError:
                res.attempts.append("timeout")
                if not retried:
                    retried = True
                    continue
                res.error = "timeout"
                break
            except Exception as e:               # network / transport bug: redacted, short
                res.attempts.append("error")
                res.error = self.redact(f"{type(e).__name__}: {e}")[:200]
                if not retried and isinstance(e, (ConnectionError, OSError)):
                    retried = True
                    continue
                break
            res.status = status
            res.attempts.append(str(status))
            text = self.redact(text)
            if status == 200:
                self._parse_ok(text, res)
                break
            if (400 <= status < 500 and status != 429 and self.schema_mode == "json_schema"
                    and _mentions_schema(text)):
                self.schema_mode = "json_object"  # permanent: this endpoint won't take the schema
                continue
            if (status == 429 or status >= 500) and not retried:
                retried = True
                time.sleep(0.5)
                continue
            res.error = f"HTTP {status}: {text[:160]}"
            break
        res.latency_s = round(time.monotonic() - t0, 3)
        res.cost_usd = estimate_cost(self.model, res.tokens_in, res.tokens_out)
        return res

    def _parse_ok(self, text: str, res: ChatResult) -> None:
        try:
            payload = json.loads(text)
            choice = payload["choices"][0]
            msg = choice["message"]
            content = msg.get("content") or ""
            res.reasoning_chars = len(msg.get("reasoning_content") or msg.get("reasoning") or "")
            res.finish_reason = str(choice.get("finish_reason") or "")
            usage = payload.get("usage") or {}
            res.tokens_in = int(usage.get("prompt_tokens") or 0)
            res.tokens_out = int(usage.get("completion_tokens") or 0)
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            res.error = f"unexpected response shape: {text[:160]}"
            return
        res.text = content
        res.ok = bool(content.strip())
        if not res.ok:
            res.error = f"empty content (finish_reason={res.finish_reason or '?'})"
