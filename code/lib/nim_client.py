"""Thin client for NVIDIA NIM (build.nvidia.com) chat completions, OpenAI-compatible schema.

Includes a simple on-disk JSON cache keyed by request hash so repeated runs / evaluation
iterations don't re-spend tokens or calls, plus retry-with-backoff for transient errors.
"""
import hashlib
import json
import os
import time

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache")

VISION_MODEL = os.environ.get("NIM_VISION_MODEL", "meta/llama-3.2-11b-vision-instruct")
TEXT_MODEL = os.environ.get("NIM_TEXT_MODEL", "meta/llama-3.1-8b-instruct")


class NimError(Exception):
    pass


def _api_key():
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        raise NimError("NVIDIA_API_KEY is not set (load it via code/.env)")
    return key


def _cache_path(payload):
    h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.json")


class CallStats:
    """Process-wide counters for the operational analysis in the eval report.

    `calls` counts real API calls (cache misses) actually billed/rate-limited.
    `logical_calls`/`prompt_tokens`/`completion_tokens` count all calls including cache
    hits, since that reflects the true workload a fresh run (e.g. on the test set) would
    need -- caching only saves cost/latency on *repeated* runs of the same input.
    """
    calls = 0
    cache_hits = 0
    logical_calls = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_latency_s = 0.0


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type((requests.RequestException, NimError)),
)
def _post(payload):
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    start = time.time()
    resp = requests.post(INVOKE_URL, headers=headers, json=payload, timeout=60)
    CallStats.total_latency_s += time.time() - start
    if resp.status_code == 429 or resp.status_code >= 500:
        raise NimError(f"retryable status {resp.status_code}: {resp.text[:200]}")
    if resp.status_code != 200:
        raise NimError(f"NIM call failed ({resp.status_code}): {resp.text[:500]}")
    return resp.json()


def chat(messages, model, max_tokens=1024, temperature=0.0, use_cache=True):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "stream": False,
    }

    os.makedirs(CACHE_DIR, exist_ok=True)
    cpath = _cache_path(payload)
    if use_cache and os.path.exists(cpath):
        CallStats.cache_hits += 1
        CallStats.logical_calls += 1
        with open(cpath, encoding="utf-8") as f:
            cached = json.load(f)
        usage = cached.get("usage", {})
        CallStats.prompt_tokens += usage.get("prompt_tokens", 0)
        CallStats.completion_tokens += usage.get("completion_tokens", 0)
        return cached["content"]

    data = _post(payload)
    CallStats.calls += 1
    CallStats.logical_calls += 1
    usage = data.get("usage", {})
    CallStats.prompt_tokens += usage.get("prompt_tokens", 0)
    CallStats.completion_tokens += usage.get("completion_tokens", 0)
    content = data["choices"][0]["message"]["content"]

    if use_cache:
        with open(cpath, "w", encoding="utf-8") as f:
            json.dump({"content": content, "usage": usage}, f)
    return content


def vision_chat(text_prompt, image_data_uris, model=None, max_tokens=600, system=None):
    """image_data_uris: list of data URIs. Most NIM vision-instruct models accept one
    image per call reliably, so callers typically pass a single-element list."""
    content = [{"type": "text", "text": text_prompt}]
    for uri in image_data_uris:
        content.append({"type": "image_url", "image_url": {"url": uri}})
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    return chat(messages, model=model or VISION_MODEL, max_tokens=max_tokens)


def text_chat(prompt, model=None, max_tokens=800, system=None):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return chat(messages, model=model or TEXT_MODEL, max_tokens=max_tokens)
