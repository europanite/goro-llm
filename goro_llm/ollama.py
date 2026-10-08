from __future__ import annotations

import json
import os
from urllib.parse import urlsplit

import httpx


def chat_json(system: str, user: dict, *, temperature: float = 0.8) -> dict:
    model = os.getenv("GORO_MODEL", "").strip()
    if not model:
        raise RuntimeError("Set GORO_MODEL to an installed Ollama model, e.g. qwen3:8b")
    base = os.getenv("GORO_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
    parsed = urlsplit(base)
    if parsed.scheme != "http" or parsed.hostname not in {
        "127.0.0.1",
        "localhost",
        "ollama",
        "host.docker.internal",
    }:
        raise RuntimeError("GORO_OLLAMA_URL must point to a local HTTP host")

    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature, "num_ctx": 8192},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
    }
    with httpx.Client(timeout=180, trust_env=False) as client:
        response = client.post(base + "/api/chat", json=payload)
        response.raise_for_status()
    envelope = response.json()
    content = envelope["message"]["content"]
    obj = json.loads(content)
    if not isinstance(obj, dict):
        raise ValueError("model did not return a JSON object")
    return obj
