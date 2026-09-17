"""Thin client for the local Ollama runtime.

Every call goes to 127.0.0.1 and is written to the audit log with its
destination, so a reviewer can verify no inference ever left the machine.
"""
import base64
import json
import time

import httpx

import audit

OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
OLLAMA_URL = "http://{}:{}".format(OLLAMA_HOST, OLLAMA_PORT)


def _payload(model, prompt, image_path=None):
    body = {"model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.1}}
    if image_path:
        with open(image_path, "rb") as f:
            body["images"] = [base64.b64encode(f.read()).decode()]
    return body


def generate(model, prompt, image_path=None, timeout=300):
    t0 = time.time()
    try:
        r = httpx.post(OLLAMA_URL + "/api/generate",
                       json=_payload(model, prompt, image_path), timeout=timeout)
        r.raise_for_status()
        text = r.json()["response"]
        ok, err = True, None
    except Exception as e:
        text, ok, err = "", False, str(e)
    audit.record("model_call", model=model, dest="{}:{}".format(OLLAMA_HOST, OLLAMA_PORT),
                 ms=int((time.time() - t0) * 1000), ok=ok, error=err,
                 prompt_chars=len(prompt), has_image=bool(image_path),
                 response_preview=text[:200])
    if not ok:
        raise RuntimeError("local inference failed: {}".format(err))
    return text


def generate_stream(model, prompt, image_path=None, timeout=300):
    """Yield response chunks as they are produced, so the UI is never silent."""
    t0, acc = time.time(), []
    body = _payload(model, prompt, image_path)
    body["stream"] = True
    with httpx.stream("POST", OLLAMA_URL + "/api/generate", json=body, timeout=timeout) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            piece = chunk.get("response", "")
            if piece:
                acc.append(piece)
                yield piece
            if chunk.get("done"):
                break
    audit.record("model_call", model=model, dest="{}:{}".format(OLLAMA_HOST, OLLAMA_PORT),
                 ms=int((time.time() - t0) * 1000), ok=True, streamed=True,
                 prompt_chars=len(prompt), response_preview="".join(acc)[:200])


def list_models():
    try:
        r = httpx.get(OLLAMA_URL + "/api/tags", timeout=10)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def available():
    return bool(list_models())
