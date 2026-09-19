"""Thin client for the local Ollama runtime.

Calls use a loopback endpoint and ignore proxy environment variables. Metadata
is written to the audit log; prompts and generated content are not logged.
"""
import base64
import json
import math
import re
import time
from urllib.parse import urlsplit

import httpx

import audit

OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
OLLAMA_URL = "http://{}:{}".format(OLLAMA_HOST, OLLAMA_PORT)
MAX_RESPONSE_CHARS = 200000


class InferenceError(RuntimeError):
    pass


def _url(path):
    try:
        parsed = urlsplit(OLLAMA_URL)
        port = parsed.port
    except ValueError:
        raise InferenceError("Invalid local inference endpoint") from None
    if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment or port == 0):
        raise InferenceError("Inference must use a loopback HTTP endpoint without a path, query or fragment")
    if parsed.username or parsed.password:
        raise InferenceError("Credentials are not supported in the inference URL")
    return OLLAMA_URL.rstrip("/") + path


def _local_name(model):
    return (isinstance(model, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}", model)
            and ".." not in model and not re.search(r"(?:^|[-:])cloud(?:[:/-]|$)", model, re.I))


def _payload(model, prompt, image_path=None):
    if not _local_name(model):
        raise InferenceError("Select an installed local model; cloud identifiers are not allowed")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 64000:
        raise InferenceError("Use a nonempty prompt of at most 64000 characters")
    required = "vision" if image_path else "completion"
    if required not in model_capabilities(model):
        raise InferenceError("The selected local model does not support {}.".format(required))
    body = {"model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.1, "num_ctx": 8192, "num_predict": 4096}}
    if image_path:
        with open(image_path, "rb") as source:
            data = source.read(20 * 1024 * 1024 + 1)
        if len(data) > 20 * 1024 * 1024:
            raise InferenceError("Image exceeds 20 MiB")
        body["images"] = [base64.b64encode(data).decode("ascii")]
    return body


def _failure(exc):
    if isinstance(exc, httpx.HTTPStatusError):
        if exc.response.status_code == 404:
            return "The selected local model is not installed. Check the model registry."
        return "Local inference returned HTTP {}.".format(exc.response.status_code)
    if isinstance(exc, httpx.TimeoutException):
        return "Local inference timed out. Try a smaller model or a shorter request."
    if isinstance(exc, httpx.RequestError):
        return "The local model runtime is unavailable. Start Ollama and install the required models."
    return str(exc) if isinstance(exc, InferenceError) else "Local inference returned an invalid response."


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(text):
    """Remove Qwen3-style reasoning blocks so callers see only the answer.

    An unterminated <think> tail (model ran out of tokens mid-reasoning) is
    dropped as well. Applied before validation so a think-only response is
    reported as unusable rather than returned as an empty answer."""
    cleaned = _THINK_RE.sub("", text)
    tail = cleaned.lower().find("<think>")
    if tail != -1:
        cleaned = cleaned[:tail]
    return cleaned.strip()


def generate(model, prompt, image_path=None, timeout=120, output_format=None):
    started, ok = time.monotonic(), False
    try:
        body = _payload(model, prompt, image_path)
        if output_format:
            body["format"] = output_format
        with httpx.Client(trust_env=False, follow_redirects=False, timeout=timeout) as client:
            response = client.post(_url("/api/generate"), json=body)
            response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise InferenceError("The local model returned an invalid response object.")
        text = data.get("response")
        if isinstance(text, str):
            text = _strip_think(text)
        if (data.get("error") or data.get("done") is not True or not isinstance(text, str)
                or not text.strip() or len(text) > MAX_RESPONSE_CHARS):
            raise InferenceError("The local model returned incomplete or unusable content.")
        if data.get("done_reason") == "length":
            raise InferenceError("Local generation hit its output limit. Request a shorter result.")
        ok = True
        return text
    except (httpx.HTTPError, ValueError, OSError, InferenceError) as exc:
        raise InferenceError(_failure(exc)) from None
    finally:
        audit.record("model_call", model=model, dest="{}:{}".format(OLLAMA_HOST, OLLAMA_PORT),
                     ms=round((time.monotonic() - started) * 1000), ok=ok,
                     prompt_chars=len(prompt) if isinstance(prompt, str) else 0, has_image=bool(image_path))


def generate_stream(model, prompt, image_path=None, timeout=120):
    """Yield content chunks and report malformed or interrupted responses."""
    started, completed, chars = time.monotonic(), False, 0
    try:
        body = _payload(model, prompt, image_path)
        body["stream"] = True
        with httpx.Client(trust_env=False, follow_redirects=False, timeout=timeout) as client:
            with client.stream("POST", _url("/api/generate"), json=body) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    if len(line) > MAX_RESPONSE_CHARS:
                        raise InferenceError("Local model stream exceeded the response limit.")
                    chunk = json.loads(line)
                    if not isinstance(chunk, dict) or chunk.get("error"):
                        raise InferenceError("The local model reported an invalid stream event.")
                    piece = chunk.get("response", "")
                    if not isinstance(piece, str):
                        raise InferenceError("Invalid model stream content.")
                    chars += len(piece)
                    if chars > MAX_RESPONSE_CHARS or chunk.get("done_reason") == "length":
                        raise InferenceError("Local generation hit its output limit.")
                    if piece:
                        yield piece
                    if chunk.get("done") is True:
                        completed = True
                        break
        if not completed or not chars:
            raise InferenceError("The local model stream ended without a complete response.")
    except (httpx.HTTPError, ValueError, OSError, InferenceError) as exc:
        raise InferenceError(_failure(exc)) from None
    finally:
        audit.record("model_call", model=model, dest="{}:{}".format(OLLAMA_HOST, OLLAMA_PORT),
                     ms=round((time.monotonic() - started) * 1000), ok=completed,
                     streamed=True, prompt_chars=len(prompt) if isinstance(prompt, str) else 0)


def embed(text, model="nomic-embed-text"):
    started, ok = time.monotonic(), False
    try:
        if not isinstance(text, str) or not text.strip() or len(text) > 32000:
            raise InferenceError("Embedding input must contain 1 to 32000 characters")
        if "embedding" not in model_capabilities(model):
            raise InferenceError("The selected local model does not support embeddings.")
        with httpx.Client(trust_env=False, follow_redirects=False, timeout=30) as client:
            response = client.post(_url("/api/embed"), json={"model": model, "input": text})
            response.raise_for_status()
        data = response.json()
        vectors = data.get("embeddings") if isinstance(data, dict) else None
        vector = vectors[0] if isinstance(vectors, list) and vectors else None
        if (not isinstance(vector, list) or not 1 <= len(vector) <= 16384
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) for value in vector)):
            raise InferenceError("The local runtime returned an invalid embedding.")
        ok = True
        return vector
    except (httpx.HTTPError, ValueError, OSError, InferenceError) as exc:
        raise InferenceError(_failure(exc)) from None
    finally:
        audit.record("embedding_call", model=model, dest="{}:{}".format(OLLAMA_HOST, OLLAMA_PORT),
                     ms=round((time.monotonic() - started) * 1000), ok=ok,
                     prompt_chars=len(text) if isinstance(text, str) else 0)


def runtime_status():
    try:
        with httpx.Client(trust_env=False, follow_redirects=False, timeout=3) as client:
            response = client.get(_url("/api/tags"))
            response.raise_for_status()
        data = response.json()
        models = data.get("models") if isinstance(data, dict) else None
        if not isinstance(models, list):
            raise ValueError("Invalid model list")
        names, rejected = [], []
        for model in models:
            if not isinstance(model, dict) or not isinstance(model.get("name"), str):
                raise ValueError("Invalid model entry")
            if model.get("remote_model") or model.get("remote_host") or not _local_name(model["name"]):
                rejected.append(model["name"])
            else:
                names.append(model["name"])
        return {"reachable": True, "models": names, "rejected_remote_models": rejected}
    except (httpx.HTTPError, ValueError, InferenceError) as exc:
        return {"reachable": False, "models": [], "error": _failure(exc)}


def model_capabilities(model):
    if not _local_name(model):
        raise InferenceError("Only installed local model identifiers are supported.")
    state = runtime_status()
    if not state["reachable"]:
        raise InferenceError(state.get("error") or "The local model runtime is unavailable.")
    if model not in state["models"]:
        raise InferenceError("The model is not installed locally, or is a remote alias.")
    try:
        with httpx.Client(trust_env=False, follow_redirects=False, timeout=5) as client:
            response = client.post(_url("/api/show"), json={"model": model})
            response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or data.get("remote_model") or data.get("remote_host"):
            raise InferenceError("Remote or invalid model metadata is not allowed.")
        capabilities = data.get("capabilities")
        if not isinstance(capabilities, list) or not capabilities or any(not isinstance(c, str) for c in capabilities):
            raise InferenceError("The runtime did not report model capabilities. Check the local Ollama version.")
        return set(capabilities)
    except (httpx.HTTPError, ValueError, InferenceError) as exc:
        raise InferenceError(_failure(exc)) from None


def list_models():
    return runtime_status()["models"]


def available():
    return runtime_status()["reachable"]
