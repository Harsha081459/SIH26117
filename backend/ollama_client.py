"""Thin local client for Ollama — no external network, ever."""
import base64
import httpx

OLLAMA_URL = "http://127.0.0.1:11434"


def generate(model: str, prompt: str, image_path: str | None = None, timeout: float = 300) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    if image_path:
        with open(image_path, "rb") as f:
            payload["images"] = [base64.b64encode(f.read()).decode()]
    r = httpx.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["response"]


def list_models() -> list[str]:
    r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=10)
    return [m["name"] for m in r.json().get("models", [])]
