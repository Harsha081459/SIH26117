"""Task router: classify the user's task, pick the right open-weight model."""
import yaml
from pathlib import Path

_CFG = yaml.safe_load((Path(__file__).parent / "models.yaml").read_text())

_KEYWORDS = {
    "write_code": ["code", "script", "python", "function", "program", "debug", "fix", "compile"],
    "image_understanding": ["image", "photo", "picture", "scan", "scanned", "drawing", "handwritten", "diagram"],
    "draft_document": ["draft", "write a", "approval note", "letter", "report", "memo", "note"],
    "summarize": ["summar", "tldr", "key points", "findings", "extract"],
}


def classify(message: str, has_attachment: bool = False) -> str:
    if has_attachment:
        return "image_understanding"
    m = message.lower()
    for task, keys in _KEYWORDS.items():
        if any(k in m for k in keys):
            return task
    return "general"


def pick_model(task: str) -> tuple[str, str]:
    for entry in _CFG["models"]:
        if task in entry["tasks"]:
            return entry["id"], entry["reason"]
    return _CFG["default"], "default model for unclassified tasks"


def route(message: str, has_attachment: bool = False) -> dict:
    task = classify(message, has_attachment)
    model, reason = pick_model(task)
    return {"task": task, "model": model, "reason": reason}
