"""Task router: classify the task, then pick the open-weight model for it.

Two stages. Keyword rules resolve the obvious cases instantly and work with
no model loaded; anything ambiguous is classified by the default model itself,
so adding a new task type does not mean writing new keyword lists. Models are
declared in models.yaml -- adding a model is a config change, not a code
change, which is what the problem statement asks for.
"""
import os

import yaml

_CFG_PATH = os.path.join(os.path.dirname(__file__), "models.yaml")
with open(_CFG_PATH) as _f:
    _CFG = yaml.safe_load(_f)

TASKS = sorted({t for m in _CFG["models"] for t in m["tasks"]})

_KEYWORDS = {
    "write_code": ["code", "script", "python", "function", "program", "debug",
                   "fix the", "compile", "regex"],
    "image_understanding": ["image", "photo", "picture", "scan", "scanned",
                            "drawing", "handwritten", "diagram", ".png", ".jpg"],
    "draft_document": ["draft", "approval note", "letter", "memo", "write a note",
                       "write a report", "prepare a note", "docx", "word file",
                       "presentation", "slide", "pptx", "excel", "xlsx"],
    "summarize": ["summar", "tldr", "key points", "findings", "extract", "brief"],
    "analyze": ["analyse", "analyze", "compare", "trend", "calculate", "compute",
                "how many", "total"],
}


def classify_keywords(message):
    m = (message or "").lower()
    for task in ("image_understanding", "write_code", "draft_document",
                 "summarize", "analyze"):
        if any(k in m for k in _KEYWORDS[task]):
            return task
    return None


def classify_llm(message):
    """Ask the default model to label the task. Used only when rules abstain."""
    from ollama_client import generate
    prompt = ("Classify the user's request into exactly one label from this list:\n"
              + ", ".join(TASKS)
              + "\n\nRequest: " + message
              + "\n\nAnswer with the label only, no punctuation or explanation.")
    model, _ = _resolve(_CFG["default"])
    try:
        raw = generate(model, prompt, timeout=60).strip().lower()
    except Exception:
        return None
    for t in TASKS:
        if t in raw:
            return t
    return None


def classify(message, has_attachment=False, use_llm=True):
    if has_attachment:
        return "image_understanding", "attachment present"
    task = classify_keywords(message)
    if task:
        return task, "keyword rules"
    if use_llm:
        task = classify_llm(message)
        if task:
            return task, "model-based classification"
    return "general", "fallback"


def pick_model(task):
    for entry in _CFG["models"]:
        if task in entry["tasks"]:
            return entry["id"], entry["reason"]
    return _CFG["default"], "default model for unclassified tasks"


def _resolve(model_id):
    """Map a configured model onto one the runtime actually has.

    The venue may not host the same models as the development box -- the
    problem statement explicitly allows a smaller model there -- so a
    configured id that is not loaded falls back to the closest match rather
    than failing the request.
    """
    from ollama_client import list_models
    loaded = list_models()
    if not loaded or model_id in loaded:
        return model_id, None
    base = model_id.split(":")[0]
    same_family = [m for m in loaded if m.split(":")[0] == base]
    if same_family:
        return same_family[0], "{} not loaded; using {}".format(model_id, same_family[0])
    if _CFG["default"] in loaded:
        return _CFG["default"], "{} not loaded; using default".format(model_id)
    return loaded[0], "{} not loaded; using {}".format(model_id, loaded[0])


VISION_TASKS = ("image_understanding", "ocr_handwriting", "drawing_review")


def route(message, has_attachment=False, use_llm=True, resolve=True):
    """Pick the model for the task, and the model that should drive the loop.

    These are not always the same. A vision-language model is the right thing to
    read a scan, but a poor choice to plan a multi-step job -- in testing it
    produced the document and then kept calling tools aimlessly. So image tasks
    keep the vision model for the reading (reached through the ocr_doc tool) and
    let the reasoning model orchestrate.
    """
    task, how = classify(message, has_attachment, use_llm)
    model, reason = pick_model(task)
    orchestrator = model
    orch_note = None
    if task in VISION_TASKS:
        orchestrator, _ = pick_model("general")
        orch_note = "vision model reads the image; reasoning model drives the loop"

    note = None
    if resolve:
        model, note = _resolve(model)
        orchestrator, _ = _resolve(orchestrator)

    out = {"task": task, "model": model, "orchestrator": orchestrator,
           "reason": reason, "classified_by": how}
    if orch_note:
        out["orchestration"] = orch_note
    if note:
        out["substitution"] = note
    return out


def registry():
    return _CFG
