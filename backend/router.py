"""Task router: classify the task, then pick the open-weight model for it.

Two stages. Keyword rules resolve the obvious cases instantly and work with
no model loaded; anything ambiguous is classified by the default model itself,
so adding a new task type does not mean writing new keyword lists. Models are
declared in models.yaml -- adding a model is a config change, not a code
change, which is what the problem statement asks for.
"""
import os
import re

import yaml

_CFG_PATH = os.path.join(os.path.dirname(__file__), "models.yaml")
with open(_CFG_PATH, encoding="utf-8") as _f:
    _CFG = yaml.safe_load(_f)

TASKS = sorted({t for m in _CFG["models"] for t in m["tasks"]})
VISION_TASKS = ("image_understanding", "ocr_handwriting", "drawing_review")
CODE_TASKS = ("write_code", "debug", "explain_code")

_KEYWORDS = {
    "write_code": ["code", "scripts?", "python", "debug", "compile", "regex",
                   "write a function", "write a program"],
    "image_understanding": ["images?", "photos?", "pictures?", "scans?", "scanned",
                            "drawings?", "handwritten", "diagrams?", "png", "jpe?g"],
    "draft_document": ["draft", "approval note", "letter", "memo", "write a note",
                       "write a report", "prepare a note", "docx", "word file",
                       "presentations?", "slides?", "decks?", "pptx", "excel", "xlsx"],
    "summarize": ["summari[sz]e", "summari[sz]ing", "summary", "tldr", "key points",
                  "findings", "extract", "brief"],
    "analyze": ["analyse", "analyze", "compare", "trend", "calculate", "compute",
                "how many", "total"],
}


def classify_keywords(message):
    for task in ("write_code", "draft_document", "image_understanding", "summarize", "analyze"):
        if any(re.search(r"\b(?:" + keyword + r")\b", message or "", re.I)
               for keyword in _KEYWORDS[task]):
            return task
    return None


def classify_llm(message):
    """Ask the default model to label the task. Used only when rules abstain."""
    from ollama_client import generate
    prompt = ("Classify the user's request into exactly one label from this list:\n"
              + ", ".join(TASKS)
              + "\n\nRequest: " + message
              + "\n\nAnswer with the label only, no punctuation or explanation.")
    try:
        model, _ = _resolve(_CFG["default"], capability="text")
        raw = generate(model, prompt, timeout=60).strip().lower().strip("` .\n")
    except Exception:
        return None
    return raw if raw in TASKS else None


def classify(message, has_attachment=False, use_llm=True, attachment_kind=None):
    # Only an image attachment implies a vision task. A text file or a
    # spreadsheet is read by its own tool, and forcing the vision model on it
    # produces a failure the model then tries to explain.
    if has_attachment and attachment_kind == "image":
        return "image_understanding", "image attached"
    task = classify_keywords(message)
    if has_attachment and task == "image_understanding":
        task = "analyze"
    if task:
        return task, "keyword rules"
    if has_attachment:
        return "analyze", "{} attached".format(attachment_kind or "file")
    if use_llm:
        task = classify_llm(message)
        if task:
            return task, "model-based classification"
    return "general", "fallback"


def _task_capability(task):
    return "vision" if task in VISION_TASKS else "code" if task in CODE_TASKS else "text"


def pick_model(task, capability=None):
    from ollama_client import InferenceError
    capability = capability or _task_capability(task)
    compatible = [entry for entry in _CFG["models"] if capability in entry.get("capabilities", [])]
    for entry in compatible:
        if task in entry["tasks"]:
            return entry["id"], entry["reason"]
    for entry in compatible:
        if entry["id"] == _CFG["default"]:
            return entry["id"], "default model for unclassified tasks"
    if compatible:
        return compatible[0]["id"], compatible[0]["reason"]
    raise InferenceError("No {}-capable model is configured in the registry.".format(capability))


def _resolve(model_id, capability="text", state=None):
    """Map a configured model onto a compatible installed registry entry.

    Missing capabilities fail explicitly. Unregistered, embedding-only and
    cloud model identifiers are never used as arbitrary fallbacks. Installed
    models are not necessarily resident in GPU memory.
    """
    from ollama_client import InferenceError, model_capabilities, runtime_status
    state = runtime_status() if state is None else state
    if not state["reachable"]:
        raise InferenceError(state.get("error") or "The local model runtime is unavailable.")
    installed = state["models"]
    candidates = [entry["id"] for entry in _CFG["models"]
                  if capability in entry.get("capabilities", []) and entry["id"] in installed
                  and not entry["id"].lower().endswith(("-cloud", ":cloud"))]
    base = model_id.split(":")[0]
    candidates.sort(key=lambda name: (name != model_id, name.split(":")[0] != base, name != _CFG["default"]))
    required = "vision" if capability == "vision" else "completion"
    detail = ""
    for candidate in candidates:
        try:
            supported = model_capabilities(candidate)
        except InferenceError as exc:
            detail = " " + str(exc)
            continue
        if required in supported:
            note = None if candidate == model_id else "{} unavailable for {}; using {}".format(model_id, capability, candidate)
            return candidate, note
    raise InferenceError("No installed, registered {}-capable model was confirmed by the runtime. "
                         "Install a compatible local model and check models.yaml.{}".format(capability, detail))


def route(message, has_attachment=False, use_llm=True, resolve=True,
          attachment_kind=None, capability=None):
    """Pick the model for the task, and the model that should drive the loop.

    These are not always the same. A vision-language model is the right thing to
    read a scan, but a poor choice to plan a multi-step job -- in testing it
    produced the document and then kept calling tools aimlessly. So image tasks
    keep the vision model for the reading (reached through the ocr_doc tool) and
    let the reasoning model orchestrate.
    """
    task, how = classify(message, has_attachment, use_llm, attachment_kind)
    capability = capability or _task_capability(task)
    model, reason = pick_model(task, capability)
    orchestrator = model
    orch_note = None
    if capability == "vision":
        orchestrator, _ = pick_model("general", "text")
        orch_note = "vision model reads the image; reasoning model drives the loop"

    note, orchestration_note = None, None
    if resolve:
        from ollama_client import runtime_status
        state = runtime_status()
        model, note = _resolve(model, capability, state)
        if capability == "vision":
            orchestrator, orchestration_note = _resolve(orchestrator, "text", state)
        else:
            orchestrator = model

    out = {"task": task, "model": model, "orchestrator": orchestrator,
           "reason": reason, "classified_by": how, "capability": capability}
    if orch_note:
        out["orchestration"] = orch_note
    if note:
        out["substitution"] = note
    if orchestration_note:
        out["orchestrator_substitution"] = orchestration_note
    return out


def registry():
    return _CFG
