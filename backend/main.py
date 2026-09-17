"""FastAPI backend for the sovereign on-premise agentic workbench."""
import json
import os
import shutil
from typing import Optional

import psutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import agent
import audit
import router as model_router
from ollama_client import list_models
from tools import OUT, WORKSPACE, call

app = FastAPI(title="Sovereign AI Workbench - SIH26117")

# Anything outside these ranges counts as leaving the premises.
LOCAL_PREFIXES = ("127.", "10.", "172.16.", "172.17.", "172.18.", "192.168.",
                  "169.254.", "::1", "fe80:")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
SHEET_EXT = (".xlsx", ".xlsm")
TEXT_EXT = (".txt", ".md", ".csv", ".log", ".json", ".yaml", ".yml", ".ini", ".py")


def _reader_for(name):
    """Choose the right tool for an attachment, and name its kind.

    Dispatching on the file type matters: sending a text file to the vision
    model produces a confusing failure, and the model will then try to explain
    that failure instead of answering the question.
    """
    low = (name or "").lower()
    if low.endswith(IMAGE_EXT):
        return "ocr_doc", "image"
    if low.endswith(".pdf"):
        return "pdf_read", "pdf"
    if low.endswith(SHEET_EXT):
        return "sheet_read", "spreadsheet"
    if low.endswith(TEXT_EXT):
        return "fs_read", "text"
    return "fs_read", "file"


class Chat(BaseModel):
    message: str
    attachment: Optional[str] = None


@app.post("/api/chat")
def chat(body: Chat):
    name = body.attachment
    reader, kind = _reader_for(name) if name else (None, None)
    r = model_router.route(body.message, has_attachment=bool(name),
                           attachment_kind=kind)

    # An attachment is read with the tool that suits its type, then the
    # question is answered with that content in hand.
    if name:
        extracted = call(reader, {"path": name})
        task = ("The following content was read from the attached {} '{}':\n\n{}\n\n"
                "Using that content, answer the user's request: {}").format(
                    kind, name, extracted[:6000], body.message)
        try:
            out = agent.run(task, r.get("orchestrator") or r["model"], max_steps=6)
        except Exception as e:
            return {**r, "answer": extracted, "trace": [
                {"step": 1, "type": "tool", "tool": reader,
                 "args": {"path": name}, "result": extracted[:600]}],
                "files": [], "steps": 1, "secs": 0, "note": str(e)}
        out["trace"] = [{"step": 0, "type": "tool", "tool": reader,
                         "args": {"path": name}, "result": extracted[:600]}] + out["trace"]
        return {**r, **out}

    try:
        out = agent.run(body.message, r.get("orchestrator") or r["model"])
    except Exception as e:
        return {**r,
                "answer": "Local inference is not reachable. Start the runtime with "
                          "`ollama serve`, then pull a model. Details: {}".format(e),
                "trace": [], "files": [], "steps": 0, "secs": 0}
    return {**r, **out}


@app.post("/api/chat/stream")
def chat_stream(body: Chat):
    """Same as /api/chat but emits each tool call as it happens (SSE).

    Keeps the interface informative while a local model is thinking, and lets a
    reviewer watch the agent actually take steps.
    """
    name = body.attachment
    reader, kind = _reader_for(name) if name else (None, None)
    r = model_router.route(body.message, has_attachment=bool(name),
                           attachment_kind=kind)
    task = body.message
    prelude = []

    if name:
        extracted = call(reader, {"path": name})
        prelude.append({"step": 0, "type": "step", "tool": reader,
                        "args": {"path": name}, "result": extracted[:600]})
        task = ("The following content was read from the attached {} '{}':\n\n{}\n\n"
                "Using that content, answer the user's request: {}").format(
                    kind, name, extracted[:6000], body.message)

    def events():
        yield "data: " + json.dumps({"type": "route", **r}) + "\n\n"
        for ev in prelude:
            yield "data: " + json.dumps(ev) + "\n\n"
        try:
            for ev in agent.iter_run(task, r.get("orchestrator") or r["model"]):
                if ev.get("type") == "final" and prelude:
                    ev["trace"] = [dict(p, type="tool") for p in prelude] + ev["trace"]
                yield "data: " + json.dumps(ev, default=str) + "\n\n"
        except Exception as e:
            yield "data: " + json.dumps({
                "type": "final", "answer": "Local inference is not reachable. Start it "
                "with `ollama serve` and pull a model. Details: {}".format(e),
                "trace": [], "files": [], "steps": 0, "secs": 0}) + "\n\n"
        yield "data: " + json.dumps({"type": "done"}) + "\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    safe = os.path.basename(file.filename or "upload.bin").replace(" ", "_")
    dest = WORKSPACE / safe
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    audit.record("tool_call", tool="upload", args={"name": safe},
                 dest="local-process", ok=True)
    kind = ("image" if safe.lower().endswith(IMAGE_EXT)
            else "pdf" if safe.lower().endswith(".pdf") else "file")
    return {"saved": safe, "kind": kind, "bytes": dest.stat().st_size}


def _is_external(ip):
    return not any(ip.startswith(p) for p in LOCAL_PREFIXES)


@app.get("/api/egress")
def egress(scope: str = "process"):
    """Outbound connections to addresses outside the premises.

    scope=process : this backend and its children (works on a shared machine)
    scope=machine : every process on the box (use on the dedicated deployment)
    """
    pids = None
    if scope == "process":
        me = psutil.Process(os.getpid())
        pids = {me.pid} | {c.pid for c in me.children(recursive=True)}
    hits = []
    try:
        for c in psutil.net_connections(kind="inet"):
            if pids is not None and c.pid not in pids:
                continue
            if c.status == "ESTABLISHED" and c.raddr and _is_external(c.raddr.ip):
                hits.append("{}:{}".format(c.raddr.ip, c.raddr.port))
    except (psutil.AccessDenied, PermissionError) as e:
        return {"external_calls": -1, "scope": scope, "note": "permission denied: {}".format(e)}
    return {"external_calls": len(hits), "scope": scope, "detail": sorted(set(hits))[:20]}


@app.post("/api/egress/probe")
def egress_probe():
    """Actively try to leave the machine and report the result.

    The passive counter shows that nothing went out; this shows that nothing
    *can*. Returns the per-target outcome plus a pass/fail verdict.
    """
    out = call("egress_probe", {})
    denied = "ALLOWED" not in out
    rows = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith(("DENIED", "ALLOWED")):
            state, _, detail = line.partition(" ")
            rows.append({"state": state, "detail": detail.strip()})
    return {"denied": denied, "rows": rows, "raw": out,
            "verdict": out.strip().splitlines()[-1] if out.strip() else ""}


@app.get("/api/audit")
def audit_log(n: int = 50):
    return {"summary": audit.summary(), "recent": audit.tail(n)}


@app.get("/api/models")
def models():
    loaded = list_models()
    reg = model_router.registry()
    declared = [m["id"] for m in reg["models"]]
    return {"runtime_models": loaded, "registry": reg,
            "missing": [m for m in declared if m not in loaded],
            "runtime_up": bool(loaded)}


@app.get("/api/health")
def health():
    loaded = list_models()
    return {"status": "ok" if loaded else "runtime_down",
            "models_loaded": len(loaded),
            "workspace_files": sorted(p.name for p in WORKSPACE.iterdir()
                                      if p.is_file() and not p.name.startswith("_")),
            "outputs": sorted(p.name for p in OUT.iterdir() if p.is_file())}


@app.get("/api/download/{name}")
def download(name: str):
    p = OUT / os.path.basename(name)
    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p, filename=p.name)


FRONTEND = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="ui")
