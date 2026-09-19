"""FastAPI backend for the sovereign on-premise agentic workbench."""
import asyncio
import hashlib
import ipaddress
import json
import os
import queue
import threading
import time
import uuid
from pathlib import Path, PureWindowsPath
from typing import Optional
from urllib.parse import urlsplit

import psutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

import agent
import audit
import router as model_router
import tools
from ollama_client import runtime_status
from tools import OUT, WORKSPACE, call

# Only loopback hosts are accepted by this single-user POC.
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
SHEET_EXT = (".xlsx", ".xlsm")
TEXT_EXT = (".txt", ".md", ".csv", ".log", ".json", ".yaml", ".yml", ".ini", ".py")
OFFICE_EXT = (".docx", ".pptx")
MAX_UPLOAD = 20 * 1024 * 1024
_SLOTS = threading.BoundedSemaphore(1)
_RUNS = {}
_RUN_LOCK = threading.Lock()


def _disk_build_id():
    root = Path(__file__).resolve().parent.parent
    names = ["backend/" + name for name in ("main.py", "agent.py", "router.py", "tools.py", "sandbox.py",
             "calc.py", "audit.py", "ollama_client.py", "models.yaml")]
    digest = hashlib.sha256()
    for name in [*names, "frontend/index.html", "requirements.txt"]:
        digest.update(name.encode("utf-8"))
        digest.update((root / name).read_bytes())
    return digest.hexdigest()[:16]


BUILD_ID = _disk_build_id()


def _body_limit(path):
    return MAX_UPLOAD + 1024 * 1024 if path == "/api/upload" else 64 * 1024


class RequestBodyLimit:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > _body_limit(scope["path"]):
                response = JSONResponse({"detail": "Request body exceeds the size limit"}, status_code=413)
                return await response(scope, receive, send)
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body, delivered = b"".join(chunks), False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


app = FastAPI(title="Sovereign AI Workbench - SIH26117", docs_url=None, redoc_url=None)
app.add_middleware(RequestBodyLimit)


@app.middleware("http")
async def request_safety(request: Request, next_handler):
    if request.url.hostname not in LOCAL_HOSTS:
        return JSONResponse({"detail": "This POC accepts loopback hosts only"}, status_code=400)
    if request.method == "POST":
        origin = request.headers.get("origin")
        if origin:
            try:
                parsed = urlsplit(origin)
                target_port = request.url.port or (443 if request.url.scheme == "https" else 80)
                origin_port = parsed.port or (443 if parsed.scheme == "https" else 80)
                same_origin = (parsed.scheme == request.url.scheme and parsed.hostname == request.url.hostname
                               and origin_port == target_port and not parsed.username and not parsed.password)
            except ValueError:
                same_origin = False
            if not same_origin:
                return JSONResponse({"detail": "Cross-origin writes are not allowed"}, status_code=403)
        if request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail": "Cross-site writes are not allowed"}, status_code=403)
        try:
            length = int(request.headers.get("content-length", "0"))
        except ValueError:
            return JSONResponse({"detail": "Invalid request length"}, status_code=400)
        if length < 0 or length > _body_limit(request.url.path):
            return JSONResponse({"detail": "Request body exceeds the size limit"}, status_code=413)
    response = await next_handler(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
        "connect-src 'self'; img-src 'self' data:; font-src 'self'; object-src 'none'; "
        "base-uri 'none'; frame-ancestors 'self'; form-action 'self'")
    return response


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
    if low.endswith(OFFICE_EXT):
        return "office_read", "document"
    if low.endswith(TEXT_EXT):
        return "fs_read", "text"
    return None, "unsupported"


class Chat(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    attachment: Optional[str] = Field(default=None, max_length=240)

    @field_validator("message")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("Enter a request")
        return value.strip()


def _check_attachment(body):
    if not body.attachment:
        return
    try:
        p = tools.workspace_path(body.attachment)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not p.is_file():
        raise HTTPException(404, "The attachment is missing. Please upload it again.")
    if _reader_for(body.attachment)[0] is None:
        raise HTTPException(415, "Unsupported attachment type")
    if p.stat().st_size > MAX_UPLOAD:
        raise HTTPException(413, "Attachment exceeds 20 MiB")


def _events(body, request_id, cancelled):
    started = time.monotonic()
    trace = []
    yield {"type": "start", "request_id": request_id}
    try:
        if cancelled.is_set():
            raise InterruptedError("Request cancelled")
        yield {"type": "progress", "message": "Selecting a local model"}
        name = body.attachment
        reader, kind = _reader_for(name) if name else (None, None)
        r = model_router.route(body.message, has_attachment=bool(name), attachment_kind=kind)
        yield {"type": "route", **r}
        if cancelled.is_set():
            raise InterruptedError("Request cancelled")
        source = ""
        # An attachment is read with the tool that suits its type, then the
        # question is answered with that content in hand.
        if name:
            yield {"type": "progress", "message": "Reading the attachment"}
            with tools.task_scope(files=[name], model=r["orchestrator"], request=body.message, cancel_event=cancelled):
                source = call(reader, {"path": name})
            entry = {"step": 1, "type": "tool", "tool": reader,
                     "args": {"path": name}, "result": source[:1800], "ok": tools.succeeded(source)}
            trace.append(entry)
            yield {**entry, "type": "step"}
            if cancelled.is_set():
                raise InterruptedError("Request cancelled")
            if not tools.succeeded(source):
                raise ValueError("The attachment could not be read. " + source[6:].strip())
        for event in agent.iter_run(body.message, r["orchestrator"], attachment=name,
                                    source=source, initial_trace=trace, cancel_event=cancelled):
            if event.get("type") == "final":
                event["secs"] = round(time.monotonic() - started, 1)
            yield event
    except Exception as exc:
        status = "cancelled" if isinstance(exc, InterruptedError) else "failed"
        yield {"type": "final", "status": status, "answer": str(exc), "trace": trace,
               "files": [], "artifacts": [], "steps": len(trace),
               "secs": round(time.monotonic() - started, 1), "error": type(exc).__name__}
    yield {"type": "done", "request_id": request_id}


def _start_run(body):
    _check_attachment(body)
    if not _SLOTS.acquire(blocking=False):
        raise HTTPException(409, "Another request is still running. Wait for it to finish or stop it.")
    task_id, cancelled = uuid.uuid4().hex, threading.Event()
    with _RUN_LOCK:
        _RUNS[task_id] = cancelled
    return task_id, cancelled


def _end_run(task_id):
    with _RUN_LOCK:
        ended = _RUNS.pop(task_id, None)
    if ended is not None:
        _SLOTS.release()


@app.post("/api/chat")
def chat(body: Chat):
    task_id, cancelled = _start_run(body)
    route, final = {}, {}
    try:
        with audit.request_context(task_id):
            for event in _events(body, task_id, cancelled):
                if event["type"] == "route":
                    route = {k: v for k, v in event.items() if k != "type"}
                if event["type"] == "final":
                    final = {k: v for k, v in event.items() if k != "type"}
        return {**route, **final, "request_id": task_id}
    finally:
        _end_run(task_id)


@app.post("/api/chat/stream")
def chat_stream(body: Chat):
    """Same as /api/chat but emits each tool call as it happens (SSE).

    Keeps the interface informative while a local model is thinking, and lets a
    reviewer watch the agent actually take steps.
    """
    task_id, cancelled = _start_run(body)
    messages = queue.Queue(maxsize=64)
    finished, disconnected = threading.Event(), threading.Event()

    def work():
        try:
            with audit.request_context(task_id):
                for event in _events(body, task_id, cancelled):
                    while not disconnected.is_set():
                        try:
                            messages.put(event, timeout=0.2)
                            break
                        except queue.Full:
                            continue
                    if disconnected.is_set():
                        break
        finally:
            _end_run(task_id)
            finished.set()

    try:
        threading.Thread(target=work, daemon=True, name="workbench-task").start()
    except RuntimeError as exc:
        _end_run(task_id)
        raise HTTPException(503, "The request worker could not start") from exc

    async def stream():
        last_heartbeat = time.monotonic()
        try:
            while not finished.is_set() or not messages.empty():
                try:
                    event = messages.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.2)
                    if time.monotonic() - last_heartbeat >= 10:
                        yield ": heartbeat\n\n"
                        last_heartbeat = time.monotonic()
                    continue
                yield "data: " + json.dumps(event, ensure_ascii=False, default=str) + "\n\n"
                last_heartbeat = time.monotonic()
        finally:
            disconnected.set()
            cancelled.set()

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.post("/api/chat/{task_id}/cancel")
def cancel(task_id: str):
    with _RUN_LOCK:
        running = _RUNS.get(task_id)
        if running:
            running.set()
    return {"status": "cancelling" if running else "finished"}


@app.post("/api/upload")
def upload(file: UploadFile = File(...)):
    name = PureWindowsPath((file.filename or "").replace("/", "\\")).name
    reader, kind = _reader_for(name)
    if not reader:
        raise HTTPException(415, "Unsupported file. Use text, PDF, image, DOCX, PPTX or XLSX.")
    data = file.file.read(MAX_UPLOAD + 1)
    if not data:
        raise HTTPException(400, "The uploaded file is empty")
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Upload exceeds 20 MiB")
    try:
        tools.validate_upload(data, Path(name).suffix.lower())
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc
    stem = tools._safe_name(Path(name).stem)
    saved = "{}-{}{}".format(stem, uuid.uuid4().hex[:12], Path(name).suffix.lower())
    dest = WORKSPACE / saved
    with dest.open("xb") as output:
        output.write(data)
    audit.record("tool_call", tool="upload", dest="local-process", ok=True, bytes=len(data))
    return {"saved": saved, "display_name": name, "kind": kind, "bytes": len(data)}


def _is_external(ip):
    try:
        address = ipaddress.ip_address(ip.split("%")[0])
        return not address.is_loopback
    except ValueError:
        return True


@app.get("/api/egress")
def egress(scope: str = Query(default="process", pattern="^(process|machine)$")):
    """Snapshot of non-loopback connections, not proof of historical isolation.

    scope=process : this backend and its children (not the separate inference service)
    scope=machine : every process on the deployment host, subject to permissions
    """
    hits = []
    try:
        me = psutil.Process(os.getpid())
        pids = {me.pid, *(p.pid for p in me.children(recursive=True))} if scope == "process" else None
        for connection in psutil.net_connections(kind="inet"):
            if pids is not None and connection.pid not in pids:
                continue
            if connection.status == "ESTABLISHED" and connection.raddr and _is_external(connection.raddr.ip):
                hits.append("{}:{}".format(connection.raddr.ip, connection.raddr.port))
    except (psutil.Error, OSError):
        return {"external_calls": None, "status": "unknown", "scope": scope,
                "note": "Connection snapshot unavailable with current permissions"}
    return {"external_calls": len(hits), "status": "snapshot", "scope": scope,
            "detail": sorted(set(hits))[:20], "note": "Instantaneous snapshot; not an air-gap certificate"}


@app.post("/api/egress/probe")
def egress_probe():
    """Attempt fixed network probes inside the code sandbox.

    Blocked results describe only the tested sandbox. Unavailable or failed
    probes are inconclusive, never successful isolation checks.
    """
    if not _SLOTS.acquire(blocking=False):
        raise HTTPException(409, "Another task or probe is still running")
    try:
        with audit.request_context(uuid.uuid4().hex):
            out = call("egress_probe", {})
        try:
            result = json.loads(out)
            if result.get("scope") == "sandbox" and result.get("status") in ("blocked", "reachable", "inconclusive"):
                return result
        except (ValueError, AttributeError):
            pass
        return tools.probe_report("ERROR: probe could not complete")
    finally:
        _SLOTS.release()


@app.get("/api/audit")
def audit_log(n: int = Query(default=50, ge=0, le=200)):
    return {"summary": audit.summary(), "recent": audit.tail(n)}


@app.get("/api/models")
def models():
    state = runtime_status()
    reg = model_router.registry()
    installed = state["models"]
    return {"runtime_models": installed, "registry": reg,
            "missing": [m["id"] for m in reg["models"] if m["id"] not in installed],
            "runtime_up": state["reachable"], "error": state.get("error"),
            "rejected_remote_models": state.get("rejected_remote_models", []),
            "note": "Installed local models; not all are resident in GPU memory"}


@app.get("/api/health")
def health():
    state = runtime_status()
    current = _disk_build_id()
    return {"status": "ok" if state["reachable"] else "runtime_down",
            "models_installed": len(state["models"]), "version": "local-hardening-v2", "api_version": 2,
            "build_id": BUILD_ID, "disk_build_id": current, "restart_required": current != BUILD_ID,
            "workspace_files": sorted(p.name for p in WORKSPACE.iterdir()
                                      if p.is_file() and not p.is_symlink() and not p.name.startswith(("_", "."))),
            "outputs": sorted(p.name for p in OUT.iterdir() if p.is_file() and not p.is_symlink())}


@app.get("/api/download/{name}")
def download(name: str, sha256: Optional[str] = Query(default=None, pattern="^[a-f0-9]{64}$")):
    if name != Path(name).name or PureWindowsPath(name).drive or "\\" in name or ":" in name:
        raise HTTPException(400, "Invalid output filename")
    info = tools.artifact_info(name, require_content=True)
    if info is None:
        raise HTTPException(404, "The generated file is missing or failed format validation")
    if sha256 is not None and info["sha256"] != sha256:
        raise HTTPException(409, "The file changed after verification. Regenerate it before downloading.")
    return FileResponse(OUT / name, filename=name)


FRONTEND = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="ui")
