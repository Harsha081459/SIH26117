"""FastAPI backend for the sovereign on-prem agentic workbench (POC)."""
import os
import psutil
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import agent
import router as model_router
from ollama_client import list_models
from tools import OUT

app = FastAPI(title="Sovereign AI Workbench — SIH26117 POC")

BOOT_TIME = __import__("time").time()
LOCAL_NETS = ("127.", "10.", "172.16.", "172.17.", "192.168.", "::1")


class Chat(BaseModel):
    message: str
    attachment: str | None = None


@app.post("/api/chat")
def chat(body: Chat):
    r = model_router.route(body.message, has_attachment=bool(body.attachment))
    if body.attachment:
        from tools import call
        r["answer"] = call("ocr_doc", {"path": body.attachment})
        r["trace"] = [{"step": 1, "type": "tool", "tool": "ocr_doc",
                       "args": {"path": body.attachment}, "result": r["answer"][:500]}]
        return r
    try:
        out = agent.run(body.message, r["model"])
    except Exception as e:
        return {**r, "answer": f"ERROR reaching local inference (is Ollama running? `ollama serve`): {e}",
                "trace": [], "steps": 0, "secs": 0}
    return {**r, **out}


@app.get("/api/egress")
def egress():
    """Outbound connections from THIS process to non-local addresses — must stay 0.
    On the dedicated air-gapped box, monitor the whole machine instead."""
    me = psutil.Process(os.getpid())
    pids = {me.pid} | {c.pid for c in me.children(recursive=True)}
    hits = []
    try:
        for c in psutil.net_connections(kind="inet"):
            if c.pid in pids and c.status == "ESTABLISHED" and c.raddr:
                ip = c.raddr.ip
                if not ip.startswith(LOCAL_NETS):
                    hits.append(f"{ip}:{c.raddr.port}")
    except Exception as e:
        return {"external_calls": -1, "note": str(e)}
    return {"external_calls": len(hits), "detail": hits[:20]}


@app.get("/api/models")
def models():
    return {"loaded": list_models(), "registry": model_router._CFG}


@app.get("/api/download/{name}")
def download(name: str):
    p = OUT / name
    return FileResponse(p) if p.exists() else {"error": "not found"}


FRONTEND = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="ui")
