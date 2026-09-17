"""Append-only audit log.

Every model inference and every tool call is recorded locally. Together with
the egress monitor this is the evidence trail the problem statement asks for:
"show, through logs or a visible network monitor, that no external calls are
made at any point".

Each record notes the destination of the call so a reviewer can confirm that
inference only ever went to 127.0.0.1 (the local Ollama runtime).
"""
import json
import os
import threading
import time
from pathlib import Path

LOG_PATH = Path(__file__).parent.parent / "logs" / "audit.jsonl"
LOG_PATH.parent.mkdir(exist_ok=True)
_lock = threading.Lock()


def record(kind, **fields):
    """kind: 'model_call' | 'tool_call' | 'session'."""
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "kind": kind,
        "pid": os.getpid(),
        **fields,
    }
    line = json.dumps(entry, default=str)
    with _lock:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    return entry


def tail(n=100):
    if not LOG_PATH.exists():
        return []
    lines = LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    out = []
    for line in lines[-n:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def summary():
    """Counts by kind + proof that every outbound destination was localhost."""
    entries = tail(100000)
    kinds, dests = {}, {}
    for e in entries:
        kinds[e.get("kind", "?")] = kinds.get(e.get("kind", "?"), 0) + 1
        d = e.get("dest")
        if d:
            dests[d] = dests.get(d, 0) + 1
    external = {d: c for d, c in dests.items()
                if not (d.startswith("127.0.0.1") or d.startswith("localhost")
                        or d == "local-process")}
    return {
        "total_records": len(entries),
        "by_kind": kinds,
        "destinations": dests,
        "external_destinations": external,
        "all_local": not external,
    }
