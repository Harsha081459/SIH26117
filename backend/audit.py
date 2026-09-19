"""Append-only application audit log.

Every instrumented model inference and tool call records local metadata.
Document contents and generated code are not retained in new records. These
records support debugging; they are not an independent network monitor.

Each record notes the destination of the instrumented call. Calls outside this
application, or uninstrumented native code, are not covered by this log.
"""
import hashlib
import ipaddress
import json
import os
import threading
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

LOG_PATH = Path(__file__).parent.parent / "logs" / "audit.jsonl"
LOG_PATH.parent.mkdir(exist_ok=True)
_lock = threading.Lock()
_REQUEST = ContextVar("audit_request", default=None)
_METADATA_FIELDS = {"ts", "kind", "pid", "request_id", "tool", "model", "dest", "ms", "secs", "steps",
                    "event", "status", "scope", "ok", "bytes", "prompt_chars", "has_image", "streamed",
                    "deliverable_attempted"}


@contextmanager
def request_context(request_id):
    token = _REQUEST.set(request_id)
    try:
        yield
    finally:
        _REQUEST.reset(token)


def _metadata(fields):
    cleaned = {}
    for key, value in fields.items():
        if any(part in key.lower() for part in ("password", "secret", "token", "key")):
            cleaned[key] = "[redacted]"
        elif key == "error":
            cleaned[key] = bool(value)
        elif key in _METADATA_FIELDS or (key.endswith("_metadata") and isinstance(value, dict)
                                        and set(value) == {"bytes", "sha256"}):
            cleaned[key] = value
        else:
            encoded = json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
            cleaned[key + "_metadata"] = {"bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}
    return cleaned


def record(kind, **fields):
    """kind: 'model_call' | 'tool_call' | 'session'."""
    entry = {"ts": datetime.now(timezone.utc).isoformat(), "kind": kind, "pid": os.getpid(),
             "request_id": _REQUEST.get(), **_metadata(fields)}
    line = json.dumps(entry, ensure_ascii=False, default=str)
    with _lock:
        with LOG_PATH.open("a", encoding="utf-8") as output:
            output.write(line + "\n")
    return entry


def tail(n=100):
    if not isinstance(n, int) or n <= 0 or not LOG_PATH.exists():
        return []
    with _lock, LOG_PATH.open(encoding="utf-8", errors="replace") as source:
        lines = deque(source, maxlen=min(n, 50000))
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(_metadata(row))
        except ValueError:
            continue
    return rows


def _local_destination(destination):
    if destination == "local-process":
        return True
    try:
        host = urlsplit("//" + destination).hostname
        return host == "localhost" or ipaddress.ip_address(host).is_loopback
    except (TypeError, ValueError):
        return False


def summary():
    """Summarise instrumented calls, without certifying machine-wide locality."""
    entries = tail(50000)
    kinds, destinations = {}, {}
    for entry in entries:
        kind = entry.get("kind", "?")
        kinds[kind] = kinds.get(kind, 0) + 1
        destination = entry.get("dest")
        if isinstance(destination, str):
            destinations[destination] = destinations.get(destination, 0) + 1
    external = {d: n for d, n in destinations.items() if not _local_destination(d)}
    return {"total_records": len(entries), "by_kind": kinds, "destinations": destinations,
            "external_destinations": external, "all_local": not external if destinations else None,
            "scope": "instrumented application calls", "record_limit": 50000,
            "note": "Metadata view; historical files are not rewritten. Not whole-machine isolation evidence."}
