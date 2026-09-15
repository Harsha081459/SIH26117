"""Tool registry — the local tools the agent can call.

Sandbox: Docker `--network none` container when Docker is available
(generated code physically cannot reach a network), subprocess+timeout
fallback otherwise. KB search: local embeddings via Ollama
(nomic-embed-text) with a keyword fallback — never leaves the box.
"""
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).parent.parent / "workspace"
CORPUS = Path(__file__).parent.parent / "corpus"
OUT = Path(__file__).parent.parent / "outputs"
EMBED_CACHE = Path(__file__).parent / ".embed_cache.json"
for d in (WORKSPACE, CORPUS, OUT):
    d.mkdir(exist_ok=True)

EMBED_MODEL = "nomic-embed-text"


def _safe_workspace(path: str) -> Path:
    p = (WORKSPACE / path).resolve()
    if not str(p).startswith(str(WORKSPACE.resolve())):
        raise ValueError("path outside workspace")
    return p


# ── filesystem ──────────────────────────────────────────────────────────
def fs_read(path: str) -> str:
    try:
        return _safe_workspace(path).read_text(errors="replace")[:8000]
    except FileNotFoundError:
        return f"ERROR: {path} not found in workspace"


def fs_write(path: str, content: str) -> str:
    try:
        p = _safe_workspace(path)
    except ValueError:
        return "ERROR: path outside workspace"
    p.write_text(content)
    return f"WROTE {p.name} ({len(content)} chars)"


def fs_list() -> str:
    files = [f.name for f in WORKSPACE.iterdir() if f.is_file() and not f.name.startswith("_")]
    return "workspace files: " + (", ".join(files) or "(empty)")


# ── sandboxed code execution ────────────────────────────────────────────
def run_python(code: str, timeout: int = 30) -> str:
    script = WORKSPACE / "_sandbox_run.py"
    script.write_text(code)
    if shutil.which("docker"):
        try:
            r = subprocess.run(
                ["docker", "run", "--rm", "--network", "none",
                 "--memory", "256m", "--cpus", "1",
                 "-v", f"{WORKSPACE}:/work", "-w", "/work",
                 "python:3.11-slim", "python", "_sandbox_run.py"],
                capture_output=True, text=True, timeout=timeout + 10)
            out = (r.stdout + r.stderr).strip()[:4000] or "(no output)"
            return f"[docker-sandbox, network=none] {out}"
        except Exception:
            pass  # fall back to subprocess
    try:
        r = subprocess.run([sys.executable, str(script)], capture_output=True,
                           text=True, timeout=timeout, cwd=str(WORKSPACE))
        out = (r.stdout + r.stderr).strip()[:4000] or "(no output)"
        return f"[local-sandbox] {out}"
    except subprocess.TimeoutExpired:
        return f"[local-sandbox] ERROR: execution timed out after {timeout}s"


# ── multimodal / OCR ────────────────────────────────────────────────────
def ocr_doc(path: str, vl_model: str = "qwen2-vl:7b") -> str:
    """Read a scanned doc / photo / drawing via the local VL model."""
    from ollama_client import generate
    try:
        p = _safe_workspace(path)
    except ValueError:
        return "ERROR: path outside workspace"
    if not p.exists():
        return f"ERROR: {path} not found in workspace"
    return generate(vl_model,
                    "Extract all text and key information from this document image. "
                    "List findings as bullet points.",
                    image_path=str(p))


# ── knowledge base: local embeddings + cosine, keyword fallback ─────────
def _embed(text: str) -> list[float]:
    import httpx
    r = httpx.post("http://127.0.0.1:11434/api/embed",
                   json={"model": EMBED_MODEL, "input": text[:2000]}, timeout=60)
    r.raise_for_status()
    return r.json()["embeddings"][0]


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-9)


def kb_search(query: str, top_k: int = 3) -> str:
    docs = {f.name: f.read_text(errors="replace") for f in CORPUS.glob("*.txt")}
    if not docs:
        return "Knowledge base is empty."
    try:
        cache = json.loads(EMBED_CACHE.read_text()) if EMBED_CACHE.exists() else {}
        vecs = {}
        for name, text in docs.items():
            if name not in cache or cache[name]["len"] != len(text):
                cache[name] = {"len": len(text), "vec": _embed(text)}
            vecs[name] = cache[name]["vec"]
        EMBED_CACHE.write_text(json.dumps(cache))
        qv = _embed(query)
        ranked = sorted(docs, key=lambda n: _cosine(qv, vecs[n]), reverse=True)
        mode = "semantic"
    except Exception:
        # keyword fallback (works even with Ollama down)
        qw = set(query.lower().split())
        ranked = sorted(docs, key=lambda n: sum(docs[n].lower().count(w)
                                                for w in qw if len(w) > 3),
                        reverse=True)
        ranked = [n for n in ranked if sum(docs[n].lower().count(w)
                                           for w in qw if len(w) > 3) > 0] or ranked[:top_k]
        mode = "keyword"
    body = "\n---\n".join(f"[{n}]\n{docs[n][:1200]}" for n in ranked[:top_k])
    return f"[kb:{mode}] {body}"


# ── deliverable generators ──────────────────────────────────────────────
def _safe_name(title: str) -> str:
    return "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:60]


def make_docx(title: str, body: str) -> str:
    from docx import Document
    doc = Document()
    doc.add_heading(title, 0)
    for para in body.split("\n\n"):
        if para.strip():
            doc.add_paragraph(para.strip())
    out = OUT / f"{_safe_name(title)}.docx"
    doc.save(out)
    return f"FILE:{out.name}"


def make_xlsx(title: str, csv_rows: str) -> str:
    """csv_rows: newline-separated rows, commas between cells."""
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = _safe_name(title)[:30] or "Sheet1"
    for row in csv_rows.strip().splitlines():
        ws.append([c.strip() for c in row.split(",")])
    out = OUT / f"{_safe_name(title)}.xlsx"
    wb.save(out)
    return f"FILE:{out.name}"


def make_pptx(title: str, slides: str) -> str:
    """slides: one slide per line 'Slide Title | bullet; bullet; bullet'."""
    from pptx import Presentation
    prs = Presentation()
    for line in slides.strip().splitlines():
        if not line.strip():
            continue
        t, _, bullets = line.partition("|")
        s = prs.slides.add_slide(prs.slide_layouts[1])
        s.shapes.title.text = t.strip()
        for b in bullets.split(";"):
            if b.strip():
                s.placeholders[1].text_frame.add_paragraph().text = b.strip()
    out = OUT / f"{_safe_name(title)}.pptx"
    prs.save(out)
    return f"FILE:{out.name}"


def sheet_read(path: str) -> str:
    """Read first sheet of a workspace .xlsx as CSV-ish text."""
    from openpyxl import load_workbook
    try:
        p = _safe_workspace(path)
    except ValueError:
        return "ERROR: path outside workspace"
    try:
        ws = load_workbook(p).active
    except Exception as e:
        return f"ERROR: cannot read {path}: {e}"
    rows = [[str(c.value or "") for c in r] for r in ws.iter_rows(max_row=50)]
    rows = [r for r in rows if any(c.strip() for c in r)]
    return "\n".join(",".join(r) for r in rows)


TOOLS = {
    "fs_list":    {"fn": fs_list,    "args": [],                    "desc": "list files in the workspace"},
    "fs_read":    {"fn": fs_read,    "args": ["path"],              "desc": "read a file in the workspace"},
    "fs_write":   {"fn": fs_write,   "args": ["path", "content"],   "desc": "write a file in the workspace"},
    "run_python": {"fn": run_python, "args": ["code"],              "desc": "execute python code in a network-less sandbox and return output"},
    "ocr_doc":    {"fn": ocr_doc,    "args": ["path"],              "desc": "extract text/findings from a scanned doc, photo, or drawing image"},
    "kb_search":  {"fn": kb_search,  "args": ["query"],             "desc": "search internal manuals/SOPs/correspondence"},
    "sheet_read": {"fn": sheet_read, "args": ["path"],              "desc": "read a workspace .xlsx as rows"},
    "make_docx":  {"fn": make_docx,  "args": ["title", "body"],     "desc": "produce a Word document deliverable"},
    "make_xlsx":  {"fn": make_xlsx,  "args": ["title", "csv_rows"], "desc": "produce an Excel deliverable"},
    "make_pptx":  {"fn": make_pptx,  "args": ["title", "slides"],   "desc": "produce a PowerPoint deliverable ('Title | bullet; bullet' per line)"},
}

TOOL_SPEC = "\n".join(f"- {name}({', '.join(t['args'])}): {t['desc']}" for name, t in TOOLS.items())


def call(name: str, args: dict) -> str:
    if name not in TOOLS:
        return f"ERROR: unknown tool {name}"
    try:
        return str(TOOLS[name]["fn"](**args))
    except TypeError as e:
        return f"ERROR: bad args for {name}: {e}"
    except Exception as e:
        return f"ERROR in {name}: {e}"
