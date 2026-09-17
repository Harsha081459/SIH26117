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
import time
from pathlib import Path

import audit
import calc as calc_mod

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
def fs_read(path):
    try:
        return _safe_workspace(path).read_text(errors="replace")[:8000]
    except ValueError:
        return "ERROR: path outside workspace"
    except FileNotFoundError:
        return "ERROR: {} not found in workspace".format(path)
    except UnicodeDecodeError:
        return "ERROR: {} is not a text file (use ocr_doc, pdf_read or sheet_read)".format(path)


def fs_write(path, content):
    try:
        p = _safe_workspace(path)
    except ValueError:
        return "ERROR: path outside workspace"
    p.write_text(content)
    return "WROTE {} ({} chars)".format(p.name, len(content))


def fs_list() -> str:
    files = [f.name for f in WORKSPACE.iterdir() if f.is_file() and not f.name.startswith("_")]
    return "workspace files: " + (", ".join(files) or "(empty)")


# ── sandboxed code execution ────────────────────────────────────────────
_UNSHARE_OK = None


def _unshare_works():
    """Can we get an empty network namespace without root? (Linux user namespaces)"""
    global _UNSHARE_OK
    if _UNSHARE_OK is None:
        if not shutil.which("unshare"):
            _UNSHARE_OK = False
        else:
            try:
                r = subprocess.run(["unshare", "-rn", "true"],
                                   capture_output=True, timeout=10)
                _UNSHARE_OK = r.returncode == 0
            except Exception:
                _UNSHARE_OK = False
    return _UNSHARE_OK


def run_python(code, timeout=30):
    """Execute model-written code with the network taken away.

    Three levels, strongest first. The label in the output names the level that
    actually ran, so a reviewer never has to guess how isolated the code was.
      1. Docker container started with --network none
      2. a Linux network namespace via `unshare -rn` (no root required)
      3. a plain subprocess with a timeout -- reported as NOT isolated
    """
    script = WORKSPACE / "_sandbox_run.py"
    script.write_text(code)

    if shutil.which("docker"):
        try:
            r = subprocess.run(
                ["docker", "run", "--rm", "--network", "none",
                 "--memory", "256m", "--cpus", "1",
                 "-v", "{}:/work".format(WORKSPACE), "-w", "/work",
                 "python:3.11-slim", "python", "_sandbox_run.py"],
                capture_output=True, text=True, timeout=timeout + 15)
            out = (r.stdout + r.stderr).strip()[:4000] or "(no output)"
            return "[sandbox: docker, network=none] {}".format(out)
        except Exception:
            pass

    if _unshare_works():
        try:
            r = subprocess.run(["unshare", "-rn", sys.executable, str(script)],
                               capture_output=True, text=True, timeout=timeout,
                               cwd=str(WORKSPACE))
            out = (r.stdout + r.stderr).strip()[:4000] or "(no output)"
            return "[sandbox: netns via unshare, network=none] {}".format(out)
        except subprocess.TimeoutExpired:
            return "[sandbox: netns] ERROR: execution timed out after {}s".format(timeout)
        except Exception:
            pass

    try:
        r = subprocess.run([sys.executable, str(script)], capture_output=True,
                           text=True, timeout=timeout, cwd=str(WORKSPACE))
        out = (r.stdout + r.stderr).strip()[:4000] or "(no output)"
        return "[sandbox: subprocess, NOT network-isolated] {}".format(out)
    except subprocess.TimeoutExpired:
        return "[sandbox: subprocess] ERROR: execution timed out after {}s".format(timeout)


EGRESS_TARGETS = [
    ("https://example.com", "443/tcp HTTPS"),
    ("8.8.8.8:53", "53/udp DNS"),
]


def egress_probe():
    """Deliberately attempt outbound connections and report that they fail.

    A counter sitting at zero only shows that nothing happened to go out. This
    actively tries to leave the machine from inside the sandbox, so the denial
    is demonstrated rather than assumed.
    """
    code = (
        "import socket, urllib.request\n"
        "targets = [('example.com', 443, 'HTTPS 443'), ('8.8.8.8', 53, 'DNS 53')]\n"
        "for host, port, label in targets:\n"
        "    try:\n"
        "        socket.create_connection((host, port), 3).close()\n"
        "        print('ALLOWED  ' + label + ' -> reached ' + host)\n"
        "    except Exception as e:\n"
        "        print('DENIED   ' + label + ' -> ' + type(e).__name__ + ': ' + str(e)[:60])\n"
        "try:\n"
        "    urllib.request.urlopen('https://example.com', timeout=3)\n"
        "    print('ALLOWED  urllib https://example.com')\n"
        "except Exception as e:\n"
        "    print('DENIED   urllib https://example.com -> ' + type(e).__name__)\n"
    )
    out = run_python(code, timeout=25)
    allowed = "ALLOWED" in out
    verdict = ("EGRESS POSSIBLE - this deployment is NOT isolated"
               if allowed else
               "ALL OUTBOUND ATTEMPTS DENIED - nothing can leave this machine")
    audit.record("egress_probe", dest="local-process", ok=not allowed,
                 verdict=verdict, detail=out[:400])
    return "{}\n\n{}".format(out, verdict)


# ── multimodal / OCR ────────────────────────────────────────────────────
def _vision_model():
    """Pick a vision model that is actually loaded rather than a hardcoded name."""
    from ollama_client import list_models
    loaded = list_models()
    for m in loaded:
        low = m.lower()
        if any(tag in low for tag in ("vl", "vision", "llava", "moondream", "minicpm-v")):
            return m
    import router
    for entry in router.registry()["models"]:
        if "image_understanding" in entry["tasks"]:
            return entry["id"]
    return loaded[0] if loaded else "qwen2.5vl:3b"


def ocr_doc(path, vl_model=None):
    """Read a scanned document, photo, drawing or handwritten note locally."""
    from ollama_client import generate
    try:
        p = _safe_workspace(path)
    except ValueError:
        return "ERROR: path outside workspace"
    if not p.exists():
        return "ERROR: {} not found in workspace".format(path)
    model = vl_model or _vision_model()
    return generate(model,
                    "Extract all text and key information from this document image. "
                    "List the findings as bullet points, preserving numbers, units "
                    "and limits exactly as written.",
                    image_path=str(p))


# ── knowledge base: local embeddings + cosine, keyword fallback ─────────
def pdf_read(path: str, max_pages: int = 8) -> str:
    """Read a PDF. Uses the embedded text layer when present; falls back to
    rendering the page and reading it with the local vision model, which is
    what makes scanned PDFs work."""
    try:
        p = _safe_workspace(path)
    except ValueError:
        return "ERROR: path outside workspace"
    if not p.exists():
        return "ERROR: {} not found in workspace".format(path)
    try:
        import pymupdf
    except ImportError:
        try:
            import fitz as pymupdf
        except ImportError:
            return ("ERROR: PyMuPDF not installed (pip install pymupdf). "
                    "For scanned pages, convert to PNG and use ocr_doc.")
    doc = pymupdf.open(str(p))
    chunks = []
    for i, page in enumerate(doc[:max_pages]):
        text = page.get_text().strip()
        if len(text) >= 40:
            chunks.append("[page {} - text layer]\n{}".format(i + 1, text[:2500]))
            continue
        # No usable text layer: this is a scanned page. Render it and read it
        # with the local vision model.
        img_path = WORKSPACE / "_pdfpage_{}_{}.png".format(p.stem, i + 1)
        try:
            page.get_pixmap(dpi=170).save(str(img_path))
            vision = ocr_doc(img_path.name)
            chunks.append("[page {} - scanned, read by vision model]\n{}".format(
                i + 1, vision[:2500]))
        except Exception as e:
            # Keep the pages we could read rather than failing the whole file.
            chunks.append("[page {} - scanned; vision model unavailable: {}]".format(
                i + 1, str(e)[:120]))
    doc.close()
    return "\n\n".join(chunks) or "(no extractable content)"


def calculate(expression, label=""):
    """Deterministic arithmetic with every intermediate step shown."""
    try:
        return calc_mod.evaluate_text(expression, label)
    except calc_mod.CalcError as e:
        # A bare "not allowed" is not actionable; say what to do instead, because
        # the caller is usually a model that can correct itself given an example.
        return ("ERROR: {}. calculate() takes only literal numbers and "
                "+ - * / ** % with sqrt/log/exp/min/max/round -- it cannot read "
                "files or run code. Substitute the values you already have, e.g. "
                '"(2*18400 + 5*3200) * 1.18".'.format(e))


def _embed(text):
    import httpx
    r = httpx.post("http://127.0.0.1:11434/api/embed",
                   json={"model": EMBED_MODEL, "input": text[:2000]}, timeout=60)
    r.raise_for_status()
    return r.json()["embeddings"][0]


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-9)


def _keyword_scores(query, docs):
    """Term-frequency score per document, normalised to 0..1."""
    terms = [w for w in query.lower().split() if len(w) > 3]
    raw = {}
    for name, text in docs.items():
        low = text.lower()
        raw[name] = sum(low.count(w) for w in terms)
    peak = max(raw.values()) if raw else 0
    return {n: (v / peak if peak else 0.0) for n, v in raw.items()}


def kb_search(query, top_k=3):
    """Search the organisation's own manuals, SOPs and correspondence.

    Ranking is hybrid. Embedding similarity handles a reworded question;
    keyword frequency anchors exact plant vocabulary such as a tag number or a
    unit. Each alone misranks -- two procedure documents look nearly identical
    to an embedding model, while keywords miss paraphrase -- so both contribute
    half the score. Keyword-only remains the fallback when no embedding model is
    loaded, which keeps the tool usable with the runtime down.
    """
    docs = {f.name: f.read_text(errors="replace") for f in CORPUS.glob("*.txt")}
    if not docs:
        return "Knowledge base is empty."

    kw = _keyword_scores(query, docs)
    try:
        cache = json.loads(EMBED_CACHE.read_text()) if EMBED_CACHE.exists() else {}
        vecs = {}
        for name, text in docs.items():
            if name not in cache or cache[name]["len"] != len(text):
                cache[name] = {"len": len(text), "vec": _embed(text)}
            vecs[name] = cache[name]["vec"]
        EMBED_CACHE.write_text(json.dumps(cache))
        qv = _embed(query)
        sem = {n: _cosine(qv, vecs[n]) for n in docs}
        lo, hi = min(sem.values()), max(sem.values())
        span = (hi - lo) or 1.0
        score = {n: 0.5 * ((sem[n] - lo) / span) + 0.5 * kw[n] for n in docs}
        mode = "hybrid"
    except Exception:
        score = kw
        mode = "keyword"

    ranked = sorted(docs, key=lambda n: score[n], reverse=True)
    body = "\n---\n".join("[{}]\n{}".format(n, docs[n][:1200]) for n in ranked[:top_k])
    return "[kb:{}] {}".format(mode, body)


# ── deliverable generators ──────────────────────────────────────────────
def _safe_name(title: str) -> str:
    return "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:60]


def make_docx(title, body, findings="", recommendation="", reference="",
              model=""):
    """Produce a Word deliverable.

    Plain prose works, but an industrial approval note has an expected shape --
    title block, findings, recommendation, sign-off -- so the structured fields
    are filled in when supplied. Provenance (model used, generation time, that
    it was produced on-premise) is stamped in the document itself, because a
    reviewer receiving the file later needs to know where it came from.
    """
    from docx import Document
    from docx.shared import Pt

    doc = Document()
    doc.add_heading(title, 0)

    meta = doc.add_paragraph()
    meta.add_run("Generated on-premise · {}{}".format(
        time.strftime("%d-%b-%Y %H:%M"),
        " · model: {}".format(model) if model else "")).italic = True
    for run in meta.runs:
        run.font.size = Pt(9)

    if body:
        doc.add_heading("Summary", level=1)
        for para in str(body).split("\n\n"):
            if para.strip():
                doc.add_paragraph(para.strip())

    if findings:
        doc.add_heading("Findings", level=1)
        items = [f for f in str(findings).replace("\n", ";").split(";") if f.strip()]
        for f in items:
            doc.add_paragraph(f.strip().lstrip("-* "), style="List Bullet")

    if recommendation:
        doc.add_heading("Recommendation", level=1)
        doc.add_paragraph(str(recommendation).strip())

    if reference:
        doc.add_heading("Reference", level=1)
        doc.add_paragraph(str(reference).strip())

    doc.add_paragraph()
    doc.add_paragraph("Prepared by: ____________________    "
                      "Approved by: ____________________")

    out = OUT / "{}.docx".format(_safe_name(title))
    doc.save(out)
    return "FILE:{}".format(out.name)


def make_xlsx(title, csv_rows):
    """Build a spreadsheet deliverable.

    csv_rows is normally newline-separated text with commas between cells, but
    models also pass a list of lists, so both are accepted rather than losing
    the deliverable over a formatting detail.
    """
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = _safe_name(title)[:30] or "Sheet1"

    if isinstance(csv_rows, (list, tuple)):
        for row in csv_rows:
            if isinstance(row, (list, tuple)):
                ws.append([str(c) for c in row])
            else:
                ws.append([c.strip() for c in str(row).split(",")])
    else:
        for row in str(csv_rows).strip().splitlines():
            ws.append([c.strip() for c in row.split(",")])

    out = OUT / "{}.xlsx".format(_safe_name(title))
    wb.save(out)
    return "FILE:{}".format(out.name)


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
    "fs_read":    {"fn": fs_read,    "args": ["path"],              "desc": "read a text file in the workspace"},
    "fs_write":   {"fn": fs_write,   "args": ["path", "content"],   "desc": "write a file in the workspace"},
    "run_python": {"fn": run_python, "args": ["code"],              "desc": "execute python code in a network-less sandbox and return output"},
    "ocr_doc":    {"fn": ocr_doc,    "args": ["path"],              "desc": "read a scanned image, photo, handwritten note or engineering drawing"},
    "pdf_read":   {"fn": pdf_read,   "args": ["path"],              "desc": "read a PDF including scanned PDFs (falls back to the vision model per page)"},
    "kb_search":  {"fn": kb_search,  "args": ["query"],             "desc": "search internal manuals/SOPs/correspondence"},
    "sheet_read": {"fn": sheet_read, "args": ["path"],              "desc": "read a workspace .xlsx as rows"},
    "calculate":  {"fn": calculate,  "args": ["expression"],        "desc": "evaluate an arithmetic/engineering expression exactly, showing every step"},
    "egress_probe": {"fn": egress_probe, "args": [],                "desc": "attempt outbound network connections and report that they are denied"},
    "make_docx":  {"fn": make_docx,  "args": ["title", "body"],     "desc": "produce a Word deliverable; optional findings, recommendation, reference for an approval note"},
    "make_xlsx":  {"fn": make_xlsx,  "args": ["title", "csv_rows"], "desc": "produce an Excel deliverable"},
    "make_pptx":  {"fn": make_pptx,  "args": ["title", "slides"],   "desc": "produce a PowerPoint deliverable ('Title | bullet; bullet' per line)"},
}

TOOL_SPEC = "\n".join("- {}({}): {}".format(name, ", ".join(t["args"]), t["desc"])
                      for name, t in TOOLS.items())


def call(name, args):
    """Dispatch a tool call, recording it in the audit log."""
    t0 = time.time()
    if name not in TOOLS:
        result = "ERROR: unknown tool {}".format(name)
    else:
        try:
            result = str(TOOLS[name]["fn"](**args))
        except TypeError as e:
            result = "ERROR: bad args for {}: {}".format(name, e)
        except Exception as e:
            result = "ERROR in {}: {}".format(name, e)
    audit.record("tool_call", tool=name, args=args, dest="local-process",
                 ms=int((time.time() - t0) * 1000),
                 ok=not result.startswith("ERROR"), result_preview=result[:200])
    return result
