# Sovereign On-Premise Agentic AI Workbench

**SIH 2026 · Problem Statement `SIH26117`**  
**Organisation:** Mangalore Refinery and Petrochemicals Ltd (MRPL)  
**Theme:** Smart Automation · **Team:** LatentX · **Institute:** IIIT Bangalore

> A **multi-model agent harness** that runs entirely on the organisation's own GPU box.  
> It routes each task to the right open-weight model, calls real local tools, and can **show** that nothing left the machine.

```
  THIS IS                          THIS IS NOT
  -------------------------        -------------------------
  local agent + tools              a ChatGPT skin
  model router (≥2 types)          one frozen model forever
  PDF/scan → Word deliverable      chat text pretending to be a note
  Docker / netns sandbox           code running on the host with Wi-Fi
  live egress DENY probe           "trust us, we are air-gapped" slide
```

## The industrial problem

Refinery and PSU knowledge work is routine and confidential at the same time:

- inspection PDFs and handwritten shift notes  
- P&ID drawings and tag lists  
- SOPs, LOTO procedures, vendor letters  
- spare-cost sheets and short analysis scripts  

None of that can be pasted into a public cloud assistant. Today the choice is slow manual drafting, or silent leakage into SaaS tools. The PS asks for a **sovereign on-premise agentic workbench** on mid-range GPU (or smaller open models), with auto model selection, OCR-to-document agents, sandboxed coding, multimodal reads, and visible proof of zero egress.

We build the **harness around open models**. We do not train a foundation model.

## System at a glance

```mermaid
flowchart TB
  U[Engineer in browser] --> UI[Frontend: chat + live trace + upload + egress badge]
  UI -->|SSE| API[FastAPI]
  API --> R[Router models.yaml]
  R -->|task label| A[Agent loop plan / tool / observe]
  A --> T[Tool registry]
  T --> OCR[ocr_doc / pdf_read]
  T --> SB[run_python sandbox]
  T --> DOC[make_docx / xlsx / pptx]
  T --> KB[kb_search corpus]
  T --> CALC[calculate AST]
  T --> EG[egress_probe]
  A --> OLL[Ollama on 127.0.0.1]
  OCR --> OLL
  API --> AUD[audit.jsonl]
  API --> MON[egress monitor]
  DOC --> OUT[outputs/ downloadable files]
```

ASCII twin (renders everywhere):

```
                 ┌──────────────────────────────────────┐
                 │  Browser UI                          │
                 │  chat · SSE trace · upload · badges  │
                 └──────────────────┬───────────────────┘
                                    │
                 ┌──────────────────▼───────────────────┐
                 │  FastAPI                             │
                 │  route → agent → tools → audit       │
                 └─┬───────────┬─────────────┬──────────┘
                   │           │             │
           ┌───────▼──┐  ┌─────▼─────┐  ┌───▼────────┐
           │  Router  │  │  Agent    │  │  Egress +  │
           │  yaml    │  │  ≤10 step │  │  audit log │
           └───────┬──┘  └─────┬─────┘  └────────────┘
                   │           │
                   │     ┌─────▼──────────────────────┐
                   │     │ Tools: OCR PDF sandbox KB  │
                   │     │ sheet calc docx xlsx pptx  │
                   │     └─────┬──────────────────────┘
                   │           │
                   └─────►┌────▼──────────────────────┐
                          │ Ollama  127.0.0.1:11434    │
                          │ qwen3 · coder · qwen3-vl   │
                          └───────────────────────────┘
```

## Mapped to the problem statement

| PS requirement | How we meet it | Status |
| :---: | :--- | :---: |
| Multi-model backend, auto-select | `router.py` + `models.yaml`; docs / code / vision differ | live |
| New models without redesign | add a YAML block; `_resolve()` substitutes if missing | live |
| Agentic multi-step work | `agent.py` plan → tool → observe, max 10, UI trace | live |
| Local tools | 13 tools in `tools.py` (fs, sandbox, OCR, PDF, KB, sheets, calc, deliverables, probe) | live |
| Scanned docs / handwriting / drawings | `ocr_doc`, `pdf_read` (+ optional PaddleOCR) | live |
| Real deliverables | `make_docx` / `make_xlsx` / `make_pptx` → download | live |
| Grounded in local manuals | `kb_search` over `corpus/` (embed + keyword) | live |
| Proof of no external calls | `/api/egress` + `/api/egress/probe` + `audit.jsonl` | live |

## Four demos that win the room

```mermaid
sequenceDiagram
  participant E as Engineer
  participant UI as UI
  participant R as Router
  participant A as Agent
  participant T as Tools
  participant O as Ollama

  E->>UI: upload inspection PDF + ask for approval note
  UI->>R: classify task
  R-->>UI: draft_document / vision split
  UI->>A: run loop
  A->>T: pdf_read / ocr_doc
  T->>O: VL + optional PaddleOCR
  O-->>T: findings
  A->>T: make_docx
  T-->>UI: FILE:Approval Note….docx
  UI-->>E: download + full tool trace
```

1. **Router proof**  
   Ask a SOP summary, then a coding task. Header shows two different model ids.

2. **Document proof**  
   Attach `workspace/inspection_report.pdf` (or any scan). Trace shows extract → draft → `.docx` download.

3. **Sandbox proof**  
   “Count ERROR lines in `log_sample.txt`.” Result prefix names the isolation tier (`docker` / `unshare` / honest non-isolated).

4. **Egress proof**  
   Click the probe. Rows show DENIED for HTTPS/DNS. Badge stays at external calls = 0 during normal work.

Detailed click order: [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md).

## How a single request is decided

Two decisions, not one:

```
                 message + optional file
                           |
                           v
                 ┌─────────────────────┐
                 │  classify task      │
                 │  keywords first,    │
                 │  else local LLM     │
                 └──────────┬──────────┘
                            |
              ┌─────────────┴─────────────┐
              v                           v
     pick MODEL for task          pick ORCHESTRATOR
     (vl for image labels)        (reasoning model drives loop
                                   when task is vision)
              |                           |
              └─────────────┬─────────────┘
                            v
                   agent JSON actions
                   tool | final answer
```

Why the split: a vision model is strong at reading a P&ID and weak at running a five-step job. The VL model reads through `ocr_doc`; the reasoning model plans. Both appear in the API response and UI.

## Model registry (preferred)

| Role | Ollama tag | Notes |
| :---: | :--- | :--- |
| Orchestration / drafts | `qwen3:8b` | default in `models.yaml` |
| Coding | `qwen2.5-coder:7b` | write_code / debug |
| Vision | `qwen3-vl:8b` | scans, drawings, handwriting |
| Vision fallback | `qwen3-vl:4b` | tighter VRAM |
| Embeddings | `nomic-embed-text` | local KB |
| Legacy substitutes | `qwen2.5:7b-instruct`, `qwen2.5vl:3b`, … | still listed so old pulls keep working |

Optional OCR: `pip install paddlepaddle paddleocr`. If present, scans get a deterministic Paddle pass before VL. If absent, vision-only (no crash).

## Sandbox tiers (named in the tool output)

```
  run_python(code)
        |
        +-->[1] docker run … network=none     strongest
        |
        +-->[2] unshare -rn …                 no root needed
        |
        +-->[3] subprocess + timeout          labelled NOT network-isolated
```

Tier 3 is never sold as air-gap. The string in the result is the proof.

## Measured outcomes (target workstation)

**Hardware:** RTX 4060 Ti 16 GB · Ollama local  
**Stack preference now:** `qwen3:8b` + `qwen3-vl:8b` (or `4b`). If those tags are not pulled yet, router substitution keeps the same tools green.

| Task | Outcome |
| :--- | :--- |
| Sandbox: count `ERROR` in `log_sample.txt` | **3** lines · isolation prefix shown |
| KB: vibration alert limit for pump P-201 | **4.5 mm/s RMS** · SOP-MECH-041 |
| `spares.xlsx` + 18% GST → approval Word file | **Rs 62304** · downloadable `.docx` |
| `scanned_report.png` | seal **14 drops/min** (limit 10) · bearing **74.8 C** |
| `inspection_report.pdf` (text + scanned pages) | both layers recovered |
| `pid_crude_transfer.png` | tags P-201, limits, line design data |
| `handwritten_shift_note.png` | shift time, actions, signer |
| Egress probe | **ALL OUTBOUND ATTEMPTS DENIED** |

## Run in three steps

```bash
bash scripts/setup.sh
cd backend && ../.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

Manual model pulls:

```bash
ollama pull qwen3:8b
ollama pull qwen2.5-coder:7b
ollama pull qwen3-vl:8b
ollama pull qwen3-vl:4b
ollama pull nomic-embed-text
```

Offline suite (no GPU required):

```bash
python tests/test_offline.py
```

## Repository map

```
SIH26117/
├── backend/
│   ├── main.py            API: chat, stream, upload, egress, audit
│   ├── agent.py           plan → act → observe
│   ├── router.py          task classify + model resolve
│   ├── tools.py           13 local tools
│   ├── calc.py            safe AST math
│   ├── audit.py           every call + destination
│   ├── ollama_client.py   127.0.0.1 only
│   └── models.yaml        registry (edit here to add models)
├── frontend/index.html    UI + SSE trace + probe
├── corpus/                synthetic SOPs / correspondence
├── workspace/             demo PDF, P&ID, scan, log, xlsx
├── outputs/               generated deliverables
├── scripts/setup.sh       runtime + pulls + tests
├── docs/
│   ├── DEMO_SCRIPT.md     judge run order
│   └── ARCHITECTURE.md    design decisions
└── tests/test_offline.py  routing, sandbox labels, parser shapes
```

Sample workspace files are **synthetic**, written for the PS rule of open/sample data only. No proprietary MRPL documents are in this repo.

## What we refuse to overclaim

- We prove **process / sandbox / probe** isolation. We do not claim a formal air-gap certification.
- Machine-wide egress scope counts every process on the host. Use a dedicated demo laptop for that mode.
- Small local models can emit messy JSON; the parser retries known shapes. That is why the offline tests exist.
- This nomination POC is the harness. Plant DCS/SCADA connectors, SSO, and K8s are out of scope for the sheet deadline.

## Team · LatentX (IIIT Bangalore)

Anish Reddy · P Lohith · Harsha Vardhan D · P Prajwal · Munjikesh · Hasini

**Docs:** [Demo script](docs/DEMO_SCRIPT.md) · [Architecture](docs/ARCHITECTURE.md)
