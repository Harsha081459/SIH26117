# Sovereign On-Premise Agentic AI Workbench

**SIH 2026 · `SIH26117` · MRPL · Smart Automation**  
**Team LatentX · IIIT Bangalore**

> Local multi-model agent harness for confidential plant work.  
> Not a chat UI. Not a cloud wrapper. Not a fine-tune.

```
  confidential PDF / P&ID / log / xlsx
                 |
                 v
     +---------------------------+
     |  ROUTER  (models.yaml)    |  task -> open-weight model
     +-------------+-------------+
                   |
                   v
     +---------------------------+
     |  AGENT  plan -> tool ->   |  live trace in UI (SSE)
     |         observe (max 10)  |
     +------+------+------+------+
            |      |      |
         OCR/PDF  SANDBOX  DOCX/KB/CALC
            |      |      |
            +------+------+--> outputs/  +  egress DENY proof
                              all on 127.0.0.1
```

## Why this exists

MRPL staff draft approval notes, read inspection scans, check SOPs, and run small analysis scripts on material that must not leave the fence. Cloud assistants are off-limits. Manual work is slow. Quiet pastes into public tools are worse.

This repo is the on-prem answer the PS asks for: open weights, real tools, agent loop, and a monitor that can show nothing left.

## What judges should see (4 proofs)

1. **Model auto-select (≥2 types)**  
   Docs vs code vs vision pick different ids.  
   `router.py` + `models.yaml`

2. **Agentic PDF/scan → Word**  
   Upload → `pdf_read` / `ocr_doc` → `make_docx`.  
   Live agent trace + download link.

3. **Sandboxed coding**  
   `run_python` with network removed.  
   Docker `network=none` or `unshare -rn`.

4. **Zero egress**  
   Badge stays 0. Probe button shows DENY.  
   `/api/egress` + `/api/egress/probe`

Also in the tree: local KB over SOPs, spreadsheet read, AST calculator (no `eval`), audit log of every model/tool destination.

## Request path

```
Browser  =SSE=>  FastAPI
                     |
                     +--> route(task) --> model + orchestrator
                     |
                     +--> agent loop
                     |      |
                     |      +--> tools (13): fs, sandbox, ocr, pdf,
                     |      |              kb, sheet, calc, egress_probe,
                     |      |              make_docx / xlsx / pptx
                     |      |
                     |      +--> Ollama 127.0.0.1:11434
                     |
                     +--> audit.jsonl
                     +--> egress counter (process or machine scope)
```

Vision tasks: VL model reads the image; reasoning model drives the loop. Both show in the response so the UI does not hide the split.

## Models (config, not code)

Preferred pulls (mid GPU / 16 GB class):

```
orchestration / drafts     qwen3:8b
coding                     qwen2.5-coder:7b
vision / scans / P&ID      qwen3-vl:8b   (fallback qwen3-vl:4b)
KB embeddings              nomic-embed-text
```

Add a model by editing `backend/models.yaml`. If a tag is missing, `router._resolve()` substitutes same-family, then default, then whatever is loaded. Legacy qwen2.5 ids remain listed so an existing box keeps running.

Optional: install `paddleocr` for a deterministic OCR pass before the VL model. No paddle → vision-only path, unchanged.

## Sandbox honesty

Generated code tries isolation in order. The tool result names which tier ran:

1. Docker with `network=none`
2. `unshare -rn` network namespace (no root)
3. Timed subprocess labelled `NOT network-isolated`

We do not claim tier 3 is air-gapped.

## Measured on target box

Workstation: **RTX 4060 Ti 16 GB**, Ollama local. Outcomes below are from the live agent (same tool chain as today). Registry now prefers qwen3; if those tags are not pulled yet, substitution keeps the loop green.

```
sandbox ERROR count in log_sample.txt     ->  3
KB vibration limit pump P-201             ->  4.5 mm/s RMS (SOP-MECH-041)
spares.xlsx + 18% GST -> Word note        ->  Rs 62304 + downloadable .docx
scanned_report.png                        ->  seal 14/min (limit 10); bearing 74.8 C
mixed PDF / P&ID / handwritten note       ->  text + tags recovered
egress probe                              ->  ALL OUTBOUND DENIED
```

Full click path: [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md)  
Design notes: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

## Run

```bash
bash scripts/setup.sh
cd backend && ../.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

Manual pulls if you skip setup:

```bash
ollama pull qwen3:8b
ollama pull qwen2.5-coder:7b
ollama pull qwen3-vl:8b
ollama pull qwen3-vl:4b
ollama pull nomic-embed-text
```

Offline checks (no GPU needed):

```bash
python tests/test_offline.py
```

## Layout

```
backend/     agent · router · tools · calc · audit · models.yaml
frontend/    chat · live trace · upload · egress probe
corpus/      sample SOPs / correspondence (synthetic)
workspace/   log, xlsx, scan, mixed PDF, P&ID, handwriting
outputs/     generated docx/xlsx/pptx
scripts/     setup.sh
docs/        DEMO_SCRIPT.md · ARCHITECTURE.md
tests/       test_offline.py
```

Sample files are synthetic, drawn for the PS expectation of open/sample material. No proprietary MRPL data.

## Limits (said plainly)

- Small models sometimes emit odd JSON; the parser accepts the shapes we saw and retries.
- Machine-wide egress scope also counts other users on a shared host. Use a dedicated demo box for that mode.
- PaddleOCR and qwen3 tags are preferred, not mandatory on day one. Substitution + vision-only keep the demo alive.

## Team · LatentX (IIIT-B)

Anish Reddy · P Lohith · Harsha Vardhan D · P Prajwal · Munjikesh · Hasini
