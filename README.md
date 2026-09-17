# Sovereign On-Premise Agentic AI Workbench

**SIH 2026 · Problem Statement `SIH26117`**
Sovereign On-Premise Agentic AI Workbench using Open-Weight Multimodal LLMs for Confidential Industrial Work
**Organisation:** Mangalore Refinery & Petrochemicals Ltd (MRPL) · **Theme:** Smart Automation
**Team:** LatentX, IIIT Bangalore

---

## The problem

Refineries, PSUs and government offices produce a large amount of routine but
sensitive knowledge work: approval notes, engineering calculations, inspection
report reviews, internal tooling code. None of it can go through a cloud AI
assistant, because the underlying material is confidential — P&IDs, financials,
vendor negotiations, internal correspondence. So the work is either done by hand,
or confidential material quietly gets pasted into public tools anyway.

## What this is

A self-hosted AI workbench that runs entirely on the organisation's own GPU
machine. It plans multi-step work, calls local tools, reads scans and drawings,
and produces real files — and it can demonstrate that nothing ever left the
machine.

## Mapping to the problem statement

| Requirement in the PS | Where it lives | Status |
|---|---|---|
| Multiple open-weight models, automatic selection per task | `router.py`, `models.yaml` | verified — coding, document and vision tasks route to different models; an image task uses the vision model to read and the reasoning model to orchestrate |
| New models addable without redesign | `models.yaml` (config only) + `router._resolve()` | verified — unlisted/absent models substitute automatically |
| Agent that plans multi-step work and iterates | `agent.py` (plan → act → observe) | verified — 4-step chain on real model |
| Local tools: file read/write, sandboxed code, spreadsheets, doc search | `tools.py` (12 tools) | verified |
| Multimodal: scanned PDFs, handwriting, drawings, photos | `ocr_doc`, `pdf_read` (text layer → vision fallback per page) | verified on all four: scan, mixed PDF, P&ID, handwritten note |
| Real deliverables (Word/Excel/PPT), not chat replies | `make_docx`, `make_xlsx`, `make_pptx` | verified — downloadable files |
| Calculations with steps shown | `calc.py` (AST evaluator, no `eval`) | verified |
| Grounded in local manuals, SOPs, correspondence | `kb_search` over `corpus/`, hybrid local-embedding + keyword ranking | verified — correct document for direct and paraphrased questions |
| **Proof that no external calls are made** | `/api/egress` monitor, `/api/egress/probe` active test, `audit.jsonl` log | verified — probe reports all outbound attempts DENIED |

## Architecture

```
      Browser UI  (chat · live execution trace · egress + audit badges)
           |  SSE
      FastAPI  ──  Router ─ picks model per task from models.yaml
           |        Agent  ─ plan → act → observe, max 10 steps
           |        Tools  ─ fs · sandbox · ocr · pdf · kb · sheets · calc · docx/xlsx/pptx
           |
      Ollama on 127.0.0.1  (qwen2.5 · qwen2.5-coder · qwen2.5vl)
           |
      Audit log (every model + tool call, with destination)
      Egress monitor (outbound connections outside the premises)
```

No component reaches the internet. Model-written code runs with its network
removed, so even a hostile snippet cannot phone home. Three levels are attempted
in order and the response states which one ran:

1. Docker container started with `--network none`
2. a Linux network namespace via `unshare -rn` — no root required
3. a plain subprocess with a timeout, labelled `NOT network-isolated`

## Verified run

Measured on the deployment target: a workstation with an RTX 4060 Ti (16 GB),
`qwen2.5:7b-instruct` + `qwen2.5vl:3b` served locally by Ollama.

```
TASK: How many ERROR lines are in log_sample.txt? Count them with the sandbox.
  routed: analyze -> qwen2.5:7b-instruct (keyword rules)
  step 1  run_python {"code": "print(sum(1 for l in open('log_sample.txt') if 'ERROR' in l))"}
       -> [sandbox: netns via unshare, network=none] 3
  ANSWER: log_sample.txt contains 3 ERROR lines.                    (2 steps, 1.2 s)

TASK: Search the knowledge base for the vibration alert limit for pump P-201.
  routed: analyze -> qwen2.5:7b-instruct (model-based classification)
  step 1  kb_search {"query": "vibration alert limit for pump P-201"}
       -> [kb] sop_pump_maintenance.txt ...
  ANSWER: 4.5 mm/s RMS, per SOP-MECH-041.                           (2 steps, 1.5 s)

TASK: Compute the total value of spares.xlsx including 18% GST, then draft a
      short approval note as a Word file.
  routed: draft_document -> qwen2.5:7b-instruct (keyword rules)
  step 1  fs_list {}          -> inspection_report.pdf, log_sample.txt, ...
  step 2  sheet_read {"path": "spares.xlsx"} -> item,qty,unit_rate | ...
  step 3  run_python {...}    -> [sandbox: netns, network=none] 62304.0
  step 4  make_docx {"title": "Approval Note for Spares Purchase", ...}
       -> FILE:Approval Note for Spares Purchase.docx
  ANSWER: Approval note created; total including 18% GST is 62304 INR.
                                                                    (5 steps, 6.7 s)

MULTIMODAL - scanned_report.png read by the local vision model:
  Bearing temp (DE): 74.8 C - near limit 75 C
  Mechanical seal leakage: 14 drops/min - EXCEEDS 10/min limit
  Vibration post seal change: 5.2 mm/s - resolved to 2.8 mm/s   ... (all findings)

MULTIMODAL - inspection_report.pdf (mixed text + scanned pages):
  [page 1 - text layer]            MRPL INSPECTION SUMMARY - CDU-2 ...
  [page 2 - scanned, read by vision model]
                                   Observed: minor oil seepage near NDE bearing ...

MULTIMODAL - pid_crude_transfer.png (P&ID engineering drawing):
  TK-101 CRUDE STORAGE | P-201 CENTRIFUGAL PUMP | V-101 | VI 201 | TI 201
  PI 202 | NRV-201 | FCV-203 | E-301 FEED PREHEATER | TO COLUMN C-401
  1. P-201 SEAL FLUSH PER SOP-MECH-041; VIBRATION ALERT LIMIT 4.5 mm/s RMS.
  4. LINE SIZE 6 IN SCH 40; DESIGN PRESSURE 19 barg; DESIGN TEMP 120 C.

MULTIMODAL - handwritten_shift_note.png (fitter's night shift log):
  Date: 12-Sep-2026  Time: 22:40 hrs | Pump: P-201 | Oil Seepage: ~3 drops/min
  Action Taken: wiped and tightened gland nut by 1/4 turn
  Vibration: 2.8 mm/s (limit 4.5) - within range | Bearing temp: 71 C stable
  Recommendation: monitor each shift for 3 days | Signer: R. Kamath, Fitter Gr-I

EGRESS PROBE - deliberately attempts to leave the machine:
  DENIED  HTTPS 443  -> gaierror: [Errno -2] Name or service not known
  DENIED  DNS 53     -> OSError: [Errno 101] Network is unreachable
  DENIED  urllib https://example.com -> URLError
  ALL OUTBOUND ATTEMPTS DENIED - nothing can leave this machine

/api/egress  {"external_calls": 0, "scope": "machine", "detail": []}
/api/audit   {"destinations": {"local-process": 15, "127.0.0.1:11434": 21},
              "external_destinations": {}, "all_local": true}
```

Reproduce with `python tests/test_offline.py` for the offline checks; the run
above is produced by driving `/api/chat` and `/api/chat/stream` on the box.

## Running it

Requires Python 3.8+. No root needed — the setup script installs the Ollama
runtime into `$HOME`.

```bash
bash scripts/setup.sh        # runtime + models + deps + isolation check + tests
cd backend && ../.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

Manual equivalent:

```bash
pip install -r requirements.txt
ollama serve &
ollama pull qwen2.5:7b-instruct     # reasoning, drafting, orchestration
ollama pull qwen2.5-coder:7b        # coding tasks
ollama pull qwen2.5vl:3b            # scans, drawings, handwriting
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000
```

Any subset of models works: a configured model that is not loaded is
substituted with the closest one available, which is why the same code runs on a
laptop and on a GPU server.

## Tests

```bash
python tests/test_offline.py     # 19 checks, no model runtime needed
```

Covers model routing, calculator safety (rejects `__import__`, `open`, division
by zero), sandbox timeouts and isolation labelling, workspace path-escape
attempts, deliverable generation, audit completeness, the loop guard that stops a
stuck model repeating itself, and the malformed-action shapes local models
actually emit — including tool names placed in `"action"` and single-quoted
values that are valid Python but invalid JSON.

## Layout

```
backend/    main.py · agent.py · router.py · tools.py · calc.py · audit.py
            ollama_client.py · models.yaml
frontend/   index.html          chat, live trace, egress probe, badges
corpus/     internal SOPs and correspondence for the knowledge base
workspace/  documents the agent can read:
            log_sample.txt · spares.xlsx · scanned_report.png
            inspection_report.pdf (text + scanned pages)
            pid_crude_transfer.png (P&ID) · handwritten_shift_note.png
scripts/    setup.sh            runtime + models + deps, no root
docs/       DEMO_SCRIPT.md      run order for evaluators
            ARCHITECTURE.md     design decisions and why
tests/      test_offline.py     19 checks
logs/       audit.jsonl         created at runtime
```

The workspace documents are synthetic, drawn for this demonstration — the
problem statement expects open/sample material, so nothing proprietary is
included.

## Known limits

- Small models occasionally emit malformed actions. The parser accepts the shapes
  we observed in real runs and retries with corrective feedback, but a 3B model
  needs more retries than a 7B one.
- Network isolation for generated code needs Docker or Linux user namespaces.
  Where neither exists the sandbox degrades to a timed subprocess and says so
  explicitly rather than implying isolation it does not have.
- `kb_search` ranks with local embeddings and keyword frequency combined; with no
  embedding model loaded it degrades to keyword-only and reports which mode ran.
- `scope=machine` on the egress endpoint reports every process on the host, so on
  a shared machine it will also count other users' connections; the dedicated
  deployment is the intended target for that mode.

## Team

Anish Reddy · P Lohith · Harsha Vardhan D · P Prajwal · Munjikesh · Hasini
