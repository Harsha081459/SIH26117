# Sovereign On-Premise Agentic AI Workbench — POC

**SIH 2026 · Problem Statement `SIH26117`** — Sovereign On-Premise Agentic AI Workbench using Open-Weight Multimodal LLMs for Confidential Industrial Work
**Organization:** Mangalore Refinery & Petrochemicals Ltd (MRPL) · **Theme:** Smart Automation
**Team:** LatentX · IIIT-B

---

## The problem

Refineries, PSUs and government offices handle confidential knowledge work — engineering drawings, financials, vendor negotiations, inspection reports — that **cannot** go through cloud AI assistants. Today employees either work manually (slow) or quietly leak confidential data into public AI tools (unsafe). No deployable on-premise alternative exists.

## Our solution (POC)

A **fully air-gapped agentic AI workbench** running on a single on-premise GPU machine. Nothing ever leaves the premises — and we *prove* it, not just claim it.

### What this POC demonstrates (mapped to the PS's own acceptance criteria)

| PS requirement | Implemented in this POC |
|---|---|
| Multi-model backend, auto model selection | `router.py` + `models.yaml` — tasks routed to qwen2.5:14b (docs), qwen2.5-coder:7b (code), qwen2-vl:7b (vision); new models = one YAML row |
| Agentic multi-step work | `agent.py` — plan → act → observe loop, up to 8 steps, full execution trace |
| Local tools | `tools.py` — fs_read/write, run_python (sandbox), ocr_doc (VL model), kb_search, make_docx |
| Real deliverables | `.docx` output via python-docx (xlsx/pptx in pipeline) |
| Scanned docs / handwriting / drawings | `ocr_doc` routes images to the local vision-language model |
| Local knowledge base | `kb_search` over `corpus/` (SOPs, manuals, past correspondence) — upgrade path: ChromaDB + local embeddings |
| **Proof of zero external calls** | `/api/egress` endpoint + live dashboard widget counting outbound connections — shows **0** |

## Architecture

```
Browser UI ──► FastAPI ──► Orchestrator (router → planner → executor)
                              │
            ┌─────────────────┼──────────────────┐
            ▼                 ▼                  ▼
      Tool registry      Ollama (local)     Egress watchdog
      fs / sandbox /     qwen2.5:14b        outbound conns = 0
      ocr / kb / docx    qwen2.5-coder:7b
                         qwen2-vl:7b
```

## Tech stack

FastAPI · Ollama (open-weight models, local inference) · python-docx/openpyxl/pptx · psutil (egress monitor) · single-page vanilla JS UI

## Run it

```bash
pip install -r requirements.txt
ollama pull qwen2.5:14b-instruct qwen2.5-coder:7b qwen2-vl:7b   # any 2 suffice for routing demo
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

## Demo tasks (try in the UI)

1. **Agentic + deliverable:** *"Summarize the pump SOP and draft an approval note as a docx"* → kb_search → summarize → make_docx → downloadable Word file
2. **Coding + sandbox:** *"Write and run a Python script that counts ERROR lines in log_sample.txt"* → routes to coder model → executes → shows output
3. **Multimodal:** *"Read scanned_report.png and list the findings"* → VL model OCRs the scanned inspection report

## Roadmap to full solution

- [x] Docker `--network none` sandbox (auto-falls back to subprocess when Docker absent)
- [x] Semantic KB via local embeddings (nomic-embed-text, cached) + keyword fallback
- [x] xlsx/pptx generators + spreadsheet read tool
- [ ] iptables-based egress logging (kernel-level proof on the dedicated box)
- [ ] Streaming responses + file upload in UI
- [ ] More corpus coverage + per-model load balancing

## Team — LatentX (IIIT-B)

Anish Reddy · P Lohith · Harsha Vardhan D · P Prajwal · Munjikesh · Hasini
