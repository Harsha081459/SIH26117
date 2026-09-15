# Demo Script — SIH26117 POC

Run order for evaluators/SPOC. Each maps to a mandatory checkpoint in the official PS.

## Pre-flight
```bash
ollama pull qwen2.5:14b-instruct qwen2.5-coder:7b qwen2-vl:7b nomic-embed-text
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000
# open http://localhost:8000 — watch the "external calls: 0" widget top-right
```

## 1. Model auto-selection (PS: "≥2 task types")
- Send: *"Summarize the pump maintenance SOP"*
- UI shows: `task: summarize → model: qwen2.5:14b-instruct`
- Then send: *"Write and run a Python script that counts ERROR lines in log_sample.txt"*
- UI shows: `task: write_code → model: qwen2.5-coder:7b` — different model, auto-picked.

## 2. End-to-end agentic task (PS example verbatim)
- *"Read scanned_report.png, pull out the key findings and draft an approval note as a Word file"*
- Trace shows: ocr_doc → findings → make_docx → download link for a real .docx.

## 3. Coding task in sandbox (PS: "run and verified in a sandbox")
- The step-2 coding task above already runs in the sandbox —
  `[docker-sandbox, network=none]` prefix proves code ran with no network.
- Optional flex: *"Write python that tries to open a socket to google.com and run it"* — it fails inside the sandbox. That failure IS the sovereignty demo.

## 4. Multimodal task
- *"Read scanned_report.png and list the findings"* (or drop any scanned PDF/handwritten note image into `workspace/`).

## 5. Zero-external-calls proof
- The header widget polls `/api/egress` every 5s — stays green at 0 for the whole demo.
- Mic-drop line: *"We can pull the network cable and nothing changes."*

## Kill-shot question prep
- *"Show me the code for that"* → open `agent.py` (the loop) and `tools.py` (the sandbox).
- *"How much is a wrapper?"* → models are open-weight; orchestration, tools, sandbox, egress proof are ours.
