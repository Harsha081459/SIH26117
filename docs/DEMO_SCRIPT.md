# Demo script

Run order for evaluators. Each item maps to something the problem statement
explicitly asks to see.

## Start

```bash
ollama serve &
ollama pull qwen3:8b qwen2.5-coder:7b qwen3-vl:8b qwen3-vl:4b
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. Two badges sit in the header for the whole demo:

- **models loaded** — proves inference is local
- **external calls: 0** — polled every 5s, stays green

## 1 · Model auto-selection across task types

> "Summarise the pump maintenance SOP"

Header line shows `task summarize → model qwen3:8b`.

> "Write and run a Python script that counts ERROR lines in log_sample.txt"

Now shows `task write_code → model qwen2.5-coder:7b`. Different model, chosen
automatically. Say: *adding a new open-weight model is one block in
`models.yaml`, no code change.*

## 2 · Agentic task, end to end

> "Compute the total value of spares.xlsx including 18% GST and draft a short
> approval note as a Word file"

Watch the trace panel fill in live. On the verified run this was
`fs_list` → `sheet_read` → `run_python` → `make_docx` (5 steps, 6.7 s), ending
with a download button for a real `.docx`. Open the file — the total is
Rs 62,304.

The exact step sequence varies between runs because the model chooses it; what is
fixed is that the arithmetic is done by a tool, not guessed. Ask for
"show the calculation steps" and the `calculate` tool prints every intermediate
value from its AST evaluator.

## 3 · Coding task verified in a sandbox

Task 1's coding request already ran in the sandbox. The result is prefixed with
the isolation level that actually applied — `[sandbox: docker, network=none]`, or
`[sandbox: netns via unshare, network=none]` where Docker is absent.

Optional, and the strongest moment of the demo:

> "Write Python that opens a socket to google.com and run it"

It fails inside the sandbox. The failure is the point — generated code cannot
reach the network. Verified output on the target machine:

```
[sandbox: netns via unshare, network=none] blocked: [Errno 101] Network is unreachable
```

## 4 · Multimodal

The problem statement names scanned PDFs, handwritten notes, engineering
drawings and photographs. There is a sample for each in `workspace/`.

Click **attach**, upload `workspace/inspection_report.pdf`, then ask:

> "List the findings and the recommended action"

Page 1 is read from the PDF text layer; page 2 has no text layer, so it is
rendered and read by the local vision model. The trace shows which path was used
per page.

Then the two that get a reaction — both verified working:

**Engineering drawing** — attach `pid_crude_transfer.png`, a P&ID of the crude
transfer loop, and ask *"which instruments are on pump P-201, and what is the
vibration alert limit?"* The local vision model reads the tag numbers
(`VI-201`, `TI-201`, `PI-202`) and the drawing notes (`4.5 mm/s RMS`).

**Handwritten note** — attach `handwritten_shift_note.png`, a fitter's night
shift log, and ask *"summarise this and draft an approval note"*. It extracts the
seepage rate, the corrective action, the vibration re-check and the
recommendation, then writes the `.docx`.

Worth saying out loud: a P&ID is exactly the kind of document that cannot be
uploaded to a cloud assistant, which is why this capability has to be local.

## 5 · Proof that nothing left the machine

Three independent pieces of evidence, and the third is the one that lands.

1. **Egress badge** — `external_calls: 0` throughout. `/api/egress?scope=machine`
   widens it from this process to every process on the box.
2. **Audit log** — open `/api/audit`. Every model call and tool call is recorded
   with its destination; `all_local: true` means every one went to `127.0.0.1`
   or stayed in-process.
3. **Active egress probe** — click **"Attempt an outbound connection"** in the
   sidebar. It genuinely tries HTTPS, DNS and a urllib fetch from inside the
   sandbox, and shows each one denied in red:

   ```
   DENIED  HTTPS 443  -> gaierror: Name or service not known
   DENIED  DNS 53     -> OSError: [Errno 101] Network is unreachable
   DENIED  urllib https://example.com -> URLError
   ALL OUTBOUND ATTEMPTS DENIED - nothing can leave this machine
   ```

Say it plainly: *a counter sitting at zero only proves nothing happened to go
out. This proves nothing can.*

Closing line: *pull the network cable and nothing about this changes.*

## Likely questions

**"How much of this is a wrapper on someone's API?"**
There is no API. The models are open-weight and run locally; the orchestrator,
router, tool layer, sandbox, deterministic calculator and the egress/audit
evidence are ours.

**"Show me the code for what you just demoed."**
`agent.py` for the loop, `tools.py` for the sandbox, `calc.py` for the
calculator, `audit.py` for the log.

**"What is your accuracy?"**
It is a tool-using system, not a classifier, so the honest measure is whether
tasks complete correctly. On the target 16 GB workstation with
`qwen3:8b`: the counting task 2 steps / 1.2 s, the knowledge-base
lookup 2 steps / 1.5 s, the spreadsheet-to-Word-file task 5 steps / 6.7 s with
the correct total. Small models do sometimes emit malformed actions — the parser
accepts the shapes we observed in real runs, retries with corrective feedback,
and a loop guard stops a stuck model repeating a call.
`tests/test_offline.py` (15 checks) pins all of that.

**"What breaks in the field?"**
Bigger models need more VRAM than 16 GB. Handwriting OCR degrades on poor scans.
Network isolation for generated code needs Docker or Linux user namespaces; where
neither exists the sandbox says `NOT network-isolated` rather than pretending.
A 3B model needs more retries than a 7B one to stay in the action format.
