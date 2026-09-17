# Architecture

How the workbench is put together, and why each piece is the way it is.

---

## Constraint that drives everything

The material this system handles cannot leave the premises. That single
constraint rules out the usual design — a thin client calling a hosted model —
and forces every capability to be local: inference, OCR, document search, code
execution, file generation. It also means a *claim* of locality is not enough;
the system has to be able to show it.

---

## Request path

```
 Browser (single page)
    |
    |  POST /api/chat/stream          server-sent events
    v
 FastAPI  (backend/main.py)
    |
    +--> Router      (router.py)   which model should handle this?
    |
    +--> Agent loop  (agent.py)    plan -> act -> observe, repeat
    |        |
    |        +--> Tool registry (tools.py)   13 local tools
    |        |         fs_list / fs_read / fs_write
    |        |         run_python        network-isolated sandbox
    |        |         ocr_doc           local vision model
    |        |         pdf_read          text layer, else vision per page
    |        |         kb_search         local embeddings over corpus/
    |        |         sheet_read
    |        |         calculate         AST evaluator, shows every step
    |        |         egress_probe      actively proves isolation
    |        |         make_docx / make_xlsx / make_pptx
    |        |
    |        +--> Ollama client (ollama_client.py) --> 127.0.0.1:11434
    |
    +--> Audit log   (audit.py)    every model + tool call, with destination
    +--> Egress      (main.py)     outbound connections, process or machine
```

---

## Model routing

Two decisions are made per request, and they are not the same decision.

**Which model handles the task.** `router.py` classifies first with keyword
rules, which are instant and work with no model loaded. Only if the rules
abstain does it ask the default model to label the request. The label is looked
up in `models.yaml` to get a model id.

**Which model drives the loop.** A vision-language model is the right tool to
read a scan and a poor one to plan a five-step job — in testing, a VL model
produced the requested document and then kept calling tools aimlessly. So for
image tasks the vision model stays responsible for reading (reached through the
`ocr_doc` tool) while the reasoning model orchestrates. Both are reported in the
response, so the interface shows two models cooperating on one request.

**Models are configuration, not code.** `models.yaml` maps task labels to model
ids. Adding a newly released open-weight model is a new block in that file.
Nothing imports a model name.

**Absent models are substituted.** The machine that runs the demo may not hold
the same models as the machine it was built on, so `router._resolve()` maps a
configured id onto the closest one the runtime actually has — same family first,
then the default, then whatever is loaded. This is why the same checkout runs on
a 16 GB workstation and on a larger GPU server.

---

## The agent loop

Each turn the model returns one action: a tool call or a final answer. The result
of a tool call is appended to a running scratchpad and fed back in. The loop ends
on a final action, on the step limit, or when the model stops producing usable
actions.

Three things make this survive small local models:

**A tolerant action parser.** Models drift from the requested format in
predictable ways. Observed in real runs and handled: the tool name placed in
`"action"` instead of `"tool"`; arguments under `arguments` / `parameters` /
inlined beside the tool name; and values quoted with apostrophes, which is valid
Python and invalid JSON. The parser extracts the first balanced object, tries
`json.loads`, then `ast.literal_eval`.

**Corrective retries.** A reply that cannot be parsed is echoed back to the model
with the list of valid tool names and the two acceptable shapes.

**A loop guard.** A model that repeats a call with byte-identical arguments is
stuck. Re-running it burns the step budget and returns the same failure, so the
second attempt is answered with a redirection instead of being executed. This
turned a run that wandered for nine steps and produced a wrong total into a
correct five-step run.

Tool errors are written to be actionable rather than merely correct. `calculate`
refusing an expression says what it does accept and gives an example, because the
caller is a model that can correct itself if told how.

---

## Sandboxing generated code

Model-written code runs with the network taken away. Three levels are attempted
in order, and **the label in the output names the level that actually ran**:

| Level | Mechanism | Requires |
|---|---|---|
| 1 | `docker run --network none`, memory and CPU capped | Docker |
| 2 | `unshare -rn` — empty network namespace | Linux user namespaces, no root |
| 3 | subprocess with timeout, labelled `NOT network-isolated` | nothing |

Level 2 matters in practice: it gives genuine isolation on a machine where you
cannot install Docker. Verified — code attempting an outbound connection gets
`[Errno 101] Network is unreachable`.

Reporting the level honestly is deliberate. A system that claims isolation it
does not have is worse than one that says so.

---

## Evidence of locality

Three independent mechanisms, because the claim is the whole point of the
problem statement.

**Egress monitor** (`/api/egress`) counts established connections to addresses
outside private ranges. `scope=process` covers this backend and its children;
`scope=machine` covers every process on the host.

**Audit log** (`audit.py`) appends one JSON line per model call and per tool call,
each recording its destination. `summary()` aggregates destinations and reports
`all_local`. Inference is always `127.0.0.1:11434`; tools are `local-process`.

**Active probe** (`/api/egress/probe`) is the one that convinces. It deliberately
attempts HTTPS, DNS and a urllib fetch from inside the sandbox and reports each
denial. A counter reading zero shows nothing happened to go out; the probe shows
nothing can.

---

## Knowledge grounding

`corpus/` holds the organisation's own material — SOPs, safety procedures, past
correspondence. `kb_search` embeds each document with a local embedding model,
caches the vectors, and ranks by cosine similarity. If no embedding model is
loaded it falls back to keyword scoring, so the tool never hard-fails; the
response says which mode ran.

---

## Deliverables

Chat replies are not the product. `make_docx` produces a structured approval note
— summary, findings, recommendation, reference, sign-off block — and stamps
provenance into the document itself: generated on-premise, timestamp, model used.
A reviewer opening the file later can tell where it came from. `make_xlsx` and
`make_pptx` cover spreadsheets and decks, and accept both text and list inputs
because models supply either.

Arithmetic never goes through the model. `calc.py` walks a Python AST with a
whitelist of operators and functions — no `eval` — and records every intermediate
step, so an engineer can audit the number rather than trust it.

---

## What is deliberately not here

Authentication, multi-user isolation, conversation memory across requests, model
lifecycle management, and horizontal scaling. All are needed for a production
deployment inside a refinery; none are needed to demonstrate that the sovereign
agentic workbench is buildable, and each would have cost time better spent making
the core path work on real hardware.
