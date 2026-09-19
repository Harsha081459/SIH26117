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
    |        +--> Tool registry (tools.py)   15 local tools; every call runs
    |        |                               inside a per-request file scope
    |        |         fs_list / fs_read / fs_write   scope-enforced
    |        |         run_python        docker / bubblewrap, no host fallback
    |        |         ocr_doc           vision model + optional PaddleOCR
    |        |         pdf_read          text layer, else OCR per page
    |        |         office_read       DOCX / PPTX attachments
    |        |         kb_search         local embeddings over corpus/
    |        |         sheet_read        preserves 0 / False / quoted commas
    |        |         calculate         AST evaluator, shows every step
    |        |         egress_probe      blocked / reachable / inconclusive
    |        |         make_docx / make_xlsx / make_pptx / compose_deck
    |        |                           verified on disk before success
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

**Absent models are substituted — within capabilities.** The machine that runs
the demo may not hold the same models as the machine it was built on, so
`router._resolve()` maps a configured id onto a *registered, compatible* model
the runtime actually has. The `capabilities` field in `models.yaml` is the
gate: a vision request never resolves to a text-only or embedding model, and
when nothing compatible is installed the request fails with an explicit error
rather than a silent downgrade. Runtime unavailable, registry mismatch and
no-compatible-model are reported as three distinct errors.

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

## Request scoping and verified success

Two integrity rules run underneath the loop — both enforced in code, not in the
system prompt.

**Per-request file scope.** Every tool call executes inside a scope resolved
from the request: the attached file plus workspace files the user explicitly
named. Reads outside that scope fail in `_safe_workspace`, so a file merely
*mentioned inside* a document's text never becomes readable, `fs_list` shows
only authorised files unless the request asks for a listing, knowledge-base
search runs only when the request calls for it, and sandbox staging copies only
authorised inputs. The scope is a context variable, restored when the request
ends — one request cannot widen another's access.

**Verified artifacts.** A final answer that claims a deliverable is accepted
only after the file verifies: it exists inside `outputs/`, is not a symlink, is
within size limits, and parses as its claimed format — DOCX / XLSX / PPTX are
opened and checked, decks against the requested slide count. Artifacts are
re-verified immediately before the success response, valid partial results are
reported as `partial` rather than success, and a forged `FILE:` line cannot
adopt an older request's output.

---

## Sandboxing generated code

Model-written code runs with the network taken away and the filesystem
restricted. Two isolation tiers are attempted in order, and **the label in the
output names the tier that actually ran**:

| Tier | Mechanism | Requires |
|---|---|---|
| 1 | `docker run --network none`, memory/CPU/pids capped, container removed after | Docker |
| 2 | `bubblewrap` — restricted mounts, no network access | Linux user namespaces, no root |

There is deliberately no third tier. An earlier design fell back to a plain
subprocess labelled "not network-isolated"; that was removed because a labelled
non-sandbox is still a non-sandbox. If no tier exists, `run_python` returns an
explicit error and the code does not execute. Resource limits cover CPU,
memory, process count, file size, open files, output size and wall time;
timeouts kill the whole process group or container; only files inside the
request's scope are staged in.

Historical note: an earlier build used `unshare -rn` (an empty network
namespace). That isolates networking but not the filesystem, which is why it
was replaced — verified at the time by a blocked socket (`Errno 101`), but it
was never filesystem isolation.

---

## Evidence of locality

Three independent mechanisms, because the claim is the whole point of the
problem statement.

**Egress monitor** (`/api/egress`) counts established connections to addresses
outside private ranges. `scope=process` covers this backend and its children;
`scope=machine` covers every process on the host.

**Audit log** (`audit.py`) appends one JSON line per model call and per tool
call, recording destination, timing and content *metadata* — byte counts and
SHA-256 digests — never raw prompts, documents or results; fields that look
like credentials are redacted. `summary()` aggregates destinations and reports
`all_local`. Inference is always `127.0.0.1:11434`; tools are `local-process`.
The log covers instrumented application calls only — it is evidence for the
demo, not machine-wide monitoring.

**Active probe** (`/api/egress/probe`) is the one that convinces. It deliberately
attempts outbound connections from inside the sandbox and reports each row as
blocked, reachable or inconclusive. An inconclusive row means the probe could
not complete — it is never reported as a denial. A counter reading zero shows
nothing happened to go out; a completed, denied probe shows the sandboxed code
could not.

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
