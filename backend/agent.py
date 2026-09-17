"""Plan -> act -> observe loop.

Each turn the model emits one action: either a tool call or a final answer.
Results are fed back in and the loop continues, which is what lets a single
request read a scanned report, do arithmetic and write a Word file rather
than answering once and stopping.

Everything the loop does is recorded in the trace and in the audit log so a
reviewer can see which model ran, which tools fired and with what arguments.
"""
import ast
import json
import time

import audit
from ollama_client import generate
from tools import TOOLS, TOOL_SPEC, call

MAX_STEPS = 10
MAX_PARSE_RETRIES = 2

SYSTEM = """You are an air-gapped industrial AI workbench for a refinery. You run
entirely on local hardware and have no internet access. You may use ONLY these
local tools:

{tools}

Reply with EXACTLY ONE JSON object and nothing else. No prose, no markdown.

To use a tool:
{{"action": "tool", "tool": "TOOL_NAME", "args": {{"ARG": "VALUE"}}}}

When the work is complete:
{{"action": "final", "answer": "the complete answer for the user"}}

Worked example
User task: how many ERROR lines are in log_sample.txt?
Your 1st reply: {{"action": "tool", "tool": "run_python", "args": {{"code": "print(sum(1 for l in open('log_sample.txt') if 'ERROR' in l))"}}}}
(tool returns: 3)
Your 2nd reply: {{"action": "final", "answer": "log_sample.txt contains 3 ERROR lines."}}

Grounding -- read this before choosing any tool
- Do NOT call ocr_doc, pdf_read, fs_read or sheet_read unless the user attached
  that file or named it in their request. Opening an unrelated document pulls
  confidential material into a task it has nothing to do with. This is the single
  most important rule here.
- If the user's request names no file and needs none, do not go looking for one.
- If the request does need source material and you were given none, say so with the
  "final" action and state what you would need. That is a correct answer.
- Never let content from one document appear in a deliverable about something else.

Rules
- Use the results you have already been given. If a tool has shown you the numbers
  or text you need, work from those values -- do not re-read the same file.
- calculate takes literal numbers only, e.g. "(2*18400 + 5*3200) * 1.18". Substitute
  the actual values you have seen. It cannot read files or run code.
- run_python is for counting, parsing or processing a file. Never use it merely to
  print text you already wrote -- that accomplishes nothing.
- Keep the final answer short: state what you did and where the result is. Do not
  paste the whole document back to the user.
- kb_search searches internal SOPs, manuals and correspondence.
- ocr_doc reads images/scans/drawings; pdf_read reads PDFs; sheet_read reads .xlsx.
- Use fs_list only to confirm the name of a file the user referred to, never to go
  looking for something to write about.
- If the user wants a document, note, report, spreadsheet or deck AND you have the
  source material for it, finish by calling make_docx / make_xlsx / make_pptx so a
  real file is produced. Do not produce a file out of unrelated content.
- Never repeat a tool call that already succeeded; use the result you were given.
- After you have what you need, reply with the "final" action.
- This is an Indian refinery: amounts are in rupees (Rs / INR), never dollars.
"""


def _system_prompt():
    return SYSTEM.format(tools=TOOL_SPEC)


def _loads(blob):
    """Parse an object that is usually JSON but not always.

    Local models frequently quote values with apostrophes -- valid Python,
    invalid JSON -- so a literal_eval pass catches those instead of throwing the
    whole action away.
    """
    try:
        return json.loads(blob)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        obj = ast.literal_eval(blob)
        return obj if isinstance(obj, dict) else None
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None


def _extract_json(text):
    """Pull the first balanced object out of a model response."""
    start = text.find("{")
    while start != -1:
        depth, quote, esc = 0, None, False
        for i in range(start, len(text)):
            ch = text[i]
            if quote:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == quote:
                    quote = None
                continue
            if ch in "\"'":
                quote = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    obj = _loads(text[start:i + 1])
                    if obj is not None:
                        return obj
                    break
        start = text.find("{", start + 1)
    return None


def _parse_action(text):
    """Accept the several shapes small models actually emit, not just the ideal one."""
    obj = _extract_json(text)
    if not isinstance(obj, dict):
        return None

    action = obj.get("action")

    # Ideal shape.
    if action == "final":
        answer = obj.get("answer") or obj.get("text") or obj.get("response") or ""
        return {"action": "final", "answer": answer}
    if action == "tool":
        tool = obj.get("tool") or obj.get("name") or obj.get("tool_name")
        if tool:
            return {"action": "tool", "tool": tool, "args": _args_of(obj)}

    # Common drift: the tool name is put straight into "action".
    if isinstance(action, str) and action in TOOLS:
        return {"action": "tool", "tool": action, "args": _args_of(obj)}

    # Or there is no "action" key at all, just a tool name.
    for key in ("tool", "name", "tool_name", "function"):
        cand = obj.get(key)
        if isinstance(cand, str) and cand in TOOLS:
            return {"action": "tool", "tool": cand, "args": _args_of(obj)}

    # A bare {"answer": "..."} means it is done.
    if "answer" in obj and not action:
        return {"action": "final", "answer": obj["answer"]}
    return None


def _args_of(obj):
    """Arguments may be nested under args/arguments/parameters, or inlined."""
    for key in ("args", "arguments", "parameters", "params", "input"):
        val = obj.get(key)
        if isinstance(val, dict):
            return val
    inline = {k: v for k, v in obj.items()
              if k not in ("action", "tool", "name", "tool_name", "function",
                           "args", "arguments", "parameters", "params", "input")}
    return inline


def iter_run(task, model, max_steps=MAX_STEPS):
    """Execute a task, yielding each event as it happens.

    Yields dicts: {"type": "step", ...} for every tool call and
    {"type": "final", ...} once, carrying the completed result. Streaming these
    lets the interface show the agent working instead of sitting silent while a
    local model thinks.
    """
    trace, scratch, files = [], "", []
    seen, tool_uses = {}, {}
    t0 = time.time()
    audit.record("session", event="task_start", model=model, task=task[:300])

    step = 0
    while step < max_steps:
        step += 1
        prompt = "{}\n\nUSER TASK: {}\n\nWORK SO FAR:\n{}\n\nNext action (single JSON object):".format(
            _system_prompt(), task, scratch or "(nothing yet)")

        act, raw = None, ""
        for attempt in range(MAX_PARSE_RETRIES + 1):
            attempt_prompt = prompt
            if attempt:
                attempt_prompt += (
                    "\n\nYour previous reply could not be used:\n"
                    + raw.strip()[:400]
                    + "\n\nReply with ONE JSON object in exactly this form:\n"
                      '{"action": "tool", "tool": "<one of: '
                    + ", ".join(TOOLS) + '>", "args": {...}}\n'
                      'or {"action": "final", "answer": "..."}')
            raw = generate(model, attempt_prompt)
            act = _parse_action(raw)
            if act:
                break
        if not act:
            # The model will not emit structured actions; surface its prose so the
            # user still gets an answer instead of an error.
            text = raw.strip() or "The local model did not return a usable response."
            trace.append({"step": step, "type": "final", "text": text[:4000]})
            yield _final_event(text, trace, files, step, t0)
            return

        if act["action"] == "final":
            answer = str(act.get("answer", "")).strip()
            trace.append({"step": step, "type": "final", "text": answer})
            yield _final_event(answer, trace, files, step, t0)
            return

        tool = act.get("tool", "")
        args = act.get("args") or {}
        if not isinstance(args, dict):
            args = {}

        # A model that repeats a call verbatim is stuck. Re-running it wastes the
        # step budget and produces the same failure, so answer with a redirection
        # instead of executing again.
        key = (tool, json.dumps(args, sort_keys=True, default=str))
        seen[key] = seen.get(key, 0) + 1
        tool_uses[tool] = tool_uses.get(tool, 0) + 1

        if seen[key] > 1:
            result = ("ERROR: {}() was already called with these exact arguments and "
                      "returned the result above. Do not repeat it. Either use that "
                      "result to continue, try a different tool, or finish with "
                      '{{"action": "final", ...}}.'.format(tool))
        elif tool_uses[tool] > 3:
            # Same tool, slightly different arguments, over and over: the model is
            # iterating on wording rather than making progress.
            result = ("ERROR: {}() has now been called {} times. Stop calling it and "
                      "finish with {{\"action\": \"final\", \"answer\": \"...\"}} using "
                      "what you already have.".format(tool, tool_uses[tool]))
        else:
            result = call(tool, args)

        entry = {"step": step, "type": "tool", "tool": tool, "args": args,
                 "result": result[:600]}
        trace.append(entry)
        yield dict(entry, type="step")

        if result.startswith("FILE:"):
            # "FILE:name.pptx (3 slides)" -> "name.pptx"
            files.append(result[5:].split(" (")[0].strip())
        scratch += "\n[step {}] {}({}) -> {}\n".format(
            step, tool, json.dumps(args)[:300], result[:1500])

    # Out of steps: ask for a closing summary rather than failing silently.
    try:
        answer = generate(model,
                          "{}\n\nUSER TASK: {}\n\nWORK DONE:\n{}\n\n"
                          "Summarise the outcome for the user in plain prose. No JSON.".format(
                              _system_prompt(), task, scratch))
    except Exception:
        answer = "Reached the step limit. Work completed so far is in the trace."
    trace.append({"step": step + 1, "type": "final", "text": answer.strip()})
    yield _final_event(answer.strip(), trace, files, step + 1, t0)


def run(task, model, max_steps=MAX_STEPS):
    """Execute a task to completion. Returns answer, trace, files and timings."""
    last = None
    for event in iter_run(task, model, max_steps):
        last = event
    if last is None:
        return {"answer": "No response produced.", "trace": [], "files": [],
                "steps": 0, "secs": 0}
    return {k: v for k, v in last.items() if k != "type"}


def _final_event(answer, trace, files, steps, t0):
    secs = round(time.time() - t0, 1)
    audit.record("session", event="task_end", steps=steps, secs=secs, files=files)
    return {"type": "final", "answer": answer, "trace": trace, "files": files,
            "steps": steps, "secs": secs}
