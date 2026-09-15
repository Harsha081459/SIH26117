"""Hand-rolled plan -> act -> observe agent loop.

Each turn the model either calls one tool or emits a final answer.
The full trace is returned so the UI can show the agent 'acting', not
just answering — this is what separates an agent from a chatbot.
"""
import json
import re
import time

from ollama_client import generate
from tools import TOOL_SPEC, call

MAX_STEPS = 8

SYSTEM = f"""You are an air-gapped industrial AI workbench. You may use ONLY these local tools:

{TOOL_SPEC}

Reply with EXACTLY ONE action as JSON, no other text:
{{"action": "tool", "tool": "<name>", "args": {{...}}}}
or, when the task is complete:
{{"action": "final", "answer": "<your full answer to the user>"}}

Workspace files already available: scanned_report.png, log_sample.txt
(call fs_list to discover others). Internal docs live in the knowledge base — use kb_search.
Rules: use tools to do real work; if asked for a document, end by calling make_docx/make_xlsx/make_pptx; show calculation steps explicitly."""


def _parse_action(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"action": "final", "answer": text}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"action": "final", "answer": text}


def run(task: str, model: str) -> dict:
    trace, scratch = [], ""
    t0 = time.time()
    for step in range(MAX_STEPS):
        prompt = f"{SYSTEM}\n\nUSER TASK: {task}\n\nWORK SO FAR:\n{scratch}\n\nNext action (JSON only):"
        raw = generate(model, prompt)
        act = _parse_action(raw)

        if act.get("action") == "final":
            trace.append({"step": step + 1, "type": "final", "text": act.get("answer", "")})
            return {"answer": act.get("answer", ""), "trace": trace,
                    "steps": step + 1, "secs": round(time.time() - t0, 1)}

        tool, args = act.get("tool", ""), act.get("args", {})
        result = call(tool, args)
        trace.append({"step": step + 1, "type": "tool", "tool": tool,
                      "args": args, "result": result[:500]})
        scratch += f"\n[step {step+1}] {tool}({json.dumps(args)}) -> {result[:1500]}\n"
        if result.startswith("FILE:"):
            act = {"action": "final",
                   "answer": f"Done. Deliverable created: {result[5:]}"}
            trace.append({"step": step + 2, "type": "final", "text": act["answer"]})
            return {"answer": act["answer"], "trace": trace,
                    "steps": step + 2, "secs": round(time.time() - t0, 1)}

    return {"answer": "Stopped: max steps reached.", "trace": trace,
            "steps": MAX_STEPS, "secs": round(time.time() - t0, 1)}
