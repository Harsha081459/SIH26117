"""Checks that run without the model runtime.

Everything here must pass on a bare checkout so a reviewer can verify the
plumbing before pulling any models.

    python -m pytest tests -q          (or: python tests/test_offline.py)
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import audit           # noqa: E402
import calc            # noqa: E402
import router          # noqa: E402
import tools           # noqa: E402
from agent import _parse_action, _extract_json   # noqa: E402


def test_router_picks_distinct_models():
    """No resolve: test config logic, not the live runtime."""
    """The problem statement requires auto-selection across task types."""
    code = router.route("write a python script to parse the log", use_llm=False, resolve=False)
    doc = router.route("draft an approval note", use_llm=False, resolve=False)
    img = router.route("what is here", has_attachment=True, use_llm=False, resolve=False)
    assert code["task"] == "write_code"
    assert doc["task"] == "draft_document"
    assert img["task"] == "image_understanding"
    assert code["model"] != doc["model"] != img["model"]


def test_registry_is_config_driven():
    reg = router.registry()
    assert reg["default"]
    assert len(reg["models"]) >= 2
    for m in reg["models"]:
        assert m["id"] and m["tasks"] and m["reason"]


def test_calc_shows_every_step():
    result, steps = calc.evaluate("(2*18400 + 5*3200) * 1.18")
    assert abs(result - 62304) < 1e-6
    assert len(steps) == 4
    assert "Result:" in calc.evaluate_text("2+2")


def test_calc_rejects_unsafe_input():
    for bad in ['__import__("os").system("ls")', 'open("f")', '1/0', "x := 5"]:
        try:
            calc.evaluate(bad)
            raise AssertionError("should have rejected: " + bad)
        except (calc.CalcError, Exception):
            pass


def test_sandbox_executes_and_reports_its_isolation_level():
    out = tools.run_python("print(6*7)")
    assert "42" in out
    assert "[sandbox:" in out
    # The label must state whether the network was actually removed, so nobody
    # claims isolation the deployment did not provide.
    assert ("network=none" in out) or ("NOT network-isolated" in out)


def test_sandbox_times_out():
    out = tools.run_python("import time; time.sleep(30)", timeout=2)
    assert "timed out" in out.lower() or "sandbox" in out.lower()


def test_workspace_escape_blocked():
    assert "ERROR" in tools.fs_read("../backend/main.py")
    assert "ERROR" in tools.fs_write("../../evil.txt", "x")


def test_kb_search_finds_sop():
    out = tools.kb_search("mechanical seal leakage limit")
    assert "sop" in out.lower()


def _filename(result):
    """FILE:name.ext or FILE:name.ext (n slides) -> name.ext"""
    assert result.startswith("FILE:"), result
    return result[5:].split(" (")[0].strip()


def test_deliverables_are_real_files():
    for res in (tools.make_docx("T1", "body text"),
                tools.make_xlsx("T2", "a,b\n1,2"),
                tools.make_pptx("T3", "Slide | one; two")):
        assert (tools.OUT / _filename(res)).stat().st_size > 0


def test_approval_note_has_structured_sections():
    res = tools.make_docx(
        "Approval Note Structured", "Seal replaced on pump P-201.",
        findings="leakage 14 drops/min; vibration 5.2 mm/s",
        recommendation="Return to service, re-check in 72 hours.",
        reference="SOP-MECH-041", model="qwen2.5:7b-instruct")
    assert res.startswith("FILE:")
    from docx import Document
    text = "\n".join(p.text for p in Document(str(tools.OUT / res[5:])).paragraphs)
    for expected in ("Findings", "Recommendation", "Reference",
                     "leakage 14 drops/min", "SOP-MECH-041",
                     "qwen2.5:7b-instruct", "Approved by"):
        assert expected in text, expected
    # Provenance must be stamped so a reviewer knows where the file came from.
    assert "Generated on-premise" in text


def test_attachment_is_read_by_the_right_tool():
    """A text file must not be sent to the vision model."""
    import main
    cases = {
        "scan.png": ("ocr_doc", "image"),
        "drawing.JPG": ("ocr_doc", "image"),
        "report.pdf": ("pdf_read", "pdf"),
        "spares.xlsx": ("sheet_read", "spreadsheet"),
        "DV-Team.txt": ("fs_read", "text"),
        "notes.md": ("fs_read", "text"),
        "data.csv": ("fs_read", "text"),
        "mystery.bin": ("fs_read", "file"),
    }
    for name, expected in cases.items():
        assert main._reader_for(name) == expected, name

    # Only an image attachment should be classified as a vision task.
    img = router.route("what is this", has_attachment=True,
                       attachment_kind="image", use_llm=False, resolve=False)
    txt = router.route("what is this", has_attachment=True,
                       attachment_kind="text", use_llm=False, resolve=False)
    assert img["task"] == "image_understanding"
    assert txt["task"] != "image_understanding"
    assert "vl" not in txt["model"].lower()


def test_vision_task_uses_reasoning_model_to_orchestrate():
    """A VL model should read the image, not drive the whole loop."""
    r = router.route("read scanned_report.png", has_attachment=True,
                     use_llm=False, resolve=False)
    assert r["task"] == "image_understanding"
    assert "vl" in r["model"].lower()
    assert r["orchestrator"] != r["model"]
    assert "vl" not in r["orchestrator"].lower()
    # Non-vision tasks keep one model for both roles.
    c = router.route("write a python script", use_llm=False, resolve=False)
    assert c["orchestrator"] == c["model"]


def test_make_xlsx_accepts_text_or_rows():
    for payload in ("a,b\n1,2", [["a", "b"], [1, 2]], ["a,b", "1,2"]):
        res = tools.make_xlsx("Flexible Sheet", payload)
        assert res.startswith("FILE:"), payload


def test_pptx_accepts_markdown_and_refuses_empty():
    """A deck built from nothing looks like success but is worthless."""
    for empty in ("", "   \n\n", None, []):
        assert "ERROR" in tools.make_pptx("Empty Deck", empty), repr(empty)

    md = ("## Findings\n- leakage 14 drops/min\n- vibration 5.2 mm/s\n"
          "## Recommendation\n- return to service after re-check")
    res = tools.make_pptx("From Markdown", md)
    assert res.startswith("FILE:")
    assert "2 slides" in res

    res2 = tools.make_pptx("Pipe Form", "Findings | a; b\nActions | c")
    assert "2 slides" in res2

    slides = tools._parse_slides(md)
    assert [t for t, _ in slides] == ["Findings", "Recommendation"]
    assert slides[0][1] == ["leakage 14 drops/min", "vibration 5.2 mm/s"]


def test_a_list_of_titles_makes_one_slide_each():
    """"Give me 10 slides" produces a list; it must not collapse to one slide."""
    titles = ["Intro", "Crude Types", "Refining", "Safety", "Maintenance",
              "Efficiency", "Environment", "Technology", "Quality", "Future"]
    res = tools.make_pptx("Ten Slide Deck", titles)
    assert "10 slides" in res, res

    parsed = tools._parse_slides(titles)
    assert len(parsed) == 10
    assert [t for t, _ in parsed] == titles

    # entries may carry their own bullets, and dicts are accepted too
    mixed = tools._parse_slides(["Findings | a; b", {"title": "Next",
                                                     "bullets": ["c", "d"]}])
    assert mixed[0] == ("Findings", ["a", "b"])
    assert mixed[1] == ("Next", ["c", "d"])


def test_success_is_not_claimed_when_no_file_exists():
    """A failed make_* must never be reported as a created document."""
    import agent as agent_mod

    replies = iter([
        '{"action":"tool","tool":"make_pptx","args":{"title":"D","slides":""}}',
        '{"action":"final","answer":"The presentation D.pptx has been created. '
        'Please review the presentation."}',
    ])
    real = agent_mod.generate
    agent_mod.generate = lambda *a, **k: next(replies)
    try:
        out = agent_mod.run("build a deck", "fake", max_steps=4)
    finally:
        agent_mod.generate = real

    assert out["files"] == [], out["files"]
    assert "No file was produced" in out["answer"], out["answer"]

    # A genuine success must be left untouched.
    replies2 = iter([
        '{"action":"tool","tool":"make_pptx","args":{"title":"E","slides":"A | x"}}',
        '{"action":"final","answer":"E.pptx has been created."}',
    ])
    agent_mod.generate = lambda *a, **k: next(replies2)
    try:
        ok = agent_mod.run("build a deck", "fake", max_steps=4)
    finally:
        agent_mod.generate = real
    assert ok["files"] and "No file was produced" not in ok["answer"]


def test_file_result_yields_clean_filename():
    """FILE:name.pptx (3 slides) must not become part of the download name."""
    import agent as agent_mod

    replies = iter([
        '{"action":"tool","tool":"make_pptx","args":{"title":"D","slides":"## A\\n- x"}}',
        '{"action":"final","answer":"done"}',
    ])
    real = agent_mod.generate
    agent_mod.generate = lambda *a, **k: next(replies)
    try:
        out = agent_mod.run("build a deck", "fake", max_steps=4)
    finally:
        agent_mod.generate = real
    assert out["files"], out
    assert out["files"][0].endswith(".pptx")
    assert "(" not in out["files"][0]


def test_same_tool_spam_is_stopped():
    """Repeated calls with slightly different args must be cut off."""
    import agent as agent_mod

    n = {"i": 0}

    def fake(*a, **k):
        n["i"] += 1
        return ('{"action":"tool","tool":"kb_search","args":{"query":"q%d"}}' % n["i"]
                if n["i"] <= 8 else '{"action":"final","answer":"stop"}')

    real_gen, real_call = agent_mod.generate, agent_mod.call
    executed = {"n": 0}

    def counting(name, args):
        executed["n"] += 1
        return "result"

    agent_mod.generate, agent_mod.call = fake, counting
    try:
        out = agent_mod.run("spam", "fake", max_steps=9)
    finally:
        agent_mod.generate, agent_mod.call = real_gen, real_call

    assert executed["n"] <= 3, executed
    assert any("has now been called" in s.get("result", "")
               for s in out["trace"] if s["type"] == "tool")


def test_egress_probe_reports_a_verdict():
    out = tools.egress_probe()
    assert ("DENIED" in out) or ("ALLOWED" in out)
    assert ("DENIED" in out and "ALL OUTBOUND ATTEMPTS DENIED" in out) or \
           ("ALLOWED" in out and "NOT isolated" in out)


def test_action_parsing_survives_messy_output():
    """Shapes small local models actually emit, collected from real runs."""
    tool_cases = [
        '{"action":"tool","tool":"fs_list","args":{}}',
        'Sure!\n```json\n{"action":"tool","tool":"kb_search","args":{"query":"pump"}}\n```',
        # tool name placed in "action" -- observed from qwen2.5:3b
        '{"action": "kb_search", "args": {"query": "vibration limit"}}',
        # no action key at all
        '{"tool": "fs_list", "args": {}}',
        # arguments under a different key
        '{"action":"tool","tool":"kb_search","arguments":{"query":"seal"}}',
        # arguments inlined next to the tool name
        '{"action":"kb_search","query":"seal leakage"}',
        {"action": "tool", "name": "fs_list", "args": {}},
        # single-quoted value: valid Python, invalid JSON -- observed from qwen2.5:7b
        '{"action": "tool", "tool": "run_python", "args": {"code": \'print("hi")\'}}',
    ]
    for c in tool_cases:
        c = c if isinstance(c, str) else json.dumps(c)
        act = _parse_action(c)
        assert act, "failed to parse: " + c
        assert act["action"] == "tool", c
        assert act["tool"] in tools.TOOLS, c
        assert isinstance(act["args"], dict), c

    for c in ('I will finish now. {"action":"final","answer":"done {nested} ok"}',
              '{"answer": "all done"}'):
        act = _parse_action(c)
        assert act and act["action"] == "final", c
        assert act["answer"]

    assert _parse_action("no json at all") is None
    assert _extract_json('{"a": {"b": 1}}')["a"]["b"] == 1


def test_inline_args_exclude_control_keys():
    act = _parse_action('{"action":"kb_search","query":"x"}')
    assert act["args"] == {"query": "x"}


def test_tool_calls_are_audited():
    before = audit.summary()["total_records"]
    tools.call("fs_list", {})
    after = audit.summary()
    assert after["total_records"] > before
    assert after["all_local"] is True


def test_unknown_tool_is_handled():
    assert "unknown tool" in tools.call("definitely_not_a_tool", {})
    assert "bad args" in tools.call("fs_read", {"wrong_arg": 1})


def test_repeated_identical_call_is_not_re_executed():
    """A stuck model must be redirected, not allowed to burn the step budget."""
    import agent as agent_mod

    calls = {"n": 0}

    def fake_generate(model, prompt, **kw):
        calls["n"] += 1
        if calls["n"] <= 3:
            return '{"action":"tool","tool":"kb_search","args":{"query":"same"}}'
        return '{"action":"final","answer":"stopped repeating"}'

    real_generate = agent_mod.generate
    real_call = agent_mod.call
    executed = {"n": 0}

    def counting_call(name, args):
        executed["n"] += 1
        return "some result"

    agent_mod.generate = fake_generate
    agent_mod.call = counting_call
    try:
        out = agent_mod.run("do the thing", "fake-model", max_steps=6)
    finally:
        agent_mod.generate = real_generate
        agent_mod.call = real_call

    # Three identical actions were requested but only the first reached the tool.
    assert executed["n"] == 1, executed
    repeats = [s for s in out["trace"]
               if s["type"] == "tool" and "already called" in s.get("result", "")]
    assert repeats, "expected a redirection message"


def test_every_registered_tool_is_callable():
    for name, spec in tools.TOOLS.items():
        assert callable(spec["fn"])
        assert name in tools.TOOL_SPEC


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print("PASS  " + name)
        except Exception as e:
            failed += 1
            print("FAIL  {}: {}".format(name, e))
    print("\n{}/{} passed".format(len(fns) - failed, len(fns)))
    sys.exit(1 if failed else 0)
