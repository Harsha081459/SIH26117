import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from playwright.sync_api import expect, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import agent
import audit
import main
import ollama_client
import tools


@pytest.fixture(scope="session")
def browser():
    with sync_playwright() as driver:
        channel = os.environ.get("WORKBENCH_BROWSER_CHANNEL", "msedge" if os.name == "nt" else "chromium")
        instance = driver.chromium.launch(channel=channel, headless=True,
                                         args=["--disable-background-networking", "--disable-component-update"])
        yield instance
        instance.close()


@pytest.fixture
def environment(tmp_path, monkeypatch):
    for name in ("workspace", "outputs", "corpus", "logs"):
        (tmp_path / name).mkdir()
    for module in (tools, main):
        monkeypatch.setattr(module, "WORKSPACE", tmp_path / "workspace")
        monkeypatch.setattr(module, "OUT", tmp_path / "outputs")
    monkeypatch.setattr(tools, "CORPUS", tmp_path / "corpus")
    monkeypatch.setattr(tools, "EMBED_CACHE", tmp_path / "embeddings.json")
    monkeypatch.setattr(audit, "LOG_PATH", tmp_path / "logs" / "audit.jsonl")
    monkeypatch.setattr(main, "_RUNS", {})
    monkeypatch.setattr(main, "_SLOTS", threading.BoundedSemaphore(1))
    installed = ["qwen2.5:7b-instruct", "qwen2.5-coder:7b", "qwen2.5vl:3b"]
    runtime = {"reachable": True, "models": installed}
    monkeypatch.setattr(main, "runtime_status", lambda: runtime)
    monkeypatch.setattr(ollama_client, "runtime_status", lambda: runtime)
    monkeypatch.setattr(ollama_client, "list_models", lambda: runtime["models"])
    monkeypatch.setattr(ollama_client, "model_capabilities", lambda name:
                        {"completion", "vision"} if "vl" in name else {"completion"})
    answer = lambda *a, **k: '{"action":"final","answer":"A synthetic test response."}'
    monkeypatch.setattr(agent, "generate", answer)
    monkeypatch.setattr(ollama_client, "generate", answer)
    (tools.WORKSPACE / "log_sample.txt").write_text("ERROR one\nINFO normal\nERROR two\n", encoding="utf-8")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=port,
                                          log_level="error", access_log=False))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started, "The isolated test server did not start"
    try:
        yield {"url": "http://127.0.0.1:" + str(port), "runtime": runtime, "root": tmp_path}
    finally:
        with main._RUN_LOCK:
            for event in main._RUNS.values():
                event.set()
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()
        assert not thread.is_alive(), "The isolated test server did not stop"
        assert not main._RUNS, "A request worker leaked after browser test cleanup"


@pytest.fixture
def page(browser, environment):
    context = browser.new_context(viewport={"width": 1280, "height": 900}, accept_downloads=True)
    remote, errors = [], []

    def network(route):
        if not route.request.url.startswith(environment["url"] + "/"):
            remote.append(route.request.url)
            route.abort()
        else:
            route.continue_()

    context.route("**/*", network)
    page = context.new_page()
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(environment["url"])
    expect(page.locator("#buildbox")).to_contain_text("local-hardening-v2")
    try:
        yield page
    finally:
        context.close()
        assert not errors, errors
        assert not remote, "The page attempted a non-local request: " + repr(remote)


def submit(page, message):
    page.locator("#msg").fill(message)
    page.locator("#send").click()


def sse(*events):
    return ": heartbeat\n\n" + "".join("data: " + json.dumps(event) + "\n\n" for event in events)


def test_browser_basic_chat_finishes_and_resets_controls(page):
    submit(page, "Summarise a simple test request")
    expect(page.locator(".turn.bot .status.completed")).to_have_text("Completed")
    expect(page.locator(".turn.bot .md")).to_contain_text("synthetic test response")
    expect(page.locator(".think")).to_have_count(0)
    expect(page.locator("#stop")).to_be_hidden()
    page.locator("#msg").fill("Summarise another request")
    expect(page.locator("#send")).to_be_enabled()


def test_browser_download_opens_as_a_real_word_document(page, monkeypatch):
    replies = iter([
        '{"action":"tool","tool":"make_docx","args":{"title":"Browser Draft","body":"BROWSER_DOCUMENT_SENTINEL"}}',
        '{"action":"final","answer":"An invented filename.docx has been created."}'])
    monkeypatch.setattr(agent, "generate", lambda *a, **k: next(replies))
    submit(page, "Create a Word file")
    expect(page.locator(".status.completed")).to_be_visible()
    link = page.locator(".turn.bot a.file")
    expect(link).to_have_count(1)
    with page.expect_download() as download:
        link.click()
    from docx import Document
    text = "\n".join(p.text for p in Document(download.value.path()).paragraphs)
    assert "BROWSER_DOCUMENT_SENTINEL" in text
    assert "General-knowledge draft" in text
    assert "invented filename" not in page.locator(".turn.bot .md").inner_text()


def test_browser_text_upload_is_read_without_vision(page, monkeypatch):
    prompts = []

    def reply(model, prompt, **kwargs):
        prompts.append(prompt)
        return '{"action":"final","answer":"Read the uploaded text."}'

    def no_vision(*args, **kwargs):
        raise AssertionError("A text upload reached the vision reader")

    monkeypatch.setattr(agent, "generate", reply)
    monkeypatch.setattr(tools, "_describe_image", no_vision)
    page.locator("#file").set_input_files({"name": "team.txt", "mimeType": "text/plain", "buffer": b"BROWSER_TEXT_SOURCE"})
    expect(page.locator(".chip")).to_contain_text("team.txt")
    submit(page, "Summarise the attached text")
    expect(page.locator(".status.completed")).to_be_visible()
    expect(page.locator(".step .tname")).to_have_text("fs_read()")
    assert prompts and "BROWSER_TEXT_SOURCE" in prompts[0]


def test_browser_upload_rejection_is_visible_and_preserves_previous_attachment(page):
    page.locator("#file").set_input_files({"name": "good.txt", "mimeType": "text/plain", "buffer": b"usable text"})
    expect(page.locator(".chip")).to_contain_text("good.txt")
    page.locator("#file").set_input_files({"name": "bad.exe", "mimeType": "application/octet-stream", "buffer": b"MZ invalid"})
    expect(page.locator(".turn.err")).to_contain_text("Unsupported file")
    expect(page.locator(".chip")).to_contain_text("good.txt")
    expect(page.locator("#attachBtn")).to_be_enabled()
    assert not list(tools.WORKSPACE.glob("*.exe"))


def test_browser_stop_cancels_via_the_api_and_retains_partial_files(page, monkeypatch):
    entered = threading.Event()
    calls = {"n": 0}

    def reply(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return '{"action":"tool","tool":"make_docx","args":{"title":"Partial","body":"Completed before cancellation"}}'
        if calls["n"] == 2:
            entered.set()
            with main._RUN_LOCK:
                event = next(iter(main._RUNS.values()))
            assert event.wait(8), "The browser did not send a cancellation request"
        return '{"action":"final","answer":"A synthetic response."}'

    monkeypatch.setattr(agent, "generate", reply)
    submit(page, "Create a Word file")
    expect(page.locator(".step .tname")).to_have_text("make_docx()")
    assert entered.is_set()
    page.locator("#stop").click()
    expect(page.locator(".status.cancelled")).to_have_text("Cancelled")
    expect(page.locator(".turn.bot a.file")).to_have_count(1)
    expect(page.locator(".think")).to_have_count(0)
    expect(page.locator("#stop")).to_be_hidden()
    submit(page, "Summarise the next request")
    expect(page.locator(".status.completed")).to_have_count(1)


@pytest.mark.parametrize("status", ["blocked", "reachable", "inconclusive"])
def test_browser_probe_states_are_distinct(page, monkeypatch, status):
    report = {"status": status, "scope": "sandbox", "denied": {"blocked": True, "reachable": False}.get(status),
              "rows": [{"target": "TCP", "state": status, "detail": "synthetic probe result"}],
              "verdict": "Synthetic probe verdict", "limitation": "Not whole-machine evidence."}
    monkeypatch.setattr(main, "call", lambda *a, **k: json.dumps(report))
    page.locator("#probe").click()
    expect(page.locator("#probebox .verdict." + status)).to_contain_text(status.upper())
    expect(page.locator("#probebox")).to_contain_text("Not whole-machine evidence")
    expect(page.locator("#probe")).to_be_enabled()


def test_browser_truncated_stream_keeps_steps_and_clears_busy_state(page):
    events = sse({"type": "start", "request_id": "0" * 32},
                 {"type": "step", "step": 1, "tool": "calculate", "args": {"expression": "2+2"}, "result": "4", "ok": True})
    page.route("**/api/chat/stream", lambda route: route.fulfill(body=events, content_type="text/event-stream"))
    submit(page, "Calculate 2+2")
    expect(page.locator(".status.failed")).to_contain_text("Completion is unconfirmed")
    expect(page.locator(".step .tname")).to_have_text("calculate()")
    expect(page.locator(".think")).to_have_count(0)
    expect(page.locator("#stop")).to_be_hidden()
    page.locator("#msg").fill("Retry")
    expect(page.locator("#send")).to_be_enabled()


def test_browser_malformed_event_after_final_does_not_erase_download(page):
    name = tools.make_docx("Preserved", "A verified existing test artifact")[5:]
    info = tools.artifact_info(name)
    events = sse({"type": "start", "request_id": "1" * 32},
                 {"type": "final", "status": "completed", "answer": "Verified draft available.",
                  "files": [name], "artifacts": [info], "trace": [], "steps": 1, "secs": 0})
    page.route("**/api/chat/stream", lambda route: route.fulfill(body=events + "data: not-json\n\n",
                                                               content_type="text/event-stream"))
    submit(page, "Create a Word file")
    expect(page.locator(".status.completed")).to_be_visible()
    expect(page.locator(".status.partial")).to_contain_text("Final result retained")
    expect(page.locator(".turn.bot a.file")).to_have_count(1)
    expect(page.locator("#stop")).to_be_hidden()


def test_browser_sse_parser_handles_utf8_crlf_and_heartbeats(page):
    payload = ": heartbeat\r\n\r\ndata: " + json.dumps({"type": "final", "answer": "Total ₹62,304"}, ensure_ascii=False) + "\r\n\r\n"
    result = page.evaluate("""async text => {
        const bytes = new TextEncoder().encode(text), events = [];
        const stream = new ReadableStream({start(controller) {
            for (const byte of bytes) controller.enqueue(new Uint8Array([byte]));
            controller.close();
        }});
        await readEventStream(stream, event => events.push(event));
        return events;
    }""", payload)
    assert result == [{"type": "final", "answer": "Total ₹62,304"}]


def test_browser_sse_idle_timeout_is_an_error(page):
    result = page.evaluate("""async () => {
        try { await readEventStream(new ReadableStream(), () => {}, 20); }
        catch (error) { return error.message; }
        return 'unexpected success';
    }""")
    assert "No stream activity" in result


def test_browser_markdown_and_trace_do_not_execute_html(page, monkeypatch):
    payload = '<svg onload="window.attacked=true"></svg>\n<img src=x onerror="window.attacked=true">\n\x00B99\x00\n```html\n<script>window.attacked=true</script>\n```'
    monkeypatch.setattr(agent, "generate", lambda *a, **k: json.dumps({"action": "final", "answer": payload}))
    submit(page, "Summarise this test")
    expect(page.locator(".status.completed")).to_be_visible()
    expect(page.locator(".turn.bot .md svg,.turn.bot .md img,.turn.bot .md script")).to_have_count(0)
    assert page.evaluate("window.attacked === undefined")
    assert "undefined" not in page.locator(".turn.bot .md").inner_text()
    result = page.evaluate("""() => {
        const element = stepEl({step:'<img src=x>', tool:'<svg onload=alert(1)>', args:{x:'<script>'}, result:'<img src=x>'}, false);
        return element.querySelectorAll('img,svg,script').length;
    }""")
    assert result == 0


@pytest.mark.parametrize("reachable,expected", [(False, "runtime unavailable"), (True, "0 models installed")])
def test_browser_runtime_unavailable_and_empty_are_not_green(page, environment, reachable, expected):
    environment["runtime"].update(reachable=reachable, models=[], error="Synthetic runtime unavailable")
    page.evaluate("poll()")
    expect(page.locator("#runtime")).to_contain_text(expected)
    expect(page.locator("#runtime")).to_have_class("pill warn")
    submit(page, "Summarise this request")
    expect(page.locator(".status.failed")).to_be_visible()
    expect(page.locator(".think")).to_have_count(0)


def test_browser_unknown_snapshot_and_old_backend_are_not_success(page):
    page.route("**/api/egress", lambda route: route.fulfill(json={"external_calls": None, "status": "unknown"}))
    page.route("**/api/health", lambda route: route.fulfill(json={"workspace_files": [], "status": "ok"}))
    page.evaluate("poll()")
    expect(page.locator("#egress")).to_have_text("connections unknown")
    expect(page.locator("#buildbox")).to_contain_text("version mismatch")
    page.locator("#msg").fill("Should not reach an old backend")
    expect(page.locator("#send")).to_be_disabled()


def test_browser_busy_error_is_visible_and_request_is_recoverable(page):
    task_id, _ = main._start_run(main.Chat(message="Summarise another request"))
    try:
        submit(page, "Summarise my request")
        expect(page.locator(".status.failed")).to_contain_text("Another request is still running")
        expect(page.locator(".think")).to_have_count(0)
        expect(page.locator("#msg")).to_have_value("Summarise my request")
    finally:
        main._end_run(task_id)
    page.locator("#send").click()
    expect(page.locator(".status.completed")).to_have_count(1)


def test_browser_layout_has_no_horizontal_overflow(page, environment):
    submit(page, "Summarise a simple test request")
    expect(page.locator(".status.completed")).to_be_visible()
    for width in (1280, 430):
        page.set_viewport_size({"width": width, "height": 900})
        overflow = page.evaluate("""() => [...document.querySelectorAll('body *')]
            .map(element => ({name: element.id || element.className || element.tagName,
                              right: Math.round(element.getBoundingClientRect().right)}))
            .filter(element => element.right > window.innerWidth).slice(0, 12)""")
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), (width, overflow)
    page.set_viewport_size({"width": 1280, "height": 900})
    screenshot = environment["root"] / "workbench-browser.png"
    page.screenshot(path=str(screenshot), full_page=True)
    print("Browser screenshot:", screenshot)
