import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

IMAGE = "python:3.11-slim"
MAX_OUTPUT = 64 * 1024
MAX_INPUT = 32 * 1024 * 1024
MAX_INPUT_FILES = 32


def _docker_prefix():
    host = os.environ.get("DOCKER_HOST") or ("npipe:////./pipe/docker_engine" if os.name == "nt"
                                            else "unix:///var/run/docker.sock")
    if "\x00" in host or not (host.startswith("unix:///") or host.startswith("npipe:////./pipe/")):
        return None
    return ["docker", "--host", host]


def _docker_env():
    return {key: value for key, value in os.environ.items()
            if key not in ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH")}


def _bwrap_base():
    command = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session"]
    for name in ("/usr", "/bin", "/lib", "/lib64"):
        if Path(name).exists():
            command += ["--ro-bind", name, name]
    # No --size on tmpfs: it needs bubblewrap >= 0.4.1; prlimit --fsize already
    # caps per-file writes inside the sandbox.
    command += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/dev/shm",
                "--remount-ro", "/dev", "--tmpfs", "/tmp"]
    return command


# --clearenv needs bubblewrap >= 0.5; `env -i` clears the child environment on
# every released version, so the sandbox does not inherit variables or secrets.
_BWRAP_ENV = ["/usr/bin/env", "-i", "PATH=/usr/bin:/bin", "HOME=/tmp", "PYTHONUTF8=1"]


def available_mode():
    prefix = _docker_prefix()
    if prefix and shutil.which("docker"):
        try:
            info = subprocess.run(prefix + ["info", "--format", "{{.OSType}}"],
                                  capture_output=True, timeout=8, env=_docker_env())
            image = subprocess.run(prefix + ["image", "inspect", IMAGE],
                                   capture_output=True, timeout=8, env=_docker_env())
            if info.returncode == image.returncode == 0 and info.stdout.strip() == b"linux":
                return "docker"
        except (OSError, subprocess.TimeoutExpired):
            pass
    base_python = Path(getattr(sys, "_base_executable", sys.executable)).resolve()
    if (sys.platform.startswith("linux") and str(base_python).startswith("/usr/")
            and shutil.which("bwrap") and shutil.which("prlimit")):
        try:
            probe = subprocess.run(_bwrap_base() + ["--remount-ro", "/", "--"] + _BWRAP_ENV
                                   + [str(base_python), "-I", "-B", "-c", "pass"],
                                   capture_output=True, timeout=8)
            if probe.returncode == 0:
                return "bubblewrap"
        except (OSError, subprocess.TimeoutExpired):
            pass
    return None


def _kill(process):
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass


def _execute(command, timeout, cancel_event=None, env=None):
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, start_new_session=os.name == "posix", env=env)
    output = bytearray()
    exceeded = threading.Event()

    def drain():
        while True:
            data = process.stdout.read(4096)
            if not data:
                return
            room = MAX_OUTPUT - len(output)
            output.extend(data[:room])
            if len(data) > room:
                exceeded.set()
                _kill(process)
                return

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    deadline = time.monotonic() + timeout
    try:
        while process.poll() is None:
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("execution cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("execution timed out")
            try:
                process.wait(timeout=min(0.1, remaining))
            except subprocess.TimeoutExpired:
                continue
    except (InterruptedError, TimeoutError):
        _kill(process)
        process.wait(timeout=5)
        raise
    finally:
        reader.join(timeout=5)
        process.stdout.close()
    if exceeded.is_set():
        raise RuntimeError("output limit exceeded")
    return process.returncode, output.decode("utf-8", errors="replace").strip()


def _remove_container(name, prefix=None):
    prefix = prefix or _docker_prefix()
    if prefix:
        try:
            subprocess.run(prefix + ["rm", "--force", name], capture_output=True, timeout=8, env=_docker_env())
        except (OSError, subprocess.TimeoutExpired):
            pass


def execute(code, files=(), timeout=30, cancel_event=None):
    if not isinstance(code, str) or not code.strip() or len(code.encode("utf-8")) > 64 * 1024:
        return "ERROR: code must be nonempty text smaller than 64 KiB"
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 60:
        return "ERROR: sandbox timeout must be an integer between 1 and 60 seconds"
    if cancel_event is not None and cancel_event.is_set():
        return "ERROR: sandbox execution cancelled before it started"
    files = list(files)
    if len(files) > MAX_INPUT_FILES:
        return "ERROR: sandbox accepts at most 32 input files"
    mode = available_mode()
    if mode not in ("docker", "bubblewrap"):
        return ("ERROR: secure sandbox unavailable. Code was not executed. Use local Docker with "
                "a preloaded python:3.11-slim image, or Linux bubblewrap and prlimit. "
                "Remote Docker transports and network-only namespaces are not supported.")
    prefix = _docker_prefix() if mode == "docker" else None
    if mode == "docker" and not prefix:
        return "ERROR: remote Docker transports are not permitted"

    name = "workbench-" + uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="workbench-code-") as temp:
        root = Path(temp)
        root.chmod(0o755)
        total = 0
        for path in files:
            path = Path(path)
            if not path.is_file() or path.is_symlink():
                return "ERROR: invalid sandbox input file"
            total += path.stat().st_size
            if total > MAX_INPUT:
                return "ERROR: sandbox inputs exceed 32 MiB"
            target = root / path.name
            if target.exists() or target.name == "program.py":
                return "ERROR: duplicate or reserved sandbox input name"
            shutil.copyfile(path, target)
            target.chmod(0o444)
        script = root / "program.py"
        script.write_text(code, encoding="utf-8")
        script.chmod(0o444)

        if mode == "docker":
            command = prefix + ["run", "--rm", "--pull", "never", "--name", name,
                       "--network", "none", "--read-only", "--cap-drop", "ALL",
                       "--security-opt", "no-new-privileges", "--pids-limit", "64",
                       "--memory", "256m", "--cpus", "1", "--user", "65534:65534",
                       "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=32m",
                       "--mount", "type=bind,src={},dst=/work,readonly".format(root),
                       "--workdir", "/work", "--env", "PYTHONUTF8=1", IMAGE,
                       "python", "-I", "-B", "/work/program.py"]
        else:
            executable = str(Path(getattr(sys, "_base_executable", sys.executable)).resolve())
            # prlimit runs INSIDE the sandbox: wrapping bwrap itself with a low
            # --nproc makes user-namespace creation fail (EAGAIN) before the
            # payload ever starts.
            command = _bwrap_base() + ["--ro-bind", str(root), "/work", "--remount-ro", "/",
                                       "--chdir", "/work", "--",
                                       "/usr/bin/prlimit", "--as=536870912",
                                       "--cpu={}".format(timeout), "--nproc=64",
                                       "--fsize=1048576", "--nofile=64", "--"]
            command += _BWRAP_ENV + [executable, "-I", "-B", "/work/program.py"]
        try:
            returncode, output = _execute(command, timeout, cancel_event=cancel_event,
                                          env=_docker_env() if mode == "docker" else None)
            if returncode:
                return "ERROR: sandbox exited with code {}\n{}".format(returncode, output)
            return "[sandbox: {}, network=none]\n{}".format(mode, output or "(no output)")
        except (OSError, TimeoutError, RuntimeError) as exc:
            return "ERROR: sandbox {}".format(exc)
        finally:
            if mode == "docker":
                _remove_container(name, prefix)
