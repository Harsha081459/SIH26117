#!/usr/bin/env bash
# One-command environment setup. Installs the Ollama runtime into $HOME (no
# root required), starts it, pulls the models and installs Python deps.
#
#   bash scripts/setup.sh
#
# Re-running is safe: existing installs and pulled models are left alone.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${OLLAMA_PREFIX:-$HOME/.local}"
export PATH="$PREFIX/bin:$PATH"

say() { printf '\n== %s\n' "$1"; }

# ── 1. Python ───────────────────────────────────────────────────────────
say "python environment"
PY=""
for c in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$c" >/dev/null 2>&1; then
    v=$("$c" -c 'import sys; print(sys.version_info >= (3, 8))' 2>/dev/null || echo False)
    [ "$v" = "True" ] && { PY="$c"; break; }
  fi
done
[ -z "$PY" ] && { echo "need Python 3.8+"; exit 1; }
echo "using $PY ($($PY --version 2>&1))"

if [ ! -d "$ROOT/.venv" ]; then
  "$PY" -m venv "$ROOT/.venv"
fi
"$ROOT/.venv/bin/pip" -q install --upgrade pip
"$ROOT/.venv/bin/pip" -q install -r "$ROOT/requirements.txt"
echo "deps installed"

# ── 2. Ollama runtime ───────────────────────────────────────────────────
say "ollama runtime"
if command -v ollama >/dev/null 2>&1; then
  echo "already present: $(ollama --version 2>&1 | head -1)"
else
  mkdir -p "$PREFIX"
  URL=https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst
  echo "downloading (~1.4 GB)"
  if curl -fsSL "$URL" -o /tmp/ollama.tar.zst; then
    if command -v zstd >/dev/null 2>&1; then
      zstd -d -c /tmp/ollama.tar.zst | tar -x -C "$PREFIX"
    else
      tar --zstd -xf /tmp/ollama.tar.zst -C "$PREFIX"
    fi
    rm -f /tmp/ollama.tar.zst
    echo "installed to $PREFIX/bin/ollama"
  else
    echo "download failed -- install manually from https://ollama.com/download"
    exit 1
  fi
fi

if ! pgrep -u "$(id -u)" ollama >/dev/null 2>&1; then
  mkdir -p "$ROOT/logs"
  setsid nohup ollama serve > "$ROOT/logs/ollama_serve.log" 2>&1 < /dev/null &
  sleep 8
fi
echo "runtime up: $(ollama list >/dev/null 2>&1 && echo yes || echo no)"

# ── 3. Models ───────────────────────────────────────────────────────────
say "models"
# Order matters: the first two give the model-auto-selection demo, the third
# handles scans and drawings. Any subset works -- an absent model is
# substituted at runtime by router._resolve().
for m in qwen2.5:7b-instruct qwen2.5-coder:7b qwen2.5vl:3b nomic-embed-text; do
  if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$m"; then
    echo "have $m"
  else
    echo "pulling $m"
    ollama pull "$m" || echo "could not pull $m (continuing)"
  fi
done

# ── 4. Sandbox isolation check ──────────────────────────────────────────
say "sandbox isolation"
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  echo "docker available -- code runs with --network none (strongest)"
elif command -v unshare >/dev/null 2>&1 && unshare -rn true 2>/dev/null; then
  echo "unshare available -- code runs in an empty network namespace"
else
  echo "WARNING: neither docker nor unshare usable."
  echo "Generated code will run WITHOUT network isolation and will say so."
fi

# ── 5. Verify ───────────────────────────────────────────────────────────
say "tests"
( cd "$ROOT" && "$ROOT/.venv/bin/python" tests/test_offline.py ) | tail -3

cat <<EOF

Setup complete. Start the workbench with:

  cd backend && ../.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000

then open http://localhost:8000
EOF
