#!/usr/bin/env python3
"""Start an uncensored llama.cpp model on a free Colab T4 and expose it as an OpenAI endpoint.

    ./colab-llm.py                 # reuse the existing session, else create one, then print the URL
    ./colab-llm.py --stop          # stop the VM (do this; idle VMs burn compute units)
    ./colab-llm.py --model 14b     # swap the served model
    ./colab-llm.py --ctx 32768     # shrink context to trade KV cache for speed
    ./colab-llm.py --no-register   # skip writing the omp provider

State is remote: /content survives between runs inside one session, so a second invocation
reuses the downloaded llama.cpp build, the model weights, and the running server. If the
session is gone, everything is rebuilt from scratch.

The script also writes/updates the `uncensored` provider so both agents can use the model:
omp via ~/.omp/agent/models.yml (then `omp --model uncensored/abliterated`) and fx via
~/.fx/settings.json (then `fx provider uncensored`).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MODELS_YML = Path.home() / ".omp" / "agent" / "models.yml"
FX_SETTINGS = Path.home() / ".fx" / "settings.json"
PROVIDER_ID = "uncensored"
MODEL_ALIAS = "abliterated"
BUILD = "b11696"

# name -> (hf url, quant label, approx GiB). Q4_K_M keeps a 9B well inside the T4's 15 GB
# even at 128k context, because Qwen3.5's hybrid attention makes its KV cache unusually cheap.
MODELS = {
    "9b": (
        "https://huggingface.co/mradermacher/Huihui-Qwen3.5-9B-abliterated-GGUF/resolve/main/"
        "Huihui-Qwen3.5-9B-abliterated.Q4_K_M.gguf",
        5.24,
    ),
    "14b": (
        "https://huggingface.co/mradermacher/Huihui-Qwen3-14B-abliterated-v2-GGUF/resolve/main/"
        "Huihui-Qwen3-14B-abliterated-v2.Q4_K_M.gguf",
        8.38,
    ),
}

REMOTE = r'''
import json, os, re, subprocess, sys, time

BUILD      = {build!r}
MODEL_URL  = {model_url!r}
CTX        = {ctx}
CUDART     = f"/content/llama/cudart-llama-{{BUILD}}-bin-ubuntu-cuda-12.8-x64"
BIN        = f"/content/llama/llama-{{BUILD}}"
# /usr/lib64-nvidia holds libcuda.so.1. Assigning LD_LIBRARY_PATH instead of appending
# drops it, libcuda resolves to "not found", and llama.cpp silently runs on CPU.
LDP        = f"/usr/lib64-nvidia:{{CUDART}}:{{BIN}}"
MODEL_PATH = "/content/models/model.gguf"
PORT       = 8000  # Colab itself owns 8080
LOGF       = "/content/server.log"

os.environ["LD_LIBRARY_PATH"] = LDP
os.makedirs("/content/llama", exist_ok=True)
os.makedirs("/content/models", exist_ok=True)

def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()

CURL_CODE = "%{{http_code}}"   # curl's status-code placeholder, kept out of an f-string

def health():
    return sh(f"curl -s -o /dev/null -w '{{CURL_CODE}}' http://127.0.0.1:{{PORT}}/health") == "200"

def download(url, dest, minimum):
    if os.path.exists(dest) and os.path.getsize(dest) >= minimum:
        return "cached"
    subprocess.run(f"curl -sL --retry 3 -o {{dest}} '{{url}}'", shell=True, check=True)
    return "downloaded"

# ---- llama.cpp ----
if not os.path.exists(f"{{BIN}}/llama-server"):
    base = f"https://github.com/ggml-org/llama.cpp/releases/download/{{BUILD}}"
    for name in (f"cudart-llama-{{BUILD}}-bin-ubuntu-cuda-12.8-x64.tar.gz",
                 f"llama-{{BUILD}}-bin-ubuntu-cuda-12.8-x64.tar.gz"):
        subprocess.run(f"cd /content/llama && curl -sL -O {{base}}/{{name}} && tar xzf {{name}}",
                       shell=True, check=True)

dev = sh(f"{{BIN}}/llama-server --list-devices")
if "CUDA0" not in dev:
    print(json.dumps({{"error": "no CUDA device; LD_LIBRARY_PATH broke", "devices": dev}})); sys.exit(1)

# ---- model ----
dl = download(MODEL_URL, MODEL_PATH, int({min_bytes}))
size_gib = os.path.getsize(MODEL_PATH) / 2**30
if size_gib < {min_gib} - 0.5:
    print(json.dumps({{"error": f"model truncated: {{size_gib:.2f}} GiB"}})); sys.exit(1)

# ---- server: restart only if the running one does not match ----
want = f"--ctx-size {{CTX}}"
running = sh("pgrep -af 'llama-server --model'")
if not (health() and want in running):
    subprocess.run("pkill -f llama-server || true", shell=True); time.sleep(3)
    subprocess.run(
        f"cd /content && LD_LIBRARY_PATH={{LDP}} setsid nohup {{BIN}}/llama-server "
        f"--model {{MODEL_PATH}} --host 0.0.0.0 --port {{PORT}} --alias {alias!r} "
        f"--ctx-size {{CTX}} --n-gpu-layers 999 --threads 2 --jinja "
        f"> {{LOGF}} 2>&1 < /dev/null &", shell=True)
    for _ in range(60):
        time.sleep(5)
        if health(): break
    else:
        print(json.dumps({{"error": "server never became healthy",
                           "log": open(LOGF, errors="ignore").read()[-600:]}})); sys.exit(1)

# ---- cloudflared ----
CF = "/content/cloudflared"
if not os.path.exists(CF):
    subprocess.run(f"curl -sL -o {{CF}} https://github.com/cloudflare/cloudflared/releases/latest/"
                   "download/cloudflared-linux-amd64 && chmod +x " + CF, shell=True, check=True)

url = None
log = open("/content/cf.log", errors="ignore").read() if os.path.exists("/content/cf.log") else ""
m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", log)
# Reuse only on a normal bootstrap. A live cloudflared PROCESS is not proof the
# tunnel WORKS, so recovery (FORCE_TUNNEL) always mints a fresh one.
if m and not {FORCE_TUNNEL} and sh("pgrep -f 'cloudflared tunnel'"):
    url = m.group(0)
else:
    subprocess.run("pkill -f 'cloudflared tunnel' || true", shell=True); time.sleep(2)
    subprocess.run(f"cd /content && setsid nohup {{CF}} tunnel --url http://localhost:{{PORT}} "
                   "--no-autoupdate > /content/cf.log 2>&1 < /dev/null &", shell=True)
    for _ in range(40):
        time.sleep(3)
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com",
                      open("/content/cf.log", errors="ignore").read())
        if m: url = m.group(0); break

if not url:
    print(json.dumps({{"error": "no tunnel URL"}})); sys.exit(1)

props = json.loads(sh(f"curl -s http://127.0.0.1:{{PORT}}/props"))
n_ctx = props.get("default_generation_settings", {{}}).get("n_ctx")
vram = sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader")
print("__RESULT__" + json.dumps(
    {{"url": url, "n_ctx": n_ctx, "model_gib": round(size_gib, 2),
      "download": dl, "vram": vram, "ok": True}}))
'''


def colab(*args: str, timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["colab", *args], capture_output=True, text=True, timeout=timeout)


def session_exists(name: str) -> bool:
    r = colab("sessions", timeout=120)
    return bool(re.search(rf"^\[{re.escape(name)}\]", r.stdout, re.M))


def ensure_session(name: str, gpu: str) -> None:
    if session_exists(name):
        print(f"[llm] reusing session '{name}'", flush=True)
        return
    print(f"[llm] creating session '{name}' ({gpu})", flush=True)
    r = colab("new", "-s", name, "--gpu", gpu, timeout=600)
    if r.returncode != 0:
        sys.exit(f"colab new failed:\n{r.stdout}\n{r.stderr}")


def remote_bootstrap(name: str, model: str, ctx: int, force_tunnel: bool = False) -> dict:
    url, approx = MODELS[model]
    script = REMOTE.format(
        build=BUILD, model_url=url, ctx=ctx, alias=MODEL_ALIAS,
        min_bytes=int(approx * 0.98 * 2**30), min_gib=approx,
        FORCE_TUNNEL=force_tunnel,
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(script)
        path = fh.name
    try:
        print("[llm] bootstrapping runtime (downloads on first use)", flush=True)
        r = colab("exec", "-s", name, "-f", path, "--timeout", "1800", timeout=2400)
    finally:
        os.unlink(path)
    out = r.stdout + r.stderr
    for line in out.splitlines():
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])
    sys.exit(f"bootstrap failed:\n{out[-1500:]}")


def register_omp(url: str, ctx: int) -> None:
    import yaml

    MODELS_YML.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if MODELS_YML.exists():
        data = yaml.safe_load(MODELS_YML.read_text()) or {}
    providers = data.setdefault("providers", {})
    providers[PROVIDER_ID] = {
        "baseUrl": f"{url}/v1",
        "api": "openai-completions",
        "auth": "none",  # llama-server is keyless
        "discovery": {"type": "openai-models-list", "injectV1": False},  # baseUrl already ends in /v1
        "models": [{
            "id": MODEL_ALIAS,
            "name": "Colab uncensored (abliterated)",
            "reasoning": False,
            "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": ctx,
            "maxTokens": 8192,
        }],
    }
    if MODELS_YML.exists():
        MODELS_YML.with_suffix(".yml.bak").write_text(MODELS_YML.read_text())
    MODELS_YML.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=False))
    print(f"[llm] registered provider '{PROVIDER_ID}' in {MODELS_YML}")


def register_fx(url: str, ctx: int) -> None:
    """Write a custom model connection into ~/.fx/settings.json."""
    if not shutil.which("fx"):
        print("[llm] fx not on PATH; skipping fx registration")
        return

    FX_SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if FX_SETTINGS.exists():
        try:
            data = json.loads(FX_SETTINGS.read_text()) or {}
        except json.JSONDecodeError as e:
            print(f"[llm] {FX_SETTINGS} is not valid JSON ({e}); leaving it untouched")
            return

    data.setdefault("providers", {})[PROVIDER_ID] = {
        "protocol": "openai-chat-completions",
        "base_url": f"{url}/v1",  # fx appends /chat/completions
        "auth": {"type": "none"},  # llama-server is keyless
        "model_metadata": {
            # Without context_window fx cannot compute its auto-compaction threshold.
            # supports_tool_use is verified against this endpoint, not assumed.
            MODEL_ALIAS: {
                "context_window": ctx,
                "max_output_tokens": 8192,
                "supports_tool_use": True,
            }
        },
    }
    data.setdefault("models", {})[PROVIDER_ID] = MODEL_ALIAS
    # Deliberately NOT touching data["provider"]: that would hijack the active provider
    # (currently the Codex subscription). Activate with `fx provider uncensored`.

    if FX_SETTINGS.exists():
        FX_SETTINGS.with_suffix(".json.bak").write_text(FX_SETTINGS.read_text())
    FX_SETTINGS.write_text(json.dumps(data, indent=2) + "\n")
    print(f"[llm] registered connection '{PROVIDER_ID}' in {FX_SETTINGS}")


def probe(url: str, timeout: int = 30) -> bool:
    """True when the tunnel answers /health with 200."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{url}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def poke_kernel(session: str) -> bool:
    """Run a trivial cell so Colab's idle timer sees an active kernel.

    Colab reclaims a runtime based on KERNEL activity. Serving HTTP on port 8000
    through the tunnel never touches the kernel, so a busy model still looks idle.
    Without this heartbeat the VM is reaped mid-session.
    """
    r = subprocess.run(["colab", "exec", "-s", session], input="print(1)\n",
                       capture_output=True, text=True, timeout=240)
    return r.returncode == 0


def watch(session: str, gpu: str, model: str, ctx: int, interval: int, register: bool) -> int:
    r = remote_bootstrap(session, model, ctx)
    url = r["url"]
    if register:
        register_omp(url, ctx)
        register_fx(url, ctx)
    print(f"\n[watch] serving {url}/v1  (heartbeat every {interval}s, Ctrl-C to stop)", flush=True)
    failures = 0
    while True:
        try:
            time.sleep(interval)
            alive = poke_kernel(session)
            healthy = probe(url)
            if alive and healthy:
                failures = 0
                print(f"[watch] ok  kernel=up  endpoint=up", flush=True)
                continue
            failures += 1
            print(f"[watch] degraded (kernel={'up' if alive else 'DOWN'}, "
                  f"endpoint={'up' if healthy else 'DOWN'}) — recovering", flush=True)
            if not session_exists(session):
                print("[watch] session was reclaimed; creating a new one", flush=True)
                colab("new", "-s", session, "--gpu", gpu, timeout=600)
            r = remote_bootstrap(session, model, ctx, force_tunnel=True)
            url = r["url"]
            if register:
                register_omp(url, ctx)
                register_fx(url, ctx)
            print(f"[watch] recovered: {url}/v1", flush=True)
        except KeyboardInterrupt:
            print("\n[watch] stopped; the VM is still running — stop it with "
                  f"'colab stop -s {session}'", flush=True)
            return 0
        except Exception as e:  # network hiccup, CLI error, etc.
            failures += 1
            print(f"[watch] error ({e}); retrying", flush=True)
            time.sleep(min(60, 5 * failures))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", default="llm")
    p.add_argument("--gpu", default="T4")
    p.add_argument("--model", default="9b", choices=sorted(MODELS))
    p.add_argument("--ctx", type=int, default=204800)
    p.add_argument("--stop", action="store_true", help="stop the session and exit")
    p.add_argument("--watch", action="store_true",
                   help="keep the VM alive with a kernel heartbeat and auto-recover when it dies")
    p.add_argument("--interval", type=int, default=300, help="heartbeat interval in seconds (default 300)")
    p.add_argument("--register", dest="register", action="store_true", default=True)
    p.add_argument("--no-register", dest="register", action="store_false")
    a = p.parse_args()

    if a.stop:
        print(colab("stop", "-s", a.session, timeout=300).stdout.strip() or "stopped")
        return 0

    if a.watch:
        ensure_session(a.session, a.gpu)
        return watch(a.session, a.gpu, a.model, a.ctx, a.interval, a.register)

    ensure_session(a.session, a.gpu)
    r = remote_bootstrap(a.session, a.model, a.ctx)

    print(f"\nBASE_URL={r['url']}/v1")
    print(f"MODEL={MODEL_ALIAS}")
    print(f"context={r['n_ctx']}  model={r['model_gib']} GiB ({r['download']})  ", end="")
    print(f"VRAM={r['vram']}\n")

    if a.register:
        register_omp(r["url"], a.ctx)
        register_fx(r["url"], a.ctx)
        print(f"\nexport OPENAI_BASE_URL={r['url']}/v1")
        print(f"export OPENAI_API_KEY=none")
        print(f"\n  omp --model {PROVIDER_ID}/{MODEL_ALIAS}   # directly")
        print(f"  fx provider {PROVIDER_ID}                # then just: fx")
    print(f"\nstop it with: {shlex.join([sys.argv[0], '--session', a.session, '--stop'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
