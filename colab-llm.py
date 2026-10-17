#!/usr/bin/env python3
"""Start an uncensored llama.cpp model on a free Colab T4 and expose it as an OpenAI endpoint.

    ./colab-llm.py                 # reuse the running server (instant), else set it up, then print the URL
    ./colab-llm.py --status        # check the endpoint without touching Colab
    ./colab-llm.py --stop          # close the tunnel and stop the VM (idle VMs burn compute units)
    ./colab-llm.py --watch         # keep the VM alive and repair the tunnel/server/session when they die
    ./colab-llm.py --model 14b     # swap the served model (both stay cached on the VM)
    ./colab-llm.py --ctx 32768     # override the per-model context default
    ./colab-llm.py --think         # let the model reason before it answers (slower, default off)
    ./colab-llm.py --tunnel cloudflare   # public https URL instead of the local ssh tunnel
    ./colab-llm.py --no-register   # skip writing the omp/fx provider

State is remote: /content survives between runs inside one session, so a second invocation
reuses the downloaded llama.cpp build, the model weights, and the running server. If the
session is gone, everything is rebuilt from scratch.

The default tunnel is `colab ssh` with a local port forward, so the endpoint is always
http://127.0.0.1:PORT/v1: it never changes, and it is not public. The ssh master runs in the
background after this script exits; `--stop` closes it.

The script also writes/updates the `uncensored` provider so the omp and fx coding agents can
use the model: omp via ~/.omp/agent/models.yml (then `omp --model uncensored/abliterated`) and
fx via ~/.fx/settings.json (then `fx provider uncensored`). It writes them only on change.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

MODELS_YML = Path.home() / ".omp" / "agent" / "models.yml"
FX_SETTINGS = Path.home() / ".fx" / "settings.json"
STATE_DIR = Path.home() / ".cache" / "colab-llm"
# ssh control sockets must fit in sun_path (108 bytes), so keep them in the short runtime dir.
SOCK_DIR = Path(os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir())
PROVIDER_ID = "uncensored"
MODEL_ALIAS = "abliterated"
BUILD = "b11696"
CLOUDFLARED = "2026.10.0"
# f16, not q8_0: on a T4 a q8_0 cache generates 33% slower at 98k context (12.8 vs 19.3 tok/s)
# and saves VRAM that both models do not need at their defaults. See README "Benchmarks".
KV_TYPE = "f16"
REMOTE_PORT = 8000  # Colab itself owns 8080

# name -> hf url, approx file GiB, default context. Q4_K_M weights. Each default ctx is the largest
# that fits a T4 with >= 1 GiB free, keeps generation at 75% fill >= half the empty-context speed,
# and stays within the trained context (see README "Choosing the context size").
MODELS = {
    "9b": {
        "url": "https://huggingface.co/mradermacher/Huihui-Qwen3.5-9B-abliterated-GGUF/resolve/main/"
               "Huihui-Qwen3.5-9B-abliterated.Q4_K_M.gguf",
        "gib": 5.24,
        "ctx": 262144,  # trained context; Qwen3.5 hybrid attention keeps the KV cache small
    },
    "14b": {
        "url": "https://huggingface.co/mradermacher/Huihui-Qwen3-14B-abliterated-v2-GGUF/resolve/main/"
               "Huihui-Qwen3-14B-abliterated-v2.Q4_K_M.gguf",
        "gib": 8.38,
        "ctx": 32768,  # 40960 (trained) fits only with 0.5 GiB free in f16
    },
}

# Runs inside the Colab kernel. Parameters arrive as LLM_* env vars (colab exec --env), so this
# is plain Python with no template escaping. Lines starting with [vm] stream back as progress.
REMOTE = r'''
import json, os, re, subprocess, time, urllib.request

def _llm_main():
    E = os.environ
    BUILD, ALIAS, KV = E["LLM_BUILD"], E["LLM_ALIAS"], E["LLM_KV"]
    MODEL_URL, MIN_GIB, CTX = E["LLM_MODEL_URL"], float(E["LLM_MIN_GIB"]), int(E["LLM_CTX"])
    TUNNEL, PREFIX, PORT = E["LLM_TUNNEL"], E.get("LLM_PREFIX", ""), int(E["LLM_PORT"])
    FORCE_TUNNEL = E.get("LLM_FORCE_TUNNEL") == "1"
    CUDART = f"/content/llama/cudart-llama-{BUILD}-bin-ubuntu-cuda-12.8-x64"
    BIN = f"/content/llama/llama-{BUILD}"
    # /usr/lib64-nvidia holds libcuda.so.1. Assigning LD_LIBRARY_PATH instead of appending
    # drops it, libcuda resolves to "not found", and llama.cpp silently runs on CPU.
    LDP = f"/usr/lib64-nvidia:{CUDART}:{BIN}"
    MODEL_PATH = f"/content/models/{os.path.basename(MODEL_URL)}"
    LOGF = "/content/server.log"
    BASE = f"http://127.0.0.1:{PORT}{PREFIX}"
    t0 = time.time()

    def log(msg):
        print(f"[vm] {msg}", flush=True)

    def result(**kw):
        print("__RESULT__" + json.dumps(kw), flush=True)

    def sh(cmd):
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()

    def get(url, timeout=5):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read() if r.status == 200 else None
        except Exception:
            return None

    os.makedirs("/content/llama", exist_ok=True)
    os.makedirs("/content/models", exist_ok=True)

    # ---- model: start first, it is the long pole; everything else downloads meanwhile ----
    dl, part = "cached", MODEL_PATH + ".part"
    curl = None
    if not os.path.exists(MODEL_PATH):
        dl = "downloaded"
        # -C - resumes a .part left by an interrupted run; the rename marks a complete file.
        curl = subprocess.Popen(["curl", "-fsSL", "--retry", "5", "-C", "-", "-o", part, MODEL_URL])
        log(f"model: downloading {os.path.basename(MODEL_PATH)} (~{MIN_GIB:.1f} GiB)")

    # ---- llama.cpp ----
    if not os.path.exists(f"{BIN}/llama-server"):
        log(f"llama.cpp: downloading {BUILD}")
        base = f"https://github.com/ggml-org/llama.cpp/releases/download/{BUILD}"
        for name in (f"cudart-llama-{BUILD}-bin-ubuntu-cuda-12.8-x64.tar.gz",
                     f"llama-{BUILD}-bin-ubuntu-cuda-12.8-x64.tar.gz"):
            subprocess.run(f"cd /content/llama && curl -fsSL -O {base}/{name} && tar xzf {name} && rm {name}",
                           shell=True, check=True)
    env = dict(os.environ, LD_LIBRARY_PATH=LDP)
    dev = subprocess.run([f"{BIN}/llama-server", "--list-devices"], env=env,
                         capture_output=True, text=True).stdout
    if "CUDA0" not in dev:
        return result(error="no CUDA device; LD_LIBRARY_PATH broke or the VM has no GPU", devices=dev)

    # ---- cloudflared (only for the public tunnel) ----
    CF = "/content/cloudflared"
    if TUNNEL == "cloudflare" and not os.path.exists(CF):
        subprocess.run(f"curl -fsSL -o {CF} https://github.com/cloudflare/cloudflared/releases/download/"
                       f"{E['LLM_CLOUDFLARED']}/cloudflared-linux-amd64 && chmod +x {CF}",
                       shell=True, check=True)

    if curl:
        while curl.poll() is None:
            time.sleep(10)
            got = os.path.getsize(part) / 2**30 if os.path.exists(part) else 0
            log(f"model: {got:.1f} / ~{MIN_GIB:.1f} GiB ({time.time() - t0:.0f}s)")
        if curl.returncode:
            return result(error=f"model download failed (curl exit {curl.returncode}); rerun to resume")
        os.replace(part, MODEL_PATH)
    size_gib = os.path.getsize(MODEL_PATH) / 2**30
    if size_gib < MIN_GIB - 0.5:
        os.remove(MODEL_PATH)
        return result(error=f"model truncated: {size_gib:.2f} GiB; deleted, rerun to download again")

    # ---- server: restart only if the running one was started with different arguments ----
    args = [f"{BIN}/llama-server", "--model", MODEL_PATH, "--host", "127.0.0.1", "--port", str(PORT),
            "--alias", ALIAS, "--ctx-size", str(CTX), "--n-gpu-layers", "999", "--threads", "2",
            "--jinja", "--flash-attn", "on", "--cache-type-k", KV, "--cache-type-v", KV]
    if PREFIX:
        args += ["--api-prefix", PREFIX]
    if E.get("LLM_THINK") != "1":
        # Qwen3/3.5 think before each answer by default: ~5x slower replies in agent loops.
        # A request can still turn it on with chat_template_kwargs.
        args += ["--chat-template-kwargs", '{"enable_thinking":false}']
    running = sh("pgrep -a -f '^/content/llama/.*/llama-server'").splitlines()
    if not (len(running) == 1 and running[0].split(" ", 1)[1] == " ".join(args) and get(BASE + "/health")):
        if running:
            log("server: restarting with new arguments")
            subprocess.run("pkill -f '^/content/llama/.*/llama-server'", shell=True)
            for _ in range(20):  # the VRAM must be free before the new server allocates it
                if not sh("pgrep -f '^/content/llama/.*/llama-server'"):
                    break
                time.sleep(0.5)
            else:
                subprocess.run("pkill -9 -f '^/content/llama/.*/llama-server'", shell=True)
                time.sleep(1)
        with open(LOGF, "w") as out:
            proc = subprocess.Popen(args, env=env, stdout=out, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
        log(f"server: loading {os.path.basename(MODEL_PATH)} with ctx {CTX}")
        start, shown = time.time(), 0
        while time.time() - start < 600:
            if get(BASE + "/health", timeout=2):
                break
            if proc.poll() is not None:
                return result(error="server exited during load", log=open(LOGF, errors="ignore").read()[-1500:])
            if time.time() - start >= shown + 15:
                shown += 15
                log(f"server: still loading ({shown}s)")
            time.sleep(1)
        else:
            return result(error="server never became healthy", log=open(LOGF, errors="ignore").read()[-1500:])

    # ---- cloudflared tunnel ----
    url = None
    if TUNNEL != "cloudflare":
        # A tunnel left from an earlier --tunnel cloudflare run would now expose the server
        # without its secret prefix.
        subprocess.run("pkill -f '^/content/cloudflared tunnel'", shell=True)
    else:
        cflog = "/content/cf.log"
        pat = r"https://[a-z0-9-]+\.trycloudflare\.com"
        m = re.search(pat, open(cflog, errors="ignore").read()) if os.path.exists(cflog) else None
        # A live cloudflared PROCESS is not proof the tunnel WORKS, so recovery
        # (FORCE_TUNNEL) always mints a fresh one.
        if m and not FORCE_TUNNEL and sh("pgrep -f '^/content/cloudflared tunnel'"):
            url = m.group(0)
        else:
            subprocess.run("pkill -f '^/content/cloudflared tunnel'", shell=True)
            time.sleep(1)
            with open(cflog, "w") as out:
                subprocess.Popen([CF, "tunnel", "--url", f"http://127.0.0.1:{PORT}", "--no-autoupdate"],
                                 stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 start_new_session=True)
            for _ in range(120):
                time.sleep(0.5)
                m = re.search(pat, open(cflog, errors="ignore").read())
                if m:
                    url = m.group(0)
                    break
            if not url:
                return result(error="no tunnel URL", log=open(cflog, errors="ignore").read()[-800:])
            # Wait until the edge serves it. Probing a fresh hostname from the client too early
            # makes the local resolver cache NXDOMAIN, which then hides the tunnel for ~90s.
            log("tunnel: waiting for the Cloudflare edge")
            for _ in range(60):
                if get(url + PREFIX + "/health"):
                    break
                time.sleep(1)

    meta = (json.loads(get(BASE + "/v1/models") or b"{}").get("data") or [{}])[0].get("meta", {})
    result(ok=True, url=url, n_ctx=meta.get("n_ctx"), n_ctx_train=meta.get("n_ctx_train"),
           model_gib=round(size_gib, 2), download=dl,
           vram=sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader"))

_llm_main()
'''


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def colab(*args: str, timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["colab", *args], capture_output=True, text=True, timeout=timeout)


def session_exists(name: str) -> bool:
    r = colab("sessions", timeout=120)
    return bool(re.search(rf"^\[{re.escape(name)}\]", r.stdout, re.M))


def ensure_session(name: str, gpu: str) -> None:
    if session_exists(name):
        log(f"reusing session '{name}'")
        return
    log(f"creating session '{name}' ({gpu})")
    r = colab("new", "-s", name, "--gpu", gpu, timeout=600)
    if r.returncode != 0:
        out = r.stdout + r.stderr
        if "Service Unavailable" in out or "Too Many Requests" in out:
            raise RuntimeError(f"Colab has no {gpu} for this account right now (capacity or free-tier "
                               "usage limit). Retry later, or use --watch to keep retrying.")
        raise RuntimeError(f"colab new failed:\n{out[-1500:]}")


def model_file(model: str) -> str:
    return MODELS[model]["url"].rsplit("/", 1)[1]


def remote_bootstrap(a: argparse.Namespace, prefix: str, force_tunnel: bool = False) -> dict:
    m = MODELS[a.model]
    env = {
        "LLM_BUILD": BUILD, "LLM_ALIAS": MODEL_ALIAS, "LLM_KV": KV_TYPE, "LLM_MODEL_URL": m["url"],
        "LLM_MIN_GIB": str(m["gib"]), "LLM_CTX": str(a.ctx), "LLM_TUNNEL": a.tunnel,
        "LLM_PREFIX": prefix, "LLM_PORT": str(REMOTE_PORT), "LLM_CLOUDFLARED": CLOUDFLARED,
        "LLM_FORCE_TUNNEL": "1" if force_tunnel else "0", "LLM_THINK": "1" if a.think else "0",
    }
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(REMOTE)
        path = fh.name
    cmd = ["colab", "exec", "-s", a.session, "-f", path, "--timeout", "2400"]
    for k, v in env.items():
        cmd += ["--env", f"{k}={v}"]
    log("bootstrapping runtime (downloads on first use)")
    tail: list[str] = []
    try:
        # colab exec streams the kernel's output live, so progress shows up as it happens.
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as p:
            for line in p.stdout:
                line = line.rstrip("\n")
                if line.startswith("__RESULT__"):
                    res = json.loads(line[len("__RESULT__"):])
                    if "error" in res:
                        raise RuntimeError("bootstrap failed: " + "\n".join(
                            str(v) for v in (res["error"], res.get("log"), res.get("devices")) if v))
                    return res
                if line.startswith("[vm] "):
                    log(line)
                tail = (tail + [line])[-40:]
    finally:
        os.unlink(path)
    raise RuntimeError("bootstrap failed:\n" + "\n".join(tail))


# ---- ssh tunnel: one background ssh master per session, addressed by its control socket ----

def ssh(session: str, *opts: str) -> list[str]:
    """ssh argv for this session's master; options must precede the host or ssh runs them remotely."""
    return ["ssh", "-S", str(SOCK_DIR / f"colab-llm-{session}.sock"),
            "-o", f"ProxyCommand=colab ssh --proxy-mode -s {shlex.quote(session)}",
            # Every VM has a new host key, and the proxy already authenticates through Google.
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR", "-o", "BatchMode=yes", *opts, f"root@colab-{session}"]


def tunnel_up(session: str) -> bool:
    return subprocess.run(ssh(session, "-O", "check"), capture_output=True).returncode == 0


def stop_tunnel(session: str) -> None:
    subprocess.run(ssh(session, "-O", "exit"), capture_output=True)


def start_tunnel(session: str, port: int) -> None:
    stop_tunnel(session)
    # -f returns once the forward is listening. ServerAlive drops a dead link after ~45s,
    # so a stale master never blocks the next run. The forked master keeps any pipe open,
    # so its stderr goes to a file, not to capture_output.
    with tempfile.TemporaryFile("w+") as err:
        r = subprocess.run(ssh(session, "-M", "-f", "-N", "-o", "ExitOnForwardFailure=yes",
                               "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
                               "-L", f"127.0.0.1:{port}:127.0.0.1:{REMOTE_PORT}"),
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=err, timeout=120)
        err.seek(0)
        if r.returncode != 0:
            raise RuntimeError(f"ssh tunnel failed: {err.read().strip()}")


def props(base: str, timeout: int = 15) -> dict | None:
    """llama-server /props, or None when the endpoint does not answer."""
    try:
        with urllib.request.urlopen(f"{base}/props", timeout=timeout) as r:
            return json.load(r) if r.status == 200 else None
    except Exception:
        return None


def matches(p: dict | None, a: argparse.Namespace) -> bool:
    return bool(p) and (os.path.basename(p.get("model_path", "")) == model_file(a.model)
                        and p.get("default_generation_settings", {}).get("n_ctx") == a.ctx)


# ---- local state: the endpoint base and the cloudflare path secret ----

def load_state(session: str) -> dict:
    try:
        return json.loads((STATE_DIR / f"{session}.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(session: str, st: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / f"{session}.json").write_text(json.dumps(st))


# ---- agent configs ----

def write_config(path: Path, text: str) -> None:
    """Atomic write that keeps the file mode and backs up the user's original once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    if path.exists():
        bak = path.with_name(path.name + ".bak")
        if not bak.exists():
            shutil.copy2(path, bak)
        shutil.copymode(path, tmp)
    os.replace(tmp, path)


def register_omp(url: str, ctx: int, think: bool) -> None:
    if yaml is None:
        log("PyYAML is not installed; skipping omp registration (pip install pyyaml)")
        return
    data = yaml.safe_load(MODELS_YML.read_text()) or {} if MODELS_YML.exists() else {}
    providers = data.setdefault("providers", {})
    entry = {
        "baseUrl": url,
        "api": "openai-completions",
        "auth": "none",  # llama-server is keyless; the ssh tunnel is local, cloudflare uses a secret path
        "discovery": {"type": "openai-models-list", "injectV1": False},  # baseUrl already ends in /v1
        "models": [{
            "id": MODEL_ALIAS,
            "name": "Colab uncensored (abliterated)",
            "reasoning": think,
            "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": ctx,
            "maxTokens": 8192,
        }],
    }
    if providers.get(PROVIDER_ID) == entry:
        return
    providers[PROVIDER_ID] = entry
    # Note: safe_dump drops comments in models.yml. Writing only on change keeps that rare.
    write_config(MODELS_YML, yaml.safe_dump(data, sort_keys=False, default_flow_style=False))
    log(f"registered provider '{PROVIDER_ID}' in {MODELS_YML}")


def register_fx(url: str, ctx: int) -> None:
    """Write a custom model connection into ~/.fx/settings.json."""
    if not shutil.which("fx"):
        log("fx not on PATH; skipping fx registration")
        return
    data: dict = {}
    if FX_SETTINGS.exists():
        try:
            data = json.loads(FX_SETTINGS.read_text()) or {}
        except json.JSONDecodeError as e:
            log(f"{FX_SETTINGS} is not valid JSON ({e}); leaving it untouched")
            return
    entry = {
        "protocol": "openai-chat-completions",
        "base_url": url,  # fx appends /chat/completions
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
    if data.get("providers", {}).get(PROVIDER_ID) == entry and data.get("models", {}).get(PROVIDER_ID) == MODEL_ALIAS:
        return
    data.setdefault("providers", {})[PROVIDER_ID] = entry
    data.setdefault("models", {})[PROVIDER_ID] = MODEL_ALIAS
    # Deliberately NOT touching data["provider"]: that would hijack the active provider
    # that the user already chose. Activate with `fx provider uncensored`.
    write_config(FX_SETTINGS, json.dumps(data, indent=2) + "\n")
    log(f"registered connection '{PROVIDER_ID}' in {FX_SETTINGS}")


# ---- the whole bring-up ----

def serve(a: argparse.Namespace, force_tunnel: bool = False) -> tuple[str, dict]:
    """Make the endpoint answer with the requested model and ctx. Returns (base, bootstrap info)."""
    st = load_state(a.session)
    if a.tunnel == "ssh":
        base = f"http://127.0.0.1:{a.port}"
        # Fast path: the tunnel and server already match, so no Colab call at all.
        if (not force_tunnel and st.get("base") == base and st.get("think") == a.think
                and tunnel_up(a.session) and matches(props(base), a)):
            return base, {}
        ensure_session(a.session, a.gpu)
        info = remote_bootstrap(a, prefix="")
        if not (tunnel_up(a.session) and props(base)):
            log(f"ssh tunnel: forwarding 127.0.0.1:{a.port}")
            start_tunnel(a.session, a.port)
    else:
        prefix = st.get("prefix") or "/" + secrets.token_urlsafe(16)
        if (not force_tunnel and st.get("tunnel") == "cloudflare" and st.get("think") == a.think
                and matches(props(st.get("base", "")), a)):
            return st["base"], {}
        ensure_session(a.session, a.gpu)
        info = remote_bootstrap(a, prefix=prefix, force_tunnel=force_tunnel)
        base = info["url"] + prefix
        st["prefix"] = prefix
    if not props(base, timeout=30):
        raise RuntimeError(f"the server is up on the VM, but {base} does not answer")
    save_state(a.session, {**st, "tunnel": a.tunnel, "base": base, "think": a.think})
    if a.register:
        register_omp(f"{base}/v1", info.get("n_ctx") or a.ctx, a.think)
        register_fx(f"{base}/v1", info.get("n_ctx") or a.ctx)
    return base, info


def report(a: argparse.Namespace, base: str, info: dict) -> None:
    print(f"\nBASE_URL={base}/v1")
    print(f"MODEL={MODEL_ALIAS}")
    if info:
        train = info.get("n_ctx_train")
        print(f"context={info['n_ctx']} (trained {train})  model={info['model_gib']} GiB ({info['download']})  "
              f"VRAM={info['vram']}")
        if train and info["n_ctx"] and info["n_ctx"] > train:
            print(f"warning: ctx {info['n_ctx']} exceeds the model's trained context {train}; "
                  "quality drops past it")
        notify(f"{base}/v1")
    else:
        print("(already running; nothing changed)")
    print(f"\nexport OPENAI_BASE_URL={base}/v1")
    print("export OPENAI_API_KEY=none")
    if a.register:
        print(f"\n  omp --model {PROVIDER_ID}/{MODEL_ALIAS}   # directly")
        print(f"  fx provider {PROVIDER_ID}                # then just: fx")
    print(f"\nstop it with: {shlex.join([sys.argv[0], '--session', a.session, '--stop'])}")


def notify(url: str) -> None:
    """Desktop ping and clipboard copy: the first bring-up takes minutes."""
    # wl-copy forks to serve the clipboard, so a captured pipe would never close.
    quiet = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if shutil.which("wl-copy"):
        subprocess.run(["wl-copy", url], **quiet)
    if shutil.which("notify-send"):
        subprocess.run(["notify-send", "-a", "colab-llm", "Colab LLM ready", f"{url} (copied)"], **quiet)


def poke_kernel(session: str) -> bool:
    """Run a trivial cell so Colab's idle timer sees an active kernel.

    Colab reclaims a runtime based on KERNEL activity. Serving HTTP on port 8000
    through the tunnel never touches the kernel, so a busy model still looks idle.
    Without this heartbeat the VM is reaped mid-session.
    """
    r = subprocess.run(["colab", "exec", "-s", session], input="print(1)\n",
                       capture_output=True, text=True, timeout=240)
    return r.returncode == 0


def watch(a: argparse.Namespace) -> int:
    failures = 0
    while True:  # first bring-up retries too, so --watch also waits out "no GPU available"
        try:
            base, info = serve(a)
            report(a, base, info)
            break
        except RuntimeError as e:
            failures += 1
            log(f"[watch] {e}; retrying in {min(300, 30 * failures)}s")
            time.sleep(min(300, 30 * failures))
    log(f"[watch] serving {base}/v1  (heartbeat every {a.interval}s, Ctrl-C to stop)")
    strikes = failures = 0
    while True:
        try:
            # After a failed check, look again soon instead of a full interval.
            time.sleep(a.interval if strikes == 0 else 30)
            alive = poke_kernel(a.session)
            healthy = props(base) is not None
            if alive and not healthy and a.tunnel == "ssh":
                log("[watch] endpoint down, kernel up: reconnecting the ssh tunnel")
                start_tunnel(a.session, a.port)
                healthy = props(base) is not None
            if alive and healthy:
                strikes = failures = 0
                log("[watch] ok  kernel=up  endpoint=up")
                continue
            strikes += 1
            log(f"[watch] degraded (kernel={'up' if alive else 'DOWN'}, "
                f"endpoint={'up' if healthy else 'DOWN'}), strike {strikes}/2")
            if strikes < 2:  # one slow probe is not worth a new VM or a new URL
                continue
            # Only cloudflare needs a fresh tunnel; ssh reconnects to the same local port.
            new_base, _ = serve(a, force_tunnel=a.tunnel == "cloudflare")
            strikes = 0
            log(f"[watch] recovered: {new_base}/v1" + ("" if new_base == base else "  (URL changed)"))
            base = new_base
        except (RuntimeError, subprocess.SubprocessError, OSError) as e:
            failures += 1
            log(f"[watch] error ({e}); retrying")
            time.sleep(min(60, 5 * failures))


def stop(a: argparse.Namespace) -> int:
    stop_tunnel(a.session)
    r = colab("stop", "-s", a.session, timeout=300)
    print((r.stdout + r.stderr).strip() or "stopped")
    (STATE_DIR / f"{a.session}.json").unlink(missing_ok=True)
    return r.returncode


def status(a: argparse.Namespace) -> int:
    base = load_state(a.session).get("base")
    p = props(base) if base else None
    if not p:
        print(f"down ({base or 'never started'})")
        return 1
    print(f"up  {base}/v1  model={os.path.basename(p.get('model_path', '?'))}  "
          f"ctx={p.get('default_generation_settings', {}).get('n_ctx')}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", default="llm")
    p.add_argument("--gpu", default="T4")
    p.add_argument("--model", default="9b", choices=sorted(MODELS))
    p.add_argument("--ctx", type=int, help="context size (default: per model, see MODELS)")
    p.add_argument("--tunnel", default="ssh", choices=["ssh", "cloudflare"],
                   help="ssh: stable local URL (default); cloudflare: public https URL with a secret path")
    p.add_argument("--port", type=int, default=8765, help="local port for the ssh tunnel (default 8765)")
    p.add_argument("--status", action="store_true", help="check the endpoint without calling Colab")
    p.add_argument("--stop", action="store_true", help="close the tunnel, stop the session, and exit")
    p.add_argument("--watch", action="store_true",
                   help="keep the VM alive with a kernel heartbeat and auto-recover when it dies")
    p.add_argument("--stop-on-exit", action="store_true", help="with --watch: stop the VM on Ctrl-C")
    p.add_argument("--interval", type=int, default=300, help="heartbeat interval in seconds (default 300)")
    p.add_argument("--think", action=argparse.BooleanOptionalAction, default=False,
                   help="let the model think before it answers: better on hard problems, ~5x slower "
                        "(default off)")
    p.add_argument("--register", action=argparse.BooleanOptionalAction, default=True,
                   help="write the omp/fx provider (default on)")
    a = p.parse_args()
    a.ctx = a.ctx or MODELS[a.model]["ctx"]

    if a.stop:
        return stop(a)
    if a.status:
        return status(a)
    try:
        if a.watch:
            return watch(a)
        base, info = serve(a)
        report(a, base, info)
        return 0
    except KeyboardInterrupt:
        if a.watch and a.stop_on_exit:
            print()
            return stop(a)
        print(f"\ninterrupted; the VM is still running. Stop it with "
              f"'{shlex.join([sys.argv[0], '--session', a.session, '--stop'])}'")
        return 130
    except (RuntimeError, subprocess.SubprocessError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    sys.exit(main())
