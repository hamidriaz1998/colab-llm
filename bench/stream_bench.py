#!/usr/bin/env python3
"""Streamed chat completions through each tunnel, interleaved, against the same llama-server.

    bench/stream_bench.py cloudflare=https://x.trycloudflare.com/v1 ssh=http://127.0.0.1:8001/v1 [--rounds 10]

TTFT = time to the first content token as the client sees it. Server-side timings come from
llama-server's `timings` field, so "client tok/s" vs "server tok/s" isolates the tunnel.
"""
import argparse
import http.client
import json
import statistics
import time
from urllib.parse import urlparse


def chat(base: str, max_tokens: int, key: str | None):
    u = urlparse(base)
    cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    c = cls(u.hostname, u.port, timeout=300)
    body = json.dumps({
        "model": "x", "stream": True, "temperature": 0, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": "Count from 1 to 500, one number per line."}],
    })
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    t0 = time.perf_counter()
    c.request("POST", u.path.rstrip("/") + "/chat/completions", body, headers)
    r = c.getresponse()
    assert r.status == 200, (r.status, r.read()[:300])
    first, last, worst, n, timings = None, None, 0.0, 0, {}
    for line in r:
        if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
            continue
        d = json.loads(line[6:])
        timings = d.get("timings", timings)
        if not d.get("choices") or not d["choices"][0]["delta"].get("content"):
            continue
        now = time.perf_counter()
        if first is None:
            first = now - t0
        else:
            worst = max(worst, now - last)
        last, n = now, n + 1
    c.close()
    gen_s = last - t0 - first
    return {"ttft": first, "total": last - t0, "gap": worst, "tokens": n,
            "client_tps": (n - 1) / gen_s if gen_s > 0 else 0,
            "server_tps": timings.get("predicted_per_second", 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+", help="label=base_url (ending in /v1)")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--key")
    a = ap.parse_args()
    targets = dict(t.split("=", 1) for t in a.targets)
    res = {k: [] for k in targets}
    for k, v in targets.items():  # warm the prompt cache so every measured round is equal
        chat(v, 8, a.key)
    for i in range(a.rounds):
        for k, v in targets.items():
            res[k].append(chat(v, a.max_tokens, a.key))
        print(f"round {i + 1}/{a.rounds}", flush=True)
    med = lambda k, f: statistics.median(r[f] for r in res[k])
    print(f"\n{'median':24}" + "".join(f"{k:>14}" for k in targets))
    for label, f, scale, fmt in [("TTFT (ms)", "ttft", 1000, ".0f"), ("total (ms)", "total", 1000, ".0f"),
                                 ("worst token gap (ms)", "gap", 1000, ".0f"),
                                 ("client tok/s", "client_tps", 1, ".1f"),
                                 ("server tok/s", "server_tps", 1, ".1f"), ("tokens", "tokens", 1, ".0f")]:
        print(f"{label:24}" + "".join(f"{med(k, f) * scale:>14{fmt}}" for k in targets))
    print("\n" + json.dumps(res))


if __name__ == "__main__":
    main()
