#!/usr/bin/env python3
"""Compare two tunnels to the same VM server, interleaving trials so both see the same network.

    bench/tunnel_bench.py cloudflare=https://x.trycloudflare.com ssh=http://127.0.0.1:8001 [--rounds 20]

The server is bench/tunnel_server.py (or llama-server for the /health rows).
"""
import argparse
import http.client
import json
import statistics
import time
from urllib.parse import urlparse


def conn(base: str) -> http.client.HTTPConnection:
    u = urlparse(base)
    cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    return cls(u.hostname, u.port, timeout=60)


def req(c, method, path, body=None):
    t = time.perf_counter()
    c.request(method, path, body=body, headers={"Content-Type": "application/octet-stream"} if body else {})
    r = c.getresponse()
    r.read()
    assert r.status == 200, r.status
    return time.perf_counter() - t


def stream(c, n, gap):
    t0 = time.perf_counter()
    c.request("GET", f"/stream?n={n}&gap={gap}")
    r = c.getresponse()
    first, last, worst, count = None, t0, 0.0, 0
    for line in r:
        if not line.startswith(b"data:"):
            continue
        now = time.perf_counter()
        first = first or now - t0
        worst = max(worst, now - last)
        last, count = now, count + 1
    assert count == n, count
    return first, last - t0, worst


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+", help="label=base_url")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--upload-kb", type=int, default=256)
    ap.add_argument("--blob-mb", type=float, default=16)
    a = ap.parse_args()
    targets = dict(t.split("=", 1) for t in a.targets)
    warm = {k: conn(v) for k, v in targets.items()}
    res = {k: {"cold": [], "warm": [], "upload": [], "ttfb": [], "stream": [], "gap": [], "dl": []}
           for k in targets}
    body = b"x" * (a.upload_kb * 1024)
    n, gap = 100, 0.02
    for i in range(a.rounds):
        for k, base in targets.items():
            c = conn(base)
            res[k]["cold"].append(req(c, "GET", "/health"))  # includes TCP+TLS handshake
            c.close()
            res[k]["warm"].append(req(warm[k], "GET", "/health"))
            res[k]["upload"].append(req(warm[k], "POST", "/echo", body))
            f, total, worst = stream(warm[k], n, gap)
            res[k]["ttfb"].append(f)
            res[k]["stream"].append(total)
            res[k]["gap"].append(worst)
            if i < 3:
                res[k]["dl"].append(a.blob_mb / req(warm[k], "GET", f"/blob?mb={a.blob_mb}"))
        print(f"round {i + 1}/{a.rounds}", flush=True)

    ms = lambda xs: f"{statistics.median(xs) * 1000:7.0f} / {pct(xs, 95) * 1000:7.0f}"
    print(f"\n{'metric (median / p95)':34}" + "".join(f"{k:>20}" for k in targets))
    rows = [
        ("health, new connection (ms)", "cold", ms),
        ("health, keep-alive (ms)", "warm", ms),
        (f"upload {a.upload_kb} KiB (ms)", "upload", ms),
        ("stream first chunk (ms)", "ttfb", ms),
        (f"stream {n} chunks, ideal {n * gap:.0f}s (ms)", "stream", ms),
        ("worst gap between chunks (ms)", "gap", ms),
        ("download (MiB/s, median)", "dl", lambda xs: f"{statistics.median(xs):17.1f}"),
    ]
    for label, key, fmt in rows:
        print(f"{label:34}" + "".join(f"{fmt(res[k][key]):>20}" for k in targets))
    print("\n" + json.dumps(res))


if __name__ == "__main__":
    main()
