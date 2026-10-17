#!/usr/bin/env python3
"""Render the README charts as animated SVGs, one light and one dark file per chart.

    python3 docs/make_charts.py        # reads bench/results/, writes docs/img/

Standard library only. Measured numbers come from bench/results/; the few values that
were measured interactively (reconnect times, thinking on/off) are constants below.
The animations are CSS inside each SVG (GitHub plays them in <img>) and switch off
under prefers-reduced-motion, where every element is already in its final state.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "bench" / "results"
OUT = ROOT / "docs" / "img"

# Measured by hand during the benchmarks (see README "Tunnel performance" / "Thinking").
RECONNECT_S = {"ssh": 3.6, "cloudflare": 15.0}
THINK = {"off": (6.6, 149), "on": (32.0, 766)}

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
THEMES = {
    "light": {
        "surface": "#fcfcfb", "border": "rgba(11,11,11,0.10)", "ink": "#0b0b0b", "ink2": "#52514e",
        "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7", "gray": "#b9b7af",
        "s": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], "good": "#0ca30c", "goodtext": "#006300",
        "danger": "#d03b3b", "track": "#efeee9",
    },
    "dark": {
        "surface": "#1a1a19", "border": "rgba(255,255,255,0.10)", "ink": "#f0efec", "ink2": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835", "gray": "#55544f",
        "s": ["#3987e5", "#d95926", "#199e70", "#c98500"], "good": "#0ca30c", "goodtext": "#0ca30c",
        "danger": "#d03b3b", "track": "#262624",
    },
}

CSS = """
text { font-family: %(font)s; font-variant-numeric: tabular-nums; }
.t1 { font-size: 22px; font-weight: 650; fill: %(ink)s; letter-spacing: -0.01em; }
.t2 { font-size: 14px; fill: %(ink2)s; }
.ax { font-size: 13px; fill: %(muted)s; }
.lb { font-size: 13px; fill: %(ink2)s; }
.lbb { font-size: 13px; fill: %(ink)s; font-weight: 600; }
.big { font-size: 40px; font-weight: 650; fill: %(ink)s; letter-spacing: -0.02em; font-variant-numeric: normal; }
.fn { font-size: 12px; fill: %(muted)s; }
.draw { stroke-dasharray: 1; animation: draw 1.6s cubic-bezier(.22,.61,.36,1) both; }
.pop { transform-box: fill-box; transform-origin: center; animation: pop .45s cubic-bezier(.34,1.56,.64,1) both; }
.fade { animation: fade .7s ease-out both; }
.grow { transform-box: fill-box; transform-origin: left center; animation: grow 1s cubic-bezier(.22,.61,.36,1) both; }
@keyframes draw { from { stroke-dashoffset: 1; } }
@keyframes pop { from { opacity: 0; transform: scale(.3); } }
@keyframes fade { from { opacity: 0; } }
@keyframes grow { from { transform: scaleX(0); } }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; } }
"""


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, cls="lb", anchor="start", extra=""):
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}" {extra}>{esc(s)}</text>'


def card(w, h, t, title, subtitle, body, desc):
    css = CSS % {**t, "font": FONT}
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-labelledby="title desc">'
            f'<title id="title">{esc(title)}</title><desc id="desc">{esc(desc)}</desc>'
            f'<style>{css}</style>'
            f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="16" fill="{t["surface"]}" '
            f'stroke="{t["border"]}"/>'
            + text(40, 52, title, "t1") + text(40, 78, subtitle, "t2") + body + "</svg>")


def monotone(pts):
    """Monotone cubic (Fritsch-Carlson) path: smooth, passes through every point, no overshoot."""
    if len(pts) < 3:
        return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    n = len(pts)
    d = [(pts[i + 1][1] - pts[i][1]) / (pts[i + 1][0] - pts[i][0]) for i in range(n - 1)]
    m = [d[0]] + [0.0 if d[i - 1] * d[i] <= 0 else (d[i - 1] + d[i]) / 2 for i in range(1, n - 1)] + [d[-1]]
    for i in range(n - 1):
        if d[i] == 0:
            m[i] = m[i + 1] = 0
        else:
            a, b = m[i] / d[i], m[i + 1] / d[i]
            s = a * a + b * b
            if s > 9:
                k = 3 / s ** 0.5
                m[i], m[i + 1] = k * a * d[i], k * b * d[i]
    out = f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"
    for i in range(n - 1):
        (x0, y0), (x1, y1) = pts[i], pts[i + 1]
        h = (x1 - x0) / 3
        out += f" C{x0 + h:.1f},{y0 + m[i] * h:.1f} {x1 - h:.1f},{y1 - m[i + 1] * h:.1f} {x1:.1f},{y1:.1f}"
    return out


def k(n):
    return "0" if n == 0 else f"{round(n / 1024)}K"


# ---------------------------------------------------------------- data

def jsonl(name):
    return [json.loads(line) for line in (RES / name).read_text().splitlines() if line.strip()]


def bench_avg(name):
    """(type_k, depth) -> {'tg': tok/s, 'pp': tok/s}, averaged over repeated runs."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in jsonl(name):
        acc[(r["type_k"], r["n_depth"])]["pp" if r["n_prompt"] else "tg"].append(r["avg_ts"])
    return {key: {m: statistics.mean(v) for m, v in d.items()} for key, d in acc.items()}


def needle():
    rows = []
    for line in (RES / "needle-9b.txt").read_text().splitlines()[1:]:
        tok, depth, res, pp, tg, wall = line.split()
        rows.append({"tokens": int(tok), "depth": float(depth), "pass": res == "PASS",
                     "pp": float(pp), "tg": float(tg), "wall": float(wall)})
    return rows


def tunnel():
    lines = (RES / "tunnel-synthetic-cpu.txt").read_text().splitlines()
    syn = json.loads(next(line for line in lines if line.startswith("{")))
    stream = json.loads(next(line for line in (RES / "stream-tiny-cpu.txt").read_text().splitlines()
                             if line.startswith("{")))
    med = lambda xs: statistics.median(xs) * 1000
    return {
        "Time to first token": {k_: med([r["ttft"] for r in stream[k_]]) for k_ in ("ssh", "cloudflare")},
        "Health check, new connection": {k_: med(syn[k_]["cold"]) for k_ in ("ssh", "cloudflare")},
        "Health check, kept-alive": {k_: med(syn[k_]["warm"]) for k_ in ("ssh", "cloudflare")},
    }


# ---------------------------------------------------------------- charts

def chart_speed(t):
    ab9, ab14 = bench_avg("ab-9b.jsonl"), bench_avg("ab-14b.jsonl")
    main9, main14 = bench_avg("speed-9b.jsonl"), bench_avg("speed-14b.jsonl")
    nd = {r["tokens"]: r["tg"] for r in needle() if r["depth"] in (0.5, 0.25)}
    n200 = statistics.mean(r["tg"] for r in needle() if r["tokens"] > 190000)
    # (depth, tok/s, interleaved?)
    f16 = [(d, ab9[("f16", d)]["tg"], True) for d in (0, 32768, 98304)] + \
          [(128007, nd[128007], False), (199959, n200, False)]
    q8 = [(d, ab9[("q8_0", d)]["tg"], True) for d in (0, 32768, 98304)] + \
         [(d, main9[("q8_0", d)]["tg"], False) for d in (131072, 196608)]
    b14 = [(d, ab14[("f16", d)]["tg"], True) for d in (0, 16384)] + [(24576, main14[("f16", 24576)]["tg"], False)]

    W, H = 960, 600
    L, R, T, B = 72, 190, 130, 470
    X = lambda v: L + v / 262144 * (W - L - R)
    Y = lambda v: B - v / 30 * (B - T)
    s = t["s"]
    body = []
    for v in (0, 10, 20, 30):
        body.append(f'<line x1="{L}" x2="{W - R + 30}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" '
                    f'stroke="{t["axis"] if v == 0 else t["grid"]}" stroke-width="1"/>')
        body.append(text(L - 14, Y(v) + 4.5, str(v), "ax", "end"))
    for v in (0, 65536, 131072, 196608, 262144):
        body.append(text(X(v), B + 26, k(v), "ax", "middle"))
    body.append(text((L + W - R) / 2, B + 52, "Tokens already in the context", "ax", "middle"))
    body.append(text(40, T - 20, "tokens / s", "ax"))

    # the rule: stay above half the empty-context speed
    half = f16[0][1] / 2
    body.append(f'<line x1="{L}" x2="{X(262144):.1f}" y1="{Y(half):.1f}" y2="{Y(half):.1f}" '
                f'stroke="{t["muted"]}" stroke-width="1" opacity=".55" class="fade" style="animation-delay:1.9s"/>')
    body.append('<g class="fade" style="animation-delay:2s">'
                + text(X(262144), Y(half) + 18, "half of the empty-context speed", "fn", "end") + "</g>")

    gid = f"wash-{id(t)}"
    pts_f16 = [(X(d), Y(v)) for d, v, _ in f16]
    area = monotone(pts_f16) + f" L{pts_f16[-1][0]:.1f},{B} L{pts_f16[0][0]:.1f},{B} Z"
    body.append(f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
                f'<stop offset="0" stop-color="{s[0]}" stop-opacity=".16"/>'
                f'<stop offset="1" stop-color="{s[0]}" stop-opacity="0"/></linearGradient></defs>')
    body.append(f'<path d="{area}" fill="url(#{gid})" class="fade" style="animation-delay:.9s"/>')

    series = [("9b · f16 KV (default)", f16, s[0], 0.0), ("9b · q8_0 KV", q8, s[1], 0.25),
              ("14b · f16 KV", b14, s[2], 0.5)]
    for name, data, col, delay in series:
        pts = [(X(d), Y(v)) for d, v, _ in data]
        body.append(f'<path d="{monotone(pts)}" fill="none" stroke="{col}" stroke-width="2.5" '
                    f'stroke-linecap="round" stroke-linejoin="round" pathLength="1" class="draw" '
                    f'style="animation-delay:{delay}s"/>')
        for i, ((x, y), (_, _, solid)) in enumerate(zip(pts, data)):
            # solid: series fill with a 2px surface ring; hollow: surface fill with a series ring
            fill, ring = (col, t["surface"]) if solid else (t["surface"], col)
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{fill}" stroke="{ring}" '
                        f'stroke-width="2" class="pop" style="animation-delay:{delay + 0.25 + i * 0.22:.2f}s"/>')
        ex, ey = pts[-1]
        val = data[-1][1]
        lx, ly0 = (ex - 6, ey + 26) if data is b14 else (ex + 14, ey - 2)
        body.append(f'<g class="fade" style="animation-delay:{delay + 1.4:.2f}s">'
                    + text(lx, ly0, name, "lbb") + text(lx, ly0 + 17, f"{val:.1f} tok/s", "lb") + "</g>")

    # call-out: the f16 vs q8_0 gap at 98K
    xg = X(98304)
    a, b = Y(ab9[("f16", 98304)]["tg"]), Y(ab9[("q8_0", 98304)]["tg"])
    gain = ab9[("f16", 98304)]["tg"] / ab9[("q8_0", 98304)]["tg"] - 1
    bx = xg + 16
    body.append(f'<g class="fade" style="animation-delay:2.3s">'
                f'<line x1="{bx:.1f}" x2="{bx:.1f}" y1="{a + 8:.1f}" y2="{b - 8:.1f}" '
                f'stroke="{t["ink2"]}" stroke-width="1.25"/>'
                f'<line x1="{bx - 5:.1f}" x2="{bx + 5:.1f}" y1="{a + 8:.1f}" y2="{a + 8:.1f}" stroke="{t["ink2"]}" stroke-width="1.25"/>'
                f'<line x1="{bx - 5:.1f}" x2="{bx + 5:.1f}" y1="{b - 8:.1f}" y2="{b - 8:.1f}" stroke="{t["ink2"]}" stroke-width="1.25"/>'
                + text(bx + 12, (a + b) / 2 - 3, f"+{gain:.0%} faster", "lbb")
                + text(bx + 12, (a + b) / 2 + 14, "with an f16 cache", "lb") + "</g>")

    # legend for the two marker kinds (series identity is direct-labeled)
    ly = H - 30
    body.append(f'<circle cx="{L + 6}" cy="{ly - 4}" r="5" fill="{t["ink2"]}"/>'
                + text(L + 18, ly, "interleaved A/B runs", "fn")
                + f'<circle cx="{L + 182}" cy="{ly - 4}" r="4" fill="{t["surface"]}" stroke="{t["ink2"]}" stroke-width="2"/>'
                + text(L + 194, ly, "single runs (needle test, back-to-back bench)", "fn")
                + text(W - 40, ly, "Colab T4 · llama.cpp b11696 · flash attention on", "fn", "end"))
    return card(W, H, t, "Generation speed as the context fills",
                "The default keeps writing at 15 tokens/s with 200K tokens in its memory.", "".join(body),
                "Line chart of generation speed against context depth for 9b with f16 and q8_0 KV cache "
                "and for 14b. 9b f16 falls from 25 to 15.5 tok/s at 200K; q8_0 falls to 8 tok/s.")


def chart_vram(t):
    fit = jsonl("fit.jsonl")
    W, H = 960, 590
    L, R, T, B = 72, 200, 130, 460
    xs = [16384, 32768, 65536, 131072, 262144]
    import math
    X = lambda v: L + (math.log2(v) - 14) / (18 - 14) * (W - L - R)
    lo, hi = 6, 15.5
    Y = lambda v: B - (v - lo) / (hi - lo) * (B - T)
    s = t["s"]
    body = []
    cap = 15360 / 1024
    # safety margin band: the last GiB below the ceiling
    body.append(f'<rect x="{L}" y="{Y(cap):.1f}" width="{W - R + 30 - L}" height="{Y(cap - 1) - Y(cap):.1f}" '
                f'fill="{t["danger"]}" opacity=".08"/>')
    for v in (6, 8, 10, 12, 14):
        body.append(f'<line x1="{L}" x2="{W - R + 30}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{t["grid"]}"/>')
        body.append(text(L - 14, Y(v) + 4.5, f"{v}", "ax", "end"))
    body.append(f'<line x1="{L}" x2="{W - R + 30}" y1="{Y(cap):.1f}" y2="{Y(cap):.1f}" stroke="{t["danger"]}" stroke-width="1.5"/>')
    body.append(text(W - R + 30, Y(cap) - 8, "T4 limit · 15 GiB", "lbb", "end"))
    body.append(text(W - R + 30, Y(cap - 1) - 8, "keep 1 GiB free", "fn", "end"))
    body.append(text(40, T - 20, "GiB used", "ax"))
    for v in xs:
        body.append(text(X(v), B + 26, k(v), "ax", "middle"))
    body.append(text((L + W - R) / 2, B + 52, "Context size (log scale)", "ax", "middle"))

    combos = [("9b", "f16", s[0], "9b · f16"), ("9b", "q8_0", s[1], "9b · q8_0"),
              ("14b", "f16", s[2], "14b · f16"), ("14b", "q8_0", s[3], "14b · q8_0")]
    defaults = {("9b", "f16"): 262144, ("14b", "f16"): 32768}
    for i, (m, kv, col, name) in enumerate(combos):
        rows = sorted((r["ctx"], int(r["mem"].split(",")[0]) / 1024) for r in fit if r["model"] == m and r["kv"] == kv)
        pts = [(X(c), Y(g)) for c, g in rows]
        delay = i * 0.2
        body.append(f'<path d="{monotone(pts)}" fill="none" stroke="{col}" stroke-width="2.5" stroke-linecap="round" '
                    f'pathLength="1" class="draw" style="animation-delay:{delay}s"/>')
        for j, ((x, y), (c, g)) in enumerate(zip(pts, rows)):
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{col}" stroke="{t["surface"]}" stroke-width="2" '
                        f'class="pop" style="animation-delay:{delay + .3 + j * .18:.2f}s"/>')
            if defaults.get((m, kv)) == c:
                body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="11" fill="none" stroke="{t["ink"]}" stroke-width="1.5" '
                            f'class="pop" style="animation-delay:2.1s"/>')
        ex, ey = pts[-1]
        # 9b lines end at the right edge; 14b lines end mid-chart, so label them above their end
        if m == "9b":
            body.append(f'<g class="fade" style="animation-delay:{delay + 1.3:.2f}s">'
                        + text(ex + 16, ey + 5, name, "lbb") + text(ex + 16, ey + 22, f"{rows[-1][1]:.1f} GiB", "lb") + "</g>")
        else:
            body.append(f'<g class="fade" style="animation-delay:{delay + 1.3:.2f}s">'
                        + text(ex + 14, ey + 5 + (10 if kv == "q8_0" else 0), name, "lbb") + "</g>")
    # default call-outs
    x9, y9 = X(262144), Y(13693 / 1024)
    x14, y14 = X(32768), Y(13583 / 1024)
    body.append(f'<g class="fade" style="animation-delay:2.3s">'
                + text(x9 - 16, y9 + 30, "default: full 256K context", "lb", "end")
                + text(x14 - 16, y14 - 16, "default 32K", "lb", "end") + "</g>")
    body.append(text(40, H - 30, "Measured after llama-server loads, with flash attention. Rings mark the defaults.", "fn"))
    return card(W, H, t, "How much context fits on a T4",
                "Qwen3.5's hybrid attention keeps the 9b cache small enough for its full 256K context.",
                "".join(body),
                "Line chart of GPU memory against context size for each model and KV cache type, with the "
                "15 GiB T4 limit. 9b f16 uses 13.4 GiB at 256K; 14b f16 reaches 14.5 GiB at 40K.")


def chart_tunnel(t):
    data = tunnel()
    W, H = 960, 490
    L, T = 300, 156
    bar, gap, group = 18, 6, 74
    maxv = 1000
    X = lambda v: L + v / maxv * (W - L - 320)
    s = t["s"]
    body = []
    for v in (0, 250, 500, 750, 1000):
        body.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{T - 14}" y2="{T + group * 3 - 14}" '
                    f'stroke="{t["axis"] if v == 0 else t["grid"]}"/>')
        body.append(text(X(v), T + group * 3 + 8, f"{v} ms" if v else "0", "ax", "middle"))
    for i, (metric, vals) in enumerate(data.items()):
        y0 = T + i * group
        body.append(text(L - 20, y0 + bar + 4, metric, "lbb", "end"))
        for j, (key, col, label) in enumerate((("ssh", s[0], "SSH"), ("cloudflare", t["gray"], "Cloudflare"))):
            y = y0 + j * (bar + gap)
            w = X(vals[key]) - L
            body.append(f'<path d="M{L},{y} h{w - 4:.1f} a4,4 0 0 1 4,4 v{bar - 8} a4,4 0 0 1 -4,4 h-{w - 4:.1f} Z" '
                        f'fill="{col}" class="grow" style="animation-delay:{i * .15 + j * .08:.2f}s"/>')
            body.append(f'<g class="fade" style="animation-delay:{.8 + i * .15:.2f}s">'
                        + text(L + w + 10, y + 13.5, f"{vals[key]:.0f} ms", "lbb" if key == "ssh" else "lb")
                        + "</g>")
    # legend
    ly = T - 40
    body.append(f'<rect x="{L}" y="{ly - 10}" width="12" height="12" rx="3" fill="{s[0]}"/>' + text(L + 18, ly, "SSH tunnel (default)", "lb")
                + f'<rect x="{L + 170}" y="{ly - 10}" width="12" height="12" rx="3" fill="{t["gray"]}"/>' + text(L + 188, ly, "Cloudflare quick tunnel", "lb"))
    # stat tiles on the right
    tx = W - 232
    tiles = [("Reconnect after a drop", f"{RECONNECT_S['ssh']:.1f} s", f"vs ~{RECONNECT_S['cloudflare']:.0f} s and a new URL"),
             ("Request time limit", "None", "vs 100 s with no reply bytes")]
    for i, (lab, val, sub) in enumerate(tiles):
        y = T - 4 + i * 118
        body.append(f'<g class="fade" style="animation-delay:{1.1 + i * .2:.2f}s">'
                    f'<rect x="{tx}" y="{y}" width="192" height="100" rx="12" fill="none" stroke="{t["border"]}"/>'
                    + text(tx + 16, y + 26, lab, "lb") + text(tx + 16, y + 66, val, "big")
                    + text(tx + 16, y + 88, sub, "fn") + "</g>")
    body.append(text(40, H - 28, "Median of 20 interleaved rounds (8 for time to first token) against the same Colab VM.", "fn"))
    return card(W, H, t, "SSH tunnel vs Cloudflare quick tunnel",
                "Shorter is better. The SSH tunnel is faster, private, and keeps one URL.", "".join(body),
                "Bar chart of latency for the SSH and Cloudflare tunnels. Time to first token is 571 ms over "
                "SSH and 976 ms over Cloudflare.")


def chart_think(t):
    W, H = 960, 300
    L, T = 210, 128
    track = W - L - 330
    dur = 4.0  # seconds of animation for the 32 s answer
    s = t["s"]
    body = []
    rows = [("Thinking off", "default", THINK["off"], s[0]), ("Thinking on", "--think", THINK["on"], t["gray"])]
    longest = THINK["on"][0]
    for i, (name, flag, (sec, tok), col) in enumerate(rows):
        y = T + i * 62
        w = sec / longest * track
        body.append(text(L - 20, y + 10, name, "lbb", "end") + text(L - 20, y + 28, flag, "fn", "end"))
        body.append(f'<rect x="{L}" y="{y}" width="{track}" height="22" rx="11" fill="{t["track"]}"/>')
        body.append(f'<rect x="{L}" y="{y}" width="{w:.1f}" height="22" rx="11" fill="{col}" class="grow" '
                    f'style="animation-duration:{sec / longest * dur:.2f}s; animation-timing-function:linear; animation-delay:.4s"/>')
        body.append(f'<g class="fade" style="animation-delay:{.4 + sec / longest * dur:.2f}s">'
                    + text(L + w + 12, y + 16, f"{sec:g} s", "lbb") + text(L + w + 12 + 44, y + 16, f"{tok} tokens", "lb")
                    + "</g>")
    speed = THINK["on"][0] / THINK["off"][0]
    body.append(f'<g class="fade" style="animation-delay:{.4 + dur:.2f}s">'
                + text(W - 40, T + 34, f"{speed:.1f}×", "big", "end")
                + text(W - 40, T + 58, "faster replies", "lb", "end") + "</g>")
    body.append(text(40, H - 26, 'One request: "Write a Python function that checks if a string is a palindrome." '
                                 "Bars fill in scaled real time.", "fn"))
    return card(W, H, t, "Thinking costs time on a T4",
                "The model reasons before each answer unless you turn it off, so it is off by default.",
                "".join(body),
                "Two bars: with thinking off the answer takes 6.6 seconds, with thinking on 32 seconds.")


def chart_needle(t):
    rows = needle()
    W, H = 960, 420
    sizes = [32000, 128000, 200000]
    depths = [0.1, 0.25, 0.5, 0.75, 0.9]
    L, T = 210, 168
    cw, ch, g = 120, 56, 8
    body = []
    for j, d in enumerate(depths):
        body.append(text(L + j * (cw + g) + cw / 2, T - 16, f"{d:.0%}", "ax", "middle"))
    body.append(text(L + (len(depths) * (cw + g) - g) / 2, T - 42, "Where the fact is hidden in the prompt", "ax", "middle"))
    for i, size in enumerate(sizes):
        y = T + i * (ch + g)
        body.append(text(L - 20, y + ch / 2 + 5, f"{size // 1000}K tokens", "lbb", "end"))
        for j, d in enumerate(depths):
            x = L + j * (cw + g)
            hit = next((r for r in rows if abs(r["tokens"] - size) < 1000 and r["depth"] == d), None)
            delay = .2 + (i * len(depths) + j) * .07
            if not hit:
                body.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="10" fill="none" '
                            f'stroke="{t["grid"]}" class="fade" style="animation-delay:{delay:.2f}s"/>')
                body.append(text(x + cw / 2, y + ch / 2 + 5, "not tested", "fn", "middle", 'opacity=".6"'))
                continue
            col, mark = (t["good"], "PASS") if hit["pass"] else (t["danger"], "FAIL")
            wall = hit["wall"]
            when = f"{wall:.0f} s" if wall < 120 else f"{wall / 60:.1f} min"
            body.append(f'<g class="pop" style="animation-delay:{delay:.2f}s">'
                        f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="10" fill="{col}" fill-opacity=".12" '
                        f'stroke="{col}" stroke-opacity=".5"/>'
                        f'<path d="M{x + 16},{y + ch / 2} l6,6 l11,-12" fill="none" stroke="{t["goodtext"]}" '
                        f'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>'
                        + text(x + 42, y + ch / 2 - 2, mark, "lbb") + text(x + 42, y + ch / 2 + 15, when, "fn")
                        + "</g>")
    passed = sum(r["pass"] for r in rows)
    body.append(text(40, H - 26, f"{passed} of {len(rows)} tests passed. Time includes reading the whole prompt with "
                                 "no cache; the 9b default (f16 KV, 256K context).", "fn"))
    return card(W, H, t, "It finds a fact anywhere in a 200K-token prompt",
                "A secret number is hidden in filler text; the model must return it.", "".join(body),
                f"Grid of needle-in-a-haystack results: {passed} of {len(rows)} tests passed at 32K, 128K "
                "and 200K tokens.")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("speed", chart_speed), ("vram", chart_vram), ("tunnel", chart_tunnel),
                     ("thinking", chart_think), ("needle", chart_needle)):
        for mode, theme in THEMES.items():
            (OUT / f"{name}-{mode}.svg").write_text(fn(theme))
    print(f"wrote {len(list(OUT.glob('*.svg')))} SVGs to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
