#!/usr/bin/env python3
"""Long-context retrieval check: hide a fact at a relative depth in filler text, ask for it back.

    bench/needle.py http://127.0.0.1:8765 --tokens 32000,128000 --depths 0.1,0.5,0.9

Speed says nothing about whether the model still *uses* a long context, and a quantized KV
cache could hurt exactly that. This reports pass/fail plus server prefill and generation speed.
"""
import argparse
import json
import random
import time
import urllib.request

FILLER = ("The grass is green. The sky is blue. The sun is yellow. Here we go. "
          "There and back again. ")


def post(base, path, body, timeout=1800):
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="server root, without /v1")
    ap.add_argument("--tokens", default="32000")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    a = ap.parse_args()
    per = len(post(a.base, "/tokenize", {"content": FILLER * 100})["tokens"]) / 100
    print(f"{'tokens':>8} {'depth':>6} {'result':>7} {'prefill tok/s':>14} {'gen tok/s':>10} {'wall s':>7}")
    for n in map(int, a.tokens.split(",")):
        for d in map(float, a.depths.split(",")):
            key = random.randint(100000, 999999)
            reps = int(n / per)
            cut = int(reps * d)
            # A fresh first line defeats the prompt cache, so the timings are a full prefill.
            text = (f"Document {random.random()}\n" + FILLER * cut
                    + f"\nThe secret number is {key}. Remember it.\n" + FILLER * (reps - cut))
            t = time.time()
            r = post(a.base, "/v1/chat/completions", {
                "temperature": 0, "max_tokens": 400,
                "chat_template_kwargs": {"enable_thinking": False},  # test retrieval, not reasoning
                "messages": [{"role": "user", "content": text + "\n\nWhat is the secret number? "
                              "Reply with the number only."}]})
            wall = time.time() - t
            answer = r["choices"][0]["message"]["content"] or ""
            tm = r.get("timings", {})
            print(f"{tm.get('prompt_n', n):>8} {d:>6.2f} {'PASS' if str(key) in answer else 'FAIL':>7} "
                  f"{tm.get('prompt_per_second', 0):>14.0f} {tm.get('predicted_per_second', 0):>10.1f} "
                  f"{wall:>7.0f}", flush=True)


if __name__ == "__main__":
    main()
