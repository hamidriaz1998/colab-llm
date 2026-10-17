"""Stand-in for llama-server on the VM: lets the tunnel benchmark run without a GPU.

    /health          200 ok
    POST /echo       reads the body, returns its length (upload cost)
    /stream?n=&gap=  SSE chunks every `gap` seconds (token streaming)
    /blob?mb=        N MiB of zeros (download throughput)
"""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def reply(self, body: bytes, ctype="text/plain"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/health":
            return self.reply(b"ok")
        if u.path == "/blob":
            n = int(float(q.get("mb", 10)) * 2**20)
            self.send_response(200)
            self.send_header("Content-Length", str(n))
            self.end_headers()
            chunk = b"\0" * 2**16
            while n > 0:
                self.wfile.write(chunk[: min(n, len(chunk))])
                n -= len(chunk)
            return
        if u.path == "/stream":
            n, gap = int(q.get("n", 200)), float(q.get("gap", 0.02))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for i in range(n):
                data = f"data: {json.dumps({'i': i, 't': time.time()})}\n\n".encode()
                self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                self.wfile.flush()
                time.sleep(gap)
            self.wfile.write(b"0\r\n\r\n")
            return
        self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        got = len(self.rfile.read(n))
        self.reply(json.dumps({"n": got}).encode(), "application/json")


ThreadingHTTPServer(("0.0.0.0", int(sys.argv[1]) if len(sys.argv) > 1 else 8000), H).serve_forever()
