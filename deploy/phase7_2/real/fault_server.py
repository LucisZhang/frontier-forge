"""Controlled fault upstream for Phase 7.2 alert and rollback drills."""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    mode = "http500"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send(200, b"ok\n", "text/plain")
        else:
            self._send(404, b"not found\n", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        if self.mode == "http500":
            self._send(500, b'{"error":"phase7_2_injected_500"}', "application/json")
            return
        time.sleep(3)
        body = json.dumps(
            {
                "id": "forge-fault-latency",
                "object": "chat.completion",
                "model": "forge-r1b",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '{"fault":"latency"}',
                        },
                    }
                ],
            },
            separators=(",", ":"),
        ).encode()
        self._send(200, body, "application/json")

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--mode", choices=("http500", "latency"), required=True)
    args = parser.parse_args()
    Handler.mode = args.mode
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
