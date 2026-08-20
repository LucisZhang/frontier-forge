"""Small OpenAI-compatible upstream used only by the CPU kind smoke."""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    mode = "healthy"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send(200, b"ok\n", "text/plain")
        else:
            self._send(404, b"not found\n", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.startswith("/v1/"):
            self._send(404, b"not found\n", "text/plain")
            return
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")
        if self.mode == "http500":
            self._send(500, b'{"error":"injected"}', "application/json")
            return
        if self.mode == "latency":
            time.sleep(3)
        content = json.dumps(
            {
                "ambiguity_flag": False,
                "company": "Experian Information Solutions Inc.",
                "issue": "Improper use of your report",
                "product": "credit_reporting",
                "tool_call": {
                    "arguments": {
                        "company": "Experian Information Solutions Inc.",
                        "issue": "Improper use of your report",
                    },
                    "name": "route_to_company",
                },
                "urgency": "medium",
            },
            separators=(",", ":"),
        )
        body = json.dumps(
            {
                "id": "forge-mock",
                "object": "chat.completion",
                "model": request.get("model", "forge-r1b"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 32, "total_tokens": 40},
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
    parser.add_argument("--mode", choices=("healthy", "http500", "latency"), default="healthy")
    args = parser.parse_args()
    Handler.mode = args.mode
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
