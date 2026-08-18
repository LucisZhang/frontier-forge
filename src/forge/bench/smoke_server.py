"""Tiny OpenAI-compatible server used only to exercise Phase 4 locally."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from forge.train.config import canonical_json


def _schema_value(schema: Mapping[str, Any]) -> Any:
    if "const" in schema:
        return schema["const"]
    values = schema.get("enum")
    if isinstance(values, list) and values:
        return values[0]
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((item for item in kind if item != "null"), "null")
    if kind == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", list(properties))
        return {
            key: _schema_value(properties[key])
            for key in required
            if isinstance(properties, Mapping) and key in properties
        }
    if kind == "array":
        return [_schema_value(schema.get("items", {"type": "string"}))]
    if kind == "integer":
        return 1
    if kind == "number":
        return 1.0
    if kind == "boolean":
        return False
    if kind == "null":
        return None
    return "smoke"


def _first_user(messages: object) -> str:
    if not isinstance(messages, list):
        return ""
    for item in messages:
        if isinstance(item, Mapping) and item.get("role") == "user":
            content = item.get("content")
            return content if isinstance(content, str) else ""
    return ""


class SmokeState:
    def __init__(self, workload: list[dict[str, Any]], *, model: str, speculative: bool) -> None:
        self.model = model
        self.speculative = speculative
        self.labels = {_first_user(row["messages"]): row["label"] for row in workload}
        self.request_count = 0
        self.output_tokens = 0
        self.lock = threading.Lock()

    def record(self, output_tokens: int) -> None:
        with self.lock:
            self.request_count += 1
            self.output_tokens += output_tokens


class SmokeHandler(BaseHTTPRequestHandler):
    server: SmokeHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, value: object, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._json({"status": "ok"})
        elif self.path == "/version":
            self._json({"version": "smoke-0"})
        elif self.path == "/v1/models":
            self._json({"object": "list", "data": [{"id": self.server.state.model}]})
        elif self.path == "/metrics":
            self._metrics()
        else:
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _metrics(self) -> None:
        state = self.server.state
        count = state.request_count
        itl_count = max(state.output_tokens - count, 0)
        lines = [
            f"vllm:time_to_first_token_seconds_count {count}",
            f"vllm:time_to_first_token_seconds_sum {count * 0.001}",
            f'vllm:time_to_first_token_seconds_bucket{{le="0.01"}} {count}',
            f'vllm:time_to_first_token_seconds_bucket{{le="+Inf"}} {count}',
            f"vllm:time_per_output_token_seconds_count {itl_count}",
            f"vllm:time_per_output_token_seconds_sum {itl_count * 0.001}",
            f'vllm:time_per_output_token_seconds_bucket{{le="0.01"}} {itl_count}',
            f'vllm:time_per_output_token_seconds_bucket{{le="+Inf"}} {itl_count}',
            f"vllm:e2e_request_latency_seconds_count {count}",
            f"vllm:e2e_request_latency_seconds_sum {count * 0.004}",
            f'vllm:e2e_request_latency_seconds_bucket{{le="0.01"}} {count}',
            f'vllm:e2e_request_latency_seconds_bucket{{le="+Inf"}} {count}',
        ]
        if state.speculative:
            lines.extend(
                [
                    f"vllm:spec_decode_num_drafts_total {count * 10}",
                    f"vllm:spec_decode_num_draft_tokens_total {count * 40}",
                    f"vllm:spec_decode_num_accepted_tokens_total {count * 28}",
                ]
            )
        body = ("\n".join(lines) + "\n").encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", "text/plain")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("content-length", "0"))
        request = json.loads(self.rfile.read(length))
        label = self.server.state.labels.get(_first_user(request.get("messages")))
        structured = request.get("structured_outputs")
        tools = request.get("tools")
        if tools and not structured and label is not None:
            self._tool_response(request, label)
            return
        schema = structured.get("json") if isinstance(structured, Mapping) else None
        if label is not None:
            output_value = label
        elif isinstance(schema, Mapping):
            output_value = _schema_value(schema)
        else:
            output_value = {}
        output = canonical_json(output_value)
        self.server.state.record(max(1, len(output) // 4))
        if request.get("stream"):
            self._stream_response(request, output)
        else:
            self._json(
                {
                    "id": "smoke-chat",
                    "object": "chat.completion",
                    "model": request.get("model"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": output},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 8,
                        "completion_tokens": max(1, len(output) // 4),
                        "total_tokens": max(1, len(output) // 4) + 8,
                    },
                }
            )

    def _tool_response(self, request: Mapping[str, Any], label: Mapping[str, Any]) -> None:
        tool_call = label["tool_call"]
        self.server.state.record(8)
        self._json(
            {
                "id": "smoke-tool",
                "object": "chat.completion",
                "model": request.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_smoke",
                                    "type": "function",
                                    "function": {
                                        "name": tool_call["name"],
                                        "arguments": canonical_json(tool_call["arguments"]),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 8, "completion_tokens": 8, "total_tokens": 16},
            }
        )

    def _stream_response(self, request: Mapping[str, Any], output: str) -> None:
        parts = [output[index : index + 12] for index in range(0, len(output), 12)]
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.end_headers()
        for part in parts:
            chunk = {
                "id": "smoke-stream",
                "object": "chat.completion.chunk",
                "model": request.get("model"),
                "choices": [{"index": 0, "delta": {"content": part}, "finish_reason": None}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.0005)
        final = {
            "id": "smoke-stream",
            "object": "chat.completion.chunk",
            "model": request.get("model"),
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": max(1, len(output) // 4),
                "total_tokens": max(1, len(output) // 4) + 8,
            },
        }
        self.wfile.write(f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n".encode())
        self.wfile.flush()


class SmokeHTTPServer(ThreadingHTTPServer):
    def __init__(self, state: SmokeState) -> None:
        super().__init__(("127.0.0.1", 0), SmokeHandler)
        self.state = state


class SmokeServer:
    def __init__(self, workload: list[dict[str, Any]], *, model: str, speculative: bool) -> None:
        self.server = SmokeHTTPServer(SmokeState(workload, model=model, speculative=speculative))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def __enter__(self) -> SmokeServer:
        self.thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
