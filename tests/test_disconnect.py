"""Client-disconnect handling for the streaming response paths.

These lock in two behaviors:
  1. `_write_sse` swallows BrokenPipe/ConnectionReset, flags the connection
     closed, and returns False instead of raising (no stray traceback).
  2. The streaming handler stops draining the Qwen upstream once a write
     fails, rather than reading the whole upstream after the client is gone.
"""

import json
import unittest

from qwen_proxy import server


class _FakeUpstream:
    """Iterable of raw SSE byte lines that records how many were consumed."""

    def __init__(self, lines):
        self._lines = lines
        self.consumed = 0

    def __iter__(self):
        for line in self._lines:
            self.consumed += 1
            yield line


def _content_line(text: str) -> bytes:
    return (
        "data: " + json.dumps({"choices": [{"delta": {"phase": "answer", "content": text}}]}) + "\n"
    ).encode()


class WriteGuardTests(unittest.TestCase):
    def test_write_sse_returns_false_and_marks_closed_on_disconnect(self):
        class Boom:
            def write(self, *_):
                raise ConnectionResetError("client gone")

            def flush(self):
                pass

        handler = object.__new__(server.ProxyHandler)
        handler.wfile = Boom()
        handler.path = "/v1/messages"
        handler.close_connection = False

        self.assertFalse(handler._write_sse(b"data: x\n\n"))
        self.assertTrue(handler.close_connection)

    def test_write_sse_returns_true_when_connected(self):
        class Sink:
            def __init__(self):
                self.data = b""

            def write(self, data):
                self.data += data

            def flush(self):
                pass

        handler = object.__new__(server.ProxyHandler)
        handler.wfile = Sink()
        handler.path = "/v1/messages"
        handler.close_connection = False

        self.assertTrue(handler._write_sse(b"data: x\n\n"))
        self.assertFalse(handler.close_connection)
        self.assertEqual(handler.wfile.data, b"data: x\n\n")


class StreamDisconnectTests(unittest.TestCase):
    def _make_handler(self, fail_after: int):
        class DisconnectHandler(server.ProxyHandler):
            def __init__(self):  # bypass real socket setup
                self.path = "/v1/chat/completions"
                self.close_connection = False
                self.writes = 0
                self._fail_after = fail_after

            def _start_sse(self):
                return True

            def _write_sse(self, data):
                self.writes += 1
                if self.writes > self._fail_after:
                    self.close_connection = True
                    return False
                return True

        return DisconnectHandler()

    def test_stops_reading_upstream_after_client_disconnect(self):
        lines = [_content_line(f"chunk{i}") for i in range(10)]
        upstream = _FakeUpstream(lines)
        # role chunk (write #1) + one content chunk (write #2) succeed, then fail.
        handler = self._make_handler(fail_after=2)

        handler._handle_stream(upstream, "id1", "qwen3.6-plus", "auto", has_tools=False)

        self.assertTrue(handler.close_connection)
        self.assertLess(
            upstream.consumed,
            len(lines),
            "handler must stop draining upstream once the client disconnects",
        )

    def test_consumes_full_upstream_when_connected(self):
        lines = [_content_line(f"chunk{i}") for i in range(10)]
        upstream = _FakeUpstream(lines)
        handler = self._make_handler(fail_after=10_000)  # never fails

        handler._handle_stream(upstream, "id2", "qwen3.6-plus", "auto", has_tools=False)

        self.assertEqual(upstream.consumed, len(lines))
        self.assertFalse(handler.close_connection)


if __name__ == "__main__":
    unittest.main()
