"""极简 MCP Streamable HTTP 桥接与注册表扩展单元测试（离线 mock）。"""

from __future__ import annotations

import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from app import config
from app.agent import Agent, make_system_prompt
from app.llm import LLMResponse
from app.mcp_bridge import (
    MCPClientError,
    MCPHttpClient,
    _parse_mcp_payload,
    _sanitize_schema,
    build_web_registry,
    mount_mcp_tools,
)
from app.session import Session
from app.tools.registry import ToolRegistry


class _FakeHTTPResponse:
    def __init__(self, body: str, *, status: int = 200, session_id: str | None = None):
        self._body = body.encode("utf-8")
        self.status = status
        self.headers = {"Mcp-Session-Id": session_id} if session_id else {}

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, *_args) -> None:
        return None


class TestParsePayload(unittest.TestCase):
    def test_parses_json_and_sse(self):
        self.assertEqual(_parse_mcp_payload('{"jsonrpc":"2.0","id":1,"result":{}}')["id"], 1)

        sse = (
            "event: message\r\n"
            'data: {"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"web_search"}]}}\r\n\r\n'
        )
        parsed = _parse_mcp_payload(sse)
        self.assertEqual(parsed["id"], 2)
        self.assertEqual(parsed["result"]["tools"][0]["name"], "web_search")


class TestMCPHttpClient(unittest.TestCase):
    def test_handshake_list_filter_and_call_with_session_retry(self):
        requests_seen: list[tuple[str, str | None]] = []
        state = {"expired_once": False}

        def fake_urlopen(req, timeout=25):  # noqa: ARG001
            payload = json.loads(req.data.decode("utf-8"))
            method = payload["method"]
            sid = req.headers.get("Mcp-session-id") or req.headers.get("Mcp-Session-Id")
            requests_seen.append((method, sid))

            if method == "initialize":
                return _FakeHTTPResponse(
                    'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-03-26"}}\n\n',
                    session_id="sid-1" if not state["expired_once"] else "sid-2",
                )
            if method == "notifications/initialized":
                return _FakeHTTPResponse("", status=202)
            if method == "tools/list":
                return _FakeHTTPResponse(
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": payload["id"],
                            "result": {
                                "tools": [
                                    {
                                        "name": "web_search",
                                        "description": "Search the web",
                                        "inputSchema": {
                                            "type": "object",
                                            "properties": {
                                                "objective": {"type": "string"},
                                                "search_queries": {"type": "array"},
                                            },
                                            "required": ["objective", "search_queries"],
                                        },
                                    },
                                    {
                                        "name": "send_feedback",
                                        "description": "Send feedback",
                                        "inputSchema": {"type": "object", "properties": {}},
                                    },
                                ]
                            },
                        }
                    )
                )
            if method == "tools/call":
                if not state["expired_once"]:
                    state["expired_once"] = True
                    raise urllib.error.HTTPError(
                        req.full_url,
                        404,
                        "Session expired",
                        hdrs=None,
                        fp=io.BytesIO(b"session expired"),
                    )
                return _FakeHTTPResponse(
                    'event: message\ndata: {"jsonrpc":"2.0","id":9,"result":{"content":[{"type":"text","text":"ok: Python 3.13"}]}}\n\n'
                )
            raise AssertionError(f"unexpected method: {method}")

        client = MCPHttpClient(
            "https://mcp.parallel.ai/mcp",
            allowed_tools=("web_search", "web_fetch"),
        )
        with patch("app.mcp_bridge.urllib.request.urlopen", side_effect=fake_urlopen):
            tools = client.list_tools()
            self.assertEqual([t["name"] for t in tools], ["web_search"])

            out = client.call_tool(
                "web_search", {"objective": "test", "search_queries": ["python 3.13"]}
            )
            self.assertEqual(out, "ok: Python 3.13")

        # 验证触发 404 后自动重新 initialize 并拿到了 sid-2
        self.assertIn(("initialize", None), requests_seen)
        self.assertEqual(requests_seen[-1], ("tools/call", "sid-2"))


class TestRegistryAndAgentIntegration(unittest.TestCase):
    def test_clone_and_mount_does_not_mutate_base_registry(self):
        base = ToolRegistry()

        @base.tool
        def local_tool(x: str) -> str:
            """本地工具。"""
            return f"local:{x}"

        client = MCPHttpClient("https://mcp.parallel.ai/mcp", allowed_tools=("web_search",))
        with (
            patch.object(
                client,
                "list_tools",
                return_value=[
                    {
                        "name": "web_search",
                        "description": "联网搜索",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    }
                ],
            ),
            patch.object(client, "call_tool", return_value="search-hit") as mock_call,
        ):
            merged, mounted, err = build_web_registry(base, client=client)
            self.assertIsNone(err)
            self.assertEqual(mounted, ["web_search"])
            self.assertNotIn("web_search", base.names())
            self.assertEqual(set(merged.names()), {"local_tool", "web_search"})

            calls = 0

            def fake_llm(messages, tools):  # noqa: ARG001
                nonlocal calls
                calls += 1
                if calls == 1:
                    return LLMResponse(
                        message={
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "web_search",
                                        "arguments": '{"query": "Parallel MCP"}',
                                    },
                                }
                            ],
                        }
                    )
                return LLMResponse(
                    message={"role": "assistant", "content": "联网结果：search-hit"}
                )

            agent = Agent(Session(), llm=fake_llm, registry=merged)
            answer = agent.ask("查一下 Parallel MCP")
            self.assertIn("search-hit", answer)
            mock_call.assert_called_once_with("web_search", {"query": "Parallel MCP"})

    def test_build_web_registry_falls_back_on_error(self):
        base = ToolRegistry()
        client = MCPHttpClient("https://mcp.parallel.ai/mcp")
        with patch.object(client, "list_tools", side_effect=MCPClientError("network down")):
            reg, mounted, err = build_web_registry(base, client=client)
        self.assertIs(reg, base)
        self.assertEqual(mounted, [])
        self.assertIn("network down", err or "")

    def test_prompt_and_chat_guard_and_config_default(self):
        from app.ui_streamlit import _can_chat

        self.assertEqual(config.MCP_PARALLEL_URL, "https://mcp.parallel.ai/mcp")
        prompt = make_system_prompt(web_search=True)
        self.assertIn("web_search", prompt)
        self.assertIn("web_fetch", prompt)
        self.assertTrue(_can_chat(False, web_search=True))

    def test_sanitize_schema_unwraps_nullable_anyof_and_drops_noise(self):
        raw = {
            "type": "object",
            "title": "v1_extract_toolArguments",
            "additionalProperties": False,
            "properties": {
                "urls": {
                    "title": "Urls",
                    "type": "array",
                    "items": {"type": "string", "title": "Url"},
                    "description": "List of URLs",
                },
                "objective": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                    "title": "Objective",
                    "description": "Search goal",
                },
                "search_queries": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "string"}},
                        {"type": "null"},
                    ],
                    "default": None,
                    "title": "Search Queries",
                },
                "full_content": {
                    "type": "boolean",
                    "default": False,
                    "title": "Full Content",
                },
            },
            "required": ["urls", "ghost_field"],
        }

        cleaned = _sanitize_schema(raw)
        self.assertNotIn("title", cleaned)
        self.assertNotIn("additionalProperties", cleaned)
        self.assertEqual(cleaned["required"], ["urls"])

        props = cleaned["properties"]
        self.assertEqual(props["objective"], {"type": "string", "description": "Search goal"})
        self.assertEqual(props["search_queries"], {"type": "array", "items": {"type": "string"}})
        self.assertEqual(props["full_content"], {"type": "boolean", "default": False})
        self.assertNotIn("title", props["urls"]["items"])


if __name__ == "__main__":
    unittest.main()
