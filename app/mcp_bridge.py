"""极简 MCP Streamable HTTP 桥接器（纯标准库实现，不依赖第三方 mcp 包）。

把远程 MCP Server（默认 Parallel 新版入口 https://mcp.parallel.ai/mcp）暴露的工具
适配成项目内置的 ToolRegistry 工具，Agent 主循环无需感知底层走的是本地函数还是 MCP。
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from typing import Any

from . import config
from .tools.registry import ToolRegistry

_PROTOCOL_VERSION = "2025-03-26"


class MCPClientError(RuntimeError):
    """MCP 通信或协议异常。"""


def _parse_mcp_payload(raw: str) -> dict[str, Any]:
    """同时兼容 application/json 与 text/event-stream (SSE) 两种 MCP HTTP 响应。"""
    text = raw.strip()
    if not text:
        return {}
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    last_obj: dict[str, Any] = {}
    data_lines: list[str] = []

    def _flush_block() -> None:
        nonlocal last_obj
        if not data_lines:
            return
        joined = "\n".join(data_lines).strip()
        data_lines.clear()
        if not joined or joined == "[DONE]":
            return
        try:
            obj = json.loads(joined)
        except json.JSONDecodeError:
            return
        if isinstance(obj, dict):
            last_obj = obj

    for line in text.splitlines():
        stripped = line.rstrip("\r")
        if not stripped:
            _flush_block()
            continue
        if stripped.startswith("data:"):
            data_lines.append(stripped[5:].lstrip())
    _flush_block()
    return last_obj


def _extract_tool_text(result: dict[str, Any]) -> str:
    """把 MCP tools/call 的 result 转成可读文本。"""
    chunks: list[str] = []
    for item in result.get("content") or []:
        if not isinstance(item, dict):
            chunks.append(str(item))
            continue
        kind = item.get("type")
        if kind == "text" and item.get("text") is not None:
            chunks.append(str(item["text"]))
        elif kind == "resource" and isinstance(item.get("resource"), dict):
            res = item["resource"]
            chunks.append(str(res.get("text") or json.dumps(res, ensure_ascii=False)))
        else:
            chunks.append(json.dumps(item, ensure_ascii=False))

    if not chunks and result.get("structuredContent") is not None:
        chunks.append(json.dumps(result["structuredContent"], ensure_ascii=False))

    text = "\n".join(c for c in chunks if c) or json.dumps(result, ensure_ascii=False)
    if result.get("isError"):
        return f"MCP 工具返回错误：{text}"
    return text


class MCPHttpClient:
    """面向 Streamable HTTP 的轻量同步 MCP 客户端。"""

    def __init__(
        self,
        url: str = config.MCP_PARALLEL_URL,
        *,
        timeout: int = config.MCP_TIMEOUT,
        allowed_tools: tuple[str, ...] = config.MCP_ALLOWED_TOOLS,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.url = url
        self.timeout = timeout
        self.allowed_tools = allowed_tools
        self.extra_headers = dict(headers or {})
        self._lock = threading.Lock()
        self._session_id: str | None = None
        self._initialized = False
        self._req_id = 0
        self._tools_cache: list[dict[str, Any]] | None = None

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _post(
        self, payload: dict[str, Any], *, use_session: bool = True
    ) -> tuple[int, str | None, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **self.extra_headers,
        }
        if use_session and self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                sid = resp.headers.get("Mcp-Session-Id")
                raw = resp.read().decode("utf-8", "ignore")
                return resp.status, sid, raw
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")[:400]
            raise MCPClientError(f"HTTP {exc.code}：{detail or exc.reason}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise MCPClientError(f"网络异常：{exc}") from exc

    def _handshake_locked(self) -> None:
        _, sid, raw = self._post(
            {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": _PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "datainsight", "version": "0.1"},
                },
            },
            use_session=False,
        )
        data = _parse_mcp_payload(raw)
        if data.get("error"):
            raise MCPClientError(f"MCP 初始化失败：{data['error']}")
        self._session_id = sid
        self._post(
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            use_session=True,
        )
        self._initialized = True

    def _rpc(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            if not self._initialized:
                self._handshake_locked()

            payload = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": method,
                "params": params or {},
            }
            try:
                _, new_sid, raw = self._post(payload, use_session=True)
            except MCPClientError as exc:
                # 服务端会话过期时会返回 400/404，自动重连一次
                msg = str(exc)
                if "HTTP 400" in msg or "HTTP 404" in msg:
                    self._initialized = False
                    self._session_id = None
                    self._handshake_locked()
                    _, new_sid, raw = self._post(payload, use_session=True)
                else:
                    raise

            if new_sid:
                self._session_id = new_sid

        data = _parse_mcp_payload(raw)
        if data.get("error"):
            err = data["error"]
            detail = err.get("message") if isinstance(err, dict) else str(err)
            raise MCPClientError(f"MCP 调用失败：{detail}")
        result = data.get("result")
        if not isinstance(result, dict):
            raise MCPClientError("MCP 返回格式异常：缺少 result 对象")
        return result

    def list_tools(self, *, refresh: bool = False) -> list[dict[str, Any]]:
        """拉取可用工具定义并按允许名单过滤。"""
        if self._tools_cache is None or refresh:
            result = self._rpc("tools/list", {})
            tools = result.get("tools") or []
            self._tools_cache = [t for t in tools if isinstance(t, dict) and t.get("name")]

        if not self.allowed_tools:
            return list(self._tools_cache)
        allowed = set(self.allowed_tools)
        return [t for t in self._tools_cache if t.get("name") in allowed]

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """执行远程 MCP 工具并提取文本结果。"""
        result = self._rpc(
            "tools/call",
            {"name": name, "arguments": arguments or {}},
        )
        return _extract_tool_text(result)


_CLIENTS: dict[tuple[str, tuple[str, ...], int], MCPHttpClient] = {}
_CLIENTS_LOCK = threading.Lock()


def get_parallel_client(
    url: str = config.MCP_PARALLEL_URL,
    *,
    allowed_tools: tuple[str, ...] = config.MCP_ALLOWED_TOOLS,
    timeout: int = config.MCP_TIMEOUT,
) -> MCPHttpClient:
    """按参数复用同一个客户端实例，避免 Streamlit 每次 rerun 重复握手。"""
    key = (url, tuple(allowed_tools), timeout)
    with _CLIENTS_LOCK:
        client = _CLIENTS.get(key)
        if client is None:
            client = MCPHttpClient(url, timeout=timeout, allowed_tools=tuple(allowed_tools))
            _CLIENTS[key] = client
        return client


_NOISY_SCHEMA_KEYS = frozenset({"title", "additionalProperties", "$schema"})


def _sanitize_schema_node(node: Any) -> dict[str, Any]:
    """把 Pydantic / MCP 风格的子 schema 清洗成与本地 schema.py 同构的极简形态。"""
    if not isinstance(node, dict):
        return {"type": "string"}

    cleaned = {k: v for k, v in node.items() if k not in _NOISY_SCHEMA_KEYS}
    if cleaned.get("default", "__keep__") is None:
        cleaned.pop("default", None)

    for union_key in ("anyOf", "oneOf"):
        variants = cleaned.pop(union_key, None)
        if isinstance(variants, list):
            non_null = [
                v for v in variants
                if not (isinstance(v, dict) and v.get("type") == "null")
            ]
            if len(non_null) == 1 and isinstance(non_null[0], dict):
                merged = dict(non_null[0])
                for k, v in cleaned.items():
                    merged.setdefault(k, v)
                return _sanitize_schema_node(merged)
            if non_null and "type" not in cleaned:
                first = _sanitize_schema_node(non_null[0])
                for k, v in cleaned.items():
                    first.setdefault(k, v)
                return first
            cleaned.setdefault("type", "string")

    if "items" in cleaned:
        cleaned["items"] = _sanitize_schema_node(cleaned["items"])

    if isinstance(cleaned.get("properties"), dict):
        cleaned["properties"] = {
            k: _sanitize_schema_node(v) for k, v in cleaned["properties"].items()
        }
        cleaned.setdefault("type", "object")
        if isinstance(cleaned.get("required"), list):
            cleaned["required"] = [
                k for k in cleaned["required"] if k in cleaned["properties"]
            ]

    if "type" not in cleaned:
        cleaned["type"] = "string"
    return cleaned


def _sanitize_schema(schema: Any) -> dict[str, Any]:
    """清洗远端 MCP 工具顶层 inputSchema，保证为标准 object schema。"""
    raw = dict(schema) if isinstance(schema, dict) else {}
    raw.setdefault("type", "object")
    if not isinstance(raw.get("properties"), dict):
        raw["properties"] = {}
    cleaned = _sanitize_schema_node(raw)
    cleaned["type"] = "object"
    cleaned.setdefault("properties", {})
    if not isinstance(cleaned.get("required"), list):
        cleaned["required"] = []
    return cleaned


def mount_mcp_tools(registry: ToolRegistry, client: MCPHttpClient) -> list[str]:
    """把 MCP 工具挂进指定 ToolRegistry，返回本次新挂上的工具名。"""
    mounted: list[str] = []
    for spec in client.list_tools():
        name = str(spec["name"])
        if registry.get(name) is not None:
            continue
        desc = str(spec.get("description") or spec.get("title") or name).strip()
        schema = _sanitize_schema(spec.get("inputSchema"))

        def _invoker(_remote_name: str = name, **kwargs: Any) -> str:
            return client.call_tool(_remote_name, kwargs)

        registry.register(name, _invoker, desc, schema)
        mounted.append(name)
    return mounted


def build_web_registry(
    base: ToolRegistry,
    *,
    client: MCPHttpClient | None = None,
) -> tuple[ToolRegistry, list[str], str | None]:
    """基于已有注册表克隆并挂载 MCP 联网工具；失败时平滑退回 base。"""
    target = client or get_parallel_client()
    merged = base.clone()
    try:
        mounted = mount_mcp_tools(merged, target)
    except MCPClientError as exc:
        return base, [], str(exc)
    return merged, mounted, None
