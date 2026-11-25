"""模型调用层。

只做三件事：发请求、重试、把返回收拾干净。
一次 HTTP 请求而已，没有任何神秘之处——面试时能说清这层，比会说"我用了 LangChain"值钱。
"""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from . import config


class LLMError(RuntimeError):
    """模型调用失败。上层必须接住，不能让整个 Agent 崩掉。

    带上 status/detail 是给"测试连接"用的：那边要把 401（Key 错）和 404（地址错）
    分开说，光有一句拼好的中文串就分不出来了。
    """

    def __init__(self, message: str, status: int = 0, detail: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail


@dataclass
class LLMResponse:
    message: dict[str, Any]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def normalize_assistant(message: dict[str, Any]) -> dict[str, Any]:
    """只保留协议认识的字段。

    有些模型会附带 reasoning_content 之类的东西，原样塞回去可能吃 400。
    """
    clean: dict[str, Any] = {
        "role": "assistant",
        "content": message.get("content") or "",
    }
    if message.get("tool_calls"):
        clean["tool_calls"] = message["tool_calls"]
    return clean


_RETRYABLE = {429, 500, 502, 503, 504}


def _post(cfg: config.ModelConfig, body: bytes, timeout: int, retries: int):
    """发请求。cfg 每轮现传进来，不是模块级常量——界面上换了模型要立刻生效。"""
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            cfg.endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cfg.api_key}",
            },
            method="POST",
        )
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")[:400]
            if exc.code in _RETRYABLE and attempt < retries:
                time.sleep(2 * (attempt + 1))
                continue
            raise LLMError(f"HTTP {exc.code}：{detail}", status=exc.code, detail=detail) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
                continue
            raise LLMError(f"网络异常：{exc}") from exc

    raise LLMError("请求失败")


def _read_stream(response, started: float, on_delta: Callable[[str], None]) -> LLMResponse:
    parts: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] = {}

    try:
        for raw in response:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
            except json.JSONDecodeError:
                continue

            if chunk.get("usage"):
                usage = chunk["usage"]

            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}

                text = delta.get("content")
                if text:
                    parts.append(text)
                    on_delta("".join(parts))

                for piece in delta.get("tool_calls") or []:
                    slot = calls.setdefault(
                        piece.get("index", 0),
                        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    if piece.get("id"):
                        slot["id"] = piece["id"]
                    function = piece.get("function") or {}
                    slot["function"]["name"] += function.get("name") or ""
                    slot["function"]["arguments"] += function.get("arguments") or ""
    except (OSError, http.client.HTTPException) as exc:
        raise LLMError(f"流式读取中断：{exc}") from exc

    message: dict[str, Any] = {"role": "assistant", "content": "".join(parts)}
    if calls:
        message["tool_calls"] = [calls[index] for index in sorted(calls)]

    return LLMResponse(
        message=message,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        latency=time.time() - started,
    )


def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    timeout: int = 90,
    retries: int = 2,
    on_delta: Callable[[str], None] | None = None,
) -> LLMResponse:
    cfg = config.current()
    if not cfg.api_key:
        raise LLMError(
            "没有配置 API Key：在页面右侧「模型设置」里填一个，"
            "或者在 datainsight/.env 里写一行 API_KEY=sk-xxxx。"
        )

    payload: dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "temperature": temperature,
    }
    # 只有显式传了才写进请求体。默认 None 时 payload 的键集合和以前一模一样，
    # 现有流式 / 协议测试不受影响；前置改写那种「一句话就能答完」的调用才用得上。
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    if on_delta is not None:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    started = time.time()

    with _post(cfg, body, timeout, retries) as response:
        if on_delta is not None:
            return _read_stream(response, started, on_delta)

        data = json.loads(response.read().decode("utf-8"))

    usage = data.get("usage") or {}
    return LLMResponse(
        message=data["choices"][0]["message"],
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        latency=time.time() - started,
    )


LLMCallable = Callable[..., LLMResponse]


# ---------------------------------------------------------------- 连通性测试
#
# "测试连接"要回答的是一个很具体的问题：这三样东西凑在一起，能不能真的换回一个
# token。所以它必须**真发一次请求**——把 base_url 拼对、把 Key 带上、让服务端亲自
# 确认模型名存在。查 DNS、拼字符串、读配置都答不了这个题。
#
# 代价是要钱：一条 1 token 的请求，几分钱不到。用 max_tokens=1 而不是空请求，
# 是因为有些网关对空 messages 直接 400，那会把"配置对不对"和"参数对不对"混在一起。


@dataclass
class ProbeResult:
    ok: bool
    detail: str
    latency: float = 0.0
    model: str = ""

    @property
    def summary(self) -> str:
        if not self.ok:
            return self.detail
        spent = f"{self.latency * 1000:.0f} ms"
        name = f"，返回模型 `{self.model}`" if self.model else ""
        return f"连接正常（{spent}{name}）"


def _explain_http(code: int, detail: str) -> str:
    """把状态码翻成人话。只说"HTTP 401"等于没说。"""
    hint = {
        400: "请求被拒。多半是模型名不对，或者这个地址不是 OpenAI 兼容接口。",
        401: "Key 无效或没有权限。",
        403: "Key 没有访问这个模型的权限（也可能是地区限制）。",
        404: "地址或模型名不存在。检查 Base URL 是不是多了/少了一段 `/v1`。",
        429: "被限流了：Key 的额度用完，或者短时间请求太密。",
    }.get(code, "服务端出错，稍后再试。")
    body = (detail or "").strip().replace("\n", " ")[:160]
    return f"HTTP {code}：{hint}" + (f"（服务端原话：{body}）" if body else "")


def probe(base_url: str, model: str, api_key: str, *, timeout: int = 20) -> ProbeResult:
    """用最小代价真连一次，把结果做成数据返回。

    刻意**不抛异常**：这个函数的结果是给人看的，成功失败都要有话说。
    抛异常会让界面变成一片 traceback，而用户此刻只想知道"到底哪一项填错了"。
    """
    try:
        cfg = config.ModelConfig(
            base_url=config.normalize_base_url(base_url),
            model=str(model or "").strip(),
            api_key=str(api_key or "").strip(),
        )
    except ValueError as exc:
        return ProbeResult(False, str(exc))

    if not cfg.model:
        return ProbeResult(False, "模型名不能为空。")
    if not cfg.api_key:
        return ProbeResult(False, "API Key 不能为空。")

    payload = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0,
        "stream": False,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    started = time.time()

    try:
        with _post(cfg, body, timeout, 0) as response:
            raw = response.read().decode("utf-8", "ignore")
    except LLMError as exc:
        if exc.status:
            return ProbeResult(False, _explain_http(exc.status, exc.detail))
        return ProbeResult(False, f"连不上：{exc}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return ProbeResult(False, f"连不上：{exc}。检查地址、网络和代理。")

    latency = time.time() - started
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return ProbeResult(False, f"返回的不是 JSON：{raw[:160]}（这个地址可能不是 OpenAI 兼容接口）")

    if isinstance(data, dict) and data.get("error"):
        message = data["error"].get("message") if isinstance(data["error"], dict) else data["error"]
        return ProbeResult(False, f"服务端报错：{str(message)[:200]}")

    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices:
        return ProbeResult(False, f"返回里没有 choices：{raw[:160]}")

    return ProbeResult(True, "连接正常", latency=latency, model=str(data.get("model") or cfg.model))
