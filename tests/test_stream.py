"""流式输出的协议层与事件层。

这里不联网：把 urlopen 换成一个按剧本吐 SSE 行的假对象，
验的是"碎片拼得对不对""token 有没有记上""Agent 有没有把半成品抛给界面"。
"""

from __future__ import annotations

import json
import unittest
from unittest import mock

from app import config, llm
from app.agent import Agent
from app.llm import LLMError, LLMResponse, chat
from app.session import Session


def _sse(*chunks: dict) -> list[bytes]:
    lines = [f"data: {json.dumps(c, ensure_ascii=False)}".encode("utf-8") for c in chunks]
    return lines + [b"data: [DONE]"]


class _FakeStream:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def __enter__(self) -> "_FakeStream":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def __iter__(self):
        return iter(self._lines)

    def read(self) -> bytes:
        return b"{}"


class TestChatStream(unittest.TestCase):
    def setUp(self) -> None:
        # 配置不再是从 .env 读进模块常量了，chat() 每轮现取 config.current()。
        # 所以这里把 current() 整个换掉，既不联网、也不依赖本机有没有 Key。
        patcher = mock.patch.object(
            config,
            "current",
            return_value=config.ModelConfig("https://example.test/v1", "test-model", "test-key"),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        self.urlopen = mock.patch.object(llm.urllib.request, "urlopen").start()
        self.addCleanup(mock.patch.stopall)
        self.payloads: list[dict] = []

    def _serve(self, lines: list[bytes]) -> None:
        def fake_urlopen(request, timeout: int = 0):
            self.payloads.append(json.loads(request.data.decode("utf-8")))
            return _FakeStream(lines)

        self.urlopen.side_effect = fake_urlopen

    def test_content_is_streamed_and_usage_recorded(self):
        self._serve(
            _sse(
                {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
                {"choices": [{"index": 0, "delta": {"content": "北"}}]},
                {"choices": [{"index": 0, "delta": {"content": "京"}}]},
                {
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
                },
            )
        )

        seen: list[str] = []
        response = chat([{"role": "user", "content": "hi"}], on_delta=seen.append)

        self.assertEqual(seen, ["北", "北京"])
        self.assertEqual(response.message["content"], "北京")
        self.assertEqual(response.message["role"], "assistant")
        self.assertEqual(response.total_tokens, 9)
        self.assertNotIn("tool_calls", response.message)

    def test_request_asks_for_stream_and_usage(self):
        self._serve(_sse({"choices": [{"index": 0, "delta": {"content": "ok"}}]}))
        chat([{"role": "user", "content": "hi"}], on_delta=lambda _: None)

        body = self.payloads[0]
        self.assertTrue(body.get("stream"))
        self.assertEqual(body.get("stream_options"), {"include_usage": True})

    def test_model_comes_from_runtime_config(self):
        """模型名不能再写死在请求里：界面换完模型，下一句就得用新的。

        这条盯的就是"改配置 → 请求体里的 model 真的变了"，而不是只改了显示。
        """
        body = {
            "model": "test-model",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

        class _FakeJSON:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return json.dumps(body).encode("utf-8")

        self.urlopen.side_effect = lambda request, timeout=0: (
            self.payloads.append(json.loads(request.data.decode("utf-8"))) or _FakeJSON()
        )

        chat([{"role": "user", "content": "hi"}])
        self.assertEqual(self.payloads[0]["model"], "test-model")

    def test_tool_call_fragments_are_joined(self):
        self._serve(
            _sse(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "run_sql", "arguments": ""},
                                    }
                                ]
                            },
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [{"index": 0, "function": {"arguments": '{"sql"'}}]
                            },
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [{"index": 0, "function": {"arguments": ': "select 1"}'}}]
                            },
                        }
                    ]
                },
            )
        )

        response = chat([{"role": "user", "content": "hi"}], on_delta=lambda _: None)

        calls = response.message["tool_calls"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["id"], "call_1")
        self.assertEqual(calls[0]["function"]["name"], "run_sql")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"sql": "select 1"})
        self.assertEqual(response.message["content"], "")

    def test_broken_stream_raises_llm_error(self):
        class _Broken(_FakeStream):
            def __iter__(self):
                yield b'data: {"choices":[{"delta":{"content":"a"}}]}'
                raise OSError("connection reset")

        self.urlopen.side_effect = lambda request, timeout=0: _Broken([])
        with self.assertRaises(LLMError):
            chat([{"role": "user", "content": "hi"}], on_delta=lambda _: None)


class _StreamingLLM:
    def __init__(self, text: str) -> None:
        self.text = text
        self.kwargs: dict = {}

    def __call__(self, messages, tools=None, *, on_delta=None):
        self.kwargs["on_delta"] = on_delta
        if on_delta is not None:
            for size in range(1, len(self.text) + 1):
                on_delta(self.text[:size])
        return LLMResponse(
            message={"role": "assistant", "content": self.text},
            prompt_tokens=3,
            completion_tokens=4,
        )


class _QuietLLM:
    def __init__(self) -> None:
        self.called = 0

    def __call__(self, messages, tools=None):
        self.called += 1
        return LLMResponse(message={"role": "assistant", "content": "ok"}, prompt_tokens=1)


class TestAgentDeltas(unittest.TestCase):
    def test_delta_events_carry_growing_text(self):
        events: list[tuple[str, dict]] = []
        agent = Agent(Session(), llm=_StreamingLLM("北京下雨"), on_event=lambda k, p: events.append((k, p)))

        answer = agent.ask("天气")

        self.assertEqual(answer, "北京下雨")
        deltas = [payload["text"] for kind, payload in events if kind == "delta"]
        self.assertEqual(deltas, ["北", "北京", "北京下", "北京下雨"])
        self.assertEqual(events[-1][0], "final")

    def test_llm_without_delta_support_still_works(self):
        agent = Agent(Session(), llm=_QuietLLM(), on_event=lambda *_: None)
        self.assertEqual(agent.ask("问题"), "ok")

    def test_no_listener_means_no_streaming(self):
        model = _StreamingLLM("甲")
        agent = Agent(Session(), llm=model)
        self.assertEqual(agent.ask("问题"), "甲")
        self.assertIsNone(model.kwargs["on_delta"])


if __name__ == "__main__":
    unittest.main()
