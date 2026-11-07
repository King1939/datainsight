"""第 2 周的自动化测试。

跑法（不需要 pytest，stdlib 自带）：
    python -m unittest discover -s tests -v

为什么要写测试：改 prompt、加工具之后，你怎么知道没把原来的功能改坏？
靠手动点几下是测不出来的。这也是简历上"自建回归测试"的落点。
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Literal, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import Agent  # noqa: E402
from app.llm import LLMResponse  # noqa: E402
from app.schema import build_parameters, parse_docstring  # noqa: E402
from app.session import Session  # noqa: E402
from app.tools import data_tools as dt  # noqa: E402
from app.tools.registry import ToolRegistry  # noqa: E402
from app.tools import registry as real_registry  # noqa: E402


def setUpModule() -> None:
    """数据层不再隐式加载示例 CSV，测试要自己声明用它。"""
    dt.bind_sample()


# ---------------------------------------------------------------- 测试替身


def _call(call_id: str, name: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }
        ],
    }


class ScriptedLLM:
    """按剧本依次返回消息的假模型，用来测循环，不发网络请求。"""

    def __init__(self, script: list[dict], tokens: int = 12) -> None:
        self.script = list(script)
        self.tokens = tokens
        self.seen: list[list[dict]] = []

    def __call__(self, messages, tools=None):
        self.seen.append([m for m in messages])
        message = self.script.pop(0) if self.script else {"role": "assistant", "content": "结束"}
        return LLMResponse(
            message=message, prompt_tokens=self.tokens, completion_tokens=0, latency=0.0
        )


# ---------------------------------------------------------------- schema


class TestSchema(unittest.TestCase):
    def test_type_mapping(self):
        def sample(
            name: str,
            count: int = 3,
            ratio: float = 0.5,
            flag: bool = False,
            kind: Literal["bar", "line"] = "bar",
            note: Optional[str] = None,
        ) -> str:
            """样例工具。

            Args:
                name: 名字
                count: 次数
            """
            return name

        params = build_parameters(sample)
        props = params["properties"]

        self.assertEqual(props["name"]["type"], "string")
        self.assertEqual(props["count"]["type"], "integer")
        self.assertEqual(props["ratio"]["type"], "number")
        self.assertEqual(props["flag"]["type"], "boolean")
        self.assertEqual(props["kind"]["enum"], ["bar", "line"])
        self.assertEqual(props["note"]["type"], "string")

        # 只有没默认值的参数才算必填
        self.assertEqual(params["required"], ["name"])
        self.assertEqual(props["name"]["description"], "名字")
        self.assertEqual(props["count"]["default"], 3)

    def test_docstring_parsing(self):
        description, args = parse_docstring(
            """第一行描述。

            继续补充细节。

            Args:
                a: 参数 A
                b (int): 参数 B

            Returns:
                随便
            """
        )
        self.assertIn("第一行描述", description)
        self.assertIn("继续补充细节", description)
        self.assertNotIn("Returns", description)
        self.assertEqual(args, {"a": "参数 A", "b": "参数 B"})


# ---------------------------------------------------------------- registry


class TestRegistry(unittest.TestCase):
    def test_auto_register(self):
        reg = ToolRegistry()

        @reg.tool
        def hello(name: str) -> str:
            """打个招呼。

            Args:
                name: 名字
            """
            return f"hi {name}"

        self.assertEqual(reg.names(), ["hello"])
        self.assertEqual(reg.call("hello", {"name": "世界"}), "hi 世界")

    def test_reject_missing_docstring(self):
        reg = ToolRegistry()
        with self.assertRaises(ValueError):

            @reg.tool
            def no_doc(name: str) -> str:
                return name

    def test_error_paths_never_raise(self):
        reg = ToolRegistry()

        @reg.tool
        def boom(x: int) -> str:
            """故意炸。

            Args:
                x: 数字
            """
            raise RuntimeError("炸了")

        self.assertIn("不存在工具", reg.call("nope", {}))
        self.assertIn("不认识参数", reg.call("boom", {"y": 1}))
        self.assertIn("缺少必填参数", reg.call("boom", {}))
        self.assertIn("炸了", reg.call("boom", {"x": 1}))
        self.assertIn("不是合法 JSON", reg.call("boom", "{bad json"))

    def test_real_tools_registered(self):
        # 用子集断言而不是相等：每加一个工具就要改一次测试，说明测试写错了。
        # 这里真正要保证的是"核心工具确实注册上了"，不是"工具总数恰好是 4"。
        self.assertLessEqual(
            {"inspect_table", "run_sql", "run_python", "make_chart"},
            set(real_registry.names()),
        )
        for spec in real_registry.specs():
            self.assertTrue(spec["function"]["description"])
            self.assertEqual(spec["function"]["parameters"]["type"], "object")

    def test_schema_always_matches_signature(self):
        """"结构只有一份真相"是可以被验证的，不是口号。

        注册表里存的 schema 必须能由函数签名原样重建。
        任何人绕过装饰器手改 schema、或改了签名没同步，这条立刻红。
        """
        for name in real_registry.names():
            tool = real_registry.get(name)
            self.assertIsNotNone(tool)
            self.assertEqual(
                build_parameters(tool.func),
                tool.parameters,
                f"{name} 的 schema 与函数签名不一致",
            )


# ---------------------------------------------------------------- agent 循环


class TestAgentLoop(unittest.TestCase):
    def test_full_loop(self):
        llm = ScriptedLLM(
            [
                _call("c1", "inspect_table", {}),
                _call(
                    "c2",
                    "run_sql",
                    {"sql": "SELECT city, ROUND(SUM(amount),2) t FROM dataset GROUP BY 1 ORDER BY 2 DESC LIMIT 1"},
                ),
                {"role": "assistant", "content": "上海最高。"},
            ]
        )
        session = Session()
        agent = Agent(session, llm=llm)

        answer = agent.ask("哪个城市卖得最好？")

        self.assertEqual(answer, "上海最高。")
        self.assertEqual(session.tool_calls, 2)
        self.assertEqual(session.turns, 1)
        roles = [m["role"] for m in session.messages]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"])

        # 观察结果真的含有数据
        tool_messages = [m for m in session.messages if m["role"] == "tool"]
        self.assertIn("上海", tool_messages[1]["content"])

        # 第二次调用模型时，它已经能看到第一次的工具结果
        self.assertEqual(len(llm.seen), 3)
        self.assertTrue(any(m["role"] == "tool" for m in llm.seen[1]))

    def test_stops_on_repeat(self):
        payload = _call("c1", "run_sql", {"sql": "SELECT 1"})
        llm = ScriptedLLM([payload, payload, payload, payload])
        agent = Agent(Session(), llm=llm)

        answer = agent.ask("死循环测试")
        self.assertIn("中止", answer)

    def test_stops_on_budget(self):
        llm = ScriptedLLM([_call("c1", "inspect_table", {})] * 5, tokens=100_000)
        agent = Agent(Session(), llm=llm, token_budget=1_000)

        answer = agent.ask("预算测试")
        self.assertIn("预算", answer)

    def test_budget_is_per_question_not_per_session(self):
        """预算按这一问算。累计 tokens 是跨轮累加的，拿它当尺子的话，
        聊久了（累计超过 80k）之后每一句都会在第一个工具回合后被掐断。
        """
        script = [
            _call("c1", "inspect_table", {}),
            {"role": "assistant", "content": "第一轮"},
            _call("c2", "inspect_table", {}),
            {"role": "assistant", "content": "第二轮"},
        ]
        session = Session()
        agent = Agent(session, llm=ScriptedLLM(script, tokens=60), token_budget=100)

        self.assertEqual(agent.ask("第一问"), "第一轮")
        self.assertGreater(session.tokens, 100)          # 累计早就超了
        self.assertEqual(agent.ask("第二问"), "第二轮")   # 但这一问只花了 120 里的一小段

    def test_long_task_is_not_cut_at_eight_steps(self):
        """十来步的活要能做完。

        步数上限曾经是 8，够跑通"查一次表就回答"的演示，但读两三份资料 + 查表 +
        改文件本来就超过 8 步，结果是活干到一半被掐断。这条盯的就是那个回归。
        """
        script = [_call(f"c{i}", "run_sql", {"sql": f"SELECT {i} AS n"}) for i in range(10)]
        script.append({"role": "assistant", "content": "做完了"})
        agent = Agent(Session(), llm=ScriptedLLM(script))

        self.assertEqual(agent.ask("一个需要十来步的任务"), "做完了")

    def test_hitting_step_ceiling_can_be_continued(self):
        """撞到上限不是死局：要明说能接着做，而且历史不能停在 tool 消息上。

        停在 tool 上的话，下一句"继续"就是 user 紧跟 tool，接口会直接 400。
        """
        script = [_call(f"c{i}", "run_sql", {"sql": f"SELECT {i} AS n"}) for i in range(4)]
        session = Session()
        agent = Agent(session, llm=ScriptedLLM(script), max_steps=2)

        answer = agent.ask("步数很少的一次")

        self.assertIn("继续", answer)
        self.assertEqual(session.messages[-1]["role"], "assistant")

    def test_multi_turn_keeps_history(self):
        llm = ScriptedLLM(
            [
                {"role": "assistant", "content": "第一次回答"},
                {"role": "assistant", "content": "第二次回答"},
            ]
        )
        session = Session()
        agent = Agent(session, llm=llm)

        agent.ask("问题一")
        agent.ask("问题二")

        self.assertEqual(session.turns, 2)
        users = [m["content"] for m in session.messages if m["role"] == "user"]
        self.assertEqual(users, ["问题一", "问题二"])
        # 第二次请求里应该带着上一轮的问答
        self.assertEqual(len(llm.seen[1]), 4)

    def test_events_emitted(self):
        events: list[str] = []
        llm = ScriptedLLM(
            [
                _call("c1", "inspect_table", {}),
                {"role": "assistant", "content": "好了"},
            ]
        )
        agent = Agent(Session(), llm=llm, on_event=lambda kind, payload: events.append(kind))
        agent.ask("事件测试")

        self.assertEqual(events, ["step", "tool_call", "observation", "step", "final"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
