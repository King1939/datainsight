"""第 3 周的自动化测试：沙箱策略 + 自修复闭环。

跑法：
    python -m unittest discover -s tests -v

这周的测试分三层，对应三个"如果坏了你会很久才发现"的地方：

    1. 策略层（check_code）：不执行代码就能判断越权，速度快，必须最先测。
    2. 执行层（sandbox.run）：超时、截断、行号定位、状态隔离。
    3. 集成层（run_python + Agent）：错误真的回到了模型手里，模型真的能改对。

第 3 层最重要——它测的不是"沙箱能挡住"，而是"挡住之后 Agent 还能继续工作"。
沙箱挡得太狠、错误信息写得太差，模型就永远修不好，那等于把工具废掉了。
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import sandbox  # noqa: E402
from app.config import DATA_FILE  # noqa: E402  固定固件，用例都跑在它上面
from app.agent import Agent  # noqa: E402
from app.llm import LLMResponse  # noqa: E402
from app.session import Session  # noqa: E402
from app.tools import registry as real_registry  # noqa: E402


def _call(call_id: str, name: str, args: dict) -> dict:
    import json

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
    def __init__(self, script: list[dict]) -> None:
        self.script = list(script)

    def __call__(self, messages, tools=None):
        message = self.script.pop(0) if self.script else {"role": "assistant", "content": "结束"}
        return LLMResponse(message=message, prompt_tokens=10, completion_tokens=0, latency=0.0)


# ---------------------------------------------------------------- 策略层


class TestGuard(unittest.TestCase):
    def test_blocks_dangerous_import(self):
        for code in (
            "import os\nos.system('echo hi')",
            "import subprocess",
            "from shutil import rmtree",
            "import sys",
        ):
            with self.subTest(code=code):
                with self.assertRaises(sandbox.SandboxError) as ctx:
                    sandbox.check_code(code)
                self.assertEqual(ctx.exception.kind, "PermissionError")

    def test_blocks_dangerous_calls(self):
        for code in (
            "open('x.txt', 'w')",
            "__import__('os')",
            "eval('1+1')",
            "exec('x=1')",
            "getattr(df, 'shape')",
        ):
            with self.subTest(code=code):
                with self.assertRaises(sandbox.SandboxError):
                    sandbox.check_code(code)

    def test_blocks_dunder_escape(self):
        for code in ("df.__class__", "df.__class__.__mro__", "pd.__dict__"):
            with self.subTest(code=code):
                with self.assertRaises(sandbox.SandboxError):
                    sandbox.check_code(code)

    def test_allows_whitelisted(self):
        for code in (
            "import pandas as pd",
            "import math\nmath.sqrt(4)",
            "from datetime import date",
            "df.groupby('city')['amount'].sum()",
        ):
            with self.subTest(code=code):
                sandbox.check_code(code)  # 不抛就是通过

    def test_syntax_error_kind_and_line(self):
        with self.assertRaises(sandbox.SandboxError) as ctx:
            sandbox.check_code("df.groupby('city'")
        # 语法错和越权要分开：模型该改语法，不该换思路
        self.assertEqual(ctx.exception.kind, "SyntaxError")
        self.assertIn("第 1 行", str(ctx.exception))

    def test_file_modules_point_to_the_right_tool(self):
        """拦下 zipfile / pptx 还不够，得告诉它该用哪个工具。

        真实事故：让它"读一下桌面那份 PPT 再往里加一页"，它 import pptx 被拒，
        接着试 zipfile、zlib、__import__，六次无效调用才绕到 sys.modules。
        白名单第一次报错就写全了，缺的是指路那一句。
        """
        for code in ("import zipfile", "import zlib", "from pptx import Presentation", "import openpyxl"):
            with self.subTest(code=code):
                with self.assertRaises(sandbox.SandboxError) as ctx:
                    sandbox.check_code(code)
                message = str(ctx.exception)
                self.assertIn("read_file", message)
                self.assertIn("edit_pptx", message)


# ---------------------------------------------------------------- 执行层


class TestSandbox(unittest.TestCase):
    def test_normal_calc(self):
        result = sandbox.run("df['amount'].sum().round(2)", DATA_FILE)
        self.assertTrue(result.ok, result.as_observation())
        # 固定数据集的总和。改数据生成脚本时这条会红——那正是它存在的意义。
        self.assertAlmostEqual(float(result.output), 812721.41, places=2)

    def test_print_and_last_expression(self):
        result = sandbox.run("print('第一步')\ndf.shape", DATA_FILE)
        self.assertTrue(result.ok)
        self.assertIn("第一步", result.output)
        self.assertIn("320", result.output)

    def test_runtime_error_reports_line(self):
        code = "x = 1\ny = 0\nprint('算到一半')\nz = x / y"
        result = sandbox.run(code, DATA_FILE)
        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "ZeroDivisionError")
        self.assertEqual(result.lineno, 4)
        self.assertIn("z = x / y", result.source_line)
        # 崩溃前的 print 不能丢——模型靠它判断"执行到哪一步了"
        self.assertIn("算到一半", result.as_observation())

    def test_timeout_kills_child(self):
        started = time.time()
        result = sandbox.run("i = 0\nwhile True:\n    i += 1", DATA_FILE, timeout=1.2)
        elapsed = time.time() - started
        self.assertEqual(result.error_type, "TimeoutError")
        self.assertLess(elapsed, 15, "超时后必须真的杀掉子进程")

    def test_output_truncated(self):
        result = sandbox.run("for i in range(20000):\n    print(i)", DATA_FILE, timeout=20)
        self.assertTrue(result.ok)
        self.assertLessEqual(len(result.output), sandbox.MAX_OBS_CHARS)

    def test_state_isolated_between_runs(self):
        """每次都是全新进程：上一次的改动不会漏到下一次。"""
        sandbox.run("df['leak'] = 1", DATA_FILE)
        result = sandbox.run("'leak' in df.columns", DATA_FILE)
        self.assertEqual(result.output, "False")

    def test_no_exception_escapes(self):
        """run() 的契约：永远返回 ExecResult，永远不抛。"""
        for code in ("import os", "1/0", "df.groupby('city'", "", "   "):
            with self.subTest(code=code):
                self.assertIsInstance(sandbox.run(code, DATA_FILE), sandbox.ExecResult)


# ---------------------------------------------------------------- 集成层


class TestToolIntegration(unittest.TestCase):
    def test_tool_returns_string(self):
        out = real_registry.call("run_python", {"code": "df['quantity'].max()"})
        self.assertIsInstance(out, str)
        self.assertTrue(out.isdigit(), out)

    def test_tool_error_is_observation(self):
        out = real_registry.call("run_python", {"code": "import os"})
        self.assertIn("不允许 import os", out)
        self.assertNotIn("Traceback", out)


class TestSelfRepair(unittest.TestCase):
    """这是本周真正的交付物：报错 -> 模型看懂 -> 改对。

    沙箱只会拒绝是不够的。错误信息如果只写"执行失败"，模型会原地重试同一段代码，
    直到撞上重复调用检测。所以断言的不只是"失败了"，还有**错误里有没有可行动信息**。
    """

    def test_agent_fixes_wrong_column(self):
        llm = ScriptedLLM(
            [
                # 第 1 次：模型猜错了列名（真实场景里非常常见）
                _call("c1", "run_python", {"code": "df['revenue'].sum()"}),
                # 第 2 次：看到 KeyError 之后改对
                _call("c2", "run_python", {"code": "df['amount'].sum().round(2)"}),
                {"role": "assistant", "content": "总销售额 1,234,567 元。"},
            ]
        )
        session = Session()
        agent = Agent(session, llm=llm)

        answer = agent.ask("总销售额是多少？")

        self.assertEqual(answer, "总销售额 1,234,567 元。")
        observations = [m["content"] for m in session.messages if m["role"] == "tool"]
        self.assertEqual(len(observations), 2)

        first, second = observations
        self.assertIn("KeyError", first)
        self.assertIn("出错位置", first)
        self.assertIn("revenue", first)          # 让它看到自己写错的那个列名
        self.assertNotIn("错误", second)          # 第二次成功
        self.assertEqual(session.tool_calls, 2)

    def test_agent_gives_up_after_repeat(self):
        """模型死不改代码时，不能在同一个错误上无限打转。"""
        same = _call("c1", "run_python", {"code": "df['revenue'].sum()"})
        llm = ScriptedLLM([same, same, same, same])
        agent = Agent(Session(), llm=llm)

        answer = agent.ask("总销售额是多少？")
        self.assertIn("中止", answer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
