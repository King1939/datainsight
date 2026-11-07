"""多标签页并发隔离的回归测试。

背景（2026-11-07 复现）：`data_tools` 里"当前用哪份数据"原来是模块级变量 `_active`。
标签页 A 问一句要跑十几秒，这期间 B 刷新页面，`main()` 里一句 `use_dataset(B)` 就把
这个变量改了；A 后面几步的 run_sql 于是查到 B 的表——不报错、不提示，只是数字换了。
run_python 后果更重：它按当前 key 找 CSV，会拿别人的文件算完一份"像模像样"的结果。

现在实现换成了 `contextvars.ContextVar`。这个文件钉住三条：
  1. 不同线程的 current_key 互不可见（含"新线程拿到的是默认值"这条容易忽略的）；
  2. 并发绑定各自的数据后，查询只看到自己的表；
  3. Agent 会把 session_key 注入它那一轮。

第 3 条只验"注入"，不验"还原"——`use_dataset` 本来就只写不还原（理由见它的
docstring），曾经有过的 reset_key 已经删掉：实测它对界面没有任何影响。

跑法：python -m unittest tests.test_isolation -v
"""

from __future__ import annotations

import json
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from _isolation import isolate_uploads  # noqa: E402
from app.agent import Agent  # noqa: E402
from app.llm import LLMResponse  # noqa: E402
from app.readers import Doc  # noqa: E402
from app.session import Session  # noqa: E402
from app.tools import data_tools as dt  # noqa: E402
from app.tools import registry as real_registry  # noqa: E402


def setUpModule() -> None:
    # bind_dataset 会写一份 csv 到 data/uploads/<key>，不隔离就会污染真实目录
    isolate_uploads()


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


class TestContextIsolation(unittest.TestCase):
    def tearDown(self) -> None:
        # 同一个线程里各用例共享一份上下文，不清干净会影响下一个用例
        dt.use_dataset(dt.DEFAULT_KEY)

    def test_new_thread_starts_from_default(self):
        """新线程拿到的是 default，不是主线程当下的值。

        这条看着像细节，其实很关键：Streamlit 每次 rerun 都是新线程，
        "继承上一次的 key"正是串数据的另一条路。
        """
        dt.use_dataset("iso_main")
        seen: list[str] = []

        def work() -> None:
            seen.append(dt.current_key())

        thread = threading.Thread(target=work)
        thread.start()
        thread.join(2)

        self.assertEqual(seen, [dt.DEFAULT_KEY])
        self.assertEqual(dt.current_key(), "iso_main")

    def test_concurrent_queries_see_own_table(self):
        """复现脚本的正式版：两边各自绑定、互相等一会儿，再各自连查三次。"""
        mine, theirs = threading.Event(), threading.Event()
        got: dict[str, list[str]] = {}

        def work(key: str, city: str, my_flag, other_flag) -> None:
            dt.bind_dataset(
                pd.DataFrame({"city": [city], "amount": [999]}), key=key, label=f"{key}.csv"
            )
            my_flag.set()
            other_flag.wait(3)
            time.sleep(0.05)  # 给对面留出"改当前 key"的机会
            rows = []
            for _ in range(3):
                rows.append(real_registry.call("run_sql", {"sql": "SELECT city FROM dataset"}))
                time.sleep(0.02)
            got[key] = rows

        a = threading.Thread(target=work, args=("iso_a", "北京", mine, theirs))
        b = threading.Thread(target=work, args=("iso_b", "上海", theirs, mine))
        a.start()
        b.start()
        a.join(10)
        b.join(10)

        try:
            self.assertTrue(got["iso_a"] and got["iso_b"], "两个线程都得跑完")
            for text in got["iso_a"]:
                self.assertIn("北京", text)
                self.assertNotIn("上海", text)
            for text in got["iso_b"]:
                self.assertIn("上海", text)
                self.assertNotIn("北京", text)
        finally:
            dt.unbind_dataset("iso_a")
            dt.unbind_dataset("iso_b")

    def test_documents_follow_context(self):
        """资料也有同一份问题：侧边栏显示别人上传的文件名。"""
        dt.register_document(Doc(name="我的笔记.txt", kind="txt", text="甲乙丙"), key="iso_doc_a")
        dt.register_document(Doc(name="别人的.txt", kind="txt", text="丁戊己"), key="iso_doc_b")

        dt.use_dataset("iso_doc_a")
        names = [d.name for d in dt.documents()]
        self.assertEqual(names, ["我的笔记.txt"])

        dt.use_dataset("iso_doc_b")
        self.assertEqual([d.name for d in dt.documents()], ["别人的.txt"])

        dt.clear_documents("iso_doc_a")
        dt.clear_documents("iso_doc_b")


class TestAgentCarriesSessionKey(unittest.TestCase):
    def tearDown(self) -> None:
        dt.use_dataset(dt.DEFAULT_KEY)
        dt.unbind_dataset("iso_agent")
        dt.unbind_dataset("iso_other")

    def test_agent_tools_use_its_own_dataset(self):
        """A 正在跑的时候，别处把"当前"改成别人的数据，A 的工具仍读自己的表。

        复现的是原始 bug 的时序：Agent 已进入循环 -> 另一个会话 use_dataset(B)
        -> A 的 run_sql 取到当前 key。改成 ContextVar 后 B 的改动落在 B 的上下文里，
        A 这边看不到。用线程模拟"另一个标签页"，因为 Streamlit 就是这么跑的。
        """
        dt.bind_dataset(pd.DataFrame({"city": ["广州"]}), key="iso_agent")
        dt.bind_dataset(pd.DataFrame({"city": ["成都"]}), key="iso_other")

        entered, switched = threading.Event(), threading.Event()

        class SlowLLM:
            """第一次返回工具调用前先停一下，好让主线程去改当前 key。"""

            def __init__(self) -> None:
                self.round = 0

            def __call__(self, messages, tools=None):
                self.round += 1
                if self.round == 1:
                    entered.set()
                    switched.wait(3)
                    message = _call("c1", "run_sql", {"sql": "SELECT city FROM dataset"})
                else:
                    message = {"role": "assistant", "content": "好了"}
                return LLMResponse(
                    message=message, prompt_tokens=10, completion_tokens=0, latency=0.0
                )

        session = Session()
        agent = Agent(session, llm=SlowLLM(), session_key="iso_agent")

        def run_agent() -> None:
            agent.ask("哪个城市？")

        thread = threading.Thread(target=run_agent)
        thread.start()
        self.assertTrue(entered.wait(3), "Agent 没能进入循环")
        dt.use_dataset("iso_other")  # 模拟另一个标签页 rerun
        switched.set()
        thread.join(10)

        observations = [m["content"] for m in session.messages if m["role"] == "tool"]
        self.assertEqual(len(observations), 1)
        self.assertIn("广州", observations[0])
        self.assertNotIn("成都", observations[0])

        # 主线程自己的选择不受影响
        self.assertEqual(dt.current_key(), "iso_other")

    def test_agent_without_key_leaves_context_alone(self):
        """不传 session_key 的 Agent（测试、脚本）不能被强制改成 default。"""
        dt.use_dataset("iso_outer")
        agent = Agent(Session(), llm=lambda *a, **k: LLMResponse(
            message={"role": "assistant", "content": "在"},
            prompt_tokens=1,
            completion_tokens=0,
            latency=0.0,
        ))

        agent.ask("随便问一句")

        self.assertEqual(dt.current_key(), "iso_outer")


if __name__ == "__main__":
    unittest.main()
