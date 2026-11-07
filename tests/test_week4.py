"""第 4 周测试：界面渲染逻辑 + 数据集切换。

这里刻意不 import streamlit —— ui_render 里全是纯函数，可以直接断言。
界面布局本身没法用 unittest 验证，那属于"打开浏览器看一眼"的范畴。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from _isolation import isolate_uploads
from app.agent import SYSTEM_PROMPT, Agent, make_system_prompt
from app.llm import LLMResponse
from app.session import Session
from app.tools import data_tools as dt
from app.tools import registry
from app.ui_render import (
    TraceLog,
    build_item,
    dataset_badge,
    extract_chart,
    messages_to_chat,
    observation_body,
    prettify_args,
    read_any_csv,
    suggest_questions,
    tool_label,
)


def setUpModule() -> None:
    """数据集切换的用例会 bind 好几份 DataFrame，每份都往 uploads 落一个 csv。

    不隔开的话，跑一次测试就在用户真实的上传目录里留下 ut_a、ut_b… 这些
    34 字节的垃圾（2026-11-05 攒到 367 个就是这么来的）。
    """
    isolate_uploads()


class TestReadCsv(unittest.TestCase):
    def test_utf8(self):
        frame = read_any_csv("城市,金额\n上海,10\n北京,20\n".encode("utf-8"))
        self.assertIsNotNone(frame)
        self.assertEqual(list(frame.columns), ["城市", "金额"])
        self.assertEqual(len(frame), 2)

    def test_utf8_with_bom(self):
        frame = read_any_csv("城市,金额\n上海,10\n".encode("utf-8-sig"))
        self.assertEqual(list(frame.columns), ["城市", "金额"])

    def test_gbk(self):
        """Excel 在中文 Windows 上导出的 CSV 默认是 GBK。"""
        frame = read_any_csv("城市,金额\n上海,10\n".encode("gbk"))
        self.assertIsNotNone(frame)
        self.assertEqual(list(frame.columns), ["城市", "金额"])

    def test_garbage(self):
        self.assertIsNone(read_any_csv(b"\x00\x01\x02 binary junk"))


class TestChartExtract(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.png = self.dir / "chart_1793628571457.png"
        self.png.write_bytes(b"fake png")

    def test_found(self):
        text = f"图表已生成：{self.png.as_posix()}\n类型 bar，3 个数据点"
        self.assertEqual(extract_chart(text), str(self.png))

    def test_file_missing(self):
        self.assertIsNone(extract_chart("图表已生成：C:/nope/chart_1.png"))

    def test_not_a_chart(self):
        self.assertIsNone(extract_chart("city | t\n上海 | 241589.06"))

    def test_empty(self):
        self.assertIsNone(extract_chart(""))


class TestArgs(unittest.TestCase):
    def test_pretty(self):
        out = prettify_args('{"sql": "SELECT 1", "max_rows": 5}')
        self.assertIn("\n", out)
        self.assertIn('"sql"', out)

    def test_bad_json_passthrough(self):
        self.assertEqual(prettify_args("not a json"), "not a json")

    def test_empty(self):
        self.assertEqual(prettify_args(""), "{}")


class TestBuildItem(unittest.TestCase):
    def test_step(self):
        item = build_item("step", {"step": 2, "tokens": 1234, "latency": 1.5})
        self.assertIn("第 2 步", item.title)
        self.assertIn("1234", item.body)

    def test_tool_call(self):
        item = build_item("tool_call", {"name": "run_sql", "args": '{"sql": "SELECT 1"}'})
        self.assertIn("执行 SQL", item.title)
        self.assertTrue(item.code)

    def test_observation_with_chart(self):
        path = Path(tempfile.mkdtemp()) / "chart_1.png"
        path.write_bytes(b"x")
        item = build_item("observation", {"name": "make_chart", "text": f"图表已生成：{path.as_posix()}"})
        self.assertEqual(item.chart, str(path))

    def test_final_warning_error(self):
        self.assertEqual(build_item("final", {"text": "答案"}).body, "答案")
        self.assertIn("中止", build_item("warning", {"message": "超出预算"}).title)
        self.assertIn("失败", build_item("error", {"message": "401"}).title)

    def test_unknown_event_ignored(self):
        self.assertIsNone(build_item("something_new", {}))

    def test_long_body_truncated(self):
        item = build_item("observation", {"name": "run_sql", "text": "x" * 5000})
        self.assertLess(len(item.to_markdown()), 2000)

    def test_observation_hides_file_content(self):
        """读文件的正文不进「思考过程」，只留首行摘要和字数。"""
        outline = (
            "`C:\\x\\汇报.pptx`（12 页 · 16:9 · 商务蓝 · 边距 0.625 英寸）\n\n"
            + "\n".join(f"【第 {i} 页】bullets：标题{i}\n  [1] lead 这一页的正文内容" for i in range(1, 12))
        )
        item = build_item("observation", {"name": "read_file", "text": outline})

        self.assertIn("12 页", item.body)          # 首行摘要还在
        self.assertIn("正文已省略", item.body)
        self.assertIn(f"{len(outline):,} 字", item.body)
        self.assertNotIn("这一页的正文内容", item.body)  # 正文没了
        self.assertLess(len(item.body), 200)

    def test_short_observation_kept_whole(self):
        """短返回整段留着：改文件的确认信息本来就一两行，藏了就看不懂了。"""
        text = "已更新：`C:\\x\\汇报.pptx`（6 页 → 7 页，2 处改动）"
        self.assertEqual(observation_body(text), text)

    def test_error_kept_even_if_long(self):
        text = "错误：读不了 `x.pptx`（Package not found）。" + "细节" * 200
        self.assertEqual(observation_body(text), text)

    def test_collapsed_body_still_extracts_chart(self):
        """正文折叠了，嵌图和下载按钮不能跟着丢——它们是从原始返回里取的。"""
        path = Path(tempfile.mkdtemp()) / "chart_long.png"
        path.write_bytes(b"x")
        text = f"图表已生成：{path.as_posix()}\n" + "数据明细" * 200
        item = build_item("observation", {"name": "make_chart", "text": text})

        self.assertEqual(item.chart, str(path))
        self.assertNotIn("数据明细", item.body)

    def test_status_return_kept_whole(self):
        """编辑类工具的返回是"改了哪几页、备份在哪"，几百字也得整段留着。"""
        text = "\n".join(
            [
                "已更新：C:\\x\\汇报.pptx",
                "12 页 → 13 页 · 3 处改动 · 沿用原文件 16:9 · 商务蓝 · 边距 0.625 英寸",
                "- 第 2 页后插入 bullets「四季度重点」",
                "- 第 5 页 [2] 已替换（11 字 → 9 字）",
                "- 第 8 页 [3] 已替换（24 字 → 31 字，字号压到 16pt 才放下）",
                "备份：C:\\x\\汇报.bak-20261105-1802.pptx",
            ]
        )
        self.assertGreater(len(text), 150)  # 几百字，但它是状态不是正文
        self.assertEqual(observation_body(text, "edit_pptx"), text)

    def test_notice_survives_collapse(self):
        """正文折叠了，「注意：压到页脚」这类告警不能跟着消失。"""
        text = "`C:\\x\\汇报.pptx`（12 页 · 16:9）\n\n" + "【第 1 页】正文内容\n" * 80 + "注意：第 3 页正文压到页脚区"
        body = observation_body(text)

        self.assertIn("注意：第 3 页正文压到页脚区", body)
        self.assertIn("正文已省略", body)
        self.assertNotIn("【第 1 页】正文内容", body)

    def test_single_line_short_enough_kept(self):
        """只有一行、又没超摘要长度：那就是全部内容，别先说一遍再说省略了。"""
        text = "这" * 190
        self.assertEqual(observation_body(text, "read_file"), text)

    def test_long_first_line_falls_back_to_count(self):
        """首行本身就是正文（没有摘要行）时，一个字都不露。"""
        text = "正" * 300 + "\n后面还有"
        self.assertEqual(
            observation_body(text, "read_file"), f"…（共 {len(text)} 字，正文已省略）"
        )

    def test_tool_label_fallback(self):
        self.assertEqual(tool_label("unknown_tool"), "unknown_tool")


class TestTraceLog(unittest.TestCase):
    def test_empty_placeholder(self):
        self.assertIn("等待", TraceLog().to_markdown())

    def test_collects_in_order(self):
        log = TraceLog()
        log.push("step", {"step": 1, "tokens": 10, "latency": 0})
        log.push("tool_call", {"name": "run_sql", "args": "{}"})
        log.push("final", {"text": "完了"})
        self.assertEqual([i.kind for i in log.items], ["step", "tool_call", "final"])
        self.assertIn("第 1 步", log.to_markdown())

    def test_charts(self):
        path = Path(tempfile.mkdtemp()) / "chart_9.png"
        path.write_bytes(b"x")
        log = TraceLog()
        log.push("observation", {"name": "make_chart", "text": f"图表已生成：{path.as_posix()}"})
        log.push("observation", {"name": "run_sql", "text": "a | b\n1 | 2"})
        self.assertEqual(log.charts, [str(path)])


class TestMessagesToChat(unittest.TestCase):
    def test_skips_protocol_noise(self):
        messages = [
            {"role": "system", "content": "系统"},
            {"role": "user", "content": "问"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "run_sql", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "结果"},
            {"role": "assistant", "content": "答"},
        ]
        turns = messages_to_chat(messages)
        self.assertEqual([t["role"] for t in turns], ["user", "assistant"])
        self.assertEqual(turns[-1]["text"], "答")


class TestBadgeAndSuggest(unittest.TestCase):
    def test_badge_with_stats(self):
        text = dataset_badge({"label": "upload.csv", "rows": 320, "columns": 8})
        self.assertIn("upload.csv", text)
        self.assertIn("320", text)

    def test_badge_without_stats(self):
        self.assertIn("a.csv", dataset_badge({"label": "a.csv", "rows": None}))

    def test_badge_without_label(self):
        """没有文件名时不能编一个出来，也不能退回成引导语。"""
        text = dataset_badge({"label": None, "rows": 10, "columns": 3})
        self.assertIn("10", text)
        self.assertNotIn("sales.csv", text)
        self.assertNotIn("上传", text)

    def test_suggest_by_columns(self):
        questions = suggest_questions(["city", "amount", "order_date", "category"])
        joined = " ".join(questions)
        self.assertIn("城市", joined)
        self.assertIn("趋势", joined)
        self.assertIn("占比", joined)

    def test_suggest_no_match(self):
        self.assertTrue(suggest_questions(["foo", "bar"]))


class TestDatasetSwitch(unittest.TestCase):
    def tearDown(self):
        dt.use_dataset("default")

    def test_bind_dataframe(self):
        dt.bind_dataset(pd.DataFrame({"a": [1, 2], "b": [3, 4]}), key="ut_a", label="mine.csv")
        self.assertEqual(dt.column_names(), ["a", "b"])
        self.assertEqual(dt.dataset_meta()["label"], "mine.csv")
        self.assertTrue(dt.active_data_file().exists())

    def test_sql_and_python_see_same_rows(self):
        """最容易出错的一条：SQL 读 duckdb，Python 读子进程里的 CSV。"""
        dt.bind_dataset(pd.DataFrame({"a": [1, 2, 3]}), key="ut_b")
        sql_rows = registry.call("run_sql", {"sql": "SELECT COUNT(*) n FROM dataset"}).splitlines()[-1]
        py_rows = registry.call("run_python", {"code": "len(df)"})
        self.assertEqual(sql_rows.strip(), "3")
        self.assertEqual(py_rows.strip(), "3")

    def test_two_sessions_isolated(self):
        dt.bind_dataset(pd.DataFrame({"a": [1]}), key="ut_c1")
        dt.bind_dataset(pd.DataFrame({"a": [1, 2, 3]}), key="ut_c2")

        dt.use_dataset("ut_c1")
        one = registry.call("run_sql", {"sql": "SELECT COUNT(*) n FROM dataset"}).splitlines()[-1]
        dt.use_dataset("ut_c2")
        three = registry.call("run_sql", {"sql": "SELECT COUNT(*) n FROM dataset"}).splitlines()[-1]
        self.assertEqual((one.strip(), three.strip()), ("1", "3"))

    def test_back_to_sample(self):
        dt.bind_dataset(pd.DataFrame({"a": [1]}), key="ut_d")
        dt.bind_sample()  # 现在必须显式加载示例数据，不再有隐式回退
        self.assertEqual(len(dt.column_names()), 8)
        self.assertEqual(dt.dataset_meta()["label"], "sales.csv")

    def test_no_silent_fallback(self):
        """没有绑定数据时，工具必须报错，绝不能偷偷去读 data/sales.csv。"""
        dt.use_dataset("ut_never_bound")
        self.assertIsNone(dt.active_data_file())
        self.assertIsNone(dt.dataset_meta()["label"])

        sql = registry.call("run_sql", {"sql": "SELECT COUNT(*) n FROM dataset"})
        self.assertIn("还没有选择数据集", sql)
        self.assertNotIn("320", sql)

        code = registry.call("run_python", {"code": "len(df)"})
        self.assertIn("还没有选择数据集", code)
        dt.use_dataset("default")

    def test_uploaded_csv_persisted(self):
        dt.bind_dataset(pd.DataFrame({"x": [9]}), key="ut_e")
        path = dt.active_data_file()
        self.assertTrue(path.name.startswith("ut_e"))
        self.assertIn("x", path.read_text(encoding="utf-8").splitlines()[0])

    def test_unbind_really_removes(self):
        """点"移除当前数据"后必须真的没了。

        只把 _active 指到别的 key 是假移除：连接还在 _conns 里，
        下次切回来旧表就复活了。
        """
        dt.bind_dataset(pd.DataFrame({"a": [1]}), key="ut_unbind")
        self.assertTrue(dt.has_dataset("ut_unbind"))

        dt.unbind_dataset("ut_unbind")
        self.assertFalse(dt.has_dataset("ut_unbind"))

        dt.use_dataset("ut_unbind")
        self.assertIn("还没有选择数据集", registry.call("run_sql", {"sql": "SELECT 1"}))
        dt.use_dataset("default")


class TestSystemPrompt(unittest.TestCase):
    def test_fallback_to_default(self):
        self.assertEqual(make_system_prompt(), SYSTEM_PROMPT)
        self.assertEqual(make_system_prompt([]), SYSTEM_PROMPT)

    def test_contains_real_columns(self):
        prompt = make_system_prompt(["产品", "销量"], 3, "demo.csv")
        self.assertIn("产品", prompt)
        self.assertIn("销量", prompt)
        self.assertIn("demo.csv", prompt)
        self.assertIn("3", prompt)

    def test_agent_uses_custom_prompt(self):
        captured: dict = {}

        def fake_llm(messages, tools=None):
            captured["system"] = messages[0]
            return LLMResponse(
                message={"role": "assistant", "content": "ok"},
                prompt_tokens=1,
                completion_tokens=1,
                latency=0.0,
            )

        agent = Agent(Session(), llm=fake_llm, system_prompt="只针对本次上传的数据")
        agent.ask("问题")
        self.assertEqual(captured["system"]["role"], "system")
        self.assertEqual(captured["system"]["content"], "只针对本次上传的数据")


if __name__ == "__main__":
    unittest.main()
