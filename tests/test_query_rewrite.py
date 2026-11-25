"""前置查询改写的自动化测试。

跑法（不需要 pytest，stdlib 自带）：
    python -m unittest tests/test_query_rewrite.py -v

全部用离线伪造的 llm 回调：不发网络请求、不依赖 API Key。
改写层本身就是"调模型 -> 解析 JSON"，所以测它的正确姿势是把模型替换成
一个固定返回某段 JSON 的函数，然后盯住三件事：
  1. 该短路的时候真的没调模型（0 token 不能被规则误伤）；
  2. 模型乱填的时候裁得干净（去重、上限、破环），而不是把脏数据透给下游；
  3. 任何失败都退成透传，绝不抛异常。
"""

from __future__ import annotations

import json
import sys
import unittest
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _isolation import isolate_uploads  # noqa: E402
from app.agent import Agent  # noqa: E402
from app.llm import LLMError, LLMResponse  # noqa: E402
from app.query_rewrite import (  # noqa: E402
    QueryRewriter,
    RewriteContext,
    RewriteStrategy,
    build_context,
    format_for_agent,
    parse_rewrite_response,
    retrieval_units,
)
from app.session import Session  # noqa: E402
from app.tools import data_tools as dt  # noqa: E402
from app.ui_render import build_item  # noqa: E402


def setUpModule() -> None:
    """数据层不再隐式加载示例 CSV，测试要自己声明用它。

    上传目录也要隔开：这份用例会往 `uploads/<key>/` 落一份资料来验"只取摘要"，
    不隔开就把测试文件写进用户真实的 data/uploads 了。
    """
    isolate_uploads("uploads-rw-")
    dt.bind_sample()
    dt.use_dataset(dt.DEFAULT_KEY)


# ---------------------------------------------------------------- 测试替身


def _payload(**fields) -> str:
    return json.dumps(fields, ensure_ascii=False)


BASIC_PAYLOAD = _payload(
    strategy="basic_rewrite",
    reason="补全指代",
    rewritten_query="北京各城市销售额是多少",
)


class FakeLLM:
    """按剧本返回固定 JSON 的假模型。calls 用来验证"有没有真的调"。"""

    def __init__(self, content: str = BASIC_PAYLOAD, *, tokens: int = 30) -> None:
        self.content = content
        self.tokens = tokens
        self.calls = 0
        self.seen: list[list[dict]] = []

    def __call__(self, messages, tools=None):
        self.calls += 1
        self.seen.append([m for m in messages])
        return LLMResponse(
            message={"role": "assistant", "content": self.content},
            prompt_tokens=self.tokens,
            completion_tokens=0,
            latency=0.0,
        )


class BoomLLM:
    """一调就炸。用来验证 rewrite 不把异常漏给主流程。"""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def __call__(self, messages, tools=None):
        self.calls += 1
        raise self.error


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
    """主循环用的假模型：按剧本依次返回消息。"""

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


def _user_texts(seen: list[list[dict]]) -> list[str]:
    """每次请求里那条 user 消息的内容（第 0 条永远是 system）。"""
    return [next(m["content"] for m in messages if m["role"] == "user") for messages in seen]


def _assert_acyclic(case: unittest.TestCase, subs) -> None:
    """每条子查询的依赖都必须落在它前面——这就是"无环"的可验证定义。"""
    seen: set[str] = set()
    for sub in subs:
        for dep in sub.depends_on:
            case.assertIn(dep, seen, f"{sub.id} 依赖了还没出现的 {dep}（成环或顺序错）")
        seen.add(sub.id)


# ---------------------------------------------------------------- 快速通道


class TestFastBypass(unittest.TestCase):
    def test_disabled_never_calls_model(self):
        llm = FakeLLM()
        result = QueryRewriter(llm=llm, enabled=False).rewrite("各城市销售额排名前五")

        self.assertEqual(result.strategy, RewriteStrategy.PASS_THROUGH)
        self.assertTrue(result.bypassed)
        self.assertEqual(llm.calls, 0)

    def test_blank_question_bypasses(self):
        for question in ("", "   ", "\n\t "):
            llm = FakeLLM()
            result = QueryRewriter(llm=llm).rewrite(question)
            self.assertTrue(result.bypassed, question)
            self.assertEqual(llm.calls, 0, question)

    def test_control_phrases_bypass(self):
        for question in ("继续", "好的。", "谢谢！", "OK", "算了"):
            llm = FakeLLM()
            result = QueryRewriter(llm=llm).rewrite(question)
            self.assertTrue(result.bypassed, question)
            self.assertEqual(result.strategy, RewriteStrategy.PASS_THROUGH)
            self.assertEqual(llm.calls, 0, question)

    def test_single_open_command_in_write_mode_bypasses(self):
        llm = FakeLLM()
        result = QueryRewriter(llm=llm).rewrite(
            "打开微信", RewriteContext(write_mode=True)
        )

        self.assertTrue(result.bypassed)
        self.assertEqual(llm.calls, 0)

    def test_open_command_with_compound_task_is_not_bypassed(self):
        """“打开微信并统计上月营收”是两件事，必须交给模型拆。"""
        llm = FakeLLM()
        result = QueryRewriter(llm=llm).rewrite(
            "打开微信并统计上月营收", RewriteContext(write_mode=True)
        )

        self.assertFalse(result.bypassed)
        self.assertEqual(llm.calls, 1)

    def test_open_command_needs_write_mode(self):
        llm = FakeLLM()
        result = QueryRewriter(llm=llm).rewrite(
            "打开微信", RewriteContext(write_mode=False)
        )

        self.assertFalse(result.bypassed)
        self.assertEqual(llm.calls, 1)

    def test_short_follow_ups_are_not_short_circuited(self):
        """短追问是最需要结合上下文的一类，绝不能因为字数少就放过。"""
        for question in ("北京呢？", "为什么？", "画个图", "占比呢"):
            llm = FakeLLM()
            result = QueryRewriter(llm=llm).rewrite(question)

            self.assertFalse(result.bypassed, question)
            self.assertEqual(llm.calls, 1, question)


# ---------------------------------------------------------------- 上下文采集


class TestBuildContext(unittest.TestCase):
    def test_only_clean_conversation_is_kept(self):
        messages = [
            {"role": "system", "content": "系统提示"},
            {"role": "user", "content": "各城市销售额排名"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "function": {"name": "run_sql", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "上海 100"},
            {"role": "assistant", "content": "上海最高。"},
            {"role": "user", "content": "北京呢？"},
        ]

        ctx = build_context(messages)

        self.assertEqual(
            ctx.history,
            (("user", "各城市销售额排名"), ("assistant", "上海最高。"), ("user", "北京呢？")),
        )

    def test_history_limit_and_truncation(self):
        messages = [
            {"role": "user", "content": f"问题{index}"} for index in range(10)
        ]

        ctx = build_context(messages, history_turns=2, max_msg_chars=5)

        self.assertEqual(len(ctx.history), 4)
        self.assertEqual(ctx.history[-1], ("user", "问题9"))

        long_ctx = build_context(
            [{"role": "user", "content": "一" * 100}], max_msg_chars=10
        )
        text = long_ctx.history[0][1]
        self.assertEqual(len(text), 11)                  # 10 个字 + 省略号
        self.assertTrue(text.endswith("…"))

    def test_table_metadata_is_collected(self):
        ctx = build_context([], session_key=dt.DEFAULT_KEY)

        meta = dt.dataset_meta()
        self.assertTrue(ctx.has_table)
        self.assertEqual(ctx.table_label, meta["label"])
        self.assertEqual(ctx.table_rows, meta["rows"])
        self.assertEqual(set(ctx.columns), set(dt.column_names()))
        self.assertFalse(ctx.write_mode)

    def test_documents_are_digest_only(self):
        # key 和文件名都带随机后缀：同名文件会被 add_document 自动改成 (2)(3)，
        # 上一次跑剩下的孤儿文件会让这条断言莫名其妙地红。
        tag = uuid.uuid4().hex[:8]
        key = f"rw-docs-{tag}"
        name = f"改写测试资料-{tag}.txt"
        body = "这段正文只应该留在文档里，绝不该出现在改写上下文里。"
        try:
            dt.add_document(name, body.encode("utf-8"), key=key)
            ctx = build_context([], session_key=key)

            self.assertEqual(len(ctx.documents), 1)
            digest = ctx.documents[0]
            self.assertEqual(set(digest), {"name", "kind", "chars", "parts"})
            self.assertEqual(digest["name"], name)
            self.assertEqual(digest["chars"], len(body))
            self.assertNotIn(body, json.dumps(ctx.documents, ensure_ascii=False))
        finally:
            dt.clear_documents(key)
            dt.use_dataset(dt.DEFAULT_KEY)

    def test_prompt_lists_sources(self):
        ctx = RewriteContext(
            history=(("user", "先看整体"),),
            has_table=True,
            table_label="我的数据.csv",
            columns=("城市", "销售额"),
            table_rows=320,
            write_mode=False,
        )

        from app.query_rewrite import build_rewrite_user_prompt

        prompt = build_rewrite_user_prompt("北京呢？", ctx)
        self.assertIn("我的数据.csv", prompt)
        self.assertIn("320", prompt)
        self.assertIn("城市、销售额", prompt)
        self.assertIn("U: 先看整体", prompt)
        self.assertIn("只读", prompt)
        self.assertIn("北京呢？", prompt)

    def test_prompt_hints_english_docs(self):
        from app.query_rewrite import build_rewrite_user_prompt

        cn_only = RewriteContext(
            documents=({"name": "区块链教程.pdf", "kind": "pdf", "chars": 1000, "parts": 2},)
        )
        with_en = RewriteContext(
            documents=(
                {"name": "区块链教程.pdf", "kind": "pdf", "chars": 1000, "parts": 2},
                {"name": "bitcoin_en.pdf", "kind": "pdf", "chars": 500, "parts": 1},
            )
        )
        self.assertNotIn(
            "english_keywords", build_rewrite_user_prompt("怎么验证支付", cn_only)
        )
        self.assertIn(
            "english_keywords", build_rewrite_user_prompt("怎么验证支付", with_en)
        )


# ---------------------------------------------------------------- 策略与裁剪


class TestStrategiesAndValidation(unittest.TestCase):
    def test_pass_through_keeps_original(self):
        raw = _payload(
            strategy="pass_through", reason="表述清晰", rewritten_query="模型乱写的东西"
        )
        result = parse_rewrite_response(raw, "这张表有多少行？")

        self.assertEqual(result.strategy, RewriteStrategy.PASS_THROUGH)
        self.assertEqual(result.rewritten_query, "这张表有多少行？")
        self.assertEqual(result.expanded_terms, ())
        self.assertEqual(result.sub_queries, ())
        self.assertFalse(result.changed)

    def test_basic_rewrite_completes_context(self):
        raw = _payload(
            strategy="basic_rewrite", reason="补全指代", rewritten_query="北京各城市销售额"
        )
        result = parse_rewrite_response(raw, "北京呢？")

        self.assertEqual(result.strategy, RewriteStrategy.BASIC_REWRITE)
        self.assertEqual(result.effective_query, "北京各城市销售额")
        self.assertTrue(result.changed)

    def test_basic_rewrite_without_change_degrades(self):
        """改完还是原话 -> 别假装做过改写，否则每轮都多一条空卡片。"""
        raw = _payload(
            strategy="basic_rewrite", reason="看不出要改什么", rewritten_query="北京 呢？"
        )
        result = parse_rewrite_response(raw, "北京呢？")

        self.assertEqual(result.strategy, RewriteStrategy.PASS_THROUGH)
        self.assertEqual(result.rewritten_query, "北京呢？")
        self.assertEqual(result.reason, "看不出要改什么")

    def test_query_expansion_cleans_terms(self):
        raw = _payload(
            strategy="query_expansion",
            reason="术语扩展",
            rewritten_query="违约金如何赔付",
            expanded_terms=[
                "违约金", "违约金", "赔偿金", "liquidated damages", "甲", "违约怎么赔",
            ],
            search_keywords=["违约金 赔付", "违约金 赔付", "", "赔偿 条款"],
        )
        result = parse_rewrite_response(raw, "违约怎么赔")

        self.assertEqual(result.strategy, RewriteStrategy.QUERY_EXPANSION)
        # 去重、丢掉单字符噪声、丢掉与原文完全相同的词
        self.assertEqual(
            result.expanded_terms, ("违约金", "赔偿金", "liquidated damages")
        )
        self.assertEqual(result.search_keywords, ("违约金 赔付", "赔偿 条款"))

    def test_query_expansion_caps_terms(self):
        raw = _payload(
            strategy="query_expansion",
            rewritten_query="供应链风险条款",
            expanded_terms=[f"同义词{index}" for index in range(10)],
            search_keywords=[f"词{index} 组" for index in range(8)],
        )
        result = parse_rewrite_response(raw, "查一下供应链风险相关条款")

        self.assertEqual(len(result.expanded_terms), 6)
        self.assertEqual(len(result.search_keywords), 4)

    def test_empty_expansion_degrades(self):
        raw = _payload(
            strategy="query_expansion",
            reason="没有可用扩展",
            rewritten_query="违约金赔付标准",
            expanded_terms=["甲", ""],
            search_keywords=[],
        )
        result = parse_rewrite_response(raw, "违约怎么赔")

        self.assertEqual(result.strategy, RewriteStrategy.BASIC_REWRITE)
        self.assertEqual(result.effective_query, "违约金赔付标准")

    def test_empty_expansion_without_change_degrades_to_pass(self):
        raw = _payload(
            strategy="query_expansion",
            rewritten_query="违约怎么赔",
            expanded_terms=[],
            search_keywords=[],
        )
        result = parse_rewrite_response(raw, "违约怎么赔")
        self.assertEqual(result.strategy, RewriteStrategy.PASS_THROUGH)

    def test_query_expansion_cleans_and_caps_english_keywords(self):
        raw = _payload(
            strategy="query_expansion",
            rewritten_query="如何验证支付",
            expanded_terms=[],
            search_keywords=[],
            english_keywords=[
                "  Simplified   Payment Verification  ",
                "Simplified Payment Verification",
                "包含中文词",
                "12345",
                "a",
                "Merkle branch proof",
                "SPV light node",
                "fourth group dropped",
            ],
        )
        result = parse_rewrite_response(raw, "如何验证支付")

        self.assertEqual(result.strategy, RewriteStrategy.QUERY_EXPANSION)
        self.assertEqual(
            result.english_keywords,
            (
                "Simplified Payment Verification",
                "Merkle branch proof",
                "SPV light node",
            ),
        )
        self.assertEqual(
            result.to_event_payload()["english_keywords"],
            list(result.english_keywords),
        )

    def test_multi_query_is_renumbered_and_decoupled(self):
        raw = _payload(
            strategy="multi_query",
            reason="多视角",
            rewritten_query="整体销售表现",
            sub_queries=[
                {"id": "A", "query": "总量趋势", "angle": "总量", "depends_on": ["B"],
                 "target_source": "table"},
                {"id": "B", "query": "区域分布", "angle": "区域", "depends_on": [],
                 "target_source": "document"},
                {"id": "C", "query": "品类结构", "angle": "品类", "depends_on": [],
                 "target_source": "瞎写"},
            ],
        )
        result = parse_rewrite_response(raw, "分析一下整体销售表现")

        self.assertEqual(result.strategy, RewriteStrategy.MULTI_QUERY)
        subs = result.sub_queries
        self.assertEqual([s.id for s in subs], ["Q1", "Q2", "Q3"])
        self.assertEqual([s.angle for s in subs], ["总量", "区域", "品类"])
        # 多路之间没有依赖：模型填了也要清掉
        self.assertTrue(all(s.depends_on == () for s in subs))
        self.assertEqual([s.target_source for s in subs], ["table", "document", "auto"])

    def test_multi_query_caps_four(self):
        raw = _payload(
            strategy="multi_query",
            rewritten_query="整体表现",
            sub_queries=[{"id": f"Q{i}", "query": f"视角{i}"} for i in range(1, 7)],
        )
        result = parse_rewrite_response(raw, "分析整体表现")

        self.assertEqual(len(result.sub_queries), 4)

    def test_multi_query_with_one_item_degrades(self):
        raw = _payload(
            strategy="multi_query",
            rewritten_query="整体表现",
            sub_queries=[{"id": "Q1", "query": "整体销售趋势与区域分布"}],
        )
        result = parse_rewrite_response(raw, "分析整体表现")

        self.assertEqual(result.strategy, RewriteStrategy.BASIC_REWRITE)
        self.assertEqual(result.effective_query, "整体销售趋势与区域分布")
        self.assertEqual(result.sub_queries, ())

    def test_decomposition_drops_bad_dependencies(self):
        raw = _payload(
            strategy="decomposition",
            rewritten_query="对比实际毛利与目标毛利",
            sub_queries=[
                {"id": "Q1", "query": "算各品类实际毛利", "purpose": "实际值",
                 "depends_on": ["Q1"], "target_source": "table"},
                {"id": "Q2", "query": "读文档里的目标毛利", "purpose": "目标值",
                 "depends_on": ["Q9"], "target_source": "document"},
                {"id": "Q3", "query": "标出未达标品类", "depends_on": ["Q1", "Q2"]},
            ],
        )
        result = parse_rewrite_response(raw, "对比各品类实际毛利和文档目标毛利")

        self.assertEqual(result.strategy, RewriteStrategy.DECOMPOSITION)
        subs = result.sub_queries
        self.assertEqual([s.id for s in subs], ["Q1", "Q2", "Q3"])
        self.assertEqual(subs[0].depends_on, ())                 # 自环被丢掉
        self.assertEqual(subs[1].depends_on, ())                 # 不存在的 ID 被丢掉
        self.assertEqual(subs[2].depends_on, ("Q1", "Q2"))
        _assert_acyclic(self, subs)

    def test_decomposition_breaks_cycles(self):
        """Q1 -> Q2 -> Q1 这种环，排不完就必须丢边，不能让下游拿到环。"""
        raw = _payload(
            strategy="decomposition",
            rewritten_query="三个互相依赖的步骤",
            sub_queries=[
                {"id": "Q1", "query": "取数", "depends_on": ["Q2"]},
                {"id": "Q2", "query": "算指标", "depends_on": ["Q3"]},
                {"id": "Q3", "query": "下结论", "depends_on": ["Q1"]},
            ],
        )
        result = parse_rewrite_response(raw, "三个互相依赖的步骤")

        subs = result.sub_queries
        self.assertEqual(len(subs), 3)
        self.assertEqual([s.id for s in subs], ["Q1", "Q2", "Q3"])
        _assert_acyclic(self, subs)

    def test_decomposition_sorts_by_dependency(self):
        """模型按重要性写、不按执行顺序写：这里必须排成能照着做的一串。"""
        raw = _payload(
            strategy="decomposition",
            rewritten_query="先找下滑城市再查原因",
            sub_queries=[
                {"id": "Q1", "query": "查文档里的原因", "depends_on": ["Q2"]},
                {"id": "Q2", "query": "找出销售额下滑最多的城市", "depends_on": []},
            ],
        )
        result = parse_rewrite_response(raw, "找出下滑城市并查原因")

        self.assertEqual([s.query for s in result.sub_queries],
                         ["找出销售额下滑最多的城市", "查文档里的原因"])
        self.assertEqual(result.sub_queries[1].depends_on, ("Q1",))

    def test_decomposition_caps_five(self):
        raw = _payload(
            strategy="decomposition",
            rewritten_query="七步任务",
            sub_queries=[{"id": f"Q{i}", "query": f"第{i}步"} for i in range(1, 8)],
        )
        result = parse_rewrite_response(raw, "七步任务")
        self.assertEqual(len(result.sub_queries), 5)

    def test_sub_queries_not_a_list_is_tolerated(self):
        raw = _payload(strategy="decomposition", sub_queries="Q1: 查表")
        result = parse_rewrite_response(raw, "查表并解释")

        self.assertEqual(result.strategy, RewriteStrategy.PASS_THROUGH)
        self.assertEqual(result.rewritten_query, "查表并解释")


# ---------------------------------------------------------------- 解析与降级


class TestEnglishTermsPerRetrievalUnit(unittest.TestCase):
    """英文检索词的作用域 = 检索单元的作用域。

    单查询策略的检索单元就是整条查询，词挂在结果级；
    多路 / 分解的检索单元是子查询，词必须挂在各自子查询上。
    有子查询却还在结果级挂一份词包，等于让几路检索互相污染——这里把这条不变量钉死。
    """

    def test_basic_rewrite_keeps_english_keywords(self):
        result = parse_rewrite_response(
            _payload(
                strategy="basic_rewrite",
                reason="补全指代",
                rewritten_query="比特币白皮书怎么防止双重支付",
                english_keywords=["bitcoin whitepaper double spending prevention"],
            ),
            "它怎么防止双花",
        )

        self.assertEqual(result.strategy, RewriteStrategy.BASIC_REWRITE)
        self.assertEqual(
            result.english_keywords, ("bitcoin whitepaper double spending prevention",)
        )

    def test_multi_query_keeps_english_terms_per_angle(self):
        result = parse_rewrite_response(
            _payload(
                strategy="multi_query",
                reason="多视角",
                rewritten_query="比特币白皮书讲了什么",
                sub_queries=[
                    {
                        "id": "Q1",
                        "query": "白皮书的共识机制",
                        "angle": "共识",
                        "english_terms": ["proof of work consensus"],
                    },
                    {
                        "id": "Q2",
                        "query": "白皮书的隐私保护",
                        "angle": "隐私",
                        "english_terms": ["transaction privacy anonymity"],
                    },
                ],
            ),
            "比特币白皮书讲了什么",
        )

        self.assertEqual(
            [sq.english_terms for sq in result.sub_queries],
            [("proof of work consensus",), ("transaction privacy anonymity",)],
        )
        self.assertEqual(result.english_keywords, ())

    def test_decomposition_puts_english_terms_on_each_subquery(self):
        result = parse_rewrite_response(
            _payload(
                strategy="decomposition",
                reason="两步检索",
                rewritten_query="对比 SPV 与全节点并说明防双花",
                # 顶层也给了词包：这是错配，必须丢掉，只认子查询上的
                english_keywords=["SPV full node double spending"],
                sub_queries=[
                    {
                        "id": "Q1",
                        "query": "提取 SPV 与全节点的差异",
                        "english_terms": ["SPV simplified payment verification light client"],
                    },
                    {
                        "id": "Q2",
                        "query": "提取防双花机制",
                        "english_terms": ["double spending prevention proof of work"],
                    },
                ],
            ),
            "对比 SPV 与全节点并说明防双花",
        )

        self.assertEqual(len(result.sub_queries), 2)
        self.assertEqual(
            result.sub_queries[0].english_terms,
            ("SPV simplified payment verification light client",),
        )
        self.assertEqual(
            result.sub_queries[1].english_terms,
            ("double spending prevention proof of work",),
        )
        self.assertEqual(result.english_keywords, ())

    def test_english_terms_are_cleaned_and_capped(self):
        result = parse_rewrite_response(
            _payload(
                strategy="decomposition",
                reason="两步检索",
                rewritten_query="对比两件事",
                sub_queries=[
                    {
                        "id": "Q1",
                        "query": "第一件事",
                        "english_terms": [
                            "  SPV   light client ",
                            "SPV light client",
                            "含中文 terms",
                            "a",
                            "second",
                            "third",
                            "dropped",
                        ],
                    },
                    {"id": "Q2", "query": "第二件事"},
                ],
            ),
            "对比两件事",
        )

        self.assertEqual(
            result.sub_queries[0].english_terms,
            ("SPV light client", "second", "third"),
        )
        self.assertEqual(result.sub_queries[1].english_terms, ())

    def test_retrieval_units_for_single_query_strategy(self):
        result = parse_rewrite_response(
            _payload(
                strategy="query_expansion",
                reason="术语扩展",
                rewritten_query="如何验证支付",
                search_keywords=["SPV 简化支付验证"],
                english_keywords=["Simplified Payment Verification"],
            ),
            "如何验证支付",
        )

        units = retrieval_units(result)
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].query, "如何验证支付")
        self.assertEqual(units[0].english_terms, ("Simplified Payment Verification",))

    def test_retrieval_units_follow_sub_queries(self):
        result = parse_rewrite_response(
            _payload(
                strategy="decomposition",
                reason="两步检索",
                rewritten_query="对比 SPV 与全节点并说明防双花",
                sub_queries=[
                    {
                        "id": "Q1",
                        "query": "提取 SPV 与全节点的差异",
                        "english_terms": ["SPV light client"],
                    },
                    {
                        "id": "Q2",
                        "query": "提取防双花机制",
                        "target_source": "document",
                        "english_terms": ["double spending prevention"],
                    },
                ],
            ),
            "对比 SPV 与全节点并说明防双花",
        )

        units = retrieval_units(result)
        self.assertEqual([u.id for u in units], ["Q1", "Q2"])
        self.assertEqual(units[0].english_terms, ("SPV light client",))
        self.assertEqual(units[1].english_terms, ("double spending prevention",))
        self.assertEqual(units[1].target_source, "document")

    def test_retrieval_units_empty_for_pass_through(self):
        result = parse_rewrite_response(
            _payload(strategy="pass_through", reason="清晰", rewritten_query=""),
            "好的",
        )

        self.assertEqual(retrieval_units(result), ())

    def test_format_for_agent_renders_per_subquery_english_terms(self):
        result = parse_rewrite_response(
            _payload(
                strategy="decomposition",
                reason="两步检索",
                rewritten_query="对比 SPV 与全节点并说明防双花",
                sub_queries=[
                    {
                        "id": "Q1",
                        "query": "提取 SPV 与全节点的差异",
                        "english_terms": ["SPV simplified payment verification"],
                    },
                    {
                        "id": "Q2",
                        "query": "提取防双花机制",
                        "english_terms": ["double spending prevention"],
                    },
                ],
            ),
            "对比 SPV 与全节点并说明防双花",
        )

        text = format_for_agent(result)
        self.assertIn("英文检索词（搜英文资料时用）", text)
        self.assertIn("SPV simplified payment verification", text)
        self.assertIn("double spending prevention", text)

    def test_basic_rewrite_packet_carries_english_block(self):
        result = parse_rewrite_response(
            _payload(
                strategy="basic_rewrite",
                reason="补全指代",
                rewritten_query="比特币白皮书怎么防止双重支付",
                english_keywords=["bitcoin double spending prevention"],
            ),
            "它怎么防止双花",
        )

        text = format_for_agent(result)
        self.assertIn("建议英文检索词组合", text)
        self.assertIn("- bitcoin double spending prevention", text)


class TestParserAndFallback(unittest.TestCase):
    def test_fenced_json_is_accepted(self):
        raw = "```json\n" + _payload(
            strategy="basic_rewrite", rewritten_query="各城市销售额"
        ) + "\n```"
        result = parse_rewrite_response(raw, "各城市呢")

        self.assertEqual(result.strategy, RewriteStrategy.BASIC_REWRITE)
        self.assertEqual(result.effective_query, "各城市销售额")

    def test_json_wrapped_in_prose_is_accepted(self):
        raw = "好的，这是结果：\n" + _payload(
            strategy="basic_rewrite", rewritten_query="各城市销售额"
        ) + "\n以上。"
        result = parse_rewrite_response(raw, "各城市呢")

        self.assertEqual(result.strategy, RewriteStrategy.BASIC_REWRITE)

    def test_invalid_json_falls_back(self):
        for raw in ("", "   ", "抱歉，我不太确定", "{不合法}",
                    _payload(strategy="pass_through")[:-3]):
            result = parse_rewrite_response(raw, "原问题")
            self.assertEqual(result.strategy, RewriteStrategy.PASS_THROUGH, raw)
            self.assertEqual(result.rewritten_query, "原问题")
            self.assertTrue(result.fallback, raw)
            self.assertTrue(result.error, raw)

    def test_unknown_strategy_falls_back(self):
        raw = _payload(strategy="make_it_fancier", rewritten_query="随便")
        result = parse_rewrite_response(raw, "原问题")

        self.assertEqual(result.strategy, RewriteStrategy.PASS_THROUGH)
        self.assertTrue(result.fallback)
        self.assertIn("make_it_fancier", result.error)

    def test_missing_strategy_falls_back(self):
        result = parse_rewrite_response(_payload(rewritten_query="x"), "原问题")
        self.assertTrue(result.fallback)

    def test_llm_error_degrades_silently(self):
        llm = BoomLLM(LLMError("HTTP 500：boom"))
        result = QueryRewriter(llm=llm).rewrite("北京呢？")

        self.assertEqual(llm.calls, 1)
        self.assertEqual(result.strategy, RewriteStrategy.PASS_THROUGH)
        self.assertEqual(result.rewritten_query, "北京呢？")
        self.assertTrue(result.fallback)
        self.assertIn("HTTP 500", result.error)

    def test_timeout_degrades_silently(self):
        result = QueryRewriter(llm=BoomLLM(TimeoutError("timed out"))).rewrite("北京呢？")

        self.assertTrue(result.fallback)
        self.assertEqual(result.effective_query, "北京呢？")

    def test_unexpected_error_degrades_silently(self):
        result = QueryRewriter(llm=BoomLLM(ValueError("协议怪怪的"))).rewrite("北京呢？")
        self.assertTrue(result.fallback)
        self.assertIn("协议怪怪的", result.error)

    def test_tokens_and_latency_are_recorded(self):
        llm = FakeLLM(tokens=42)
        result = QueryRewriter(llm=llm).rewrite("北京呢？")

        self.assertEqual(result.tokens, 42)
        self.assertGreaterEqual(result.latency, 0.0)
        self.assertFalse(result.fallback)

    def test_prompt_carries_system_and_context(self):
        llm = FakeLLM()
        QueryRewriter(llm=llm).rewrite(
            "北京呢？", RewriteContext(history=(("user", "各城市销售额"),))
        )

        messages = llm.seen[0]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Query Rewriter", messages[0]["content"])
        self.assertIn("各城市销售额", messages[1]["content"])


# ---------------------------------------------------------------- 下游包装


class TestFormatForAgent(unittest.TestCase):
    def test_pass_through_is_verbatim(self):
        result = parse_rewrite_response(_payload(strategy="pass_through"), "这张表有多少行？")
        self.assertEqual(format_for_agent(result), "这张表有多少行？")

    def test_basic_rewrite_keeps_original_and_intent(self):
        result = parse_rewrite_response(
            _payload(strategy="basic_rewrite", rewritten_query="北京销售额"), "北京呢？"
        )
        text = format_for_agent(result)

        self.assertIn("【用户原始问题】", text)
        self.assertIn("北京呢？", text)
        self.assertIn("完整意图：北京销售额", text)

    def test_query_expansion_lists_terms_and_keywords(self):
        result = parse_rewrite_response(
            _payload(
                strategy="query_expansion",
                rewritten_query="违约金赔付",
                expanded_terms=["赔偿金", "liquidated damages"],
                search_keywords=["违约金 赔付"],
            ),
            "违约怎么赔",
        )
        text = format_for_agent(result)

        self.assertIn("违约怎么赔", text)
        self.assertIn("同义/扩展术语：赔偿金, liquidated damages", text)
        self.assertIn("- 违约金 赔付", text)
        self.assertIn("search_document", text)
        self.assertNotIn("建议英文检索词组合", text)

    def test_format_for_agent_includes_english_keywords_when_present(self):
        result = parse_rewrite_response(
            _payload(
                strategy="query_expansion",
                rewritten_query="如何验证支付",
                expanded_terms=["SPV"],
                search_keywords=["简化支付验证"],
                english_keywords=["Simplified Payment Verification Merkle"],
            ),
            "如何验证支付",
        )
        text = format_for_agent(result)
        self.assertIn("建议英文检索词组合", text)
        self.assertIn("- Simplified Payment Verification Merkle", text)

    def test_multi_query_lists_angles(self):
        result = parse_rewrite_response(
            _payload(
                strategy="multi_query",
                rewritten_query="整体表现",
                sub_queries=[
                    {"query": "总量趋势", "angle": "总量", "target_source": "table"},
                    {"query": "区域分布", "angle": "区域", "target_source": "table"},
                ],
            ),
            "分析整体销售表现",
        )
        text = format_for_agent(result)

        self.assertIn("【查询改写（多路查询）】", text)
        self.assertIn("总目标：整体表现", text)
        self.assertIn("[Q1 | 视角:总量 | 数据源:table] 总量趋势", text)
        self.assertIn("并行", text)

    def test_decomposition_lists_dependencies(self):
        result = parse_rewrite_response(
            _payload(
                strategy="decomposition",
                rewritten_query="查原因并成文",
                sub_queries=[
                    {"query": "找下滑城市", "purpose": "定位异常", "target_source": "table"},
                    {"query": "读文档原因", "purpose": "解释", "depends_on": ["Q1"],
                     "target_source": "document"},
                ],
            ),
            "找出下滑城市，查报告原因",
        )
        text = format_for_agent(result)

        self.assertIn("【查询改写（查询分解）】", text)
        self.assertIn("[Q1 | 依赖:无 | 数据源:table] 找下滑城市", text)
        self.assertIn("[Q2 | 依赖:Q1 | 数据源:document] 读文档原因", text)
        self.assertIn("（目标：解释）", text)


# ---------------------------------------------------------------- Agent 集成


class TestAgentIntegration(unittest.TestCase):
    def test_no_rewriter_keeps_old_behaviour(self):
        events: list[str] = []
        llm = ScriptedLLM(
            [_call("c1", "run_sql", {"sql": "SELECT 1 AS n"}),
             {"role": "assistant", "content": "好了"}]
        )
        agent = Agent(Session(), llm=llm, on_event=lambda k, p: events.append(k))

        self.assertEqual(agent.ask("查一下"), "好了")
        self.assertNotIn("rewrite", events)
        self.assertEqual(events, ["step", "tool_call", "observation", "step", "final"])
        self.assertEqual(_user_texts(llm.seen), ["查一下", "查一下"])

    def _run(self, payload: str, question: str) -> tuple[Agent, Session, ScriptedLLM, list[str], str]:
        rewriter_llm = FakeLLM(payload)
        rewriter = QueryRewriter(llm=rewriter_llm)
        events: list[str] = []
        llm = ScriptedLLM(
            [_call("c1", "run_sql", {"sql": "SELECT 1 AS n"}),
             {"role": "assistant", "content": "做完了"}]
        )
        session = Session()
        agent = Agent(
            session,
            llm=llm,
            session_key=dt.DEFAULT_KEY,
            rewriter=rewriter,
            on_event=lambda k, p: events.append(k),
        )
        answer = agent.ask(question)
        return agent, session, llm, events, answer

    def test_all_four_rewrites_reach_the_main_model(self):
        payloads = {
            "basic_rewrite": _payload(
                strategy="basic_rewrite", rewritten_query="北京各城市销售额"
            ),
            "query_expansion": _payload(
                strategy="query_expansion",
                rewritten_query="违约金赔付标准",
                expanded_terms=["赔偿金"],
                search_keywords=["违约金 赔付"],
            ),
            "multi_query": _payload(
                strategy="multi_query",
                rewritten_query="整体表现",
                sub_queries=[{"query": "总量趋势"}, {"query": "区域分布"}],
            ),
            "decomposition": _payload(
                strategy="decomposition",
                rewritten_query="找下滑城市并查原因",
                sub_queries=[{"query": "找下滑城市"}, {"query": "读文档原因",
                                                         "depends_on": ["Q1"]}],
            ),
        }
        for name, payload in payloads.items():
            question = "北京呢？" if name == "basic_rewrite" else "分析一下整体表现"
            _, session, llm, events, answer = self._run(payload, question)

            expected = format_for_agent(parse_rewrite_response(payload, question))
            self.assertEqual(answer, "做完了", name)
            # 主循环每一步看到的都是包装后的完整指令（含工具回合之后的那一次）
            self.assertEqual(_user_texts(llm.seen), [expected, expected], name)
            # 持久化历史里只有原话，一个字都没被替换
            self.assertEqual(session.messages[1]["content"], question, name)
            self.assertEqual(events[0], "rewrite", name)
            self.assertEqual(
                events, ["rewrite", "step", "tool_call", "observation", "step", "final"], name
            )

    def test_rewrite_tokens_count_toward_session(self):
        _, session, _, _, _ = self._run(BASIC_PAYLOAD, "北京呢？")

        # 改写 30 + 主循环两次各 12
        self.assertEqual(session.tokens, 54)
        self.assertEqual(session.steps, 2)     # 步数只算主循环
        self.assertEqual(session.turns, 1)

    def test_rewrite_tokens_count_toward_round_budget(self):
        rewriter = QueryRewriter(llm=FakeLLM(BASIC_PAYLOAD, tokens=50))
        llm = ScriptedLLM(
            [_call("c1", "run_sql", {"sql": "SELECT 1 AS n"}),
             {"role": "assistant", "content": "不该走到这里"}],
            tokens=12,
        )
        agent = Agent(
            Session(), llm=llm, session_key=dt.DEFAULT_KEY, rewriter=rewriter,
            token_budget=10,
        )

        answer = agent.ask("北京呢？")
        self.assertIn("预算", answer)

    def test_bypassed_rewrite_still_emits_event(self):
        """快速通道也要发事件——界面靠它判断"这次没改写"，好决定不画卡片。"""
        sessions: list[tuple[str, dict]] = []
        agent = Agent(
            Session(),
            llm=ScriptedLLM([{"role": "assistant", "content": "接着做完了"}]),
            session_key=dt.DEFAULT_KEY,
            rewriter=QueryRewriter(llm=BoomLLM(AssertionError("不该调模型"))),
            on_event=lambda k, p: sessions.append((k, p)) if k == "rewrite" else None,
        )

        self.assertEqual(agent.ask("继续"), "接着做完了")
        kind, payload = sessions[0]
        self.assertEqual(kind, "rewrite")
        self.assertTrue(payload["bypassed"])
        self.assertEqual(payload["strategy"], "pass_through")

    def test_rewrite_failure_does_not_block_answer(self):
        agent = Agent(
            Session(),
            llm=ScriptedLLM([{"role": "assistant", "content": "照常回答"}]),
            session_key=dt.DEFAULT_KEY,
            rewriter=QueryRewriter(llm=BoomLLM(LLMError("HTTP 500"))),
        )

        self.assertEqual(agent.ask("北京呢？"), "照常回答")

    def test_disabled_rewriter_is_skipped(self):
        rewriter_llm = FakeLLM()
        agent = Agent(
            Session(),
            llm=ScriptedLLM([{"role": "assistant", "content": "好"}]),
            session_key=dt.DEFAULT_KEY,
            rewriter=QueryRewriter(llm=rewriter_llm, enabled=False),
        )

        agent.ask("北京呢？")
        self.assertEqual(rewriter_llm.calls, 0)

    def test_system_prompt_mentions_rewrite_hint(self):
        from app.agent import make_system_prompt

        prompt = make_system_prompt(["城市", "销售额"])
        self.assertIn("查询改写", prompt)


# ---------------------------------------------------------------- 界面渲染


class TestUIRenderRewrite(unittest.TestCase):
    def test_bypassed_pass_through_is_hidden(self):
        item = build_item(
            "rewrite",
            {"bypassed": True, "strategy": "pass_through", "tokens": 0, "latency": 0.0},
        )
        self.assertIsNone(item)

    def test_normal_pass_through(self):
        item = build_item(
            "rewrite",
            {
                "bypassed": False,
                "strategy": "pass_through",
                "label": "直接透传",
                "reason": "问题表述清晰",
                "tokens": 24,
                "latency": 0.4,
            },
        )

        self.assertIsNotNone(item)
        self.assertEqual(item.title, "查询改写 · 直接透传（24 tokens，0.4s）")
        self.assertEqual(item.body, "问题表述清晰")
        self.assertFalse(item.code)

    def test_fallback_shows_reason(self):
        item = build_item(
            "rewrite",
            {"strategy": "pass_through", "fallback": True, "error": "HTTP 500"},
        )

        self.assertIsNotNone(item)
        self.assertEqual(item.title, "查询改写 · 已降级透传")
        self.assertIn("HTTP 500", item.body)
        self.assertIn("按原始问题继续", item.body)
        self.assertFalse(item.code)

    def test_basic_rewrite_body(self):
        item = build_item(
            "rewrite",
            {
                "strategy": "basic_rewrite",
                "label": "基础改写",
                "reason": "补全指代",
                "rewritten_query": "北京销售额",
                "tokens": 30,
                "latency": 0.5,
            },
        )

        self.assertEqual(item.title, "查询改写 · 基础改写（30 tokens，0.5s）")
        self.assertIn("策略理由：补全指代", item.body)
        self.assertIn("改写问题：北京销售额", item.body)
        self.assertFalse(item.code)

    def test_query_expansion_body(self):
        item = build_item(
            "rewrite",
            {
                "strategy": "query_expansion",
                "label": "查询扩展",
                "rewritten_query": "违约金赔付",
                "expanded_terms": ["赔偿金", "liquidated damages"],
                "search_keywords": ["违约金 赔付"],
                "tokens": 40,
                "latency": 0.6,
            },
        )

        self.assertIn("查询扩展", item.title)
        self.assertIn("`赔偿金`", item.body)
        self.assertIn("建议检索词：`违约金 赔付`", item.body)
        self.assertNotIn("英文检索词", item.body)

    def test_query_expansion_body_with_english_keywords(self):
        item = build_item(
            "rewrite",
            {
                "strategy": "query_expansion",
                "label": "查询扩展",
                "rewritten_query": "如何验证支付",
                "expanded_terms": ["SPV"],
                "search_keywords": ["简化支付验证"],
                "english_keywords": ["Simplified Payment Verification Merkle"],
                "tokens": 45,
                "latency": 0.5,
            },
        )
        self.assertIn("英文检索词：`Simplified Payment Verification Merkle`", item.body)

    def test_multi_query_body(self):
        item = build_item(
            "rewrite",
            {
                "strategy": "multi_query",
                "label": "多路查询",
                "rewritten_query": "整体表现",
                "sub_queries": [
                    {"id": "Q1", "query": "总量趋势", "angle": "总量", "target_source": "table"},
                    {"id": "Q2", "query": "区域分布", "angle": "", "target_source": "auto"},
                ],
                "tokens": 50,
                "latency": 0.7,
            },
        )

        self.assertIn("多路查询", item.title)
        self.assertIn("视角：总量", item.body)
        self.assertIn("数据源：table", item.body)
        self.assertIn("总量趋势", item.body)
        self.assertIn("未标注", item.body)

    def test_decomposition_body(self):
        item = build_item(
            "rewrite",
            {
                "strategy": "decomposition",
                "label": "查询分解",
                "rewritten_query": "对比毛利",
                "sub_queries": [
                    {"id": "Q1", "query": "查实际毛利", "purpose": "实际值",
                     "target_source": "table", "depends_on": []},
                    {"id": "Q2", "query": "查目标毛利", "purpose": "目标值",
                     "target_source": "document", "depends_on": ["Q1"]},
                ],
                "tokens": 60,
                "latency": 0.8,
            },
        )

        self.assertIn("查询分解", item.title)
        self.assertIn("依赖：无", item.body)
        self.assertIn("依赖：Q1", item.body)
        self.assertIn("目标：实际值", item.body)
        self.assertFalse(item.code)


class TestWebSearchRewrite(unittest.TestCase):
    def test_prompt_reflects_web_search_flag(self):
        from app.query_rewrite import build_rewrite_user_prompt

        off = build_rewrite_user_prompt("最新价格", RewriteContext(web_search=False))
        on = build_rewrite_user_prompt("最新价格", RewriteContext(web_search=True))
        self.assertIn("联网搜索：未开启", off)
        self.assertIn("联网搜索：已开启", on)

    def test_web_search_strategy_parsed_and_formatted_when_enabled(self):
        raw = _payload(
            strategy="web_search",
            reason="需要实时外部信息",
            rewritten_query="2025年比亚迪新能源车销量",
            search_keywords=["2025 BYD EV sales", "比亚迪 2025 新能源销量"],
            english_keywords=["BYD 2025 EV sales"],
        )
        result = parse_rewrite_response(
            raw, "查一下比亚迪去年销量", web_search_enabled=True
        )
        self.assertEqual(result.strategy, RewriteStrategy.WEB_SEARCH)
        self.assertEqual(result.search_keywords, ("2025 BYD EV sales", "比亚迪 2025 新能源销量"))
        self.assertEqual(result.english_keywords, ("BYD 2025 EV sales",))

        packaged = format_for_agent(result)
        self.assertIn("联网检索", packaged)
        self.assertIn("web_search", packaged)
        self.assertIn("无需先在本地资料文件中试探检索", packaged)

    def test_web_search_strategy_degrades_when_disabled(self):
        raw = _payload(
            strategy="web_search",
            reason="想联网",
            rewritten_query="2025年比亚迪新能源车销量",
            search_keywords=["比亚迪销量"],
        )
        result = parse_rewrite_response(
            raw, "查一下比亚迪去年销量", web_search_enabled=False
        )
        self.assertNotEqual(result.strategy, RewriteStrategy.WEB_SEARCH)
        self.assertEqual(result.strategy, RewriteStrategy.BASIC_REWRITE)

    def test_sub_query_web_source_guarded_by_flag(self):
        raw = _payload(
            strategy="decomposition",
            rewritten_query="对比内部销售与行业增速",
            sub_queries=[
                {"id": "Q1", "query": "统计内部销售增速", "target_source": "table", "depends_on": []},
                {"id": "Q2", "query": "查询外部行业平均增速", "target_source": "web", "depends_on": []},
            ],
        )
        enabled = parse_rewrite_response(raw, "对比行业增速", web_search_enabled=True)
        self.assertEqual(enabled.sub_queries[1].target_source, "web")
        self.assertIn("数据源:web", format_for_agent(enabled))

        disabled = parse_rewrite_response(raw, "对比行业增速", web_search_enabled=False)
        self.assertEqual(disabled.sub_queries[1].target_source, "auto")

    def test_agent_passes_registry_web_search_state_to_rewriter(self):
        from app.tools.registry import ToolRegistry

        reg = ToolRegistry()
        reg.register(
            "web_search",
            lambda **kwargs: "ok",  # noqa: ARG005
            "Search web",
            {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        )

        seen_prompts: list[str] = []

        class InspectRewriteLLM:
            def __call__(self, messages, tools=None, **_kwargs):  # noqa: ARG002
                seen_prompts.append(messages[1]["content"])
                return LLMResponse(
                    message={
                        "role": "assistant",
                        "content": _payload(
                            strategy="web_search",
                            reason="需要联网查最新信息",
                            rewritten_query="查询最新 Python 3.13 发布日期",
                            search_keywords=["Python 3.13 release date"],
                        ),
                    }
                )

        main_seen: list[list[dict[str, Any]]] = []

        def main_llm(messages, tools=None):  # noqa: ARG001
            main_seen.append(messages)
            return LLMResponse(message={"role": "assistant", "content": "已查到。"})

        agent = Agent(
            Session(),
            llm=main_llm,
            registry=reg,
            rewriter=QueryRewriter(llm=InspectRewriteLLM()),
        )
        agent.ask("帮我联网查一下 Python 3.13 发布日期")

        self.assertIn("联网搜索：已开启", seen_prompts[0])
        self.assertIn("直接优先调用 `web_search`", main_seen[0][-1]["content"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
