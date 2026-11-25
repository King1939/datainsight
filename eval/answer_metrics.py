"""开发期生成质量评测：忠实度（Faithfulness）+ 答案相关性（Answer Relevance）。

定位：**纯开发期离线评测**。只被 `eval/eval_generation.py` 和
`tests/test_eval_generation.py` 调用，不接入 Agent 运行时链路，不改 `app/` 下任何文件，
也不引入 ragas / trulens / deepeval 这类评测框架。

两个指标各自衡量一件不同的事：

  忠实度（faithfulness）
      答案里的每一条原子事实，在**本轮工具返回的上下文**里有没有依据。
      这是"有没有幻觉"的量化：`supported` 计数 / `claims` 总数。
      注意它和"答得对不对"是两件事——上下文本身错了，忠实度照样可以是 1.0。

  答案相关性（relevance）
      答案有没有正面回答用户那句话。主路径是**逆向生成假设问题 + 向量余弦**：
      先让 Judge 只看着答案反推"这句话在回答什么问题"，再和真实问题算相似度。
      比直接让模型打一个 0~1 的分稳，因为模型给学生态答案打高分是出了名的。
      向量不可用时平滑退回 Judge 自己给的分。

两条工程上的硬约束写在这里，因为它们是这个模块存在的理由：

  1. **绝不抛异常打断整批评测。** 一条用例的 Judge 返回坏 JSON，只是这一条记
     `fallback`，剩下 15 条照跑。评测脚本最怕的就是跑到第 12 条崩掉，
     前面十几分钟的真实模型调用全白花。
  2. **拒绝回答和编造要分开算。** 资料里没有的东西，诚实说"没找到"应当拿满
     忠实度（它没编），相关性记 0（它没答）。把这两件事混成一个分数，
     就再也看不出模型是在"老实"还是在"胡编"了。
"""

from __future__ import annotations

import inspect
import json
import re
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import config, embeddings  # noqa: E402
from app.llm import LLMCallable, LLMError, LLMResponse, chat  # noqa: E402

ClaimVerdict = Literal["supported", "unsupported", "contradicted"]
EvalVerdict = Literal["pass", "warn", "fail", "skipped", "fallback"]
RelevanceMode = Literal["reverse_embedding", "llm_judge", "noncommittal", "skipped"]

# 本轮工具上下文的总预算与单条上限。默认值和计划书一致。
MAX_CONTEXT_CHARS = 12000
MAX_ITEM_CHARS = 3000

# 工具的信息密度分级：预算不够时先砍哪个，是这些集合决定的。
# 高：直接产出结论数据；中：定位线索；低：目录/打开文件，几乎不含答案。
_HIGH_VALUE_TOOLS = frozenset(
    {
        "run_sql",
        "run_python",
        "read_document",
        "read_file",
        "make_chart",
        "make_mindmap",
        "make_pptx",
        "edit_pptx",
        "write_text_file",
    }
)
_MID_VALUE_TOOLS = frozenset({"search_document", "inspect_table"})
_LOW_VALUE_TOOLS = frozenset({"list_documents", "list_files", "find_program", "open_file"})

# Agent 主动中止时写进历史的那几句话术。命中它们说明这一轮根本没跑完，
# 拿去算忠实度只会污染均值（分母是"模型想说的"，不是"模型说完的"）。
_ABORT_MARKERS: tuple[str, ...] = (
    "（模型调用失败：",
    "已强制中止",
    "token 超出预算",
    "还没收尾，先停在这里",
    "达到步数上限",
)

EVAL_SYSTEM_PROMPT = """你是检索增强生成（RAG）的严格评测员。你的任务是只依据【工具上下文】审核【答案】，并输出一个 JSON 对象。

## 任务一：把答案拆成原子事实声明（claims）

拆解规则：
- 一条 claim 只包含**一个**可验证的事实：一个数字、一个百分比、一个排名、一个日期、一个实体归属、一条文档结论。
- 数字连同单位、统计口径算同一条。等价换算视为同一事实：1254000 元 = 125.4 万元；0.3333 = 33.3%；占比 29.7% 与"约三成"等价。
- 忽略：寒暄、道歉、"我查一下"之类的过程说明、纯标题、后续操作建议。这些不构成事实声明。
- 定性结论也要拆（"该平台采用哈希值验证"是一条 claim，"效果显著提升"这种没有依据的泛泛评价也要单独成条）。

逐条判定 verdict，三选一：
- supported：上下文能直接读到，或经等价换算、简单加总/相减可比得出。必须填 evidence_source（工具名）与 evidence_quote（上下文原文片段，尽量 60 字以内）。
- unsupported：上下文里找不到依据。包括凭常识补充的背景、自行推断的因果关系、把文档里的旧数据当成本期数据。
- contradicted：与上下文中的数值、实体或结论**直接冲突**（数字算错、张冠李戴、把 A 的指标写成 B 的）。

否定性断言（"资料里没有 X""另两份资料不涉及 Y"）按检索记录判：上下文里有针对该内容的检索且无命中的返回，判 supported 并引用那条"没找到"的记录；模型没查就直接断言，判 unsupported。

如果【工具上下文】为空，则任何具体事实断言一律判 unsupported，不要凭你自己的知识判 supported。

## 任务二：判断答案有没有回答问题

- noncommittal：答案没有给出实质内容（"没找到""无法确定""请先上传文件""需要更多信息"）。**诚实地说明资料里没有，也算 noncommittal=true**。
- is_refusal：**整个答案**只是在说明"资料里没有这个信息 / 无法回答"，没有回答用户问的任何一点。如果答案正面回答了问题、只是顺带提到某处资料没写，请判 false。
- generated_questions：只根据【答案】反推 3 个它最直接回答的问题。要具体，带上答案里的实体与口径，不要照抄用户问题。
- judge_relevance：0~1，衡量答案对用户问题的覆盖度与直接性。跑题、只答一半、大量注水都要扣分。
- missing_aspects：用户问题里有、但答案没覆盖到的子问题或约束，最多 5 条。没有就给空数组。
- summary：一句话概括这条答案的问题（20 字内）。

## 输出格式

只输出一个 JSON 对象，不要解释、不要 Markdown 围栏、不要在对象前后写任何字。claims 太多时按重要性保留最关键的 12 条，不要写省略号：

{
  "is_refusal": false,
  "noncommittal": false,
  "claims": [
    {
      "statement": "原子事实陈述",
      "verdict": "supported",
      "evidence_source": "run_sql",
      "evidence_quote": "上下文中的直接依据片段",
      "reason": "简短判定理由"
    }
  ],
  "generated_questions": ["问题1", "问题2", "问题3"],
  "judge_relevance": 0.92,
  "missing_aspects": [],
  "summary": "一句话总结"
}"""


# ---------------------------------------------------------------- 数据结构


@dataclass(frozen=True)
class EvidenceItem:
    """一条工具返回。tool_name 由 tool_call_id 还原，args_summary 是压缩后的参数。"""

    tool_name: str
    args_summary: str
    content: str
    call_id: str = ""


@dataclass(frozen=True)
class ClaimCheck:
    statement: str
    verdict: ClaimVerdict
    evidence_source: str = ""
    evidence_quote: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "statement": self.statement,
            "verdict": self.verdict,
            "evidence_source": self.evidence_source,
            "evidence_quote": self.evidence_quote,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class JudgeOutcome:
    """Judge 一次调用的产物。fallback=True 表示这次调用没拿到可用结构。"""

    claims: tuple[ClaimCheck, ...] = ()
    noncommittal: bool = False
    is_refusal: bool = False
    generated_questions: tuple[str, ...] = ()
    judge_relevance: float = 0.0
    missing_aspects: tuple[str, ...] = ()
    summary: str = ""
    fallback: bool = False
    error: str = ""


@dataclass(frozen=True)
class AnswerEvalResult:
    question: str
    effective_query: str
    verdict: EvalVerdict = "skipped"
    faithfulness: float | None = None
    relevance: float | None = None
    relevance_mode: RelevanceMode = "skipped"
    claims: tuple[ClaimCheck, ...] = ()
    supported_count: int = 0
    unsupported_count: int = 0
    contradicted_count: int = 0
    noncommittal: bool = False
    is_refusal: bool = False
    generated_questions: tuple[str, ...] = ()
    missing_aspects: tuple[str, ...] = ()
    summary: str = ""
    context_count: int = 0
    context_chars: int = 0
    tokens: int = 0
    latency_ms: float = 0.0
    skipped: bool = False
    fallback: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "effective_query": self.effective_query,
            "verdict": self.verdict,
            "faithfulness": self.faithfulness,
            "relevance": self.relevance,
            "relevance_mode": self.relevance_mode,
            "claims": [c.to_dict() for c in self.claims],
            "supported_count": self.supported_count,
            "unsupported_count": self.unsupported_count,
            "contradicted_count": self.contradicted_count,
            "noncommittal": self.noncommittal,
            "is_refusal": self.is_refusal,
            "generated_questions": list(self.generated_questions),
            "missing_aspects": list(self.missing_aspects),
            "summary": self.summary,
            "context_count": self.context_count,
            "context_chars": self.context_chars,
            "tokens": self.tokens,
            "latency_ms": self.latency_ms,
            "skipped": self.skipped,
            "fallback": self.fallback,
            "error": self.error,
        }


# ---------------------------------------------------------------- 上下文提取


def _flatten(text: Any, limit: int = 160) -> str:
    flat = re.sub(r"\s+", " ", str(text or "")).strip()
    return flat if len(flat) <= limit else flat[:limit] + "…"


def summarize_args(raw: Any, *, limit: int = 120) -> str:
    """把工具参数压成一行 `k=v, k=v`。原样塞 JSON 会白占预算。"""
    payload: Any = raw
    if isinstance(raw, str):
        try:
            payload = json.loads(raw or "{}")
        except json.JSONDecodeError:
            return _flatten(raw, limit)
    if not isinstance(payload, dict):
        return _flatten(payload, limit)

    parts: list[str] = []
    for key, value in payload.items():
        if isinstance(value, str):
            shown = _flatten(value, 60)
        else:
            shown = _flatten(json.dumps(value, ensure_ascii=False), 60)
        parts.append(f"{key}={shown}")
        if sum(len(p) for p in parts) > limit:
            break
    return _flatten(", ".join(parts), limit)


def tool_priority(name: str) -> int:
    """0 = 高信息密度，1 = 中，2 = 低/未知。预算不够时按这个顺序保。"""
    key = str(name or "")
    if key in _HIGH_VALUE_TOOLS:
        return 0
    if key in _MID_VALUE_TOOLS:
        return 1
    if key in _LOW_VALUE_TOOLS:
        return 2
    return 1


def extract_turn_contexts(
    messages: Sequence[dict[str, Any]],
    *,
    user_idx: int | None = None,
    max_total_chars: int = MAX_CONTEXT_CHARS,
    max_item_chars: int = MAX_ITEM_CHARS,
) -> list[EvidenceItem]:
    """从一轮对话里捞出全部工具观察结果。

    `user_idx=None` 时自动定位**最后一条** user 消息——单轮评测就是它，
    多轮对话取末尾轮次也是它。不用调用方自己数下标。
    """
    if not messages:
        return []

    start = 0
    if user_idx is None:
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == "user":
                start = index + 1
                break
    else:
        start = max(0, int(user_idx) + 1)

    pending: dict[str, tuple[str, str]] = {}
    raw_items: list[EvidenceItem] = []
    for message in messages[start:]:
        role = message.get("role")
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") or {}
                name = str(function.get("name") or "").strip() or "unknown"
                pending[str(call.get("id") or "")] = (
                    name,
                    summarize_args(function.get("arguments")),
                )
        elif role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            name, args = pending.get(call_id, ("unknown", ""))
            raw_items.append(
                EvidenceItem(
                    tool_name=name,
                    args_summary=args,
                    content=str(message.get("content") or ""),
                    call_id=call_id,
                )
            )

    # 同一轮里重复读同一段（模型试错时很常见）只留一份，否则预算会被它挤爆。
    seen: set[tuple[str, str, str]] = set()
    unique: list[EvidenceItem] = []
    for item in raw_items:
        signature = (item.tool_name, item.args_summary, item.content)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(item)

    if not unique:
        return []

    ranked = sorted(
        enumerate(unique), key=lambda pair: (tool_priority(pair[1].tool_name), pair[0])
    )
    kept: list[tuple[int, EvidenceItem]] = []
    total = 0
    for index, item in ranked:
        text = item.content
        if len(text) > max_item_chars:
            text = text[:max_item_chars].rstrip() + "…"
        if kept and total + len(text) > max_total_chars:
            # 挤不下就跳过；因为已经按优先级排过，被跳过的是低价值的长文本，
            # 后面更短的中等价值条目还有机会进来。
            continue
        kept.append((index, replace(item, content=text)))
        total += len(text)

    kept.sort(key=lambda pair: pair[0])
    return [item for _, item in kept]


def looks_aborted(answer: str) -> bool:
    text = str(answer or "")
    return any(marker in text for marker in _ABORT_MARKERS)


# ---------------------------------------------------------------- Judge 解析


def _strip_fences(raw: str) -> str:
    text = (raw or "").strip()
    if "```" in text:
        # 先去掉围栏标记，再交给后面的首尾大括号截取兜底。顺序反过来会把
        # ```json 那行的花括号留下来，反而更难解析。
        text = re.sub(r"```[a-zA-Z0-9_-]*", "", text).replace("```", "")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return text.strip()
    return text[start : end + 1].strip()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"true", "1", "yes", "y", "是"}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(num):
        return default
    return min(1.0, max(0.0, num))


def _as_text(value: Any, limit: int = 200) -> str:
    return _flatten(value, limit)


_NORMAL_VERDICTS: dict[str, ClaimVerdict] = {
    "supported": "supported",
    "support": "supported",
    "entailed": "supported",
    "unsupported": "unsupported",
    "not_supported": "unsupported",
    "unknown": "unsupported",
    "unverifiable": "unsupported",
    "contradicted": "contradicted",
    "contradiction": "contradicted",
    "conflict": "contradicted",
}


def parse_eval_response(content: str, *, reverse_questions: int = 3) -> JudgeOutcome:
    """解析 Judge 返回。**任何异常都变成 fallback**，绝不往外抛。"""
    text = _strip_fences(content)
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        return JudgeOutcome(fallback=True, error=f"JSON 解析失败：{exc}")
    if not isinstance(payload, dict):
        return JudgeOutcome(fallback=True, error="Judge 返回的不是 JSON 对象")

    raw_claims = payload.get("claims")
    claims: list[ClaimCheck] = []
    if isinstance(raw_claims, list):
        for entry in raw_claims:
            if not isinstance(entry, dict):
                continue
            statement = _as_text(entry.get("statement"), 300)
            if not statement:
                continue
            verdict_raw = str(entry.get("verdict") or "").strip().lower()
            verdict = _NORMAL_VERDICTS.get(verdict_raw, "unsupported")
            claims.append(
                ClaimCheck(
                    statement=statement,
                    verdict=verdict,
                    evidence_source=_as_text(
                        entry.get("evidence_source") or entry.get("source"), 80
                    ),
                    evidence_quote=_as_text(entry.get("evidence_quote"), 200),
                    reason=_as_text(entry.get("reason"), 160),
                )
            )

    questions: list[str] = []
    raw_questions = payload.get("generated_questions")
    if isinstance(raw_questions, str):
        raw_questions = [raw_questions]
    if isinstance(raw_questions, list):
        for entry in raw_questions:
            question = _as_text(entry, 200)
            if question and question not in questions:
                questions.append(question)
            if len(questions) >= max(0, int(reverse_questions)):
                break

    aspects: list[str] = []
    raw_aspects = payload.get("missing_aspects")
    if isinstance(raw_aspects, str):
        raw_aspects = [raw_aspects]
    if isinstance(raw_aspects, list):
        for entry in raw_aspects:
            aspect = _as_text(entry, 120)
            if aspect and aspect not in aspects:
                aspects.append(aspect)
            if len(aspects) >= 5:
                break

    return JudgeOutcome(
        claims=tuple(claims),
        noncommittal=_as_bool(payload.get("noncommittal")),
        is_refusal=_as_bool(payload.get("is_refusal")),
        generated_questions=tuple(questions),
        judge_relevance=_as_float(payload.get("judge_relevance")),
        missing_aspects=tuple(aspects),
        summary=_as_text(payload.get("summary"), 120),
    )


# ---------------------------------------------------------------- Judge 调用


def _llm_kwargs(llm: Callable[..., Any], *, max_tokens: int, timeout: int, retries: int) -> dict[str, Any]:
    """只在对方签名真的认这些参数时才传。测试里的假模型是 `(messages, tools=None)`，
    硬塞 temperature 会当场 TypeError——那不该算"评测失败"。"""
    try:
        parameters = inspect.signature(llm).parameters
    except (TypeError, ValueError):
        return {}
    takes_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )
    wanted = {
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "retries": retries,
    }
    return {
        key: value
        for key, value in wanted.items()
        if takes_kwargs or key in parameters
    }


def build_eval_user_prompt(
    question: str,
    answer: str,
    contexts: Sequence[EvidenceItem],
    *,
    max_context_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    if contexts:
        blocks: list[str] = []
        used = 0
        for index, item in enumerate(contexts, 1):
            head = f"[{index}] 工具 {item.tool_name}"
            if item.args_summary:
                head += f"（{item.args_summary}）"
            body = item.content
            if used + len(body) > max_context_chars:
                body = body[: max(0, max_context_chars - used)] + "\n…（上下文预算已满，后续截断）"
            blocks.append(f"{head}\n{body}")
            used += len(body)
            if used >= max_context_chars:
                break
        context_block = "\n\n".join(blocks)
        total_chars = sum(len(item.content) for item in contexts)
        header = f"【工具上下文】（{len(contexts)} 段，共 {total_chars:,} 字）"
    else:
        context_block = "（本轮没有任何工具返回内容）"
        header = "【工具上下文】（空）"

    return (
        f"【用户问题】\n{question or '（空）'}\n\n"
        f"{header}\n{context_block}\n\n"
        f"【答案】\n{answer or '（空）'}"
    )


# ---------------------------------------------------------------- 计分


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _decide_verdict(
    *,
    faithfulness: float | None,
    relevance: float | None,
    contradicted: int,
    noncommittal: bool,
    expect_refusal: bool | None,
    faithfulness_threshold: float,
    relevance_threshold: float,
) -> EvalVerdict:
    """把两个分数落成一个给人看的评级。规则见计划书 3.3。"""
    if contradicted > 0:
        return "fail"

    # 拒答单独一条通道：没编造（fa=1.0 或 None）+ 相关性 0。它既不该算"答错"，
    # 也不该算"答对"——要看这条用例本来该不该有答案。
    if noncommittal:
        if expect_refusal is False:
            return "fail"        # 本来能答却回避，是真实的失败
        return "pass"            # 该拒答的拒了，或没要求时按诚实拒答处理

    if faithfulness is not None and faithfulness < 0.5:
        return "fail"
    if relevance is not None and relevance < 0.5:
        return "fail"

    ok_faith = faithfulness is None or faithfulness >= faithfulness_threshold
    ok_rel = relevance is not None and relevance >= relevance_threshold
    return "pass" if (ok_faith and ok_rel) else "warn"


class AnswerEvaluator:
    """单条样本评测器。一次 Judge 调用同时产出 claims、拒答标记与逆向问题。"""

    def __init__(
        self,
        *,
        llm: LLMCallable | None = None,
        embedder: Callable[[list[str]], Any] | None = None,
        use_embedding: bool = True,
        timeout: int = 20,
        max_tokens: int = 3000,
        retries: int = 1,
        reverse_questions: int = 3,
        faithfulness_threshold: float = 0.8,
        relevance_threshold: float = 0.7,
        max_context_chars: int = MAX_CONTEXT_CHARS,
        max_item_chars: int = MAX_ITEM_CHARS,
    ) -> None:
        self.llm = llm or chat
        self.embedder = embedder
        self.use_embedding = bool(use_embedding)
        self.timeout = int(timeout)
        # 3000 而不是 1200：长表格答案能拆出十几条声明，1200 会在 JSON 中间截断。
        # 实测 64 条里有 22 条因此 fallback，白丢样本。
        self.max_tokens = int(max_tokens)
        self.retries = int(retries)
        self.reverse_questions = max(0, int(reverse_questions))
        self.faithfulness_threshold = float(faithfulness_threshold)
        self.relevance_threshold = float(relevance_threshold)
        self.max_context_chars = int(max_context_chars)
        self.max_item_chars = int(max_item_chars)

    # -------------------------------------------------- 相关性

    def _embedding_ready(self) -> bool:
        if self.embedder is not None:
            return True
        try:
            return bool(config.embedding_current().ready)
        except Exception:  # noqa: BLE001  配置读盘失败按"不可用"处理
            return False

    def _reverse_embedding_relevance(
        self, benchmark: str, questions: Sequence[str]
    ) -> tuple[float, str]:
        """返回 (分数, 失败原因)。失败原因非空表示应当降级。"""
        texts = [benchmark, *questions]
        try:
            if self.embedder is not None:
                matrix = embeddings.l2_normalize(self.embedder(texts))
            else:
                matrix = embeddings.embed_texts(texts).vectors
        except Exception as exc:  # noqa: BLE001  向量接口任何异常都降级，不阻断评测
            return 0.0, f"{type(exc).__name__}: {exc}"[:160]

        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(texts):
            return 0.0, f"向量形状不对：{matrix.shape}，期望 ({len(texts)}, D)"
        sims = [_clamp01(float(np.dot(matrix[0], matrix[i]))) for i in range(1, len(texts))]
        if not sims:
            return 0.0, "没有可比较的逆向问题"
        return sum(sims) / len(sims), ""

    # -------------------------------------------------- 单条评测

    def evaluate(
        self,
        *,
        question: str,
        answer: str,
        contexts: Sequence[EvidenceItem] = (),
        effective_query: str = "",
        expect_refusal: bool | None = None,
    ) -> AnswerEvalResult:
        started = time.perf_counter()
        question = str(question or "").strip()
        effective = str(effective_query or "").strip() or question
        answer = str(answer or "").strip()
        items = _normalize_contexts(contexts)

        def _skipped(reason: str) -> AnswerEvalResult:
            return AnswerEvalResult(
                question=question,
                effective_query=effective,
                verdict="skipped",
                relevance_mode="skipped",
                context_count=len(items),
                context_chars=sum(len(i.content) for i in items),
                latency_ms=(time.perf_counter() - started) * 1000.0,
                skipped=True,
                error=reason,
            )

        if not answer:
            return _skipped("答案为空")
        if looks_aborted(answer):
            return _skipped("Agent 中止话术，不计入忠实度")

        messages = [
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_eval_user_prompt(
                    question, answer, items, max_context_chars=self.max_context_chars
                ),
            },
        ]
        try:
            response: LLMResponse = self.llm(
                messages,
                None,
                **_llm_kwargs(
                    self.llm,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                    retries=self.retries,
                ),
            )
        except LLMError as exc:
            return AnswerEvalResult(
                question=question,
                effective_query=effective,
                verdict="fallback",
                context_count=len(items),
                context_chars=sum(len(i.content) for i in items),
                latency_ms=(time.perf_counter() - started) * 1000.0,
                fallback=True,
                error=f"Judge 调用失败：{exc}"[:200],
            )
        except Exception as exc:  # noqa: BLE001  假模型、协议错……一律记 fallback
            return AnswerEvalResult(
                question=question,
                effective_query=effective,
                verdict="fallback",
                context_count=len(items),
                context_chars=sum(len(i.content) for i in items),
                latency_ms=(time.perf_counter() - started) * 1000.0,
                fallback=True,
                error=f"Judge 调用异常：{type(exc).__name__}: {exc}"[:200],
            )

        tokens = int(getattr(response, "total_tokens", 0) or 0)
        content = (getattr(response, "message", None) or {}).get("content") or ""
        outcome = parse_eval_response(content, reverse_questions=self.reverse_questions)
        if outcome.fallback:
            return AnswerEvalResult(
                question=question,
                effective_query=effective,
                verdict="fallback",
                context_count=len(items),
                context_chars=sum(len(i.content) for i in items),
                tokens=tokens,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                fallback=True,
                error=outcome.error or "Judge 返回结构不可用",
            )

        claims = outcome.claims
        if not items:
            # 一个工具都没调就给出具体数字，那是凭记忆编的。Judge 有时会因为
            # "这话在常识上对"而放过它，这里统一按 unsupported 处理。
            claims = tuple(
                replace(claim, verdict="unsupported", evidence_quote="", evidence_source="")
                if claim.verdict == "supported"
                else claim
                for claim in claims
            )

        supported = sum(1 for c in claims if c.verdict == "supported")
        unsupported = sum(1 for c in claims if c.verdict == "unsupported")
        contradicted = sum(1 for c in claims if c.verdict == "contradicted")

        if claims:
            faithfulness: float | None = supported / len(claims)
        elif outcome.is_refusal:
            # 什么都没说，也就什么都没编。
            faithfulness = 1.0
        else:
            faithfulness = None

        mode: RelevanceMode
        embed_error = ""
        # 拒答（明确说"资料里没有"）没有回答问题，相关性必须是 0：
        # 它答的是"能不能答"，不是用户问的那件事。两者都由 expect_refusal 决定通过与否。
        refusal_like = outcome.noncommittal or outcome.is_refusal
        if refusal_like:
            relevance: float | None = 0.0
            mode = "noncommittal"
        elif not outcome.generated_questions:
            relevance = outcome.judge_relevance
            mode = "llm_judge"
        elif not (self.use_embedding and self._embedding_ready()):
            relevance = outcome.judge_relevance
            mode = "llm_judge"
        else:
            # 用改写后的规范问题做基准：答案是在回答那个完整意图，
            # 拿用户原话去比，口语化问法会平白多丢一截分。
            benchmark = effective or question
            score, embed_error = self._reverse_embedding_relevance(
                benchmark, outcome.generated_questions
            )
            if embed_error:
                relevance = outcome.judge_relevance
                mode = "llm_judge"
            else:
                relevance = score
                mode = "reverse_embedding"

        verdict = _decide_verdict(
            faithfulness=faithfulness,
            relevance=relevance,
            contradicted=contradicted,
            noncommittal=refusal_like,
            expect_refusal=expect_refusal,
            faithfulness_threshold=self.faithfulness_threshold,
            relevance_threshold=self.relevance_threshold,
        )

        return AnswerEvalResult(
            question=question,
            effective_query=effective,
            verdict=verdict,
            faithfulness=faithfulness,
            relevance=relevance,
            relevance_mode=mode,
            claims=claims,
            supported_count=supported,
            unsupported_count=unsupported,
            contradicted_count=contradicted,
            noncommittal=outcome.noncommittal,
            is_refusal=outcome.is_refusal,
            generated_questions=outcome.generated_questions,
            missing_aspects=outcome.missing_aspects,
            summary=outcome.summary,
            context_count=len(items),
            context_chars=sum(len(i.content) for i in items),
            tokens=tokens,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error=embed_error,
        )


def _normalize_contexts(contexts: Sequence[Any]) -> list[EvidenceItem]:
    """宽容地接受 EvidenceItem、dict 或 JSON 快照里读回来的结构。"""
    items: list[EvidenceItem] = []
    for entry in contexts or ():
        if isinstance(entry, EvidenceItem):
            items.append(entry)
        elif isinstance(entry, dict):
            items.append(
                EvidenceItem(
                    tool_name=str(entry.get("tool_name") or entry.get("tool") or "unknown"),
                    args_summary=str(entry.get("args_summary") or ""),
                    content=str(entry.get("content") or ""),
                    call_id=str(entry.get("call_id") or ""),
                )
            )
    return items


# ---------------------------------------------------------------- 聚合


def _mean(values: Sequence[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _aggregate_generation(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    if count == 0:
        return {
            "count": 0,
            "evaluated_count": 0,
            "skipped_count": 0,
            "fallback_count": 0,
            "pass_count": 0,
            "warn_count": 0,
            "fail_count": 0,
            "pass_rate": 0.0,
            "avg_faithfulness": None,
            "avg_relevance": None,
            "total_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": 0,
            "contradicted_claims": 0,
            "contradiction_case_count": 0,
            "noncommittal_count": 0,
            "refusal_count": 0,
            "refusal_mismatch_count": 0,
            "avg_gen_latency_ms": 0.0,
            "avg_eval_latency_ms": 0.0,
            "avg_eval_tokens": 0.0,
        }

    faith: list[float] = []
    rel: list[float] = []
    gen_latency: list[float] = []
    eval_latency: list[float] = []
    eval_tokens: list[float] = []
    counts = {"pass": 0, "warn": 0, "fail": 0, "skipped": 0, "fallback": 0}
    contradiction_cases = 0
    refusal_mismatch = 0
    noncommittal = 0
    refusals = 0

    for rec in records:
        verdict = str(rec.get("verdict") or "skipped")
        counts[verdict] = counts.get(verdict, 0) + 1
        value = rec.get("faithfulness")
        if value is not None:
            faith.append(float(value))
        relevance = rec.get("relevance")
        if relevance is not None:
            rel.append(float(relevance))
        if int(rec.get("contradicted_count") or 0) > 0:
            contradiction_cases += 1
        if rec.get("noncommittal"):
            noncommittal += 1
        if rec.get("is_refusal"):
            refusals += 1
        if rec.get("refusal_mismatch"):
            refusal_mismatch += 1
        gen_latency.append(float(rec.get("gen_latency_ms") or 0.0))
        eval_latency.append(float(rec.get("latency_ms") or 0.0))
        eval_tokens.append(float(rec.get("tokens") or 0))

    evaluated = counts["pass"] + counts["warn"] + counts["fail"]
    return {
        "count": count,
        "evaluated_count": evaluated,
        "skipped_count": counts["skipped"],
        "fallback_count": counts["fallback"],
        "pass_count": counts["pass"],
        "warn_count": counts["warn"],
        "fail_count": counts["fail"],
        "pass_rate": (counts["pass"] / evaluated) if evaluated else 0.0,
        "avg_faithfulness": _mean(faith),
        "avg_relevance": _mean(rel),
        "total_claims": sum(int(r.get("supported_count") or 0) + int(r.get("unsupported_count") or 0) + int(r.get("contradicted_count") or 0) for r in records),
        "supported_claims": sum(int(r.get("supported_count") or 0) for r in records),
        "unsupported_claims": sum(int(r.get("unsupported_count") or 0) for r in records),
        "contradicted_claims": sum(int(r.get("contradicted_count") or 0) for r in records),
        "contradiction_case_count": contradiction_cases,
        "noncommittal_count": noncommittal,
        "refusal_count": refusals,
        "refusal_mismatch_count": refusal_mismatch,
        "avg_gen_latency_ms": _mean(gen_latency) or 0.0,
        "avg_eval_latency_ms": _mean(eval_latency) or 0.0,
        "avg_eval_tokens": _mean(eval_tokens) or 0.0,
    }


def summarize_generation_results(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """聚合总体指标与按 category 分组指标。"""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        category = str(record.get("category") or "uncategorized")
        grouped.setdefault(category, []).append(record)

    return {
        "overall": _aggregate_generation(records),
        "by_category": {
            cat: _aggregate_generation(items) for cat, items in grouped.items()
        },
    }


__all__ = [
    "AnswerEvalResult",
    "AnswerEvaluator",
    "ClaimCheck",
    "EvalVerdict",
    "EvidenceItem",
    "EVAL_SYSTEM_PROMPT",
    "JudgeOutcome",
    "MAX_CONTEXT_CHARS",
    "MAX_ITEM_CHARS",
    "RelevanceMode",
    "build_eval_user_prompt",
    "extract_turn_contexts",
    "looks_aborted",
    "parse_eval_response",
    "summarize_args",
    "summarize_generation_results",
    "tool_priority",
]
