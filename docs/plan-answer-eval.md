# 开发期生成质量评测（忠实度 Faithfulness + 答案相关性 Answer Relevance）架构与实施计划书

> 状态：待执行（可直接交付给执行 Agent 实施）。  
> 目标代码库：`e:\VScode\datainsight`（当前回归测试基线：585 项 `unittest` 全部通过）。  
> 定位：**纯开发期离线评测**——仅用于开发期量化评估模型生成答案的忠实度与相关性，**不接入 `Agent` 运行时链路，不改动 Streamlit UI，不消耗线上用户任何额外 token**，延续 [eval_retrieval.py](file:///e:/VScode/datainsight/eval/eval_retrieval.py) 的风格，且不引入任何外部评测框架。

---

## 1. 目标、非目标与硬约束

### 1.1 核心目标

在现有检索评测（`eval/eval_retrieval.py`：MRR@K / Hit@K / Recall@K）基础上，补齐 **生成阶段（Post-Generation）** 的开发期自动化评测能力，形成「检索召回 + 生成质量」完整量化闭环：

1. **生成评测核心引擎（`eval/answer_metrics.py`）**
   - **忠实度（Faithfulness）**：从本轮 Agent 对话轨迹中提取所有工具返回上下文（`tool` observations），将生成答案拆解为原子事实声明（Atomic Claims），逐条核验是否被上下文严格支撑，精准定位「无中生有（`unsupported`）」与「与依据冲突（`contradicted`）」的幻觉片段。
   - **答案相关性（Answer Relevance）**：采用「**逆向生成假设问题 + Embedding 余弦相似度**（向量模型可用时） / **Judge LLM 语义评分**（未配置向量、`--no-embedding` 或向量异常时自动平滑降级）」双模式评估答案是否直接、完整地回应了用户问题，并识别拒答回避（`noncommittal`）与遗漏维度。
2. **端到端生成评测 CLI（`eval/eval_generation.py` + `eval/generation_dataset.json`）**
   - 驱动真实 `Agent`（在独立临时会话中绑定 `data/vault` 语料与 `data/sales.csv`）生成答案并自动打分；
   - 支持 **A/B 消融对比实验**（如对比「纯 BM25 vs 混合检索」、「关闭查询改写 vs 开启查询改写」对最终回答忠实度与相关性的影响）；
   - 支持 **样本快照模式**（`--save-predictions` / `--predictions`）：生成一次答案与上下文快照后，可离线反复重跑评测或调整 Judge 模型，省去重复跑 Agent 的耗时与费用。
3. **纯离线单元测试（`tests/test_eval_generation.py`）**
   - 覆盖上下文提取、忠实度边界、相关性双模式与降级、JSON 容错、汇总指标及评测集格式校验，全部用替身模型离线运行，不发真实网络请求。

### 1.2 非目标与硬约束

- **零线上运行时侵入**：**严禁修改** `app/agent.py`、`app/ui_render.py`、`app/ui_streamlit.py`、`app/session.py`、`app/chats.py`、`app/config.py`。业务代码与 Streamlit 界面保持 100% 原样，现有 585 项回归测试零受影响风险。
- **零外部评测框架**：不引入 `ragas`、`trulens`、`deepeval`、`langchain`；仅使用 Python 标准库、`numpy` 以及项目现有的 `app/llm.py`（`chat`）、`app/embeddings.py`（`embed_texts`, `l2_normalize`）、`app/agent.py`（`Agent`, `make_system_prompt`）、`app/tools/data_tools.py`、`app/vault.py`。
- **绝对数据隔离**：开发期评测脚本运行真实 `Agent` 时，必须使用独立临时 `session_key`（如 `"eval_gen_<uuid>"`），且**不写入** `data/chats/`、`data/uploads/` 用户真实历史；运行结束后清理临时数据集与文档挂载。
- **绝不泄露密钥**：终端报告与 `--json-out` 导出文件中严禁输出任何 `api_key` 明文。

---

## 2. 针对 DataInsight 多工具 ReAct 的上下文与基准适配

DataInsight 的回答并非仅依赖单段文本召回，而是多工具 ReAct 循环的最终汇总。开发期评测脚本在**不改动 `Agent` 源码**的前提下，直接读取 `agent.session.messages` 即可无损获取全部评测要素：

1. **工具上下文无损采集（`extract_turn_contexts(messages)`）**
   - 扫描 `session.messages` 中 `role == "assistant"`（含 `tool_calls`）与 `role == "tool"` 的配对消息，通过 `tool_call_id` 还原工具名（如 `run_sql`、`run_python`、`read_document`、`search_document`、`read_file`、`inspect_table`、`make_chart` 等）与精简参数。
   - **去重与预算裁剪**：同一轮重复读取完全相同的 `(tool_name, args, content)` 只保留一次；当总字数超出评测上下文上限（默认 `12,000` 字）时，按信息密度优先级保留（`run_sql` / `run_python` / `read_document` / `read_file` > `search_document` / `inspect_table` > `list_documents`），单条截断上限默认 `3,000` 字。
2. **改写后的有效问题对齐（`effective_query`）**
   - 当评测开启前置查询改写（`QueryRewriter`）时，通过 `Agent(..., on_event=capture_events)` 捕获 `"rewrite"` 事件中的 `rewritten_query`（若有），作为多轮/口语改写场景下相关性计算的参考基准，同时保留原始 `question` 用于报告展示。
3. **诚实拒答与幻觉的正交区分**
   - 评测集专门包含 `unanswerable`（文档/表中不存在的指标或条款）用例：当工具返回空结果且模型诚实回答"未找到/无法确定"时，**忠实度应为高分（`1.0`，无编造）**，而 **`noncommittal=True` 使相关性记 `0.0`**——从而把"诚实说不知道"和"胡编乱造"清晰区分开。

---

## 3. 核心算法与计分规则

### 3.1 忠实度（Faithfulness, `[0.0, 1.0]`）

基于 RAGAS **Claim Decomposition + Verification** 思路，合并为**单次结构化 Judge LLM 调用**（一次请求同时完成原子声明拆解与逐条核验，控制评测成本与耗时）：

1. **前置过滤**：
   - 若生成答案是 `Agent` 异常中止话术（重复中止 / 步数上限 / 预算耗尽 / `（模型调用失败：…）`），标记 `aborted=True`、`verdict="skipped"`，不计入忠实度分母，单独统计 `aborted_count`。
2. **原子事实拆解与三态核验**：
   Judge LLM 将 `answer` 拆解为独立可验证的原子事实陈述 `claims`（覆盖数字、百分比、排名、日期、文档条款、实体归因；忽略礼貌寒暄与后续操作建议），逐条对照 `contexts` 判为三态之一：
   - `supported`：可由上下文直接读出或经等价换算得出（例如 `1254000` 元与 `125.4万元`、`0.3333` 与 `33.3%` 均视为等价支撑），需给出 `evidence_source` 与原文短引 `evidence_quote`；
   - `unsupported`：上下文中找不到依据（模型凭常识补充或推断）；
   - `contradicted`：与上下文中的数值、实体或结论直接冲突（或文档与数据表不一致时擅自改动其中一方）。
3. **计分公式**：
   - 当 $N_{\text{claims}} > 0$ 时：
     $$\text{faithfulness} = \frac{N_{\text{supported}}}{N_{\text{claims}}}$$
   - 若 `contexts` 为空且 `N_claims > 0`：模型在未查任何工具的情况下给出了具体事实断言，判 `unsupported`（`faithfulness = 0.0`）。
   - 若 `N_claims == 0`：
     - 若为基于空检索/无数据源的诚实拒答（`is_refusal=True`）：`faithfulness = 1.0`；
     - 若为纯元引导且非拒答：`faithfulness = None`（不拉低有效样本均值）。

### 3.2 答案相关性（Answer Relevance, `[0.0, 1.0]`）

采用**逆向问题余弦相似度为主、LLM Judge 语义评分为降级**的双模式（沿用 [hybrid_search.py](file:///e:/VScode/datainsight/app/hybrid_search.py) 的向量降级机制）：

1. **同一次 Judge LLM 调用一并产出**：
   - `noncommittal: bool`：回答是否未给出实质答案（如"未找到相关内容"、"无法确认"、"请提供更多数据"）；
   - `generated_questions: list[str]`：仅依据 `answer` 反向推导出的 $N=3$ 条它最直接回答的完整问题；
   - `judge_relevance: float`（`[0.0, 1.0]`）：Judge 从「意图覆盖度、直接性、跑题/冗余」给出的参考评分；
   - `missing_aspects: list[str]`：答案遗漏的子问题或关键约束（最多 5 条）。
2. **最终得分计算**：
   - 若 `noncommittal == True`：`relevance = 0.0`，`relevance_mode = "noncommittal"`；
   - 否则若 `use_embedding=True` 且（注入了 `embedder` 或 `config.embedding_current().ready`）且 `generated_questions` 非空：
     - 调用 `app.embeddings.embed_texts([benchmark_query, *generated_questions])`（仅 4 条短文本，单次批处理，返回已 L2 归一化向量）；
     - 计算 $\text{sim}_i = \langle \vec{e}_q, \vec{e}_{q'_i} \rangle$，截断到 `[0.0, 1.0]` 后取均值：
       $$\text{relevance} = \frac{1}{k} \sum_{i=1}^{k} \max\left(0.0, \min(1.0, \text{sim}_i)\right)$$
     - `relevance_mode = "reverse_embedding"`；
   - 否则（未配置 Embedding、`--no-embedding` 或调用抛异常）：自动平滑降级为 `relevance = judge_relevance`，`relevance_mode = "llm_judge"`。

### 3.3 单条样本评级（`verdict`）

| `verdict` | 判定条件（默认阈值 `faithfulness >= 0.8`, `relevance >= 0.7`） |
|---|---|
| `skipped` | 答案为空或属于 `Agent` 中止/报错话术 |
| `fallback` | Judge LLM 调用异常或返回非法 JSON（不阻断整批评测） |
| `pass` | `contradicted_count == 0` 且（`faithfulness is None or faithfulness >= 0.8`）且 `relevance >= 0.7` |
| `warn` | 非 `fail` 且存在部分 `unsupported` / `missing_aspects`，或分数处于 `[0.5, 阈值)` 区间 |
| `fail` | `contradicted_count > 0` 或（`faithfulness is not None and faithfulness < 0.5`）或（`relevance < 0.5` 且非 `noncommittal` 诚实拒答） |

---

## 4. 整体架构与文件清单

### 4.1 目录与文件规划

| 文件路径 | 状态 | 职责说明 |
|---|---|---|
| `eval/answer_metrics.py` | **新建** | 上下文提取器、Judge Prompt、JSON 容错解析、忠实度与相关性双模式计分器、`AnswerEvaluator`、`summarize_generation_results()` |
| `eval/eval_generation.py` | **新建** | 开发期端到端 CLI 脚本：加载评测集、搭建隔离会话驱动 `Agent.ask()`（或读快照）、执行评测、打印终端报告、导出 JSON |
| `eval/generation_dataset.json` | **新建** | 16 条真实语料与示例表端到端评测集（覆盖文档事实、表格聚合、跨源交叉、不可回答拒答四类） |
| `tests/test_eval_generation.py` | **新建** | 纯离线单元测试（`FakeLLM` / `BoomLLM` / `fake_embed` / `ScriptedLLM`），验证全部指标、降级、聚合与数据集校验 |
| `README.md` | **按需小改** | 在评测章节补充 `eval/eval_generation.py` 用法说明（仅文档，不改运行代码） |

> 再次确认：`app/` 目录下**没有任何文件需要修改**。

### 4.2 数据流示意图

```text
python eval/eval_generation.py [--mode bm25|hybrid] [--rewrite|--no-rewrite]
        │
        ├─ 方式 A：端到端真跑（默认）
        │    1. 加载 eval/generation_dataset.json 并校验语料/数据文件存在
        │    2. 为每条用例创建隔离 session_key，按 source_mode 挂载：
        │         · "vault" -> data_tools.set_documents(vault_docs)
        │         · "sales" -> data_tools.bind_sample(key)
        │         · "both"  -> 同时挂载两者
        │    3. 运行 Agent(session, system_prompt=make_system_prompt(...),
        │                 session_key=key, rewriter=...).ask(case["query"])
        │         · 通过 on_event 捕获 rewrite 的 effective_query 与步骤/耗时
        │         · 从 session.messages 提取本轮 tool contexts
        │    4. 清理临时 session_key 挂载
        │
        ├─ 方式 B：离线快照回归（--predictions <path>）
        │    · 直接读取先前 --save-predictions 导出的 (query, effective_query, contexts, answer)
        │
        ▼
  AnswerEvaluator.evaluate(question, answer, contexts, effective_query)
        │  [eval/answer_metrics.py]
        ├─ 单次 Judge LLM 调用 -> claims + noncommittal + generated_questions + judge_relevance
        └─ 若可用 -> app.embeddings.embed_texts() 计算逆向余弦相关性
        │
        ▼
  summarize_generation_results() -> 终端报告（总体 + 分 category + Bad Case 明细）+ 可选 JSON
```

---

## 5. 详细模块设计与接口契约

### 5.1 `eval/answer_metrics.py`

#### 5.1.1 核心数据结构

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

ClaimVerdict = Literal["supported", "unsupported", "contradicted"]
EvalVerdict = Literal["pass", "warn", "fail", "skipped", "fallback"]
RelevanceMode = Literal["reverse_embedding", "llm_judge", "noncommittal", "skipped"]


@dataclass(frozen=True)
class EvidenceItem:
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

    def to_dict(self) -> dict[str, Any]: ...
```

#### 5.1.2 上下文提取函数：`extract_turn_contexts`

```python
def extract_turn_contexts(
    messages: Sequence[dict[str, Any]],
    *,
    user_idx: int | None = None,
    max_total_chars: int = 12000,
    max_item_chars: int = 3000,
) -> list[EvidenceItem]:
```

设计要点：
- `user_idx is None` 时自动定位最后一条 `role == "user"` 的消息索引，从 `messages[user_idx + 1:]` 提取（天然适配单轮评测与多轮末尾轮次抽取）。
- 通过 `assistant.tool_calls[].id` 与 `tool.tool_call_id` 精确配对 `tool_name` 和精简 `args_summary`（`json.loads` 后压缩为 `k=v` 短串）。
- 对相同 `(tool_name, args_summary, content)` 去重。
- 按工具权重排序裁剪（高信息密度工具：`run_sql`, `run_python`, `read_document`, `read_file`, `make_chart`, `make_mindmap`, `make_pptx`, `edit_pptx`, `write_text_file`；中：`search_document`, `inspect_table`；低：`list_documents`, `list_files`, `find_program`, `open_file`），确保在 `max_total_chars` 预算内优先保留核心数据依据。

#### 5.1.3 Judge Prompt 与容错解析

`EVAL_SYSTEM_PROMPT` 必须明确要求返回单个 JSON 对象：

```json
{
  "is_refusal": false,
  "noncommittal": false,
  "claims": [
    {
      "statement": "原子事实陈述",
      "verdict": "supported | unsupported | contradicted",
      "evidence_source": "工具名或文档名",
      "evidence_quote": "上下文中的直接依据片段",
      "reason": "简短判定理由"
    }
  ],
  "generated_questions": ["问题1", "问题2", "问题3"],
  "judge_relevance": 0.92,
  "missing_aspects": [],
  "summary": "一句话总结"
}
```

`parse_eval_response(...)` 须做到：
- 剥离 ` ```json ... ``` ` 代码围栏，并截取首个 `{` 至最后一个 `}` 做容错反序列化；
- 非法 `verdict` 自动归一为 `"unsupported"`；`judge_relevance` clamp 到 `[0.0, 1.0]`；
- 空 `statement` 自动剔除；`generated_questions` 去重并裁剪至 `reverse_questions` 条；
- 任何解析错误返回 `fallback=True` 而绝不抛异常中断整批评测。

#### 5.1.4 评测器类与聚合统计函数

```python
class AnswerEvaluator:
    def __init__(
        self,
        *,
        llm: LLMCallable | None = None,
        embedder: EmbedderCallable | None = None,
        use_embedding: bool = True,
        timeout: int = 20,
        max_tokens: int = 1200,
        reverse_questions: int = 3,
        faithfulness_threshold: float = 0.8,
        relevance_threshold: float = 0.7,
    ) -> None: ...

    def evaluate(
        self,
        *,
        question: str,
        answer: str,
        contexts: Sequence[EvidenceItem] = (),
        effective_query: str = "",
    ) -> AnswerEvalResult: ...


def summarize_generation_results(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """聚合总体指标与按 category 分组指标。"""
```

`summarize_generation_results` 输出字段契约：
- `overall` 与 `by_category[cat]` 均包含：
  - `count`、`evaluated_count`、`skipped_count`、`fallback_count`
  - `pass_count`、`warn_count`、`fail_count`、`pass_rate`
  - `avg_faithfulness`（仅对 `faithfulness is not None` 样本求均值）、`avg_relevance`（仅对 `relevance is not None` 样本求均值）
  - `total_claims`、`supported_claims`、`unsupported_claims`、`contradicted_claims`、`contradiction_case_count`
  - `noncommittal_count`、`refusal_count`
  - `avg_gen_latency_ms`、`avg_eval_latency_ms`、`avg_eval_tokens`

---

### 5.2 `eval/generation_dataset.json`

结构沿用 [retrieval_dataset.json](file:///e:/VScode/datainsight/eval/retrieval_dataset.json) 规范，初始提供 16 条代表性用例：

```json
{
  "version": 1,
  "description": "DataInsight 开发期端到端生成质量评测集（忠实度 + 答案相关性）",
  "cases": [
    {
      "id": "g01",
      "category": "doc_fact",
      "source_mode": "vault",
      "query": "海淀信用就医的最高授信额度是多少？",
      "expect_refusal": false,
      "reference_hints": ["10000", "10000元"],
      "note": "文档精确数值定位与单位忠实度"
    },
    {
      "id": "g05",
      "category": "table_agg",
      "source_mode": "sales",
      "query": "哪个城市的总销售额最高，占整体销售额的百分比大概是多少？",
      "expect_refusal": false,
      "reference_hints": [],
      "note": "SQL 聚合 + 占比换算（检验等值换算不被误判矛盾）"
    },
    {
      "id": "g09",
      "category": "cross_source",
      "source_mode": "both",
      "query": "结合文档里的区块链存证案例和当前销售数据表，分别总结各自核心数据指标。",
      "expect_refusal": false,
      "reference_hints": [],
      "note": "跨文档与数据表交叉问答，验证工具不混淆、不张冠李戴"
    },
    {
      "id": "g13",
      "category": "unanswerable",
      "source_mode": "vault",
      "query": "资料里有没有提到 2026 年第四季度某火星基地的供电预算？",
      "expect_refusal": true,
      "reference_hints": [],
      "note": "不可回答边界：应诚实拒答（faithfulness=1.0, noncommittal=true, relevance=0.0）"
    }
  ]
}
```

### 5.3 `eval/eval_generation.py`

CLI 参数设计（与 `eval/eval_retrieval.py` 风格保持一致）：
- `--dataset`：评测集路径（默认 `eval/generation_dataset.json`）；
- `--categories`：逗号分隔过滤类别；
- `--ids`：逗号分隔只跑指定用例 id；
- `--search-mode`：`bm25` / `hybrid`（通过 `hybrid_search.set_hybrid_enabled(...)` 控制检索路径，用于对比实验）；
- `--rewrite` / `--no-rewrite`：是否启用 `QueryRewriter`；
- `--no-embedding`：强制相关性只走 `llm_judge` 模式（不调 `/embeddings`）；
- `--faithfulness-threshold`（默认 `0.8`）、`--relevance-threshold`（默认 `0.7`）；
- `--save-predictions <path>`：将端到端生成结果（含 `contexts`、`answer`、`effective_query`）落盘，便于后续离线复评；
- `--predictions <path>`：直接读取快照跑评测，跳过 `Agent.ask()`；
- `--show-all`：终端打印所有用例明细（默认仅展开 `warn`/`fail`/`fallback` 及 `expect_refusal` 不匹配的 Bad Case）；
- `--json-out <path>`：导出完整结构化报告 JSON。

**会话与测试数据保护机制**：
- 每条用例使用唯一 `key = f"eval-gen-{case['id']}-{uuid.uuid4().hex[:6]}"`；
- 在 `try ... finally` 中确保执行 `data_tools.clear_documents(key)` 与 `data_tools.unbind_dataset(key)`，绝不残留内存连接或修改 `DEFAULT_KEY`；
- 在执行真实 `Agent` 前检查 `config.current().ready`，未配置对话模型 Key 时提前给出明确错误提示退出（不吞报错）。

---

## 6. 测试与分步验收计划

### 6.1 新增离线单测：`tests/test_eval_generation.py`

仿照 `tests/test_eval_retrieval.py` 风格，全部纯离线运行：

1. **`TestExtractTurnContexts`**：
   - 从多轮 `session.messages` 中准确只取最后一轮 `user` 之后的工具观察；
   - `tool_call_id` 关联正确的 `tool_name` 与参数摘要；
   - 重复观测去重、超出 `max_total_chars` 时高优先级工具结果不被 `list_documents` 挤掉。
2. **`TestFaithfulnessScoring`**：
   - 全 `supported` -> `1.0` (`pass`)；
   - 部分 `unsupported` -> 准确比值 (`warn`)；
   - 存在 `contradicted >= 1` -> 强制 `fail`；
   - 空 `contexts` 下编造事实 -> `0.0`；诚实拒答 `is_refusal=True` -> `1.0`；
   - `Agent` 中止话术（如重复调用中止、预算耗尽）自动 `skipped`，`FakeLLM.calls == 0`。
3. **`TestRelevanceScoring`**：
   - 注入确定性 `fake_embed`：`reverse_embedding` 模式精确计算余弦相似度均值；
   - `noncommittal=True` 强制 `relevance == 0.0`；
   - `use_embedding=False`、未配置向量或 `embed_texts` 抛 `EmbeddingError` 时无缝降级为 `llm_judge`；
   - 优先采用 `effective_query` 作为逆向问题对比基准。
4. **`TestParseAndFallback`**：
   - Markdown 代码围栏 / 前后多余文本截断提取；
   - `BoomLLM` 抛 `LLMError` 或返回坏 JSON 时返回 `fallback=True`，不抛异常。
5. **`TestGenerationEvalRunnerAndDataset`**：
   - `eval/generation_dataset.json` 加载、ID 唯一性、类别覆盖（`doc_fact`, `table_agg`, `cross_source`, `unanswerable`）校验；
   - `summarize_generation_results()` 总体与分 `category` 聚合指标精确校验；
   - `eval_generation.evaluate_predictions()` 与基于 `ScriptedLLM` 驱动的端到端隔离流水线冒烟测试。

### 6.2 分步实施顺序与验收标准

1. **Step 1**：新建 `eval/answer_metrics.py` 与 `tests/test_eval_generation.py` 中的核心指标/提取/解析用例。  
   **验收**：`python -m unittest tests/test_eval_generation.py -v` 绿灯。
2. **Step 2**：新建 `eval/generation_dataset.json` 与 `eval/eval_generation.py`（含 `evaluate_predictions`、端到端 `Agent` 隔离运行、`format_generation_report` 与 CLI），并补齐 `tests/test_eval_generation.py` 对数据集与报告汇总的校验。  
   **验收**：`python -m unittest tests/test_eval_generation.py -v` 全绿；`python eval/eval_generation.py --help` 正常输出。
3. **Step 3**：按需更新 `README.md` 的开发期评测说明，运行全量回归测试：  
   ```bash
   python -m unittest discover -s tests -p "test_*.py" -v
   ```
   **验收**：原有 585 项测试 + 新增单测全部通过，确认 `app/` 下无任何非预期改动。
