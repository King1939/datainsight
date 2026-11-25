# 前置查询改写（Query Rewriting）架构与实施计划

> 状态：待执行（可直接交付给执行 Agent 实施）。
> 目标代码库：`e:\VScode\datainsight`（当前回归测试基线：456 项 `unittest` 全部通过）。

## 1. 目标与非目标

### 1.1 目标

在主 Agent 模型收到用户请求之前，引入一层**低延迟、策略自适应、可观测、可降级**的前置查询改写管线（Pre-LLM Query Rewriting）。根据问题特征与当前会话上下文，将用户提问自动路由到以下 5 种处理策略之一：

1. **直通透传（`pass_through`）**：清晰单步问题或控制类短指令，原样透传，不做多余改写。
2. **基础改写（`basic_rewrite`）**：指代消解、多轮省略补全、错别字与口语规范化、当前表列名或文档名实体对齐。
3. **查询扩展（`query_expansion`）**：同义词、业务口径、中英缩写及近义术语扩展，提升文档检索（`search_document`）和文本筛选的召回率。
4. **多路查询（`multi_query`）**：针对宽泛、开放、探索型问题，生成 2~4 条互补视角的并行查询子句，避免单一视角遗漏。
5. **查询分解（`decomposition`）**：针对复杂多跳、对比归因、跨表跨文档、含先后依赖的多步骤任务，拆解为 2~5 个带依赖关系（无环 DAG）的有序子问题。

### 1.2 非目标与硬约束

- **零外部编排框架**：延续项目手写风格，仅使用 Python 标准库与现有 `app/llm.py`，不引入 LangChain / LlamaIndex 等重框架，不新增第三方依赖。
- **100% 向后兼容现有 456 项测试**：`Agent.__init__` 默认 `rewriter=None`（不触发改写），确保现有基于 `ScriptedLLM` 的单测（如 `tests/test_week2.py` 对 `messages` 长度与事件序列的精确断言）零破坏；UI 层默认装配 `QueryRewriter`。
- **不污染 `session.messages` 协议历史**：绝不向 `session.messages` 插入非 OpenAI 标准字段（避免部分网关返回 HTTP 400），也不把冗长的改写包装模板永久写入持久化对话历史。
- **失败 100% 静默降级**：改写调用超时、网络/HTTP 报错、JSON 解析失败时，无缝降级为透传原始问题并继续主循环，绝不阻塞主问答流程。
- **不引入向量检索与重排模型**：本轮聚焦 LLM 策略路由改写与现有工具链（`run_sql` / `search_document` / `run_python` / `make_chart` 等）协同；多路检索 RRF 融合作为二期可选增强单列。

---

## 2. 整体架构与数据流

### 2.1 端到端数据流

```text
用户输入 question
      │
      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Agent.ask(question)  [app/agent.py]                                     │
│                                                                         │
│  0. 注入会话上下文: data_tools.use_dataset(self.session_key)              │
│     (必须在采集上下文元数据之前执行，防止多标签页读串表或读串文档)       │
│                                                                         │
│  1. 若 self.rewriter is not None and self.rewriter.enabled:             │
│     ├─ Stage A: build_context() 采集轻量上下文 [app/query_rewrite.py]    │
│     │    · 最近 N 轮纯文本对话 (过滤 system / tool / tool_calls)         │
│     │    · 当前表名 / 列名 / 行数 (data_tools)                           │
│     │    · 当前文档名 / 类型 / 字数 / 段数 (仅摘要，绝不读取正文)        │
│     │    · 当前本地文件权限 (files.allow_write())                        │
│     ├─ Stage B: _fast_bypass() 规则快速通道 (0 token, 0 ms)             │
│     │    · 开关关闭 / 空串 / 续跑礼貌短语 / 单步启动程序指令             │
│     ├─ Stage C: 单次 LLM 调用「策略路由 + 改写合一」                     │
│     │    · 非流式, temperature=0.0, max_tokens=600, timeout=15s         │
│     │    · 输出严格 JSON (strategy + 对应结构化改写字段)                 │
│     ├─ Stage D: parse_rewrite_response() 容错解析、裁剪、去重、DAG 破环  │
│     │    · 任何异常或非法 JSON -> 降级为 pass_through (fallback=True)    │
│     └─ Stage E: format_for_agent() 封装本轮给主模型的增强 User Prompt    │
│                                                                         │
│  2. 记账与可观测性                                                       │
│     · rw_result.tokens 累加到 self.session.tokens (计入本轮预算)         │
│     · 改写不增加 self.session.steps (步数专指主 Agent ReAct 步数)        │
│     · 发射事件 self._emit("rewrite", **rw_result.to_event_payload())    │
│                                                                         │
│  3. 隔离写入与请求发送                                                   │
│     · self.session.add_user(question)  <-- 持久化历史只存干净原始问题    │
│     · 构造局部浅拷贝 request_messages，仅在本轮循环内将 user_idx         │
│       的 content 临时替换为 packaged_question                            │
│                                                                         │
│  4. 进入原有 ReAct 循环 (run_sql / search_document / run_python ...)    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心架构决策

1. **为什么拦截点放在 `Agent.ask()` 入口，而不是 `ui_streamlit._run_turn()`？**
   - 任何调用 `Agent` 的入口（Streamlit UI、CLI 评测脚本、单元测试）都能统一复用改写能力；
   - `tests/test_week4_ui.py` 中的 `_ScriptedAgent` / `_SlowAgent` 直接替换了 `ss.agent`，将改写封装为 `Agent` 的内部前置阶段不会破坏已有 UI 冒烟测试。
2. **为什么「策略路由 + 改写执行」合并为单次 LLM 调用，而不是先分类再改写的两段串行调用？**
   - 前置改写发生在主模型首 token 输出之前，属于纯串行阻塞延迟；拆成两次串行请求会让首字节时间翻倍，并重复发送对话历史与表/文档元数据。配合规则快速通道，单次结构化 JSON 调用是延迟与成本的最优折中。
3. **为什么不直接把改写包装文本存进 `session.messages`？**
   - 避免多轮对话上下文随轮次快速膨胀；
   - 避免 `ui_render.messages_to_chat()` 把内部 prompt 模板暴露给聊天历史恢复界面；
   - 避免下一轮 `build_context()` 读到上一轮自己生成的改写模板。使用局部浅拷贝 `request_messages` 替换本轮 `user_idx`，主循环每一步工具调用都能看到完整的改写指令，循环结束后持久化历史依然完全干净。

---

## 3. 五种策略的路由规则、优先级与快速通道

### 3.1 策略选择矩阵

| 策略 (`strategy`) | 适用问题特征 | 典型示例 | 核心动作与产出 |
|---|---|---|---|
| `pass_through` | 规则命中短路；或单步事实型、对象/指标/口径已完全清晰，改写无实质增益 | `"继续"`、`"好的"`、`"打开微信"`、`"这张表有多少行？"`、`"各城市销售额排名前五"` | 原样透传，不做任何修改 |
| `basic_rewrite` | 多轮追问（代词指代、省略主语/指标/筛选条件）、口语模糊表述、错别字、未对齐实际列名/文档名 | Turn 2: `"北京呢？"`、`"把它改成折线图"`、`"卖得最差的那个品类"`、`"这份报告讲了什么"` | 结合最近历史补全主语/指标/过滤条件，并对齐当前表列名或文档名，产出单条自包含 `rewritten_query` |
| `query_expansion` | 检索型/概念型/术语型问题（尤其是查文档 `search_document` 或文本列模糊筛选），用户用词口语化或偏单一，字面匹配易漏召回 | `"违约怎么赔"`、`"文档里提到 ROI 了吗"`、`"查一下供应链风险相关条款"` | 生成规范 `rewritten_query` + `expanded_terms`（3~6 个高相关同义词/专业术语/中英缩写）+ `search_keywords`（可直接传给 `search_document` 的关键词组） |
| `multi_query` | 宽泛、开放、探索型或多视角分析/调研问题，单一查询会遗漏关键维度 | `"分析一下整体销售表现"`、`"这份年报里有哪些亮点和风险"`、`"各区域业务情况怎么样"` | 产出 2~4 条互补视角的并行 `sub_queries`（含 `id`、`query`、`angle`、`target_source`，无依赖关系） |
| `decomposition` | 复杂多跳、对比归因、跨源交叉验证（数据表 + 资料文档）、含显式先后依赖的多步任务 | `"找出销售额跌得最多的城市，查报告里的原因并做3页PPT"`、`"对比各品类实际毛利和文档目标毛利，标出未达标的"` | 产出 2~5 个有序 `sub_queries`（含 `id`、`query`、`depends_on`、`target_source`、`purpose`），构成无环 DAG |

### 3.2 策略冲突与优先级规则

当一个问题同时具备多种特征（例如既有多轮指代省略，又需要拆解执行，如 `"把刚才那两个城市按月拆开，再结合文档对比原因"`）：

- **单一主策略原则**：单次请求只允许选择一种主 `strategy`（绝不级联多次调用改写模型）；
- **路由优先级**：`decomposition` > `multi_query` > `query_expansion` > `basic_rewrite` > `pass_through`；
- **内嵌对齐要求**：在 System Prompt 中硬性规定——当选择 `query_expansion`、`multi_query` 或 `decomposition` 时，必须顺手在 `rewritten_query` 和每条 `sub_queries[].query` 内部完成指代消解与列名/文档名对齐，确保每条子查询均自包含。

### 3.3 规则快速通道（`_fast_bypass`）细则

在调用改写 LLM 之前先过确定性规则，命中即直接返回 `RewriteResult(strategy=RewriteStrategy.PASS_THROUGH, bypassed=True, rewritten_query=question)`：

1. `not self.enabled` 或 `not (question or "").strip()`；
2. 归一化后（去除首尾空白与常见中英文标点 ` \t\r\n，。！？,.!?~～`）精确命中控制/确认/礼貌短语集合：
   ```python
   _BYPASS_PHRASES = {
       "继续", "接着做", "继续做", "继续吧", "好的", "好", "ok", "okay",
       "谢谢", "感谢", "明白了", "收到", "不用了", "算了",
   }
   ```
3. 明确的单步本地打开指令：仅当 `ctx.write_mode` 为 `True`，且问题不包含复合任务连接词（`["并", "并且", "然后", "同时", "顺便", "以及", "再", "之后"]`），且匹配单步打开正则 `r"^(打开|启动)\s*\S{1,24}$"`（如 `"打开微信"`、`"打开记事本"`）；
4. **反误杀保护**：像 `"北京呢？"`、`"为什么？"`、`"画个图"`、`"占比呢"` 这类短追问**绝不可**因为字数短而被短路，必须交给 LLM 结合上下文走 `basic_rewrite`。

---

## 4. 详细模块设计与接口契约

### 4.1 新建模块：`app/query_rewrite.py`

#### 4.1.1 枚举与核心数据结构

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RewriteStrategy(str, Enum):
    PASS_THROUGH = "pass_through"
    BASIC_REWRITE = "basic_rewrite"
    QUERY_EXPANSION = "query_expansion"
    MULTI_QUERY = "multi_query"
    DECOMPOSITION = "decomposition"


STRATEGY_LABELS: dict[RewriteStrategy, str] = {
    RewriteStrategy.PASS_THROUGH: "直接透传",
    RewriteStrategy.BASIC_REWRITE: "基础改写",
    RewriteStrategy.QUERY_EXPANSION: "查询扩展",
    RewriteStrategy.MULTI_QUERY: "多路查询",
    RewriteStrategy.DECOMPOSITION: "查询分解",
}


@dataclass(frozen=True)
class SubQuery:
    id: str                                # "Q1", "Q2", ...
    query: str                             # 自包含的子查询 / 子问题
    angle: str = ""                        # multi_query 使用：视角标签，如 "总量趋势"
    purpose: str = ""                      # decomposition 使用：该步目标 / 中间产出
    depends_on: tuple[str, ...] = ()       # decomposition 使用：前置依赖 ID，如 ("Q1",)
    target_source: str = "auto"            # "table" | "document" | "mixed" | "workspace" | "auto"


@dataclass(frozen=True)
class RewriteContext:
    history: tuple[tuple[str, str], ...] = ()   # 最近 N 轮 (role, concise_text)
    has_table: bool = False
    table_label: str = ""
    columns: tuple[str, ...] = ()
    table_rows: int | None = None
    documents: tuple[dict[str, Any], ...] = ()  # [{"name", "kind", "chars", "parts"}]
    write_mode: bool = False


@dataclass(frozen=True)
class RewriteResult:
    original: str
    strategy: RewriteStrategy = RewriteStrategy.PASS_THROUGH
    reason: str = ""
    rewritten_query: str = ""              # 规范化后的主问题（pass_through 时等于 original）
    expanded_terms: tuple[str, ...] = ()   # query_expansion: 同义词 / 近义词 / 术语
    search_keywords: tuple[str, ...] = ()  # query_expansion: 可直接传给 search_document 的词组
    sub_queries: tuple[SubQuery, ...] = () # multi_query / decomposition 的子查询列表
    tokens: int = 0
    latency: float = 0.0
    bypassed: bool = False                 # True = 命中快速通道，未调用 LLM
    fallback: bool = False                 # True = 异常 / 解析失败降级透传
    error: str = ""

    @property
    def effective_query(self) -> str:
        return self.rewritten_query or self.original

    @property
    def changed(self) -> bool:
        return self.strategy != RewriteStrategy.PASS_THROUGH

    def to_event_payload(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "strategy": self.strategy.value,
            "label": STRATEGY_LABELS[self.strategy],
            "reason": self.reason,
            "rewritten_query": self.effective_query,
            "expanded_terms": list(self.expanded_terms),
            "search_keywords": list(self.search_keywords),
            "sub_queries": [
                {
                    "id": sq.id,
                    "query": sq.query,
                    "angle": sq.angle,
                    "purpose": sq.purpose,
                    "depends_on": list(sq.depends_on),
                    "target_source": sq.target_source,
                }
                for sq in self.sub_queries
            ],
            "tokens": self.tokens,
            "latency": self.latency,
            "bypassed": self.bypassed,
            "fallback": self.fallback,
            "error": self.error,
        }
```

#### 4.1.2 上下文采集：`build_context()`

```python
def build_context(
    messages: list[dict[str, Any]],
    *,
    session_key: str | None = None,
    write_mode: bool | None = None,
    history_turns: int = QUERY_REWRITE_HISTORY_TURNS,
    max_msg_chars: int = 240,
) -> RewriteContext:
```

实现要求：
1. **历史抽取**：
   - 遍历 `messages`，仅保留 `role == "user"` 以及 `role == "assistant" and not msg.get("tool_calls")` 的纯文本条目；
   - 跳过 `role == "system"`、`role == "tool"`、以及带 `tool_calls` 的中间步骤；
   - 取最后 `max(0, history_turns * 2)` 条，每条文本压平连续换行后截断至 `max_msg_chars` 字符（超出加 `"…"`）。
2. **数据源元数据抽取**：
   - 先确定 `key = session_key or data_tools.current_key()`；若传入了 `session_key`，确保查询前上下文指向该 `key`（或统一向 `has_dataset(key)` / `document_digest(key)` / `has_documents(key)` 传 `key`）；
   - 当 `data_tools.has_dataset(key)` 为真时，读取 `data_tools.dataset_meta()`（取 `label`、`rows`）和 `data_tools.column_names()`；
   - 当 `data_tools.has_documents(key)` 为真时，读取 `data_tools.document_digest(key)`（仅含 `name`、`kind`、`chars`、`parts`，**严禁读取正文**）；
   - 当 `write_mode is None` 时调用 `files.allow_write()`。

#### 4.1.3 完整 Prompt 模板

```python
REWRITE_SYSTEM_PROMPT = """你是数据分析系统的前置查询改写器（Query Rewriter）。
你的唯一任务是：结合当前会话的上下文元数据与最近对话历史，分析用户的最新提问，选择最恰当的一种处理策略并输出改写结果。

## 可选策略（只能五选一，按以下优先级判断）
1. decomposition（查询分解）：
   - 适用：包含多个先后依赖步骤、对比归因（先找异常再查原因）、跨数据源协同（既要查表又要查文档，或分析后再画图/生成PPT）的复杂任务。
   - 产出：2~5 个带依赖关系（depends_on）的有向无环子查询，且每条子查询必须自包含（消除指代）。
2. multi_query（多路查询）：
   - 适用：宽泛、开放、探索型问题，单一查询容易以偏概全（如"分析整体表现""总结亮点与风险""各区域情况怎么样"）。
   - 产出：2~4 条互补视角的并行子查询（depends_on 为空数组），并标明各自视角（angle）。
3. query_expansion（查询扩展）：
   - 适用：检索型或概念型问题（尤其是需要搜资料文档或按文本关键词筛选），用户用词单一、口语化或仅用缩写，字面匹配容易漏掉结果。
   - 产出：规范化后的 rewritten_query、3~6 个高相关同义词/业务术语/中英等价词（expanded_terms），以及 1~4 组可直接用于空格 OR 关键词检索的词组（search_keywords）。
4. basic_rewrite（基础改写）：
   - 适用：多轮追问中的指代/省略补全（如"北京呢？""把它画成折线图"）、口语化或错别字纠正、将模糊表述对齐到上下文中的真实表列名或文件名。
   - 产出：单条完整、自包含的 rewritten_query。
5. pass_through（直接透传）：
   - 适用：单步清晰、无歧义、未依赖省略信息、且无需术语扩展或多路拆解的问题。
   - 产出：无需修改，rewritten_query 填原问题或留空。

## 硬性约束
1. 必须仅输出一个合法的 JSON 对象，禁止输出 Markdown 围栏以外的任何说明文字。
2. 严禁编造元数据：只能引用上下文中真实列出的字段名与资料文件名；不确定时保留用户原始概念词。
3. 只要选择了非 pass_through 策略，必须在 rewritten_query 与所有 sub_queries[].query 中一并完成指代消解与实体对齐。
4. 保持克制：不要把简单明确的 SQL 统计问题强行拆成 multi_query 或 decomposition。
5. 输出 JSON 结构固定如下：
{
  "strategy": "pass_through | basic_rewrite | query_expansion | multi_query | decomposition",
  "reason": "不超过30字的决策理由",
  "rewritten_query": "改写后的完整清晰问题",
  "expanded_terms": ["同义词1", "术语2"],
  "search_keywords": ["关键词A 关键词B", "同义词C 术语D"],
  "sub_queries": [
    {
      "id": "Q1",
      "query": "自包含的子查询内容",
      "angle": "multi_query 时填视角标签，否则空串",
      "purpose": "decomposition 时填步骤产出目标，否则空串",
      "depends_on": [],
      "target_source": "table | document | mixed | workspace | auto"
    }
  ]
}"""
```

`build_rewrite_user_prompt(question: str, ctx: RewriteContext) -> str` 组装格式：
```text
## 当前可用数据源
- 数据表：`{table_label}`（共 {table_rows} 行，字段：{columns 用、连接}）  # 若无表则写「无数据表」
- 资料文件：{listing 或「无资料文件」}
- 本地文件权限：{"完全工作区（可读写/打开本机文件）" if ctx.write_mode else "只读"}

## 最近对话历史
{若无历史写「（首轮提问，无历史）」，否则按 `U: ...` / `A: ...` 列出}

## 用户最新提问
{question}
```

#### 4.1.4 容错解析、策略退化与 DAG 环检测：`parse_rewrite_response()`

```python
def parse_rewrite_response(
    raw: str,
    original: str,
    *,
    tokens: int = 0,
    latency: float = 0.0,
) -> RewriteResult:
```

具体处理管线：
1. **提取 JSON**：
   - 去除首尾空白；若含 ```` ``` ```` 围栏则剥离；
   - 若 `json.loads` 失败，尝试截取首个 `{` 到最后一个 `}` 之间的片段再次 `json.loads`；
   - 仍失败或结果不是 `dict` -> 返回 `RewriteResult(original=original, strategy=RewriteStrategy.PASS_THROUGH, rewritten_query=original, tokens=tokens, latency=latency, fallback=True, error="改写结果 JSON 解析失败")`。
2. **枚举校验**：
   - 读取 `strategy` 字段（小写去空白），不在 `RewriteStrategy` 值域内则降级为 `PASS_THROUGH` 且 `fallback=True`。
3. **公共字段清洗**：
   - `reason = str(data.get("reason") or "").strip()[:80]`
   - `rewritten = str(data.get("rewritten_query") or "").strip() or original`
4. **分策略裁剪与降级归一化**：
   - `PASS_THROUGH`：强制清空 `expanded_terms` / `search_keywords` / `sub_queries`，`rewritten_query = original`。
   - `BASIC_REWRITE`：若 `rewritten` 与 `original` 去除空白后完全一致，自动归一化为 `PASS_THROUGH`（保留 `reason`）。
   - `QUERY_EXPANSION`：
     - `expanded_terms`：去重保序，过滤空串、与原文完全相同的词、单字符噪声，裁剪至 `QUERY_REWRITE_MAX_EXPANSIONS`（6）；
     - `search_keywords`：去重保序，过滤空串，裁剪至 4 组；
     - 若 `expanded_terms` 与 `search_keywords` 均为空：若 `rewritten != original` 退化为 `BASIC_REWRITE`，否则退化为 `PASS_THROUGH`。
   - `MULTI_QUERY`：
     - 解析 `sub_queries`，过滤空 `query` 与完全重复项，裁剪至 `QUERY_REWRITE_MAX_MULTI`（4）；
     - 规范化 `id` 为 `Q1..Qn`，强制清空 `depends_on`；规范化 `target_source`（不在 `{"table","document","mixed","workspace","auto"}` 则回落 `"auto"`）；
     - 若有效条数 `< 2`：若仅 1 条有效且不同于原文则退化为 `BASIC_REWRITE`，否则 `PASS_THROUGH`。
   - `DECOMPOSITION`：
     - 解析 `sub_queries`，过滤空 `query` 与重复项，裁剪至 `QUERY_REWRITE_MAX_DECOMPOSE`（5）；
     - 先记录旧 ID 到新 ID（`Q1..Qn`）的映射；过滤指向不存在 ID 的 `depends_on` 及自环引用；
     - 执行 **Kahn 拓扑排序**：
       - 若无环，按拓扑顺序重排（必要时重新映射 `Q1..Qn` 或保持原拓扑顺序）；
       - 若存在环（拓扑输出条数 < 总条数），丢弃成环节点的非法环边（退化为按原序依赖上一条或清空环边），保证输出绝对无环；
     - 若有效条数 `< 2`：退化为 `BASIC_REWRITE` 或 `PASS_THROUGH`。

#### 4.1.5 下游 Prompt 封装：`format_for_agent(result: RewriteResult) -> str`

设计原则：**始终保留 `result.original`**（兜底防止改写丢失原意），并输出紧凑、清晰、可执行的工具导向提示。

- `PASS_THROUGH`：直接返回 `result.original`。
- `BASIC_REWRITE`：
  ```text
  【用户原始问题】
  {result.original}

  【查询改写（基础改写）】
  完整意图：{result.effective_query}
  请直接按上述完整意图调用工具并回答用户，无需复述改写过程。
  ```
- `QUERY_EXPANSION`：
  ```text
  【用户原始问题】
  {result.original}

  【查询改写（查询扩展）】
  规范问题：{result.effective_query}
  同义/扩展术语：{", ".join(result.expanded_terms) or "无"}
  建议检索词组合（适用于 search_document / 文本筛选）：
  - {kw1}
  - {kw2}
  请优先结合上述扩展术语与检索词组合查找资料或筛选数据；若精确词未命中可尝试同义词，最终综合给出准确回答。
  ```
- `MULTI_QUERY`：
  ```text
  【用户原始问题】
  {result.original}

  【查询改写（多路查询）】
  总目标：{result.effective_query}
  请从以下互补视角分别查询/检索（各路无依赖关系，可同一步并行调用多个工具），最后汇总去重给出全面结论：
  1. [Q1 | 视角:{angle} | 数据源:{target_source}] {query}
  2. [Q2 | 视角:{angle} | 数据源:{target_source}] {query}
  ```
- `DECOMPOSITION`：
  ```text
  【用户原始问题】
  {result.original}

  【查询改写（查询分解）】
  总目标：{result.effective_query}
  请按以下依赖顺序逐步求解（无前置依赖的步骤可并行；有前置依赖的步骤须基于前序工具返回结果再执行），最终汇总完整回答：
  1. [Q1 | 依赖:无 | 数据源:{target_source}] {query}（目标：{purpose}）
  2. [Q2 | 依赖:Q1 | 数据源:{target_source}] {query}（目标：{purpose}）
  ```

#### 4.1.6 `QueryRewriter` 类与签名探测

```python
class QueryRewriter:
    def __init__(
        self,
        *,
        llm: LLMCallable | None = None,
        enabled: bool = True,
        timeout: int = QUERY_REWRITE_TIMEOUT,
        max_tokens: int = QUERY_REWRITE_MAX_TOKENS,
        history_turns: int = QUERY_REWRITE_HISTORY_TURNS,
    ) -> None:
        self.llm = llm or chat
        self.enabled = enabled
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.history_turns = history_turns

    def rewrite(self, question: str, context: RewriteContext | None = None) -> RewriteResult:
        ...
```

关键细节：
- 使用 `inspect.signature(self.llm).parameters` 探测参数支持情况（类似 `app/agent.py` 的 `_takes_on_delta`）：仅当 `self.llm` 声明了对应参数或 `**kwargs` 时，才传入 `temperature=0.0`、`max_tokens=self.max_tokens`、`timeout=self.timeout`、`retries=0`（改写层不重试 2 次以免首屏卡顿，失败直接快速降级）。这样单元测试中只需写极简的 `lambda messages, tools=None: LLMResponse(...)` 即可无缝接入。
- `rewrite()` 内部用 `try ... except Exception as exc:` 兜底捕获 `LLMError`、`TimeoutError`、`OSError` 及任意未预期异常，记录错误摘要并返回 `RewriteResult(original=question, strategy=RewriteStrategy.PASS_THROUGH, rewritten_query=question, latency=..., fallback=True, error=str(exc)[:160])`。

---

### 4.2 修改 `app/llm.py`

在 `chat()` 中增加可选参数 `max_tokens: int | None = None`：

```python
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
```

在构造 `payload` 时：
```python
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
```
默认 `None` 时 `payload` 键集合与现有代码完全一致，不会对现有流式和协议测试产生任何影响。

---

### 4.3 修改 `app/config.py`

在 `MAX_STEPS = 100` 下方新增配置项：

```python
def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


# 前置查询改写（默认开启；可通过环境变量 QUERY_REWRITE_ENABLED=0 关闭）
QUERY_REWRITE_ENABLED = _env_bool("QUERY_REWRITE_ENABLED", True)
QUERY_REWRITE_TIMEOUT = 15
QUERY_REWRITE_MAX_TOKENS = 600
QUERY_REWRITE_HISTORY_TURNS = 4
QUERY_REWRITE_MAX_EXPANSIONS = 6
QUERY_REWRITE_MAX_MULTI = 4
QUERY_REWRITE_MAX_DECOMPOSE = 5
```

---

### 4.4 修改 `app/agent.py`

1. **`Agent.__init__`** 新增 keyword-only 参数（默认 `None`，保持 100% 向后兼容）：
   ```python
   rewriter: QueryRewriter | None = None,
   ```
   赋值 `self.rewriter = rewriter`。

2. **`Agent.ask(self, question: str) -> str`** 改造顺序：
   ```python
   if self.session_key:
       data_tools.use_dataset(self.session_key)

   spent_before = self.session.tokens

   packaged_question = question
   if self.rewriter is not None and self.rewriter.enabled:
       ctx = build_context(
           self.session.messages,
           session_key=self.session_key,
           history_turns=self.rewriter.history_turns,
       )
       rw_result = self.rewriter.rewrite(question, ctx)
       if rw_result.tokens:
           self.session.tokens += rw_result.tokens
       self._emit("rewrite", **rw_result.to_event_payload())
       packaged_question = format_for_agent(rw_result)

   self.session.add_user(question)
   if not self.session.messages or self.session.messages[0].get("role") != "system":
       self.session.messages.insert(
           0, {"role": "system", "content": self.system_prompt}
       )
   user_idx = len(self.session.messages) - 1
   ```
   在 `for step in range(1, self.max_steps + 1):` 循环内调 `self.llm` 前构造局部消息列表：
   ```python
   request_messages = self.session.messages
   if packaged_question != question:
       request_messages = list(self.session.messages)
       request_messages[user_idx] = {"role": "user", "content": packaged_question}

   response: LLMResponse = self.llm(
       request_messages, self.registry.specs(), **streaming
   )
   ```
   > **注意**：`spent_before` 必须在改写累加 `self.session.tokens` **之前**记录，这样本轮 `token_budget` 才能准确把前置改写消耗算入当轮预算。

3. **`make_system_prompt()`** 补充规则：
   - 在 `rules` 列表中增加：
     ```python
     "- 当用户消息附带「查询改写」提示（扩展词、多路视角或分解步骤）时，将其作为执行参考，直接推进工具调用与回答，不要复述改写模板；若某个扩展词或子步骤在当前数据/资料中无结果，如实说明即可。"
     ```

---

### 4.5 修改 `app/ui_render.py`

更新模块顶部事件说明注释（补充 `rewrite` 事件），并在 `build_item(kind: str, payload: dict) -> TraceItem | None` 中新增 `kind == "rewrite"` 分支：

- 若 `payload.get("bypassed")` 为 `True` 且 `payload.get("strategy") == "pass_through"`：返回 `None`（控制短语或关闭开关时不在「思考过程」刷屏）。
- 若 `payload.get("fallback")` 为 `True`：
  - `title = "查询改写 · 已降级透传"`
  - `body = f"改写未生效（{payload.get('error') or '已回退原问题'}），按原始问题继续。"`
  - `code = False`
- 若为正常 `pass_through`：
  - `title = f"查询改写 · 直接透传（{tokens} tokens，{latency:.1f}s）"`
  - `body = payload.get("reason") or "问题表述清晰，直接执行。"`
  - `code = False`
- 若为其余 4 种改写策略：
  - `title = f"查询改写 · {label}（{tokens} tokens，{latency:.1f}s）"`
  - `body` 用简洁 Markdown 分段组织：
    - 若有 `reason`：`策略理由：{reason}`
    - 若有 `rewritten_query`：`改写问题：{rewritten_query}`
    - 若有 `expanded_terms`：`扩展术语：` + 反引号包裹的词列表
    - 若有 `search_keywords`：`建议检索词：` + 反引号包裹的词组列表
    - 若有 `sub_queries`：按编号列出（`multi_query` 标出 `angle` 与 `target_source`；`decomposition` 标出 `depends_on`、`target_source` 与 `purpose`）
  - `code = False`

---

### 4.6 修改 `app/ui_streamlit.py`

1. **`_init_state()`** 新增默认状态：
   ```python
   if "query_rewrite" not in ss:
       ss.query_rewrite = config.QUERY_REWRITE_ENABLED
   ```
2. **新增统一工厂函数 `_make_agent(session: Session, prompt: str) -> Agent`**：
   ```python
   def _make_agent(session: Session, prompt: str) -> Agent:
       ss = st.session_state
       rewriter = QueryRewriter(
           enabled=bool(ss.get("query_rewrite", config.QUERY_REWRITE_ENABLED))
       )
       return Agent(session, system_prompt=prompt, session_key=ss.sid, rewriter=rewriter)
   ```
   将 `_reset_conversation()`、`_refresh_prompt()`、`_restore_chat()` 中原来的 `Agent(...)` 构造全部替换为 `_make_agent(...)`。
3. **侧边栏控制（`_render_sidebar()` 的「状态」区）**：
   在 `st.subheader("状态")` 与 `stats_slot` 附近增加开关（使用 `st.checkbox`，避免 `segmented_control` 在 `AppTest` 下的中文解析 bug）：
   ```python
   st.checkbox(
       "启用查询改写",
       key="query_rewrite",
       help="在发给主模型前自动进行指代补全、同义词扩展、多路改写或任务分解。",
   )
   ```
4. **运行前同步开关状态（`_run_turn()` 内调用 `ss.agent.ask(question)` 前，或 `_refresh_prompt()` 中）**：
   使用 `getattr` 防御性检查（兼容测试替身 `_ScriptedAgent`）：
   ```python
   rewriter = getattr(ss.agent, "rewriter", None)
   if rewriter is not None:
       rewriter.enabled = bool(ss.get("query_rewrite", config.QUERY_REWRITE_ENABLED))
   ```

---

### 4.7 可选二期增强（不阻塞一期交付）

1. **独立改写模型配置**：在 `data/model.json` 中增加可选 `rewrite_profile` 字段（如主循环用强推理模型、前置改写用轻量快速模型），通过 `QueryRewriter` 构造参数注入专属 `ModelConfig`（当前 `app/llm.py::chat` 也可增加可选 `cfg: config.ModelConfig | None = None`）。
2. **文档多路召回 RRF 融合**：在 `app/tools/doc_tools.py` 中新增 `search_document_multi(keywords: list[str], name: str = "")` 或为多关键词路做 Reciprocal Rank Fusion（`score = sum(1 / (60 + rank))`）跨路去重重排，进一步放大 `query_expansion` 与 `multi_query` 的文档检索收益。

---

## 5. 关键边界条件与防坑清单

1. **绝不破坏现有 456 项回归测试**：
   - `tests/test_week2.py` 明确断言 `roles == ["system", "user", "assistant", "tool", ...]`、`len(llm.seen) == 3`、`events == ["step", "tool_call", "observation", "step", "final"]`；
   - 因此 `Agent.__init__` 的 `rewriter` 参数默认必须为 `None`，未传入时零行为变化。
2. **UI 测试替身兼容性**：
   - `tests/test_week4_ui.py` 中的 `_ScriptedAgent` 和 `_SlowAgent` 没有 `rewriter` 属性；
   - `ui_streamlit.py` 同步开关状态时必须用 `getattr(ss.agent, "rewriter", None)` 判断，禁止直接访问 `ss.agent.rewriter`。
3. **多标签页并发隔离（ContextVar 时序）**：
   - `Agent.ask()` 中 `data_tools.use_dataset(self.session_key)` 必须在 `build_context()` **之前**执行，确保多标签页并发跑任务时改写器采集到的表字段与文档目录属于当前会话，且 `build_context` 不会在未绑定数据的线程里报错。
4. **协议兼容（严禁非标字段与硬编码 `response_format`）**：
   - `app/llm.py` 已记录部分 OpenAI 兼容网关对非标准消息字段会直接返回 400；
   - 不要向 `session.messages` 塞自定义元信息字段，也不要在改写请求里强塞 `response_format={"type": "json_object"}`（不少国产/本地网关不支持该字段），统一靠 System Prompt + 容错 JSON 提取解决。
5. **防止改写丢失关键原文与幻觉扩词**：
   - `format_for_agent()` 必须始终保留 `【用户原始问题】`，保证主模型可以核对原意；
   - System Prompt 禁止编造上下文未给出的字段名与文件名。
6. **UI 空状态测试文案禁区**：
   - `tests/test_week4_ui.py::TestEmptyState.test_no_how_it_works_text` 会检查侧边栏和主区域文案不得出现 `"会变成可查询的表"`、`"可检索的资料"`、`"还没有文件"`、`"sales.csv"`；新增 UI 控件标签与 `help` 文案不得包含这些字串。

---

## 6. 测试方案与验收标准

### 6.1 新增测试文件：`tests/test_query_rewrite.py`

全部使用离线伪造的 `llm` 回调（不发网络请求、不依赖 API Key），覆盖以下 7 个测试类：

1. **`TestFastBypass`**
   - `enabled=False`、空串、空白、`"继续"`、`"好的"` -> 返回 `PASS_THROUGH` 且 `bypassed=True`，`llm` 调用次数为 0；
   - `write_mode=True` 时 `"打开微信"` -> `bypassed=True`；但 `"打开微信并统计上月营收"` 不短路；
   - 短追问 `"北京呢？"`、`"为什么？"`、`"画个图"` **不短路**，必须调用 `llm`。
2. **`TestBuildContext`**
   - 过滤 `system`、`tool` 和带 `tool_calls` 的 `assistant` 消息，只保留干净对话；
   - 验证 `history_turns` 轮数上限与单条 `max_msg_chars` 截断；
   - 验证能正确采集 `data_tools` 的表名/列名/行数和 `document_digest` 元数据，且绝不泄露文档正文。
3. **`TestStrategiesAndValidation`**
   - `pass_through`：原样透传；
   - `basic_rewrite`：正常补全与对齐；若改写后与原文一致则自动归一化为 `pass_through`；
   - `query_expansion`：去重、裁剪到上限 6、单字噪声过滤、保留 `search_keywords`；若扩展结果全空则退化；
   - `multi_query`：2~4 条裁剪、自动重编 `Q1..Qn`、强制清空 `depends_on`；少于 2 条时降级；
   - `decomposition`：2~5 条裁剪、过滤不存在依赖与自环、**DAG 环检测与破环**（如构造 `Q1 -> Q2 -> Q1` 环，验证不死循环且输出合法无环序列）。
4. **`TestParserAndFallback`**
   - 支持 ```` ```json ... ``` ```` 代码块包裹与前后带说明文字的 JSON；
   - 非法 JSON、空串、未知 `strategy`、抛出 `LLMError`、`TimeoutError` 时，全部静默返回 `RewriteResult(strategy=PASS_THROUGH, fallback=True)`，不抛异常。
5. **`TestFormatForAgent`**
   - 验证 5 种策略的格式化输出：`pass_through` 原文无包装；其余 4 种均包含 `original` 及对应策略的结构化指令。
6. **`TestAgentIntegration`**
   - `rewriter=None`：行为与改造前完全一致，不发 `rewrite` 事件；
   - `rewriter` 启用（覆盖 `basic_rewrite`、`query_expansion`、`multi_query`、`decomposition`）：
     - 断言主模型每次 `llm()` 收到的 `user` 消息是 `format_for_agent()` 包装后的文本（跨多轮工具调用均一致）；
     - 断言结束后 `session.messages` 中存的 `user` 消息仍然是原始 `question`；
     - 断言事件序列为 `["rewrite", "step", "tool_call", "observation", "step", "final"]`；
     - 断言改写 token 累加进 `session.tokens` 并受 `token_budget` 约束，且 `session.steps` 不受改写影响。
7. **`TestUIRenderRewrite`**
   - `ui_render.build_item("rewrite", ...)` 对 `bypassed=True` 且 `pass_through` 返回 `None`；
   - 对 4 种改写策略、`pass_through` 与 `fallback=True` 均返回正确标题、Markdown 正文与 `code=False`。

### 6.2 验收命令与通过标准

```bash
# 1. 运行新增查询改写单测
python -m unittest tests/test_query_rewrite.py -v

# 2. 运行全量回归测试（含 AppTest 界面测试）
python -m unittest discover -s tests -v
```

- **通过标准**：原有 **456 项回归测试全部通过** + 新增查询改写单测全部通过，无异常、无退化跳过。

---

## 7. 分步实施顺序（执行 Agent 路线图）

| 步骤 | 改动文件 | 核心任务 | 单步验收 |
|---|---|---|---|
| **Step 1** | `app/config.py`<br>`app/llm.py` | 补齐 `_env_bool` 与 `QUERY_REWRITE_*` 配置；`llm.chat()` 新增可选 `max_tokens: int \| None = None` | `python -m unittest tests/test_stream.py -v` 全绿 |
| **Step 2** | `app/query_rewrite.py`（新建） | 实现枚举、数据结构、`build_context`、`_fast_bypass`、Prompt 模板、`parse_rewrite_response`（含 Kahn 环检测）、`format_for_agent`、`QueryRewriter` | 导入无循环依赖；新建 `tests/test_query_rewrite.py` 前 5 个测试类全绿 |
| **Step 3** | `app/agent.py` | `Agent.__init__` 增加 `rewriter=None`；`Agent.ask()` 接入前置改写、token 记账、`rewrite` 事件与 `request_messages` 局部替换；`make_system_prompt()` 补充规则 | `tests/test_week2.py` 全绿 + `TestAgentIntegration` 全绿 |
| **Step 4** | `app/ui_render.py` | `build_item()` 新增 `"rewrite"` 事件渲染分支 | `tests/test_week4.py` 全绿 + `TestUIRenderRewrite` 全绿 |
| **Step 5** | `app/ui_streamlit.py` | 新增 `query_rewrite` 状态、`_make_agent()` 统一装配、侧边栏「启用查询改写」开关与 `getattr` 防御性同步 | `tests/test_week4_ui.py`、`tests/test_chat_ui.py`、`tests/test_model_panel.py` 全绿 |
| **Step 6** | `tests/test_query_rewrite.py`<br>`README.md`（可选补说明） | 补齐全部测试用例并执行全量回归 | `python -m unittest discover -s tests -v` 全部 OK（456 + 新增用例） |
