"""Agent 主体：循环本身没变，还是第 1 周那五步。

第 2 周新增的三样东西：
  1. 工具来自注册表（不再写死一个 TOOLS 常量）
  2. Session 让多轮对话成为可能
  3. on_event 回调——UI 想展示中间过程时，不需要改这个类
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from . import config, images
from .config import IMAGE_MAX_PER_TURN, MAX_STEPS, TABLE_NAME, TOKEN_BUDGET
from .llm import LLMCallable, LLMError, LLMResponse, chat, normalize_assistant
from .query_rewrite import QueryRewriter, build_context, format_for_agent
from .session import Session
from .tools import ToolRegistry, data_tools, registry as default_registry

# 兜底 prompt：只在既没有表也没有资料时用到。
# 这里**不能**描述任何具体数据（更别说"电商订单表 sales"），否则模型会照着
# 一个不存在的表去猜列名，然后连续几步都在报错。
SYSTEM_PROMPT = """你是一个数据分析助手。当前没有可用的数据表，也没有资料文件。

硬性规则：
1. 不要假设任何字段名，不要编造任何数字。
2. 用户还没有提供数据时，直接说明需要先上传文件，不要凭常识作答。
3. 用中文回答。"""

Event = tuple[str, dict[str, Any]]


def _takes_on_delta(callable_: Callable[..., Any]) -> bool:
    try:
        parameters = inspect.signature(callable_).parameters
    except (TypeError, ValueError):
        return False
    if "on_delta" in parameters:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())


_VISION_ERROR_HINTS = (
    "image_url",
    "image",
    "vision",
    "multimodal",
    "expected a string",
    "unknown variant",
    "invalid type",
    "content",
    "media",
)


def _is_multimodal_rejection(exc: LLMError) -> bool:
    """判断异常是否因为当前模型不支持 OpenAI image_url 多模态消息。"""
    if exc.status in (400, 415, 422):
        return True
    msg = f"{exc} {exc.detail}".lower()
    return any(hint in msg for hint in _VISION_ERROR_HINTS)


def _build_request_messages(
    session_messages: list[dict[str, Any]],
    *,
    user_idx: int,
    question: str,
    packaged_question: str,
    step_attachments: dict[int, list[images.DocImage]],
) -> list[dict[str, Any]]:
    """构造本步发给模型的浅拷贝消息列表（绝不改动 session.messages 本身）。"""
    if packaged_question == question and not step_attachments:
        return session_messages

    request_messages = list(session_messages)
    if packaged_question != question:
        request_messages[user_idx] = {"role": "user", "content": packaged_question}

    for pos in sorted(step_attachments, reverse=True):
        imgs = step_attachments[pos]
        if not imgs:
            continue
        names = "、".join(
            f"{img.name}（{img.location}）" if img.location else img.name
            for img in imgs
        )
        parts: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": f"以下是读取文件时附带的 {len(imgs)} 张图片（{names}），请结合上下文内容一并分析：",
            }
        ]
        parts.extend(img.to_content_part() for img in imgs)
        request_messages.insert(pos, {"role": "user", "content": parts})

    return request_messages


def make_system_prompt(
    columns: list[str] | None = None,
    rows: int | None = None,
    label: str = "上传的数据",
    files: list[dict[str, Any]] | None = None,
    write_mode: bool = False,
    web_search: bool = False,
) -> str:
    """按当前数据表、资料文件、文件权限模式和联网开关生成 system prompt。
    """
    columns = columns or []
    files = files or []
    if not columns and not files and not write_mode and not web_search:
        return SYSTEM_PROMPT

    parts = ["你是一个数据分析助手。"]

    if write_mode:
        parts.append(
            "## 本地文件权限\n当前是「完全工作区」模式：可以读写本机任意路径，"
            "也可以打开文件和启动程序；写入**不做备份**，覆盖即永久丢失。"
        )

    if web_search:
        parts.append(
            "## 联网搜索能力（MCP）\n已挂载 Parallel 联网工具（`web_search` / `web_fetch`），"
            "可用于检索最新公开资料、行业数据与抓取指定网页正文。"
        )

    if columns:
        head = f"## 数据表\n表 `{TABLE_NAME}`，从 `{label}` 载入"
        if rows:
            head += f"，共 {rows:,} 行"
        head += f"。字段：{'、'.join(columns)}"
        parts.append(head)

    if files:
        def _file_line(f: dict[str, Any]) -> str:
            img_part = f"，含 {f['images']} 张图片" if f.get("images") else ""
            return (
                f"- `{f['name']}`（{f.get('kind', '?')}，{f.get('chars', 0):,} 字，"
                f"{f.get('parts', 1)} 段{img_part}）"
            )

        listing = "\n".join(_file_line(f) for f in files)
        parts.append(
            f"## 资料文件\n用户上传了 {len(files)} 份资料（含已上传文档或图片）。"
            f"正文与图片**不在**上下文里，要用 `read_document(name, part=1)` 或 `search_document` 去读"
            f"（**不要用 `read_file`**，后者只读本机磁盘路径）：\n{listing}"
        )

    strategy = ["## 工具使用策略"]
    if columns:
        strategy += [
            "- 不确定列名或类型 -> inspect_table（猜列名是最常见的错误来源）",
            "- 聚合、筛选、排名、占比 -> run_sql（DuckDB 语法，一次只查一件事）",
            "- SQL 表达不出来的计算（自定义指标、标准化、文本清洗）-> run_python",
        ]
    strategy.append(
        "- 要出图 -> make_chart：有数据表时传 sql；没有数据表、或数字来自文档和上一步计算时，"
        "直接传 labels 和 values，不需要先弄一份 CSV"
    )
    strategy.append(
        "- 要展示层级、目录、分类、拆解思路 -> make_mindmap：把每条「中心→分支→叶子」"
        "的路径列出来，相同前缀会自动合并。数值的涨跌对比不要用它"
    )
    strategy.append(
        "- 要成篇的多页材料（汇报、方案、总结、答辩）-> make_pptx：每页给一个 layout，"
        "内容从表或资料里取。只有一张趋势图时用 make_chart，别为一张图开一份 PPT"
    )
    strategy.append(
        "- 要往**已有的** PPT 里加页、改字、删页 -> 先 read_file 读出大纲（页码 + 块号），"
        "再 edit_pptx。不要重新 make_pptx 生成整份——重建会丢掉用户后来手动改过的东西"
    )
    strategy.append(
        "- 读本机磁盘路径上的 PDF / Word / PPT / Excel / 图片用 read_file"
        "（仅限本机目录，如桌面或 ~"
        + ("；上面「资料文件」清单里的已上传文件要用 read_document" if files else "")
        + "）。不要用 run_python 去 open 或解压它们：沙箱里没有解析库，zipfile 也不在白名单里"
    )
    if files:
        strategy += [
            "- 不确定有哪些已上传资料 -> list_documents，不要猜文件名",
            "- 已经知道要找什么（某指标、某日期、某条报错）-> search_document 定位，"
            "支持精确关键词与自然语言语义检索，默认搜全部资料；拿到段号再 read_document 读那一段",
            "- 要通读、总结或查看已上传图片 -> read_document(name, part=1) 按段读，别一次读完整份",
        ]
        if any(
            f.get("images") or f.get("kind") in {"png", "jpg", "jpeg", "webp", "gif", "bmp"}
            for f in files
        ):
            strategy.append(
                "- 上述「资料文件」清单中的文档或图片包含视觉图像时，"
                "请直接调用 read_document(name, part=1) 查看图片内容（不要调用 read_file）"
            )
    if write_mode:
        strategy += [
            "- 不知道目录里有什么 -> list_files，别凭记忆拼路径（支持 ~ 和 桌面）",
            "- 要改一个已存在的文件 -> 先 read_file 读原内容，再决定整篇重写还是追加",
            "- 新建或整体重写 -> write_text_file(mode=\"overwrite\")；只往后接 -> mode=\"append\"",
            "- 只想验证一段逻辑而不落盘 -> run_python 里直接算，不要顺手写文件",
            "- 用户说\"打开某个文件/目录\" -> open_file，路径照用户的说法给，不用先读内容",
            "- 用户说\"打开某个程序\"（微信、浏览器、记事本）-> 先 find_program 拿到路径，再 open_file",
        ]
    if web_search:
        strategy += [
            "- 需要查外部实时资讯、公开基准数据或行业资料 -> web_search；"
            "需要读取某个具体网页的完整正文时 -> web_fetch",
        ]
    strategy.append(
        "- 提问含\"今天/昨天/最近/本周/本月/上月/今年/最新\"等相对时间且需精确日期范围时 -> "
        "先调用 get_current_time 获取本机当前时间与参考区间，不要凭训练记忆猜测年份"
    )
    parts.append("\n".join(strategy))

    rules = ["## 硬性规则", "- 所有数字必须来自工具返回结果，绝不编造。"]
    if columns:
        rules.append("- 只使用上面列出的字段名，不要臆造字段。")
    if files:
        rules.append("- 引用文档内容要说明来自哪份文件；文档里没写的不要补充。")
        rules.append("- 文档说法和表里的数字不一致时，把两者都摆出来，不要自己挑一个。")
    if web_search:
        rules.append(
            "- 引用联网搜索或网页抓取内容时必须附上来源 URL，并与本地表/资料的数字明确区分。"
        )
    if write_mode:
        rules += [
            "- 只能改用户明确提到的文件。要动别的文件，先问，不要顺手整理。",
            "- 没有备份：覆盖前必须读过原内容，或者确认它就是空文件。",
            "- 写完要报告「改了哪个路径、从多少字节变成多少字节、写了什么」，不要只说已完成。",
            "- 路径尽量用绝对路径；写相对路径会被解析到用户主目录，容易建错地方。",
            "- 只在用户明确说\"打开\"时才 open_file。启动程序会抢走窗口焦点，打断用户手上的事。",
            "- open_file 之后只能说\"已请求系统打开\"，不要说\"已经打开并看到内容\"——你看不到那个窗口。",
        ]
    rules += [
        "- 当用户消息附带「查询改写」提示（扩展词、多路视角或分解步骤）时，"
        "将其作为执行参考，直接推进工具调用与回答，不要复述改写模板；"
        "若某个扩展词或子步骤在当前数据/资料中无结果，如实说明即可。",
        "- SQL 报错或结果为空时，先怀疑自己的写法，改写后重试，同一思路最多两次。",
        "- 不要在回答里贴大段 SQL，只说结论和口径。",
        "- 用中文回答，关键数字带单位和范围。",
    ]
    parts.append("\n".join(rules))

    return "\n\n".join(parts)


class Agent:
    def __init__(
        self,
        session: Session | None = None,
        *,
        llm: LLMCallable | None = None,
        registry: ToolRegistry | None = None,
        max_steps: int = MAX_STEPS,
        token_budget: int = TOKEN_BUDGET,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
        system_prompt: str | None = None,
        session_key: str | None = None,
        rewriter: QueryRewriter | None = None,
    ) -> None:
        self.session = session or Session()
        self.llm = llm or chat
        self.registry = registry or default_registry
        self.max_steps = max_steps
        self.token_budget = token_budget
        self.on_event = on_event
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        # 这个 Agent 属于哪份数据（界面上就是会话 id）。工具本身拿不到界面状态，
        # 所以在 ask() 的入口把它注入上下文，整轮循环里的工具调用都能读到；
        # 退出时还原。None = 不注入，用当前上下文的值（测试里就是这么用的）。
        self.session_key = session_key
        # 前置查询改写器。默认 None = 完全不启用：现有单测靠 ScriptedLLM 精确断言
        # 消息条数和事件序列，多一次改写调用就会全红。界面层负责装配它。
        self.rewriter = rewriter
        self._streams = _takes_on_delta(self.llm)

    def _emit(self, kind: str, **payload: Any) -> None:
        if self.on_event:
            self.on_event(kind, payload)

    def _on_delta(self, text: str) -> None:
        self._emit("delta", text=text)

    def update_system_prompt(self, text: str) -> None:
        """换一份 prompt，**保留已有对话**。

        上传新资料后必须调它：不改的话，模型看到的文件清单还是旧的，
        会坚持说"没有这份文件"。但直接清空对话又太粗暴——用户可能刚问到一半。

        system 消息只在第一轮插入一次（见 ask），所以光改 self.system_prompt 不够，
        已经插进 messages 的那条也得一起换掉。
        """
        self.system_prompt = text
        if self.session.messages and self.session.messages[0].get("role") == "system":
            self.session.messages[0]["content"] = text

    def _stop(self, message: str) -> str:
        """中止这一轮：把这句话当成回答，并且**写进历史**。

        写历史不是顺手：中止时 messages 末尾刚放进去的是 tool 消息，
        紧接着再插一条 user（用户说"继续"）就是非法序列，OpenAI 兼容接口会直接 400。
        补一条 assistant 把这一轮合上，下一句才发得出去。
        """
        self.session.add({"role": "assistant", "content": message})
        self._emit("final", text=message)
        return message

    def ask(self, question: str) -> str:
        """跑完一轮问答。

        除了循环本身，这里还负责把「这一轮属于哪份数据」注入当前上下文。
        不注入的后果是模块级全局变量那一套：A 标签页还在跑十几步，B 一 rerun
        就把当前 key 改了，A 后面几步的 run_sql / run_python 全都在 B 的数据上跑，
        而且不报错。tools/data_tools.py 里 _ACTIVE 的注释记了这一段。

        **只注入，不还原。** 曾经这里有个 try/finally 把 key 还回调用前的值，
        但那是多余的：界面每次刷新都是新线程，_init_state 第一件事就把 key 设成
        本会话的，所以"留在上下文里的值"活不过一次刷新。实测把还原改成空函数，
        449 条测试里只有两条专门测还原的会红，功能一条都没坏。
        少一层 token 传递，读代码的人就不用先理解 Token 是什么。

        装了 rewriter 之后，这里还多一段**前置改写**：进循环之前把原问题过一遍，
        换成本轮真正发给模型的增强提问。放在 ask() 而不是 UI 层，是因为 CLI 评测、
        单测、Streamlit 三个入口都能统一复用；而"改写结果不进 session.messages"
        这条边界也只有在这里才守得住。
        """
        if self.session_key:
            data_tools.use_dataset(self.session_key)

        # 预算按**这一问**算，不按整个会话算。session.tokens 是累计值（要留着看成本），
        # 拿它当尺子的话，聊到 80k 之后每一句都会在第一个工具回合后立刻被掐断——
        # 而且提示是"超出预算"，看着像模型不肯干活。
        #
        # 注意这行在**前置改写之前**：改写自己也要花 token，那部分同样属于这一问的成本，
        # 累加在 spent_before 之后才算得进本轮预算。
        spent_before = self.session.tokens

        packaged_question = question
        if self.rewriter is not None and self.rewriter.enabled:
            # 必须先 use_dataset 再采集上下文：多标签页并发时，晚一步就可能读到
            # 另一个标签页的表字段与文档目录。
            ctx = build_context(
                self.session.messages,
                session_key=self.session_key,
                history_turns=self.rewriter.history_turns,
                web_search=self.registry.get("web_search") is not None,
            )
            rw_result = self.rewriter.rewrite(question, ctx)
            if rw_result.tokens:
                self.session.tokens += rw_result.tokens
            # 改写不增加 session.steps：步数专指主 Agent 的 ReAct 步数，
            # 混进来的话"第 N 步"就再也对不上主循环里的 step 了。
            self._emit("rewrite", **rw_result.to_event_payload())
            packaged_question = format_for_agent(rw_result)

        # 历史里只存干净的原话。改写结果只活在本轮的 request_messages 里，
        # 否则多轮下来上下文会被模板越撑越大，聊天记录恢复界面也会露出内部 prompt。
        self.session.add_user(question)
        if not self.session.messages or self.session.messages[0].get("role") != "system":
            self.session.messages.insert(
                0, {"role": "system", "content": self.system_prompt}
            )
        user_idx = len(self.session.messages) - 1

        last_signature: str | None = None
        repeat = 0
        # 轮次局部的多模态图片附件表：{session.messages 插入位置索引: [DocImage, ...]}。
        # 只存进发给模型的 request_messages 浅拷贝，不写入持久化 session.messages。
        step_attachments: dict[int, list[images.DocImage]] = {}
        sent_image_ids: set[str] = set()
        vision_fallback = False

        for step in range(1, self.max_steps + 1):
            self.session.steps += 1
            streaming = {"on_delta": self._on_delta} if self._streams and self.on_event else {}

            active_attachments = (
                step_attachments
                if (config.MULTIMODAL_IMAGES_ENABLED and not vision_fallback)
                else {}
            )
            request_messages = _build_request_messages(
                self.session.messages,
                user_idx=user_idx,
                question=question,
                packaged_question=packaged_question,
                step_attachments=active_attachments,
            )

            try:
                response: LLMResponse = self.llm(
                    request_messages, self.registry.specs(), **streaming
                )
            except LLMError as exc:
                if active_attachments and _is_multimodal_rejection(exc):
                    vision_fallback = True
                    self._emit(
                        "warning",
                        message="当前模型不支持多模态图片输入，已自动降级为仅文本模式。",
                    )
                    fallback_messages = _build_request_messages(
                        self.session.messages,
                        user_idx=user_idx,
                        question=question,
                        packaged_question=packaged_question,
                        step_attachments={},
                    )
                    try:
                        response = self.llm(
                            fallback_messages, self.registry.specs(), **streaming
                        )
                    except LLMError as retry_exc:
                        self._emit("error", message=str(retry_exc))
                        return f"（模型调用失败：{retry_exc}）"
                else:
                    self._emit("error", message=str(exc))
                    return f"（模型调用失败：{exc}）"

            self.session.tokens += response.total_tokens
            self._emit(
                "step",
                step=step,
                tokens=response.total_tokens,
                latency=response.latency,
            )

            message = response.message

            # ── 出口：没有工具调用了 ──
            if not message.get("tool_calls"):
                answer = message.get("content") or ""
                self.session.add({"role": "assistant", "content": answer})
                self._emit("final", text=answer)
                return answer

            self.session.add(normalize_assistant(message))

            step_new_images: list[images.DocImage] = []
            for call in message["tool_calls"]:
                name = call["function"]["name"]
                raw_args = call["function"].get("arguments") or "{}"

                signature = f"{name}:{raw_args}"
                repeat = repeat + 1 if signature == last_signature else 1
                last_signature = signature
                if repeat >= 3:
                    self._emit("warning", message="检测到重复调用，已中止")
                    return self._stop("（Agent 在重复调用同一个工具，已强制中止。换个问法试试。）")

                self._emit("tool_call", name=name, args=raw_args)
                with images.collect_step_images() as call_images:
                    observation = self.registry.call(name, raw_args)
                self.session.tool_calls += 1

                obs_payload: dict[str, Any] = {"name": name, "text": observation}
                if call_images and config.MULTIMODAL_IMAGES_ENABLED and not vision_fallback:
                    accepted: list[images.DocImage] = []
                    for img in call_images:
                        if img.id in sent_image_ids:
                            continue
                        if len(sent_image_ids) >= IMAGE_MAX_PER_TURN:
                            break
                        if not img.to_data_url():
                            continue
                        sent_image_ids.add(img.id)
                        accepted.append(img)
                    if accepted:
                        step_new_images.extend(accepted)
                        obs_payload["images"] = [img.to_meta() for img in accepted]

                self._emit("observation", **obs_payload)

                self.session.add(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": observation,
                    }
                )

            if step_new_images:
                step_attachments[len(self.session.messages)] = step_new_images

            if self.session.tokens - spent_before > self.token_budget:
                self._emit("warning", message=f"token 预算耗尽（{self.session.tokens - spent_before}）")
                return self._stop(
                    "（这一问用掉的 token 超出预算，先停下。"
                    "这通常说明问题太宽泛，拆小一点再问会更准。）"
                )

        self._emit("warning", message=f"达到步数上限 {self.max_steps}")
        return self._stop(
            f"（这一轮已经跑了 {self.max_steps} 步还没收尾，先停在这里。"
            "前面查过的东西都还在，说「继续」可以接着做。）"
        )
