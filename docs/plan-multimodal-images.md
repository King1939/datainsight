# 多模态文档图片抽取与模型提交整改方案书

> 状态：已实施并验收（644 项 `unittest` 全量通过）。  
> 目标代码库：`e:\VScode\datainsight`（当前回归测试基线：585 项 `unittest` 全部通过）。  
> 运行环境实测：Python 3.13、`Pillow (PIL)`、`pypdf`、`python-docx`、`python-pptx`、`openpyxl` 均已就绪；对话接口为 OpenAI 兼容 `/chat/completions`（可能对接视觉模型或纯文本模型）。

---

## 1. 现状诊断与整改目标

### 1.1 现存问题定位

1. **文档解析层丢弃全部图片（`app/readers.py`）**
   - `_read_pdf()` 仅调用 `page.extract_text()`，完全忽略 `page.images`；扫描件或含图表/截图的 PDF 会丢失所有视觉信息。
   - `_read_docx()` 仅遍历 `document.paragraphs` 与 `document.tables`，忽略段落/表格/页眉页脚中嵌入的 `a:blip` / `v:imagedata` 图片。
   - `_read_pptx()` 仅读取 `has_text_frame` 与 `has_table`，忽略图片形状（`Picture`）及组合形状（`GroupShape`）内的图片；`app/tools/pptx_tools.py` 的 `describe_pptx()` 同样只输出文字大纲。
   - `_read_xlsx()` 未提取工作簿内嵌图片（`xl/media/*`）。
   - `KIND_BY_SUFFIX` 不包含独立图片后缀（`.png` / `.jpg` / `.jpeg` / `.webp` / `.gif` / `.bmp`），用户无法直接上传或用 `read_file` 读取单张图片。
2. **工具返回与模型协议层仅支持纯文本（`app/tools/doc_tools.py`、`app/tools/file_tools.py`、`app/agent.py`）**
   - `read_document` 和 `read_file` 仅返回 `str`，`Agent.ask()` 在 `request_messages` 中从未构造 OpenAI 多模态 `content` 数组（`[{"type": "text", ...}, {"type": "image_url", ...}]`）。
3. **储藏区缓存与多模型兼容空白（`app/vault.py`、`app/llm.py`）**
   - `vault.add()` 只保存 `text/<id>.txt`，重启后 `vault.open_document()` 不再读取原始文件，若不持久化图片资产会导致重启后图片丢失。
   - 当用户当前配置的是纯文本模型（如 `deepseek-chat`）时，若盲目发送 `image_url` 会收到 HTTP 400；必须具备自动降级回纯文本的能力，绝不中断主对话。

### 1.2 整改目标与硬约束

1. **全链路图片抽取与段号对齐**
   - 读取 `.pdf`、`.docx`、`.pptx`、`.xlsx`/`.xlsm` 以及独立图片文件（`.png`/`.jpg`/`.jpeg`/`.webp`/`.gif`/`.bmp`）时，提取其中存在的图片并记录所在位置（页码/节次）与对应 `Doc.chunks` 的 1-based 段号 `part`。
   - 在抽取文本对应位置插入轻量锚点标记（如 `[图片 1：第 2 页 · 800×600]`），使纯文本扫描件/纯图片文件不再被判成"无可读内容"，且模型能将图片与上下文文字对应起来。
2. **按 OpenAI Vision 规范提交图片对象**
   - `read_document` / `read_file` 读取某一段 `part` 时，将属于该段的图片通过上下文收集器附着到本步工具执行上下文；
   - `Agent.ask()` 在不污染 `session.messages`（保证 `role: "tool"` 永远为标准纯字符串、聊天落盘 JSON 不被 Base64 撑爆）的前提下，在发给模型的 `request_messages` 中于对应 `tool` 响应之后插入多模态 `role: "user"` 附件消息，携带标准 `{"type": "image_url", "image_url": {"url": "data:<mime>;base64,..."}}` 图片对象（同轮多步后续请求中持续可见，单轮总图片数受上限控制）。
3. **向后兼容与零破坏约束**
   - 保持 `ToolRegistry.call()` 永远返回 `str` 契约不变。
   - **无图片文档的输出逐字不变**，确保现有 585 项单测 100% 通过。
   - 若服务端返回不支持多模态（HTTP 400/415/422 拒绝 `image_url`），`Agent.ask()` 自动发出 `warning` 事件并平滑降级为纯文本重试，不崩溃。
   - Pillow（`PIL`）按项目规范做成**可选依赖 + 魔数头嗅探兜底**：有 Pillow 时自动转码 CMYK/TIFF/BMP 并按长边上限缩放压缩；无 Pillow 时仍支持原生 PNG/JPEG/GIF/WEBP 直接读取提交。

---

## 2. 总体架构与端到端数据流

```text
[源文件：PDF / DOCX / PPTX / XLSX / 图片]
        │
        ▼
  app/readers.py (+ app/images.py)
   ├─ 抽取文本 + 提取嵌入图片，按位置写入文本锚点 `[图片 N：...]`
   ├─ Pillow 规范化（转 RGB/RGBA、过滤 1x1 像素噪点、缩放长边 <= 1568px、转 PNG/JPEG）
   └─ split_chunks() 切块后，反查 `[图片 N` 落在哪个 1-based part，绑定 img.part
        │
        ├─ 储藏区 (vault.py)：落盘缓存 `data/vault/images/<file_hash>/manifest.json` + 图片文件
        │                     （兼容老版本无 manifest 的文件：open_document 时惰性补抽一次）
        └─ 临时上传 / 本地 read_file：随 DocImage 保存在内存或原路径按需编码
        │
        ▼
  模型调用 read_document(name, part) / read_file(path, part)
   ├─ 筛选属于本段的图片（part 匹配；上限 IMAGE_MAX_PER_CALL）
   ├─ images.attach_step_image(img)  --> 写入当前步骤 ContextVar 收集器
   └─ 返回 str 观察结果（若附带图片，末尾注明"本段附带 K 张图片，已提交给模型查看"）
        │
        ▼
  Agent.ask() [app/agent.py]
   ├─ session.messages 只存纯文本 tool 结果（干净、可持久化、兼容 query_rewrite）
   ├─ 若本步收集到图片（且 MULTIMODAL_IMAGES_ENABLED=True）：
   │    · 记录到轮次局部附件表 step_attachments[len(session.messages)] = images
   │    · 发射 `observation` 事件（附带 `images` 元信息供 UI 展示缩略图/计数）
   ├─ 组装 request_messages：在对应 step 的 tool 消息之后注入
   │    {"role": "user", "content": [{"type": "text", ...}, {"type": "image_url", ...}]}
   └─ 调用 self.llm(request_messages, ...)
        └─ 若遇 HTTP 400/415/422 多模态拒绝错误 -> 自动发射 warning 并退回纯文本重试
```

---

## 3. 详细模块设计与接口契约

### 3.1 `app/config.py` 新增配置

在 `load_env()` 之后新增多模态图片配置项（均支持 `.env` 覆盖）：

```python
MULTIMODAL_IMAGES_ENABLED = _env_bool("MULTIMODAL_IMAGES_ENABLED", True)
IMAGE_MAX_SIDE = _env_int("IMAGE_MAX_SIDE", 1568)          # 单边最大像素，超过等比缩放
IMAGE_MAX_BYTES = _env_int("IMAGE_MAX_BYTES", 2 * 1024 * 1024)  # 单张编码后体积上限（2 MB）
IMAGE_MAX_PER_DOC = _env_int("IMAGE_MAX_PER_DOC", 24)      # 单份文档最多抽取多少张图
IMAGE_MAX_PER_TURN = _env_int("IMAGE_MAX_PER_TURN", 6)     # 单轮对话最多附带多少张图给模型
```

### 3.2 新建核心模块：`app/images.py`

职责：图片数据结构、标准化转码、文件字节嗅探、以及工具到 Agent 的 `ContextVar` 收集器。

1. **数据结构 `DocImage`**：
   ```python
   @dataclass(frozen=True)
   class DocImage:
       id: str                 # 内容 sha1[:16]，用于去重与落盘文件名
       name: str               # 展示名，如 "p2_img1.png"
       mime_type: str          # "image/png" | "image/jpeg" | "image/webp" | "image/gif"
       width: int = 0
       height: int = 0
       part: int = 1           # 归属的 1-based 分段号
       location: str = ""      # 如 "第 2 页" / "文档正文" / "工作表插图"
       size: int = 0
       path: str = ""          # 落盘路径（vault 缓存时使用）
       data: bytes = field(default=b"", repr=False, compare=False)
   ```
   - `def read_bytes(self) -> bytes`：优先返回 `self.data`，为空且 `self.path` 存在时从磁盘读取。
   - `def to_data_url(self) -> str`：生成 `f"data:{self.mime_type};base64,{base64.b64encode(raw).decode('ascii')}"`。
   - `def to_content_part(self) -> dict[str, Any]`：返回标准 OpenAI Vision 结构 `{"type": "image_url", "image_url": {"url": self.to_data_url()}}`。
   - `def to_meta(self) -> dict[str, Any]`：返回不含二进制的轻量元数据（`id`, `name`, `mime_type`, `width`, `height`, `part`, `location`, `size`），用于事件发射与 `vault` 清单序列化。

2. **图片规范化 `normalize_image(raw: bytes, *, name: str = "", part: int = 1, location: str = "") -> DocImage | None`**：
   - 过滤空字节或明显像素级占位符（`max(width, height) <= 2` 的 1×1 透明分隔点）。
   - 若安装了 `PIL.Image`：
     - 打开字节流并处理 EXIF 方向（`ImageOps.exif_transpose`）；
     - 统一转为 `RGB` 或 `RGBA`（去除 CMYK / 调色板兼容性问题）；
     - 若 `max(w, h) > IMAGE_MAX_SIDE`，用 `Resampling.LANCZOS` 等比缩放；
     - 保留 PNG（含透明通道或原本是 PNG/图表且体积 <= `IMAGE_MAX_BYTES`）或转为 JPEG（quality=85）；若仍超 `IMAGE_MAX_BYTES`，逐步降质量/缩放至满足上限；
     - 无法解码的图片格式（如少数损坏图片或平台不支持的 EMF/WMF）安全返回 `None`，不崩溃。
   - 若未安装 `PIL`：按魔数头探测 PNG/JPEG/GIF/WEBP 宽高及 MIME，体积超限或不支持格式返回 `None`。

3. **步骤级图片收集器（线程/协程安全 `ContextVar`）**：
   - `_IMAGE_COLLECTOR: contextvars.ContextVar[list[DocImage] | None] = contextvars.ContextVar("datainsight.step_images", default=None)`
   - `@contextlib.contextmanager def collect_step_images() -> Iterator[list[DocImage]]`：供 `Agent.ask()` 包裹每步工具执行，收集本步内 `read_document` / `read_file` 产出的去重 `DocImage`。
   - `def attach_step_images(items: Sequence[DocImage], *, limit: int = ...) -> list[DocImage]`：工具侧调用；有激活收集器时按 `id` 去重追加并返回实际附着列表；无收集器时直接返回去重截断后的列表（方便单测）。

### 3.3 改造 `app/readers.py`

1. **支持独立图片后缀**：
   - 在 `KIND_BY_SUFFIX` 增加 `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.bmp` -> `"image"`。
   - `route(name)` 自动将图片路由到 `"document"`；`UPLOAD_SUFFIXES` 自动包含图片格式，界面上传框即可直接选择图片！
2. **`Doc` 数据类扩展（向后兼容）**：
   - 新增字段 `images: list[DocImage] = field(default_factory=list)`。
   - 新增方法 `def images_for_part(self, part: int) -> list[DocImage]`：返回 `img.part == part` 的图片列表（若文档只有 1 段，兼容全部）。
   - 修改 `__post_init__`：当 `self.text` 为空但 `self.images` 非空时，自动生成图片索引占位文本作为 `chunks[0]`，确保 `Doc.parts >= 1`。
3. **各格式图片抽取实现与锚点对齐**：
   - 统一内部辅助 `_place_anchor_and_collect(...)`：抽取到图片后先经过 `normalize_image`，若有效则赋予全局序号 `1, 2, ...`、插入 `[图片 k：<location> · <w>×<h>]` 文本锚点，并在 `split_chunks()` 之后通过正则/位置匹配每个 `[图片 k：` 落在哪一个 `chunk`（1-based），回填 `DocImage.part`！
   - **`_read_pdf(path)`**：遍历 `page.images`，对每页内嵌图片抽取 `img.data`；若某页没有文字但有图片，仍产出 `【第 N 页】\n[图片 k：第 N 页 · w×h]`，并将提示语从"需要 OCR 才能读"升级为"已抽取 M 张页面图片（共 N 页没有文字层）"。
   - **`_read_docx(path)`**：按文档流顺序遍历正文 XML（`document.element.body` 中的 `a:blip[@r:embed]` 与 `v:imagedata[@r:id]`）以及页眉/页脚，按 `rId` 从 `part.related_parts` 取出真实图片字节并去重；插图位置紧随对应段落或表格，保证图文顺序一致。
   - **`_read_pptx(path)`**：抽取每页幻灯片（含递归遍历 `GroupShape`）内的 `Picture`（`shape.image.blob`），记录所属页码 `第 N 页`；若整页无文字但有图片，不再算成完全空白页，而是生成 `【第 N 页】\n[图片 k：第 N 页 · w×h]`。
   - **`_read_xlsx(path)`**：保持 `load_workbook(..., read_only=True, data_only=True)` 的低内存表格读取不动；额外通过 `zipfile.ZipFile` 读取 `xl/media/*`（并尽可能解析 `xl/drawings/_rels/*.rels` 与 `xl/worksheets/_rels/*.rels` 关联到工作表名），零内存压力抽出 Excel 嵌图。
   - **`_read_image(path)`**：读取独立图片文件并规范化，生成单图文本描述 `[图片 1：文件名 · w×h (MIME)]` 与 1 条 `DocImage(part=1)`。
4. **`note` 摘要同步**：当存在图片时，`Doc.note` 中附带简明提示（如 `含 3 张图片`），无图片时不加任何额外文字，保持纯文本文档行为 100% 一致。

### 3.4 改造 `app/vault.py`

1. `Entry` 新增向后兼容字段 `image_count: int = 0`（老 `index.json` 缺少该键时 dataclass 默认值正常生效）。
2. 新增目录 `_images_dir(file_hash: str) -> Path`（`VAULT_DIR / "images" / file_hash`）：
   - `manifest.json` 存 `DocImage.to_meta()` 列表 + 图片文件名；
   - 实际规范化后的图片二进制存于同目录 `<img.id>.<ext>`。
3. `vault.add()` 提取文档后，若 `doc.images` 非空，落盘写入 `manifest.json` 和图片文件，记录 `image_count=len(doc.images)`；`_forget()` 在无共享引用时同步 `shutil.rmtree` 清理 `images/<file_hash>`。
4. `vault.open_document(entry)`：
   - 优先从 `images/<file_hash>/manifest.json` 恢复 `DocImage`（仅填 `path`，延迟读字节）；
   - **老文件平滑兼容**：若原始文件 `entry.path` 存在且后缀属于可含图格式，而 `manifest.json` 尚不存在（说明是升级前存入的旧 PDF/DOCX/PPTX），惰性提取一次图片并写入缓存，再构造 `Doc(..., images=images)`。

### 3.5 改造 `app/tools/doc_tools.py`、`app/tools/file_tools.py`、`app/tools/pptx_tools.py`

1. **`read_document(name: str, part: int = 1)`（`app/tools/doc_tools.py`）**：
   - 当 `not doc.text.strip() and not doc.images` 时才报"没有可读的内容"。
   - 读取第 `part` 段后，取 `doc.images_for_part(part)`，调用 `images.attach_step_images(...)` 挂载给本步模型；
   - 若实际附着了图片，在返回尾部提示 `（本段附带 K 张图片，已一并提交给模型查看）`；若本段无图片但文档其它段有图片，在 `part == 1` 提示 `（本文档另含 M 张图片，可在对应分段读取）`。
2. **`read_file(path: str, part: int = 1)`（`app/tools/file_tools.py` + `app/tools/pptx_tools.py`）**：
   - 非 `.pptx` 格式走 `readers.extract(target)` 后，同 `read_document` 按 `part` 取出 `doc.images_for_part(part)` 并调用 `images.attach_step_images(...)`。
   - `.pptx` 格式：在 `pptx_tools.describe_pptx(target)`（或辅助函数 `extract_pptx_images_by_part`）中记录每个分片包含的页码集合 `page_range`，利用 `readers` 提取出的按页标注的图片匹配对应 `part`，同样调用 `images.attach_step_images(...)`。
3. **无图片文档零变化**：当 `images_for_part(part)` 为空时，返回文本与之前完全一致。

### 3.6 改造 `app/agent.py`、`app/llm.py` 与 `app/ui_render.py`

1. **`app/agent.py`**：
   - **System Prompt（`make_system_prompt`）**：当 `files` 中存在含图文档或 `image` 类型文件时，提示词补充一句：`"部分文档或图片文件附带视觉图像，调用 read_document / read_file 读取对应分段即可直接查看图片内容。"`（纯表或无文件时不出现，确保 `tests/test_docs.py` `TestPromptWithFiles` 断言全绿）。
   - **`Agent.ask()` 多模态提交与自动降级**：
     - 维护轮次局部状态 `step_attachments: dict[int, list[DocImage]] = {}`（key 为注入位置：`session.messages` 末尾索引）、`sent_image_ids: set[str] = set()`、`multimodal_enabled = config.MULTIMODAL_IMAGES_ENABLED`、`vision_fallback = False`。
     - 每轮执行工具时用 `with images.collect_step_images() as step_imgs:` 包裹：
       - `session.add({"role": "tool", ...})` 仍只存纯字符串；
       - 若本步收集到未发送过的新图片且未超 `IMAGE_MAX_PER_TURN`，记录 `step_attachments[len(self.session.messages)] = accepted_imgs`，并在发出的 `observation` 事件或新增 `"images"` 字段里携带 `[i.to_meta() for i in accepted_imgs]`。
     - 构造发往 `self.llm` 的 `request_messages`：
       - 从 `self.session.messages` 复制（仅当存在 `packaged_question != question` 或 `step_attachments` 非空且 `not vision_fallback` 时才浅拷贝，否则零拷贝复用）；
       - 按键升序在对应 `tool` 消息之后插入多模态 `user` 消息：`{"role": "user", "content": [{"type": "text", "text": "..."}, *(img.to_content_part() for img in imgs)]}`。
     - **纯文本自动降级保护**：当本次请求携带了图片且 `self.llm(...)` 抛出 `LLMError`（`status in (400, 415, 422)` 或异常信息命中多模态不支持特征词如 `image_url`/`vision`/`multimodal`/`expected a string`/`unknown variant`/`content`），捕获后设置 `vision_fallback = True`，发射 `self._emit("warning", message="当前模型不支持多模态图片输入，已自动降级为仅文本模式。")`，并立刻以纯文本 `request_messages` 重试该步一次！
2. **`app/llm.py`**：
   - 检查 `chat()` 协议层：确保非流式与流式对数组型 `content` 透传无碍（当前 `json.dumps(payload)` 原生支持列表型 `content`，仅需确认 `_explain_http` / 重试逻辑无额外阻拦即可）。
3. **`app/ui_render.py` & `app/ui_streamlit.py`**：
   - `app/ui_render.py`：`observation` 事件若包含 `images` 元信息，在 `TraceItem` 标题或正文摘要中补充 `· 附带 K 张图片（文件名列表）`；`messages_to_chat` 与 `query_rewrite._history` 增加防御性文本提取，确保即使将来遇到 list 型 `content` 也不会崩溃。
   - `app/ui_streamlit.py`：侧边栏资料列表（`dt.document_digest()`）可显示图片数（如 `3 段 · 2 图`）；`_handle_temp_upload` 与 `_render_vault` 无需改流程即可直接接收 `.png`/`.jpg` 等图片文件。

---

## 4. 实施步骤与测试验收计划

1. **Step 1：配置与图片基础层（`app/config.py` + `app/images.py`）**
   - 实现 `DocImage`、`normalize_image`、魔数探测、及 `collect_step_images` ContextVar 收集器。
2. **Step 2：文档/图片抽取与储藏区持久化（`app/readers.py` + `app/vault.py` + `app/tools/data_tools.py`）**
   - 为 PDF / DOCX / PPTX / XLSX / 独立图片实现图片抽取、锚点插入、段号对齐与 `vault` 磁盘缓存及惰性补抽。
3. **Step 3：工具层与 Agent 多模态协议对接（`app/tools/doc_tools.py` + `app/tools/file_tools.py` + `app/tools/pptx_tools.py` + `app/agent.py` + `app/ui_render.py` + `app/query_rewrite.py`）**
   - 接入 `attach_step_images`、`request_messages` 多模态 `image_url` 构造、以及纯文本模型 HTTP 400 自动降级重试。
4. **Step 4：编写全面离线测试 `tests/test_multimodal_images.py` 并全量回归**
   - 覆盖：
     - 独立 PNG/JPEG/BMP 图片抽取与 `normalize_image`（超大尺寸缩放、CMYK/RGBA 转换、无 Pillow 兜底路径）；
     - 含真图片的 `.docx`（`python-docx` 造图）、`.pptx`（`python-pptx` 造图）、`.xlsx`（`openpyxl` 造图）、`.pdf`（真嵌图 PDF / mock `page.images`）抽取与 `part` 段号对齐；
     - 纯图片扫描件 PDF / 纯图 Word 不再返回空内容；
     - `vault.add` / `open_document` / `drop` 对图片缓存的落盘、重启恢复、老版本惰性补抽与引用计数清理；
     - `read_document` / `read_file`（含 `.pptx`）在 `collect_step_images` 下精准只附带当前 `part` 的图片；
     - `Agent.ask()` 端到端多模态提交验证：检查 `llm.seen` 中确实出现 `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`、检查 `session.messages` 保持纯字符串不被 Base64 污染、检查同轮多步持续携带图片且去重限流；
     - 纯文本模型拒绝 `image_url`（抛 `LLMError(..., status=400)`）时的自动降级重试与 `warning` 事件。
   - 执行全量测试 `python -m unittest discover -s tests -p "test_*.py" -v`，确保原有 585 项 + 新增用例 100% 通过！
