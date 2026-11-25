# L2 计划：SQLite + FTS5（中文 bigram）

> 状态：待审查。本文所有数字都是在本机真实语料上跑出来的，不是估算。

## 1. 目标与非目标

**要做**

1. 把 `data/vault/index.json` 换成 SQLite 表（`documents`）。
2. 为资料正文建 FTS5 全文索引（中文按 bigram 切），`search_document` 的候选召回改成查索引。
3. 一次性迁移 + 可重建 + 可回滚。

**本轮不做**

- `data/chats/index.json` 不动（理由见 §7 决策 A）。
- 不改 `search_document` 的返回格式、排序规则、话术。
- 不引入向量检索（那是 L3）、不加任何第三方依赖（`sqlite3` 是标准库）。

## 2. 三个实测发现（它们决定了下面的设计）

### 发现 1：当前规模下，FTS5 不会更快

真实语料 3 份 PDF / 26.4 万字 / 134 段：

| 查询 | 线性扫描（现状） | FTS5 召回 + 原打分 | 结论 |
|---|---|---|---|
| 供应链金融 | 0.77 ms | 0.88 ms | 持平 |
| 智能合约 存证 隐私计算 | 1.80 ms | 2.14 ms | 略慢 |
| 区块链创新应用 | 0.80 ms | 0.61 ms | 略快 |
| 共识机制 | 0.70 ms | 0.32 ms | 略快 |

**差的是零点几毫秒，人感觉不到。** 所以 L2 的卖点不能是"更快"，只能是"耗时与库的大小脱钩"。
把这批分块复制到 4422 段（约 100 份文档）再测：

| 段数 | 线性扫描 | FTS 召回 |
|---|---|---|
| 134（现状） | 0.7–1.8 ms | 0.6–2.1 ms |
| 4422（约 100 份） | 22–26 ms | 0.8 ms |
| 25000（约 500 份，外推） | 120–150 ms | 0.8 ms |

线性扫描是 O(总字数)，FTS 是 O(命中数，上限 200 条)。**这才是值得做的理由。**

### 发现 2：纯 bm25 排序会改变现有结果，不能直接接管

同样的 5 个查询，如果完全按 `bm25()` 排：

- 5 个里有 **2 个首选段落变了**（`智能合约 存证 隐私计算` 从第 71 段变成第 68 段；`区块链创新应用` 从第 34 段变成第 1 段）。
- 谁更对，光看数字判不出来——那要靠第 5 周的评测集。

而 `docs/manual-test-search.md` 里已经写死了"第 71/77 段"这个预期。排序一变，那份验证文档当场失效。

**所以本轮的做法是：FTS 只负责"找出含这些词的段落"（召回），排序仍然用现有的 `_score()`（命中不同词数为主权重）。**
实测这样做的 top-3 与改造前 **5/5 完全一致**。

### 发现 3：连接开销比查询本身还大

| 做法 | 单次耗时 |
|---|---|
| 每次新建连接 + 查询 | 1.99 ms |
| 复用连接查询 | 0.14 ms |

每次开连接的话，FTS 比现在的线性扫描还慢。所以连接必须缓存（进程内、按库路径缓存）。
这一条直接影响测试：现有用例靠 `mock.patch.object(vault, "VAULT_DIR", tmp)` 隔离目录，
缓存必须**按路径判定**，路径变了就重开，否则测试之间会串库。

顺带两个边界（已实测）：

- 2 字中文（「存证」）能查到；**1 字中文（「税」）查不到**——bigram 天然没有单字 token。
  需要显式话术："关键词至少两个字"。
- 建索引：134 段耗时 **111 ms**（一次性的，只在上传那份文档时发生）。
  索引文本约为正文的 **1.8 倍**（474,748 / 263,543 字符），磁盘要在文档里写明。

## 3. 数据模型

一个库：`data/vault/index.db`（跟 `index.json` 同一个位置、同一个目录生命周期）。

```sql
PRAGMA journal_mode = WAL;      -- 界面多线程读，WAL 让读不挡写
PRAGMA user_version = 1;        -- 结构版本，不匹配就整库重建（索引是缓存，可丢）

CREATE TABLE documents (        -- 替代 index.json 的 entries
    id        TEXT PRIMARY KEY,
    file      TEXT NOT NULL,    -- 原始字节的哈希，决定磁盘文件名
    text_hash TEXT NOT NULL,    -- 正文的哈希，检索索引的主键
    name      TEXT NOT NULL UNIQUE,
    size      INTEGER NOT NULL,
    suffix    TEXT NOT NULL,
    kind      TEXT NOT NULL,
    added_at  TEXT NOT NULL,
    chars     INTEGER NOT NULL DEFAULT 0,
    parts     INTEGER NOT NULL DEFAULT 0,
    note      TEXT NOT NULL DEFAULT '',
    seq       INTEGER NOT NULL  -- 插入顺序：entries() 靠它倒序，不靠时间戳
);
CREATE INDEX ix_documents_kind ON documents(kind);

CREATE TABLE chunks (           -- 每段一行，body_ft 是切好 bigram 的形态
    id        INTEGER PRIMARY KEY,
    text_hash TEXT NOT NULL,
    part      INTEGER NOT NULL,
    body_ft   TEXT NOT NULL,
    UNIQUE (text_hash, part)
);
CREATE INDEX ix_chunks_hash ON chunks(text_hash);

CREATE VIRTUAL TABLE chunks_fts USING fts5(
    body_ft, content = 'chunks', content_rowid = 'id'
);

-- 外部内容表的标配：删行时同步撤索引，省掉"整库 rebuild"
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, body_ft) VALUES (new.id, new.body_ft);
END;
CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, body_ft) VALUES ('delete', old.id, old.body_ft);
END;

CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
-- 存 chunk_chars / tokenizer=bigram-v1 / built_at
-- 任何一个与当前代码不一致 -> 自动重建索引（否则段号会错位）
```

**为什么正文不搬进库里**：`text/<hash>.txt` 仍是正文真相，`open_document` 一行不改。
库里只放索引形态（`body_ft`）。这样"删掉 .db 也能从正文重建"这句话是真的，
而不是"删了就丢数据"。代价是正文和索引各占一份磁盘。

**为什么索引按 `text_hash` 而不是按文件名**：同一份内容存两个名字（现有 `_forget` 就处理过这情况）
只索引一次；删掉其中一个名字不能把另一个的索引删掉——跟现在"没人共用才删文件"的规则一致。

## 4. 改动清单

| 文件 | 改什么 | 为什么 |
|---|---|---|
| `app/store.py`（新） | 连接缓存、建表、`bigram()`、`fts_expr()`、documents/chunks 读写 | 整个项目里唯一 import `sqlite3` 的地方；vault 已经 320 行，SQL 塞进去会失控 |
| `app/vault.py` | `_load/_save` → SQLite；`add/drop` 同步写索引；新增 `rebuild_index()`、`_migrate_from_json()` | 主战场 |
| `app/readers.py` | `Doc` 增加 `digest`（`__post_init__` 里算正文哈希） | 会话里的 Doc 要靠它映射到索引行；不新增字段就只能每次重算哈希（百份文档时是 30 MB 的哈希） |
| `app/tools/doc_tools.py` | 精确那一路改为 `FTS 召回 → _score 重排`；单字关键词报错 | 唯一改变行为的地方，但输出必须逐字不变 |
| `app/tools/data_tools.py` | 上传/挂载时建索引（`store.index_doc(doc)`） | 会话临时上传的资料也要能搜到（现在的搜索作用域就是"本会话登记的资料"） |
| `app/config.py` | 加 `VAULT_DB = VAULT_DIR / "index.db"` | 路径集中管理，测试好 patch |
| `tests/test_store.py`（新） | bigram、FTS 表达式、迁移、重建、单字报错、连接按路径切换 | 新代码要有自己的用例 |
| `README.md` / `docs/manual-test-search.md` | 补"索引存在哪、多大、怎么重建" | 不然下一个人会以为 `text/*.txt` 就是全部 |

**`vault.py` 里保留 `_LOCK`**：SQLite 自己会锁，但我们的"查-算-写"是跨多条语句的复合操作
（比如 add 要先算 quota 再插入），需要一层进程内互斥。WAL + `busy_timeout=3000` 兜住子进程/多标签页的并发。

## 5. 迁移与回滚

首次调用（惰性，不在 import 时）触发，整个过程在一个事务里：

1. `index.db` 存在 → 直接用，**完全不读 `index.json`**。
2. `index.db` 不存在、`index.json` 存在 → 读 json，插 `documents`；再逐份读 `text/<file>.txt`、
   切块、插 `chunks`（触发器自动进 FTS）。全部成功后把 `index.json` 重命名成 `index.json.migrated`。
3. 两个都没有 → 建空库。

**没有双写、也没有"读失败就回退到 json"**。双路径会让"到底哪份数据在生效"变成玄学，
这项目已经在别处为这种暧昧付过代价。

回滚到当前版本：删掉 `index.db`、把 `index.json.migrated` 改回 `index.json`，`git checkout` 旧代码即可。
`text/` 和 `files/` 全程不动，所以回滚不会丢任何一份资料。

## 6. 效果

| 指标 | 现在 | L2 后 |
|---|---|---|
| 单次检索（3 份 / 134 段） | 0.7–1.8 ms | 0.6–2.1 ms（**持平**） |
| 单次检索（约 100 份 / 4422 段） | 22–26 ms | **约 2 ms**（召回 0.8 + 重排 1.2） |
| 检索耗时与库大小 | 线性增长 | 与库大小脱钩（上限 200 条候选） |
| 上传一份 77 段 PDF | — | **+111 ms**（一次性建索引） |
| 增删一份资料写盘 | 整份 JSON 重写（O(n)） | 插入/删除若干行（O(段数)） |
| 磁盘 | 正文 1× | 正文 1× + 索引 **1.8×** |
| 检索输出 | — | **逐字不变** |

**要提前说清楚**：这轮改完，你用三份 PDF 试的时候**感觉不到任何变化**，因为本来就只有 1 毫秒。
L2 买的是"以后不会坏"，不是"现在变快"。真正能写进简历的数字得靠 L3 或评测集。

## 7. 需要你拍板的四件事

**A. chats 的 index.json 要不要一起搬？——建议不搬。**
最多 50 条、每条约 200 字节，整份重写约 1 ms；而且它已经有"索引丢了就扫内容文件重建"的机制。
搬过去要多一次迁移、多一套触发器、改 3 个测试文件，换不到任何可测量的收益。
如果你要的是"仓库里只有一个 index.json 也不留"，那就搬，我把工期算进去。

**B. 索引算不算进 200 MB 配额？——建议不算。**
算的话，用户传 100 MB 资料会被告知"超出 200 MB"（因为索引要 180 MB），这个提示没法解释。
代价是极端情况下磁盘占用接近上限的 2 倍，写进文档。

**C. 临时上传（`data/uploads/`）留下的孤儿索引行，本轮处不处理？——建议接受。**
这些行只在"会话上传了资料但没进储藏区、之后会话被清掉"时产生；正文文件本来也会被
`uploads.discard()` 删。清理它们要么加一套引用计数，要么每次清会话时算哈希，都不划算。
`rebuild_index()` 可以一次性回收。**我会在代码注释里写明这是已知取舍，不藏着。**

**D. 要不要顺带用 bm25 做次级排序？——建议不用。**
理由见发现 2：会让输出变，而变好变坏现在无从判断。等评测集能回答"哪个排第一更对"再说。

## 8. 实施顺序（每步都能单独验收）

1. `store.py` + 建表 + `bigram()` + 单测。**验收**：新用例绿，旧代码零改动。
2. `vault` 迁到 SQLite（含 json 迁移、`rebuild_index`、drop 清理）。**验收**：`test_vault` 全绿；
   迁移后 `entries()` 与 `index.json` 逐字段一致。
3. `search_document` 换召回路径。**验收**：5 个真实查询输出与改造前 `diff` 为空；`test_docs` 全绿。
4. 上传路径建索引 + `Doc.digest`。**验收**：全量测试绿；上传后能搜到，删掉 .db 后能重建。
5. 文档与手动验证脚本（含 4422 段规模对比）。**验收**：按文档执行一遍，数字与本文一致。

预计测试从 456 增至约 470。
