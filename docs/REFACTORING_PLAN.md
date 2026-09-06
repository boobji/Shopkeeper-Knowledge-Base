# 重构计划 · Shopkeeper Knowledge Base

> 基于 2026-09 全量代码走读（~9.6k 行 Python，不含前端/评测）。
> 总体评价：架构分层（api → services → processor/graph → utils）是清晰的，LangGraph 编排和节点基类的设计意图也很好；
> 主要问题集中在 **配置/连接管理散乱、导入与查询两条流水线大量复制粘贴、内存态任务存储、以及若干确定的 bug**。

---

## 一、现状问题清单

### A. 确定的 Bug（建议立即修，与重构解耦）

| # | 位置 | 问题 |
|---|------|------|
| A1 | `knowledge/services/task_service.py:28-29` | `get_task_status` 忘记 `return`，调用永远得到 `None` |
| A2 | `knowledge/utils/mongo_history_util.py:114-135` | `get_recent_messages` 注释写"倒序取最近 N 条"，实际 `sort("ts", ASCENDING).limit(limit)` 返回的是**最旧的 N 条**，且没有 reverse。多轮对话历史因此失真 |
| A3 | `knowledge/processor/import_process/main_graph.py:57-72` | `entry_node` 同时注册了条件边（`import_router`）和顺序边 `add_edge('entry_node','pdf_to_md_node')`，条件路由实际不生效（或行为取决于 LangGraph 版本），`is_md_read_enabled` 分支是死逻辑 |
| A4 | `knowledge/utils/bge_m3_embedding_util.py:24` | `use_fp16 = os.getenv('BGE_FP16', False)` 得到的是字符串，`"0"` 也是真值 → 配置写 0 仍会开 fp16 |
| A5 | `knowledge/utils/llm_client_util.py:15-53` | 客户端缓存 key 只有 `(model_name, response_format)`，**不含 temperature** → 不同温度复用同一实例；且无 model_name 时默认取 `ITEM_MODEL` 而不是 `MODEL`，语义误导 |
| A6 | `knowledge/services/query_service.py:43-52` | 查询图异常后 `finally` 里仍把任务标成 `completed`（导入侧是 `failed`），状态语义不一致；且失败原因没有写入 task result，前端无从展示 |
| A7 | `knowledge/utils/task_util.py:108` | `clear_task()` 定义了但**没有任何调用方** → `_tasks_running_list/_tasks_done_list/_tasks_result/_tasks_status` 四个全局字典无限增长（内存泄漏），SSE 队列同理 |
| A8 | `knowledge/processor/query_process/nodes/kg_search_node.py:643-645` | `find_one_hop_relations` 中单个种子节点查询异常时 `return []`，把已收集的全部关系丢掉；应 `continue` |
| A9 | `knowledge/services/import_file_service.py:48` | `file.filename` 直接拼进路径，未做 basename/路径穿越清洗；同名文件相互覆盖；MinIO object 名日期格式 `%Y%d%m` 与本地目录 `%Y%m%d` 不一致（日月颠倒） |

### B. 结构性问题（重构主体）

1. **`load_dotenv` 散布在 9+ 个文件里，策略互不相同**
   `config.py`（显式路径）/ `query_config.py`（`override=True` 但依赖 CWD）/ `llm_client_util.py`（显式路径+override）/ `milvus_util.py`、`minio_util.py`、`mongo_history_util.py`、`import_file_service.py`（裸 `load_dotenv()`）。
   加载顺序决定最终值，谁先 import 谁生效，是"改了 .env 不生效"类问题的根源。

2. **导入/查询两条流水线成对复制**
   - `import_process/base.py` 与 `query_process/base.py`：`BaseNode` 几乎相同（查询侧多了 SSE 进度推送）
   - `import_process/config.py` 与 `query_process/config.py`：LLM/Milvus/Neo4j 三段配置字段完全重复
   - `import_process/exceptions.py` 与 `query_process/exceptions.py`：同一套异常体系两份
   - 两个 `setup_logging()`、两个 `state.py` 的 `create_default_state/get_default_state`
   - 实体清洗/解析逻辑在 `knowledge_graph_node.py`（导入）与 `kg_search_node.py`（查询）各写一遍；商品名对齐逻辑在 `item_name_recognition_node.py`（导入）与 `item_name_confirm_node.py`（查询）各写一遍

3. **资源单例模式不统一、线程不安全**
   - Milvus/Neo4j/BGE/LLM：`global` 变量 + 无锁懒加载（BackgroundTasks 跑在线程池里，并发导入会竞态重复建连）
   - MinIO：`get_minio_client()` **每次调用新建客户端 + bucket_exists 检查**（最该缓存的反而没缓存）
   - Mongo：模块 import 时就实例化并建索引（import 副作用重，README FAQ 里"启动即加载全链路"即由此而来）

4. **任务状态是进程内全局字典**
   两个服务是独立进程，任务状态无法共享；重启全丢；内存无限增长（见 A7）。`TaskService` 只是对 `task_util` 函数的薄包装，抽象没有带来隔离。

5. **分层违规**
   - `import_file_service.py`（service 层）抛 `fastapi.HTTPException`，service 被架死在 FastAPI 上
   - utils 层互相掺杂业务（`bge_m3_embedding_util` 里 import 了不需要的 `langchain_openai`；节点里直接散落 Cypher 语句与 Milvus 查询编排）
   - `kg_search_node.py` 1057 行、`knowledge_graph_node.py` 819 行：单节点同时承担 LLM 调用、清洗、存储访问、编排

6. **仓库卫生**
   - 两份 `requirements.txt`（根目录 vs `knowledge/`）内容分叉
   - `knowledge/processor/import_process/nodes/test.py`（打印编码的杂物）被 git 跟踪
   - `mcp_search_node_bak.py`、`import_temp_dir/`、`temp_data/` 等留在工作区
   - `knowledge/test/` 是手工脚本集合（拼写错误目录 `neoi4j/`、`test_mongo_connetction.py`、`test_vl_flah.py`），不是 pytest 测试；无 CI、无 lint/类型检查
   - 命名问题：`graph_pineline`、`KnowLedgeGraphNode`、`kb_import__graph_app` 双下划线等

---

## 二、重构方案（按阶段执行，每阶段独立可合并、可回滚）

### Phase 0 — 安全网（0.5 天）

重构前先保证"改坏能立刻发现"：

1. 合并两份 `requirements.txt` 为根目录一份（`knowledge/requirements.txt` 保留一个指向说明或删除），**钉住关键依赖版本**（pymilvus / langgraph / FlagEmbedding，稀疏向量格式和 LangGraph API 都随版本漂移，README FAQ 已两次踩坑）。
2. 搭 pytest 骨架：把**纯函数逻辑**先补上单测（不依赖外部服务）：
   - `document_split_node` 的切分/合并（构造 md 字符串即可）
   - `rrf_node` 的融合排序、`rerank_node` 的 `_cliff_cutoff`
   - `kg_search_node` 的 `_clean_parse_llm_content`、`_build_item_entity_pairs`、`_clean_seed_rows`
   - `eval/run_eval.py --self-test` 纳入回归
3. 加 `ruff`（lint + format）与 `mypy`（宽松起步）配置，先跑通基线再逐步收严。

### Phase 1 — Bug 修复 + 仓库清理（1 天）

1. 修复 A1–A9（每条一个独立 commit，都是小改动）。
2. 清理：删除 `nodes/test.py`、`mcp_search_node_bak.py`、各 `main_graph.py` 的 `__main__` 测试代码与硬编码路径（改造成 `scripts/` 下的调试脚本或 pytest）、`knowledge/test/` 中的一次性连接脚本（有价值的迁移到 `scripts/check_env.py` 或 pytest 标记 `@pytest.mark.integration`）。
3. 统一命名：`graph_pineline→pipeline`、`KnowLedgeGraphNode→KnowledgeGraphNode`、`kb_import__graph_app→import_graph_app`、test 目录拼写修正。

### Phase 2 — 统一基础设施（2–3 天，核心阶段）

目标：消灭 B1/B2/B3 三大问题，行为不变。

```
knowledge/
├── core/
│   ├── config.py        # 新增：单一 Settings（pydantic-settings），分 import_/query_/llm_/storage_ 节
│   ├── logging.py       # 新增：唯一 setup_logging
│   ├── connections.py   # 新增：milvus/neo4j/mongo/minio/llm/bge/rerank 单例（threading.Lock 保护，懒加载）
│   └── exceptions.py    # 新增：合并两份 exceptions（ProcessError + StateFieldError + Neo4jError/MilvusError）
├── processor/
│   └── base.py          # 新增：唯一 BaseNode（含 SSE 进度推送，is_stream=False 时自动跳过）
```

要点：
- **配置**：只保留 `core/config.py` 一处 `load_dotenv`（显式指向 `knowledge/.env`，带 `override=True`）。两个 config dataclass 改为从 Settings 取字段的薄视图，节点代码不动——先合并来源，再考虑字段归并。
- **连接**：`connections.py` 模块级 `@lru_cache` + 锁；MinIO 补上缓存并去掉每次 `bucket_exists`（建桶职责移交 `docker-compose` 的 minio-init，它已经在做了）。所有"失败返回 None + 调用方层层判空"的模式收敛为"失败抛异常，BaseNode 统一捕获"，节点内十余处 `if client is None` 可删除。
- **BaseNode**：保留两侧差异作为参数（`push_progress: bool`），导入/查询节点改 import 路径即可。
- **task_util 状态对象化**：把四个全局字典封装成 `TaskStore` 类（接口不变），为 Phase 4 换存储留缝；顺手在任务完成/失败时调用 `clear_task` + TTL 兜底，堵住内存泄漏。

### Phase 3 — 去重复 + 节点瘦身（3–4 天）

1. **抽取共享领域模块**（导入/查询共用的逻辑各只剩一份）：
   - `knowledge/domain/entity_extract.py`：LLM 实体抽取 + JSON 清洗解析（`_clean_parse_llm_content`、`_clean_entities`、`_clean_relations` 归一）
   - `knowledge/domain/item_name.py`：商品名向量化对齐/打分过滤（两侧的 Aligner 合并，权重与阈值从配置读）
   - `knowledge/domain/kg_repository.py`：Cypher 语句集中（导入 Writer / 查询 Reader 共用同一套节点/关系标签常量——目前两侧白名单和 Cypher 分散在两个文件里，是最容易改出不一致的地方）
   - `knowledge/domain/milvus_repository.py`：collection 建表、hybrid search 请求、按 chunk_id 批量取回（`milvus_util.py` 的函数归类进来）
2. **大节点拆分**（保持对外 `process(state)` 不变）：
   - `kg_search_node.py`（1057 行）→ 拆为 extractor / aligner / graph_reader / backfiller 四个类移入 domain 模块，节点本体剩 ~100 行编排（文件头注释里本来就声明了这个结构，只是没有真的拆出文件）
   - `knowledge_graph_node.py`（819 行）→ 同样处理：Writer 类入 domain，清洗逻辑与查询侧合并
3. **service 层去 FastAPI 化**：`import_file_service` 抛领域异常（`StorageError` 等），router 里统一 `@app.exception_handler` 转 HTTP 响应。
4. **补齐测试**：对合并后的共享模块跑 Phase 0 的单测，导入/查询两侧行为用同一组用例验证。

### Phase 4 — 运行时与 API 层（2 天，可选增强）

1. **app factory 合并**：两个 router 的 `create_app` 抽成 `core/app_factory.py`（CORS、静态目录、异常处理器、lifespan）。lifespan 里预热并 `verify_connectivity` 各连接，把"import 阶段全链路加载"推迟到启动阶段，顺带解决 README FAQ 第一条。
2. **任务状态外置（可选）**：`TaskStore` 加一个 Mongo 实现（复用已有 MongoDB），支持重启后查询历史任务；SSE 队列同样收敛进 `SSEHub` 类，订阅时若任务已结束补发 final 事件，解决"先完成再连接收不到结果"的时序问题。
3. **非流式查询**改为后台任务 + 前端轮询，或至少在文档标注阻塞行为（当前 `/query` 同步跑完整条 LLM 链路，会长时间占住 uvicorn worker）。
4. 上传文件名清洗（`Path(filename).name` + uuid 前缀），MinIO object 路径格式对齐。

### Phase 5 — 收尾（0.5 天）

- GitHub Actions：`ruff + mypy + pytest`（单测层），eval 仍按需手动跑。
- README 项目结构图与实际目录同步。
- 删除 `.idea/` 之外的遗留（`info/`、空 `docs/` 处置）。

---

## 三、执行顺序与风险

| 阶段 | 风险 | 回滚方式 |
|---|---|---|
| P0 安全网 | 无 | — |
| P1 bug 修复 | A3（图边）改动会改变导入行为，需实测 md/pdf 两条分支 | 逐条 revert |
| P2 基础设施 | .env 加载策略统一后，**当前靠加载顺序生效的隐式配置会暴露**（如系统环境变量覆盖 .env），上线前用 `scripts/check_env.py` 核对 | 按模块 revert |
| P3 去重复 | 纯等价重构，靠共享模块单测兜底 | 按模块 revert |
| P4 运行时 | 涉及部署形态，建议单独分支验证 | 功能开关 |

原则：**每个 Phase 独立可合并**；P2/P3 期间不新增功能；`eval/run_eval.py` 的检索指标作为每次合并前后的行为对照（指标不应回退）。

## 四、明确不做的事

- 不更换技术栈（LangGraph/FastAPI/四个存储均保留）
- 不在本轮引入鉴权、多租户（Roadmap 项，待重构完成后单独立项）
- 不动 `eval/` 评测集与指标定义
