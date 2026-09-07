# 问答客服（Query 流程）智能化改造方案

> 目标：把"一次性 RAG 问答机"升级为"能澄清、能分流、能兜底"的对话式客服。
> 状态：**P0 已实施（2026-09-07）**，P1/P2 待排期。
>
> P0 实施落地说明：
> - **P0-1** `intent_route_node`（意图路由）：入口节点，LLM 分类 + 闲聊直答；`route_search_paths`
>   按意图分发检索分支（meta→仅 Web；troubleshoot→图谱+向量；其余四路全开）；
>   `INTENT_ROUTE_ENABLED=0` 可关闭回退旧行为。LLM 失败/非法输出降级 product_qa。
> - **P0-2** `clarify_output_node` + Mongo `chat_session` 集合（pending_clarify 槽位：
>   options/turn/last_query）：`item_name_confirm` 优先消费 `selected_item`（前端点击回传）→
>   槽位文本/序数词命中 → 均未命中才走 LLM 提取；轮次超限降级"请提供完整型号"。
>   向量对齐候选暂未实现（轻量文本匹配已覆盖，实测未命中偏高再升级）。
> - **P0-3** `answer_gate_node`（质量门控）：rerank 后判断 top1 分数，
>   `ANSWER_CONFIDENCE_FLOOR` 以下坦诚兜底（不调 LLM），`ANSWER_WARN_SCORE` 以下附仅供参考提示。
> - **P0-4** 答案层：ANSWER_PROMPT 加客服人设 + 引用编号要求 + `COMPARE_HINT`（compare 意图注入）；
>   答案生成后轻量调用产出 3 条建议问法（stage=`query_suggest`），非流式入 task_result、流式推
>   `suggestions` SSE 事件；chat.html 渲染可点击 chips（澄清候选 + 建议问法）。
> - 测试：`tests/test_query_p0.py`（30 例，含端到端打桩用例）。

## 一、现状诊断

当前流程（`query_process/main_graph.py`）：

```
item_name_confirm（LLM 提取商品名 → 向量对齐分桶）
    ├─ confirmed 有 → 四路并行搜索（向量/HyDE/图谱/Web） → RRF → Rerank → 生成答案
    └─ confirmed 无 → options 或 罐头回复 → 直接结束
```

四个结构性短板：

| # | 短板 | 代码位置 | 表现 |
|---|------|----------|------|
| 1 | **澄清是终态不是循环** | `item_name_confirm_node._decide` | 未确认商品时写入 `answer` 罐头话术直接 END。用户回复"我要问的是第二个"后，靠 LLM 重新从 history 里猜，没有结构化的追问-应答闭环 |
| 2 | **没有意图分流** | 无此节点 | 闲聊、售后政策、故障排查、产品对比全部走同一条重型流水线；四路搜索（Web 搜索走百炼 MCP，按次计费）固定全开 |
| 3 | **检索质量无门控** | `answer_output_node._build_prompt` | `reranked_docs` 为空/低分时 context = "无参考内容"，仍然调 LLM 强行作答；rerank 的断崖截断结果没有回传给答案层做置信判断 |
| 4 | **单轮拼接生成** | `ANSWER_PROMPT` | 无引用编号（用户无法核对出处）、无生成后自检、无后续问题引导；对比类问题（"A 和 B 哪个好"）没有拆解检索 |

已有的好底子（保留不动）：四路混合检索 + RRF + 断崖截断 rerank 的检索骨架是对的；
商品名高分对齐（0.7/0.6 分桶）机制是对的；Mongo 会话历史 + item_names 回填思路可用但实现要升级。

## 二、改造方案（按投入产出排序）

### P0-1 意图路由节点（新增，放在 item_name_confirm 之前）

一次轻量 LLM 调用输出结构化意图：

| 意图 | 后续动作 |
|------|----------|
| `chitchat` 闲聊/寒暄 | 直接小模型回复，跳过全部检索 |
| `meta` 售后政策/元问题 | 只走 Web 搜索（知识库里没有这类内容） |
| `troubleshoot` 故障排查 | 图谱 + 向量（步骤、原因类知识在图谱里最强） |
| `product_qa` 产品咨询（默认） | 现有全流程 |
| `compare` 对比咨询 | 拆解为多商品并行检索，答案层用对比模板 |

收益：闲聊不再触发 4 路搜索 + rerank，平均响应延迟和成本显著下降；
意图标签写入 state，供后续所有节点复用。

### P0-2 澄清从"终态"改成"循环"（核心体验问题）

LangGraph 加条件回边：`item_name_confirm` 未确认时 → `clarify_output` 节点
（追问 + options）→ END，但**追问内容与 options 以结构化形式存入会话槽位**
（Mongo 新增 `pending_clarify` 字段）。下一轮用户消息进来时：

1. `item_name_confirm` 先检查 `pending_clarify`：若存在且用户回复命中某个 option
   （精确匹配或向量对齐到 option），**直接锁定该商品，跳过 LLM 提取**；
2. 未命中才走正常提取。上限 2 轮，超限降级为"请提供说明书上的完整型号"。

配套前端改动：options 渲染成可点击卡片，点击后发送结构化 `{"selected_item": "..."}`
（新 query_router 入参），彻底绕开自然语言匹配的不确定性。

### P0-3 检索质量门控（防"一本正经地兜圈子"）

在 `rerank` → `answer_output` 之间加判断（复用 rerank 已有的分数）：

- `reranked_docs` 为空 或 top1 分数低于阈值（config 新增 `answer_confidence_floor`）：
  → 走"坦诚兜底"模板：说明没找到 + 给出 2~3 个建议问法 + 引导提供型号；
- 命中但分数平平：答案里附"以上内容仅供参考，建议核对说明书原文"提示。

### P0-4 答案层升级（不加节点，只改 prompt + 后处理）

1. **引用编号**：context 格式化时给每条文档加 `[1] [2]`，要求答案句尾标注来源；
2. **后续问题建议**：答案生成后追加一次轻量调用（或同调用内）产出 3 个建议问法，
   前端渲染为可点击 chips——既引导用户，也是天然的检索优化闭环；
3. **人设**：prompt 头部加客服人设与语气约束（简洁、分点、先结论后细节）。

### P1-1 查询拆解（多跳问题）

`compare` / 复合问题（"怎么校准和怎么换电池"）由 LLM 拆成子问题，
并行走检索后按子问题分节作答。放在意图路由之后、多路搜索之前。

### P1-2 会话记忆升级

- `history` 超过 6 轮时用 LLM 压缩成摘要（当前 `limit=10` 硬截断会丢关键上下文）；
- `item_names`、`pending_clarify` 做成会话槽位对象（`{confirmed: [], candidates: [], turn: n}`），
  替代现在的"回填历史消息" hack。

### P1-3 动态搜索路由

意图路由 + 意图 × 历史命中率（哪个源经常命中）决定启用哪几路搜索；
Web 搜索仅在 `meta` 意图与知识库低置信时触发（省 MCP 调用费）。

### P2-1 生成后自检（Self-RAG 式）

答案生成后一次轻量校验："答案中的每个关键论断是否都能在 context 中找到对应编号？"
不通过则带错误提示重检索一次（上限 1 次）。

### P2-2 数据回流

前端满意度反馈（👍/👎）落 Mongo，与现有的 LLM 调用留档、`eval/run_eval.py` 评测集打通：
差评 case 自动进入待补充评测用例清单。

## 三、实施顺序建议

| 阶段 | 内容 | 预估 | 依赖 |
|------|------|------|------|
| 第一批 | P0-2 澄清循环 + P0-3 质量门控 | 1~1.5 天 | 无 |
| 第二批 | P0-1 意图路由 + P0-4 答案层升级 | 1 天 | P0-2 的 state 字段 |
| 第三批 | P1-1 拆解 + P1-2 记忆 + P1-3 动态路由 | 2 天 | 意图路由 |
| 第四批 | P2 自检 + 数据回流 + 前端卡片 | 1~2 天 | 全部 |

每批完成后跑 `eval/run_eval.py` 对比 Recall/MRR，防止改动引入检索质量回退。

## 四、明确不做的事

- 不引入 Agent 框架/工具循环——当前客服场景问题固定，图编排 + 意图路由已够用，
  引入 ReAct 只会增加不可控性；
- 不做多轮主动追问"您还想问什么"之类的过度打扰设计，只在澄清和兜底场景触发。
