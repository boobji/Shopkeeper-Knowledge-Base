"""查询流程主图

使用 LangGraph 构建知识库查询工作流（P0 智能化改造后）。
"""

from typing import List

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from knowledge.processor.query_process.state import QueryGraphState

from knowledge.processor.query_process.nodes.answer_output_node import AnswerOutputNode
from knowledge.processor.query_process.nodes.answer_gate_node import AnswerGateNode
from knowledge.processor.query_process.nodes.clarify_output_node import ClarifyOutputNode
from knowledge.processor.query_process.nodes.intent_route_node import IntentRouteNode
from knowledge.processor.query_process.nodes.item_name_confirm_node import ItemNameConfirmNode
from knowledge.processor.query_process.nodes.vector_search_node import VectorSearchNode
from knowledge.processor.query_process.nodes.hyde_search_node import HyDeSearchNode
from knowledge.processor.query_process.nodes.mcp_search_node import McpSearchNode
from knowledge.processor.query_process.nodes.kg_search_node import KnowledgeGraphSearchNode
from knowledge.processor.query_process.nodes.rrf_node import RrfNode
from knowledge.processor.query_process.nodes.rerank_node import RerankNode


def route_after_intent(state: QueryGraphState) -> str:
    """意图路由后的分流。

    chitchat 已由意图节点写好答案 → 直接去 answer_output；
    其余意图 → item_name_confirm 继续确认商品。
    """
    if state.get("answer"):
        return "answer_output"
    return "item_name_confirm"


def route_after_item_confirm(state: QueryGraphState) -> str:
    """商品名称确认后的路由逻辑。

    - 已有答案且带澄清候选 → clarify_output（持久化澄清槽位后再输出追问）
    - 已有答案（降级/无法识别）→ answer_output 直接输出
    - 无答案（已确认商品）→ 继续搜索流程
    """
    if state.get("answer"):
        if state.get("clarify_options"):
            return "clarify_output"
        return "answer_output"
    return "multi_search"


def route_search_paths(state: QueryGraphState) -> List[str]:
    """按意图分发多路搜索（P0-1 收益：闲聊外的意图不必四路全开）。

    - meta（售后政策/元问题）：知识库没有这类内容，只走 Web 搜索
    - troubleshoot（故障排查）：步骤与原因类知识在图谱里最强，图谱 + 向量
    - compare / product_qa：现有全流程四路
    """
    intent = state.get("intent") or "product_qa"
    if intent == "meta":
        return ["web_search_mcp"]
    if intent == "troubleshoot":
        return ["search_embedding", "query_kg"]
    return ["search_embedding", "search_embedding_hyde", "query_kg", "web_search_mcp"]


def create_query_graph() -> CompiledStateGraph:
    """创建查询流程图。

    Returns:
        编译后的 StateGraph 实例。

    流程结构::

        intent_route
              │
              ├── (chitchat 已直答) ──────────────────> answer_output
              │                                             │
              └── (其余意图) ──> item_name_confirm          │
                                     │                      │
                                     ├─ (澄清追问) > clarify_output ──>│
                                     ├─ (降级/直答) ─────────────────>│
                                     └─ (已确认商品) > multi_search   │
                                            │                         │
                        （按意图分发检索分支）                         │
                                            │                         │
                                            v                         │
                                          join                        │
                                            │                         │
                                           rrf                        │
                                            │                         │
                                          rerank                      │
                                            │                         │
                                       answer_gate                    │
                                            │                         │
                                     answer_output <─────────────────┘
                                            │
                                            v
                                           END
    """

    # 1. 定义LangGraph工作流
    workflow = StateGraph(QueryGraphState)  # type:ignore

    # 2. 实例化节点
    nodes = {
        "intent_route": IntentRouteNode(),
        "item_name_confirm": ItemNameConfirmNode(),
        "clarify_output": ClarifyOutputNode(),
        "multi_search": lambda x: x,  # 虚拟节点
        "search_embedding": VectorSearchNode(),
        "search_embedding_hyde": HyDeSearchNode(),
        "query_kg": KnowledgeGraphSearchNode(),
        "web_search_mcp": McpSearchNode(),
        "join": lambda x: {},  # 多路搜索汇合（虚节点）
        "rrf": RrfNode(),
        "rerank": RerankNode(),
        "answer_gate": AnswerGateNode(),
        "answer_output": AnswerOutputNode()

    }

    # 3. 添加节点
    for name, node in nodes.items():
        workflow.add_node(name, node)  # type:ignore

    # 4. 设置入口点：意图路由
    workflow.set_entry_point("intent_route")

    # 5. 意图路由后分流：闲聊直答 / 其余确认商品
    workflow.add_conditional_edges(
        "intent_route",
        route_after_intent,
        {
            "answer_output": "answer_output",
            "item_name_confirm": "item_name_confirm"
        }
    )

    # 6. 商品确认后分流：澄清追问 / 降级直答 / 继续搜索
    workflow.add_conditional_edges(
        "item_name_confirm",
        route_after_item_confirm,
        {
            "clarify_output": "clarify_output",
            "answer_output": "answer_output",
            "multi_search": "multi_search"
        }
    )

    # 7. 澄清输出后仍要经过 answer_output（推送答案 + 写历史）
    workflow.add_edge("clarify_output", "answer_output")

    # 8. 多路搜索按意图分发（并行执行选中的分支）
    workflow.add_conditional_edges(
        "multi_search",
        route_search_paths,
        {
            "search_embedding": "search_embedding",
            "search_embedding_hyde": "search_embedding_hyde",
            "query_kg": "query_kg",
            "web_search_mcp": "web_search_mcp"
        }
    )

    # 9. 多路搜索汇合
    workflow.add_edge("search_embedding", "join")
    workflow.add_edge("search_embedding_hyde", "join")
    workflow.add_edge("query_kg", "join")
    workflow.add_edge("web_search_mcp", "join")

    # 10. 顺序边（rerank → 门控 → 答案输出）
    workflow.add_edge("join", "rrf")
    workflow.add_edge("rrf", "rerank")
    workflow.add_edge("rerank", "answer_gate")
    workflow.add_edge("answer_gate", "answer_output")
    workflow.add_edge("answer_output", END)

    # 11. 返回可运行的状态
    return workflow.compile()


# 创建全局图实例
query_app = create_query_graph()
