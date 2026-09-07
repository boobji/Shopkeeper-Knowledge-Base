"""查询流程状态类型定义

定义完整的查询状态结构和辅助函数。
"""

from typing import TypedDict, List, Dict
import copy


class QueryGraphState(TypedDict):
    """
    Represents the state of our query graph.
    Attributes:
    各个属性的结构
    """
    session_id: str # 会话ID
    task_id:str # 任务ID
    message_id: str # 消息ID
    original_query: str # 原始查询
    embedding_chunks: list # 已向量化的切片
    hyde_embedding_chunks: list # 已向量化的假设性问题切片
    rrf_chunks: list # rrf排序后的切片
    web_search_docs: list # 搜索结果
    reranked_docs: list  # 排序后的文档
    prompt: str  #提示词
    answer: str #答案
    item_names: List[str] # 商品名称
    rewritten_query: str  #重写答案
    history: list   # 历史对话
    is_stream: bool # 是否流式输出
    kg_chunks: list # 知识图谱切片
    kg_triples: list # 知识图谱关系
    # ==================== P0 智能化改造新增 ====================
    selected_item: str  # 前端点击澄清选项后回传的结构化商品名（优先级最高）
    intent: str  # 意图标签：chitchat/meta/troubleshoot/compare/product_qa
    clarify_options: List[str]  # 本次需要向用户澄清的候选商品（非空时走 clarify_output）
    clarify_turn: int  # 澄清轮次（从 1 计数，超过 clarify_max_turns 降级）
    answer_notice: str  # 低置信检索提示（附加在答案尾部）
    suggestions: List[str]  # 生成答案后的 3 个建议问法


# ==================== 默认状态 ====================

DEFAULT_STATE: QueryGraphState = {
    "session_id": "",               # 会话ID
    "task_id": "",               # 任务ID
    "message_id": "",               # 消息ID
    "original_query": "",           # 原始查询
    "embedding_chunks": [],         # 已向量化的切片
    "hyde_embedding_chunks": [],    # 已向量化的假设性问题切片
    "rrf_chunks": [],               # rrf排序后的切片
    "web_search_docs": [],          # 搜索结果
    "reranked_docs": [],            # 排序后的文档
    "prompt": "",                   # 提示词
    "answer": "",                   # 答案
    "item_names": [],               # 商品名称
    "rewritten_query": "",          # 重写查询
    "history": [],                  # 历史对话
    "is_stream": False,             # 是否流式输出 (默认设为 False)
    "kg_chunks": [],                # 知识图谱切片
    "kg_triples": [],               # 知识图谱关系
    "selected_item": "",            # 结构化商品选择
    "intent": "product_qa",         # 意图标签（默认产品咨询）
    "clarify_options": [],          # 澄清候选商品
    "clarify_turn": 0,              # 澄清轮次
    "answer_notice": "",            # 低置信检索提示
    "suggestions": []               # 建议问法
}

def create_default_state(**overrides) -> QueryGraphState:
    """创建默认状态，支持字段覆盖。

    Args:
        **overrides: 要覆盖的字段键值对。

    Returns:
        新的状态实例，包含默认值和覆盖值。

    """
    state = copy.deepcopy(DEFAULT_STATE)
    state.update(overrides)
    return state


def get_default_state() -> QueryGraphState:
    """获取默认状态副本。

    Returns:
        状态副本，避免修改全局默认值。
    """
    return copy.deepcopy(DEFAULT_STATE)


graph_default_state = DEFAULT_STATE
