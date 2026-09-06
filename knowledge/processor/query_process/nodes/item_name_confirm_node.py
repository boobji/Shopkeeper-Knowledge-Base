"""商品名确认节点（主编排器）。

提取与对齐组件已拆分至 knowledge/domain/item_name.py：
  ItemNameExtractor   LLM 提取商品名 + 重写问题
  ItemNameAligner     向量库对齐 + 阈值分桶 + 分差过滤
本文件只保留 LangGraph 节点编排逻辑。
"""

from typing import List

from knowledge.domain.item_name import ItemNameAligner, ItemNameExtractor
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.mongo_history_util import get_recent_messages, update_message_item_names


class ItemNameConfirmNode(BaseNode):
    name = "item_name_confirm_node"

    def __init__(self):
        super().__init__()
        self._item_name_extractor = ItemNameExtractor()
        self._item_name_aligner = ItemNameAligner(collection_name=self.config.item_name_collection)

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 获取用户的原始问题
        original_query = state.get("original_query")
        session_id = state.get('session_id')

        # 2. 构建历史对话
        chat_history = get_recent_messages(session_id, limit=10)
        history_text = ""
        for msg in chat_history:
            role = msg.get("role")
            content = msg.get("text", "")
            history_text += f"{role}: {content}\n"

        # 3. 调用LLM提取商品名（本质：是如果直接基于用户的原始问题进行检索，质量很差。而我们实际需要的是明白用户真正想问你的商品是谁。）
        clean_llm_result = self._item_name_extractor.extract_item_name(original_query, history_text)
        # 3.1 获取item_names
        item_names = clean_llm_result.get('item_names')
        # 3.2 获取rewritten_query
        rewritten_query = clean_llm_result.get('rewritten_query')

        if item_names:
            # 4. 查询向量数据库&&过滤(评分对齐&分数差异过滤)
            confirmed, options = self._item_name_aligner.match_align_filter(item_names)
        else:
            confirmed, options = [], []

        # 5. 决定state的key值（继续、结束）修改state
        self._decide(state, item_names, confirmed, options, rewritten_query)

        if confirmed:
            ids_to_update = [
                str(msg["_id"]) for msg in chat_history if not msg.get("item_names")
            ]
            if ids_to_update:
                try:
                    update_message_item_names(ids_to_update, confirmed)
                except Exception as e:
                    self.logger.warning(f"回填历史 item_names 失败: {e}")

        # 将历史对话写入 state，供下游 answer_output 使用
        state["history"] = chat_history

        return state

    def _decide(self, state: QueryGraphState, item_names: List[str], confirmed: List[str],
                options: List[str], rewritten_query: str):

        if confirmed:
            state['rewritten_query'] = rewritten_query
            state['item_names'] = confirmed

        elif options:
            state['answer'] = (f"我不确定您指的是哪款产品。"
                               f"您是在询问以下产品吗：{'、'.join(options)}？")
        else:
            state['answer'] = "抱歉，我无法识别您询问的具体产品名称，请提供更准确的产品名称或型号。"
