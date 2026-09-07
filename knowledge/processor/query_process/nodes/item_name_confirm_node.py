"""商品名确认节点（主编排器）。

提取与对齐组件已拆分至 knowledge/domain/item_name.py：
  ItemNameExtractor   LLM 提取商品名 + 重写问题
  ItemNameAligner     向量库对齐 + 阈值分桶 + 分差过滤
本文件只保留 LangGraph 节点编排逻辑。

P0-2 澄清循环：未确认商品时不再写入罐头回复直接结束，而是带上
clarify_options 走 clarify_output 节点，把候选以结构化形式存入 Mongo
会话槽位（pending_clarify）；下一轮用户消息进来时优先消费：
  1. 前端点击选项回传 selected_item（结构化，优先级最高）→ 直接锁定；
  2. 用户回复文本命中槽位候选（精确/包含/序数词匹配）→ 直接锁定；
  3. 未命中 → 走正常 LLM 提取；槽位轮次超上限后降级为"请提供完整型号"。
"""

import re
from typing import Dict, List

from knowledge.domain.item_name import ItemNameAligner, ItemNameExtractor
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.mongo_history_util import (
    clear_pending_clarify,
    get_pending_clarify,
    get_recent_messages,
    update_message_item_names,
)

# 序数词回复映射（"第一个"/"第2个"/"2" → 候选下标）
_ORDINAL_MAP = {"一": 0, "两": 1, "二": 1, "三": 2, "四": 3, "五": 4,
                "1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
_ORDINAL_RE = re.compile(r"^第?\s*([一二两三四五1-5])\s*[个条款]?$")


class ItemNameConfirmNode(BaseNode):
    name = "item_name_confirm_node"

    def __init__(self, config=None):
        super().__init__(config=config)
        self._item_name_extractor = ItemNameExtractor()
        self._item_name_aligner = ItemNameAligner(collection_name=self.config.item_name_collection)

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 基础输入 + 历史对话（后续所有分支共用）
        original_query = state.get("original_query")
        session_id = state.get('session_id')
        chat_history = get_recent_messages(session_id, limit=10) if session_id else []
        state["history"] = chat_history

        pending = get_pending_clarify(session_id) if session_id else {}

        # 2. 澄清循环消费：结构化选择优先，其次槽位文本匹配
        selected_item = (state.get("selected_item") or "").strip()
        if selected_item:
            self._lock_item(state, selected_item, pending,
                            rewritten=original_query, history_backfill=chat_history)
            return state

        matched = self._match_option(original_query, pending.get("options") or []) if pending else ""
        if matched:
            self._lock_item(state, matched, pending,
                            rewritten=original_query, history_backfill=chat_history)
            return state

        # 3. 正常流程：LLM 提取商品名 + 重写问题
        history_text = ""
        for msg in chat_history:
            role = msg.get("role")
            content = msg.get("text", "")
            history_text += f"{role}: {content}\n"

        clean_llm_result = self._item_name_extractor.extract_item_name(original_query, history_text)
        item_names = clean_llm_result.get('item_names')
        rewritten_query = clean_llm_result.get('rewritten_query')

        if item_names:
            # 4. 查询向量数据库&&过滤(评分对齐&分数差异过滤)
            confirmed, options = self._item_name_aligner.match_align_filter(item_names)
        else:
            confirmed, options = [], []

        # 5. 决定state的key值（继续、澄清、结束）
        self._decide(state, item_names, confirmed, options, rewritten_query, pending)

        if confirmed:
            ids_to_update = [
                str(msg["_id"]) for msg in chat_history if not msg.get("item_names")
            ]
            if ids_to_update:
                try:
                    update_message_item_names(ids_to_update, confirmed)
                except Exception as e:
                    self.logger.warning(f"回填历史 item_names 失败: {e}")

        return state

    # ==================== 澄清循环辅助 ====================

    def _lock_item(self, state: QueryGraphState, item: str, pending: Dict,
                   rewritten: str, history_backfill: List[Dict]):
        """锁定商品（结构化点击 / 槽位命中）：跳过 LLM 提取，清理槽位。"""
        session_id = state.get("session_id") or ""
        state['item_names'] = [item]
        last_query = (pending or {}).get("last_query") or ""
        if last_query and rewritten and self._normalize(rewritten) != self._normalize(last_query):
            # 用户在澄清后给了新说法，用"原问题 + 锁定商品"拼出完整独立问题
            state['rewritten_query'] = f"{last_query}（商品：{item}）"
        elif rewritten and rewritten.strip():
            state['rewritten_query'] = rewritten
        else:
            state['rewritten_query'] = item

        state['clarify_options'] = []
        state['clarify_turn'] = 0
        if session_id:
            clear_pending_clarify(session_id)

        ids_to_update = [
            str(msg["_id"]) for msg in (history_backfill or []) if not msg.get("item_names")
        ]
        if ids_to_update:
            try:
                update_message_item_names(ids_to_update, [item])
            except Exception as e:
                self.logger.warning(f"回填历史 item_names 失败: {e}")

        self.log_step("clarify", f"澄清命中，锁定商品：{item}")

    def _decide(self, state: QueryGraphState, item_names: List[str], confirmed: List[str],
                options: List[str], rewritten_query: str, pending: Dict = None):
        pending = pending or {}
        session_id = state.get("session_id") or ""

        if confirmed:
            state['rewritten_query'] = rewritten_query
            state['item_names'] = confirmed
            state['clarify_options'] = []
            # 已确认商品，历史澄清槽位作废
            if session_id:
                clear_pending_clarify(session_id)

        elif options:
            # 未确认但有候选 → 追问（clarify_output 负责持久化槽位）
            state['clarify_options'] = options
            state['clarify_turn'] = int(pending.get("turn") or 0) + 1
            numbered = "\n".join(f"{i}. {opt}" for i, opt in enumerate(options, 1))
            state['answer'] = (f"我不确定您指的是哪款产品，"
                               f"请回复编号或产品名称：\n{numbered}")

        else:
            # 完全没识别出商品：槽位还有候选且轮次未用完 → 重新出示候选
            pending_options = pending.get("options") or []
            pending_turn = int(pending.get("turn") or 0)
            if pending_options and pending_turn < self.config.clarify_max_turns:
                state['clarify_options'] = pending_options
                state['clarify_turn'] = pending_turn + 1
                numbered = "\n".join(f"{i}. {opt}" for i, opt in enumerate(pending_options, 1))
                state['answer'] = (f"我还是没能识别出您询问的产品。"
                                   f"您是想问以下哪款产品呢？请回复编号或名称：\n{numbered}")
            else:
                # 无槽位（首次）或轮次已用完 → 降级为请提供完整型号
                state['answer'] = ("抱歉，我无法识别您询问的具体产品名称，"
                                   "请提供产品说明书上的完整型号（如：RS-12 万用表）。")
                state['clarify_options'] = []
                if pending_options and session_id:
                    clear_pending_clarify(session_id)

    @staticmethod
    def _normalize(text: str) -> str:
        """轻量归一化：去空白/连字符/下划线，转小写。"""
        return re.sub(r"[\s\-_]+", "", str(text or "")).lower()

    def _match_option(self, query: str, options: List[str]) -> str:
        """判断用户回复是否命中某个澄清候选。

        匹配顺序：序数词回复（"第一个"/"第2个"/"2"）→ 名称包含匹配（双向）。
        命中返回候选原名，未命中返回空字符串。

        说明：方案中的"向量对齐到 option"暂不实现——候选一般只有 2~3 个，
        轻量文本匹配已覆盖绝大多数回复，省一次嵌入模型调用与延迟；
        后续若实测未命中率偏高，再升级为向量对齐。
        """
        if not options:
            return ""
        q = self._normalize(query)
        if not q:
            return ""

        # 1. 序数词回复
        m = _ORDINAL_RE.match(q)
        if m:
            idx = _ORDINAL_MAP.get(m.group(1))
            if idx is not None and idx < len(options):
                return options[idx]

        # 2. 名称包含匹配（双向）
        for opt in options:
            o = self._normalize(opt)
            if o and (o in q or q in o):
                return opt

        return ""
