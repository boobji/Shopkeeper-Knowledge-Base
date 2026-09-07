"""澄清输出节点（P0-2）。

澄清从"终态"改成"循环"的落点：item_name_confirm 未确认商品时，
追问内容与候选 options 以结构化形式存入 Mongo 会话槽位（pending_clarify），
下一轮用户消息进来时由 item_name_confirm 优先消费该槽位。

本节点只负责把槽位持久化；答案推送、历史写入等由下游 answer_output 统一处理。
槽位持久化失败只告警不阻断——最坏情况退化为旧行为（一次性澄清）。
"""

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.mongo_history_util import save_pending_clarify


class ClarifyOutputNode(BaseNode):
    name = "clarify_output_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        session_id = state.get("session_id") or ""
        options = state.get("clarify_options") or []
        turn = int(state.get("clarify_turn") or 0)

        if session_id and options:
            ok = save_pending_clarify(
                session_id,
                options=options,
                turn=turn,
                last_query=state.get("original_query") or "",
            )
            if ok:
                self.log_step("clarify", f"澄清槽位已保存（第 {turn} 轮，{len(options)} 个候选）")
            else:
                self.logger.warning("澄清槽位保存失败，退化为一次性澄清")

        return {}
