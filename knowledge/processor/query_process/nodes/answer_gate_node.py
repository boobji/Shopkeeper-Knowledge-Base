"""检索质量门控节点（P0-3）。

rerank → answer_output 之间加一道判断（复用 rerank 已有的分数）：
- reranked_docs 为空 / top1 分数低于 answer_confidence_floor：
  → 坦诚兜底模板（说明没找到 + 建议问法 + 引导提供型号），不调 LLM 强行作答；
- 命中但分数低于 answer_warn_score：
  → 在答案层附"仅供参考，建议核对说明书原文"提示（answer_notice）。

分数说明：bge-reranker-v2-m3 分数量纲约 -11 ~ +5，阈值可通过环境变量校准。
"""

from typing import Dict, Any, Optional

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState


class AnswerGateNode(BaseNode):
    name = "answer_gate_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 已有答案（闲聊直答/澄清追问等）不参与门控
        if state.get("answer"):
            return {}

        reranked_docs = state.get("reranked_docs") or []
        top1 = self._top1_score(reranked_docs)

        # 兜底一：没有可用参考内容（空列表或分数缺失视为不可信）
        if not reranked_docs or top1 is None:
            state["answer"] = self._fallback_answer(state)
            self.log_step("answer_gate", "检索结果为空/不可信 → 坦诚兜底")
            return state

        floor = self.config.answer_confidence_floor
        warn = self.config.answer_warn_score

        # 兜底二：有文档但最高分低于置信下限
        if top1 < floor:
            state["answer"] = self._fallback_answer(state)
            self.log_step("answer_gate", f"top1={top1:.4f} 低于置信下限 {floor} → 坦诚兜底")
            return state

        # 提示：分数平平（能答但不够硬），提示语由 answer_output 附加
        if top1 < warn:
            state["answer_notice"] = (
                "⚠️ 以上内容基于知识库检索结果生成，相关度一般，"
                "仅供参考，建议核对说明书原文。"
            )
            self.log_step("answer_gate", f"top1={top1:.4f} 低于提示阈值 {warn} → 附加仅供参考提示")

        return {}

    def _top1_score(self, reranked_docs: list) -> Optional[float]:
        """取 top1 分数（rerank 输出已按分数降序）。"""
        if not reranked_docs:
            return None
        first = reranked_docs[0]
        if not isinstance(first, dict):
            return None
        score = first.get("score")
        if score is None:
            return None
        try:
            return float(score)
        except (TypeError, ValueError):
            return None

    def _fallback_answer(self, state: QueryGraphState) -> str:
        """坦诚兜底：不编造，说明没找到并引导用户换问法/提供型号。"""
        item_names = state.get("item_names") or []
        product_hint = f"关于「{'、'.join(item_names)}」" if item_names else "您的问题"
        return (
            f"抱歉，我在知识库里没有找到足以回答{product_hint}的可靠资料，"
            f"为了不误导您，先不强行作答。您可以试试：\n"
            f"1. 换一种问法，聚焦具体功能或步骤（如：XX如何校准）；\n"
            f"2. 提供说明书上的完整产品型号（如：RS-12 万用表）；\n"
            f"3. 咨询其他产品的问题，我随时为您检索。"
        )
