"""意图路由节点（P0-1）。

一次轻量 LLM 调用输出结构化意图：
  chitchat     闲聊/寒暄       → 直接用同一次调用返回的 reply，跳过全部检索
  meta         售后政策/元问题   → 只走 Web 搜索（知识库里没有这类内容）
  troubleshoot 故障排查         → 图谱 + 向量（跳过 HyDE 与 Web）
  compare      对比咨询         → 全流程 + 答案层对比模板
  product_qa   产品咨询（默认） → 现有全流程

LLM 调用失败/输出非法时降级为 product_qa，保证查询永不中断。
"""

import json
import time
from json import JSONDecodeError

from langchain_core.messages import HumanMessage

from knowledge.domain.llm_parse import strip_json_fence
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompts.query.query_prompt import INTENT_ROUTE_TEMPLATE
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.utils.llm_call_logger import log_llm_call

VALID_INTENTS = ("chitchat", "meta", "troubleshoot", "compare", "product_qa")


class IntentRouteNode(BaseNode):
    name = "intent_route_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 意图路由可通过 INTENT_ROUTE_ENABLED=0 关闭（直连旧全流程）
        if not self.config.intent_route_enabled:
            return {"intent": "product_qa"}

        original_query = state.get("original_query") or ""
        intent, reply = self._route(original_query, state.get("task_id") or "")

        updates = {"intent": intent}
        if intent == "chitchat":
            # 直接回复，路由到 answer_output 走已有答案分支（历史记录照常写入）
            updates["answer"] = reply or "您好，很高兴为您服务～请问您想咨询哪款产品呢？"

        return updates

    def _route(self, original_query: str, task_id: str):
        """调用 LLM 做意图分类。

        Returns:
            (intent, reply) 元组；reply 仅 chitchat 有值。
        """
        empty_reply = ("product_qa", "")
        if not original_query.strip():
            return empty_reply

        llm_client = get_llm_client(response_format=True)
        human_prompt = INTENT_ROUTE_TEMPLATE.format(query=original_query)

        t0 = time.perf_counter()
        try:
            response = llm_client.invoke([HumanMessage(content=human_prompt)])
            llm_content = (response.content or "").strip()
        except Exception as e:
            self.logger.warning(f"意图路由 LLM 调用失败，降级为 product_qa: {e}")
            return empty_reply

        latency_ms = (time.perf_counter() - t0) * 1000
        # 留档：意图分类的输入输出（复盘误路由 badcase 用）
        log_llm_call('query_intent', prompt=human_prompt, answer=llm_content,
                     task_id=task_id, latency_ms=latency_ms,
                     meta={'stage': 'intent_route'})

        parsed = self._parse(llm_content)
        if parsed is None:
            return empty_reply

        intent, reply = parsed
        self.log_step("intent_route", f"意图识别: {intent}")
        return intent, reply

    def _parse(self, llm_content: str):
        """解析 LLM 输出；非法输出返回 None（由调用方降级）。"""
        if not llm_content:
            return None
        try:
            data = json.loads(strip_json_fence(llm_content))
        except JSONDecodeError as e:
            self.logger.warning(f"意图路由输出解析失败: {e}")
            return None

        intent = data.get("intent")
        if intent not in VALID_INTENTS:
            return None

        reply = data.get("reply") or ""
        return intent, (reply.strip() if isinstance(reply, str) else "")
