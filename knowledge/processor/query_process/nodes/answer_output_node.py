"""答案输出节点（P0-4 升级：客服人设 + 引用编号 + 建议问法 + 低置信提示）"""

import json
import time
from typing import List, Dict, Tuple
from langchain_core.messages import HumanMessage
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompts.query.query_prompt import ANSWER_PROMPT, COMPARE_HINT, SUGGEST_QUESTIONS_PROMPT
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.utils.llm_call_logger import log_llm_call
from knowledge.utils.task_util import set_task_result
from knowledge.utils.sse_util import push_sse_event, SSEEvent
from knowledge.utils.mongo_history_util import save_chat_message


class AnswerOutputNode(BaseNode):
    name = "answer_output_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:

        task_id = state.get("task_id")
        is_stream = state.get("is_stream")

        # 1. 已有答案 → 直接返回
        if state.get("answer"):
            self._push_existing_answer(state)

        # 2. 构建提示词 → 调用 LLM 生成答案
        else:
            prompt = self._build_prompt(state)
            state["prompt"] = prompt
            self._generate_answer(state, prompt)
            self._append_notice(state)
            self._generate_suggestions(state)

        # 3. 写入历史记录（用户问题 + 助手回答）
        self._write_history(state)

        # 4. 流式模式发送结束事件
        if is_stream:
            push_sse_event(task_id, SSEEvent.FINAL,
                           {"answer": state.get("answer", "")})
            if state.get("suggestions"):
                push_sse_event(task_id, "suggestions",
                               {"suggestions": state["suggestions"]})
        return state

    def _push_existing_answer(self, state: QueryGraphState):
        """非流式模式：存入任务结果；流式模式：让 FINAL 统一推送。"""
        if not state.get("is_stream"):
            set_task_result(state["task_id"], "answer", state["answer"])

    def _generate_answer(self, state, prompt):
        self.log_step("generate", "生成答案")
        llm_client = get_llm_client()

        task_id = state["task_id"]

        t0 = time.perf_counter()
        if state.get("is_stream"):
            state["answer"] = self._stream_generate(llm_client, prompt, task_id)
        else:
            state["answer"] = self._invoke_generate(prompt)
            set_task_result(task_id, "answer", state["answer"])

        # 留档：完整提示词（含检索上下文/图谱关系/历史）+ 最终答案，复盘 badcase 的关键证据
        log_llm_call('query_answer', prompt=prompt, answer=state.get("answer", ""),
                     task_id=task_id, latency_ms=(time.perf_counter() - t0) * 1000,
                     meta={'stream': bool(state.get("is_stream"))})

    def _build_prompt(self, state: QueryGraphState) -> str:
        char_budget = self.config.max_context_chars

        # 1. 获取问题和商品名
        question = state.get("rewritten_query") or state.get("original_query", "")
        item_names = state["item_names"]

        # 1.1 意图补充要求：对比类问题用对比模板引导分点作答
        intent_hint = COMPARE_HINT if state.get("intent") == "compare" else ""

        # 2. 格式化上下文文档
        context_str, char_budget = self._format_reranked_docs(
            state.get("reranked_docs") or [], char_budget
        )

        # 3. 格式化历史对话
        history_str, char_budget = self._format_chat_history(
            state.get("history") or [], char_budget
        )

        # 4. 格式化图谱关系
        graph_str, char_budget = self._format_kg_triples(
            state.get("kg_triples") or [], char_budget
        )

        # 5. 组装提示词
        return ANSWER_PROMPT.format(
            context=context_str or "无参考内容",
            history=history_str if history_str else "暂无历史对话",
            item_names=", ".join(item_names),
            graph_relation_description=graph_str or "无图谱关系",
            question=question,
            intent_hint=intent_hint,
        )

    def _append_notice(self, state: QueryGraphState):
        """低置信检索提示（answer_gate 产出）：附加到答案尾部。

        流式模式在流结束后补推一条 delta，保证前端展示完整。
        """
        notice = (state.get("answer_notice") or "").strip()
        if not notice or not state.get("answer"):
            return
        state["answer"] = f"{state['answer']}\n\n{notice}"
        if state.get("is_stream"):
            push_sse_event(state["task_id"], "delta", {"delta": f"\n\n{notice}"})

    def _generate_suggestions(self, state: QueryGraphState):
        """生成 3 个后续建议问法（轻量调用，失败静默跳过）。

        非流式：写入 task_result 供接口返回；流式：额外推送 suggestions 事件。
        """
        try:
            question = state.get("rewritten_query") or state.get("original_query", "")
            prompt = SUGGEST_QUESTIONS_PROMPT.format(
                item_names="、".join(state.get("item_names") or []) or "未知商品",
                question=question,
                answer=(state.get("answer") or "")[:800],
            )
            llm_client = get_llm_client(response_format=True)
            t0 = time.perf_counter()
            response = llm_client.invoke([HumanMessage(content=prompt)])
            latency_ms = (time.perf_counter() - t0) * 1000

            suggestions = self._parse_suggestions(response.content)
            # 留档：建议问法生成（低频轻量调用，复盘时按需查看）
            log_llm_call('query_suggest', prompt=prompt,
                         answer=getattr(response, "content", ""),
                         task_id=state.get("task_id") or "", latency_ms=latency_ms,
                         meta={'stage': 'suggest_questions'})

            if suggestions:
                state["suggestions"] = suggestions
                if not state.get("is_stream"):
                    set_task_result(state["task_id"], "suggestions", suggestions)
            self.log_step("suggest", f"生成建议问法 {len(suggestions)} 条")
        except Exception as e:
            self.logger.warning(f"生成建议问法失败（已忽略）: {e}")

    @staticmethod
    def _parse_suggestions(llm_content: str) -> List[str]:
        """解析建议问法 JSON 输出；非法输出返回空列表。"""
        from knowledge.domain.llm_parse import strip_json_fence

        try:
            data = json.loads(strip_json_fence((llm_content or "").strip()))
        except (json.JSONDecodeError, ValueError):
            return []
        raw = data.get("suggestions")
        if not isinstance(raw, list):
            return []
        cleaned = [s.strip() for s in raw if isinstance(s, str) and s.strip()]
        return cleaned[:3]

    def _format_chat_history(self, chat_history: List[Dict], char_budget: int) -> Tuple[str, int]:
        formatted_lines = []
        used_chars = 0

        role_label_map = {"user": "用户", "assistant": "助手"}

        for message in chat_history:
            role = message.get("role", "")
            text = message.get("text", "")
            if not text or role not in role_label_map:
                continue

            formatted_line = f"{role_label_map[role]}: {text}"
            used_chars += len(formatted_line) + 1

            if used_chars > char_budget:
                return "\n".join(formatted_lines), char_budget - used_chars

            formatted_lines.append(formatted_line)

        return "\n".join(formatted_lines), char_budget - used_chars

    def _format_reranked_docs(self, reranked_docs: List[Dict], char_budget: int) -> Tuple[str, int]:
        formatted_lines = []
        used_chars = 0

        for idx, doc in enumerate(reranked_docs, 1):
            content = doc.get("content", "").strip()
            if not content:
                continue

            meta_tags = [f"[{idx}]"]
            for field, template in [
                ("source", "[source={}]"), ("chunk_id", "[chunk_id={}]"),
                ("url", "[url={}]"), ("title", "[title={}]"),
            ]:
                field_value = str(doc.get(field, "")).strip()
                if field_value:
                    meta_tags.append(template.format(field_value))

            relevance_score = doc.get("score")
            if relevance_score is not None:
                meta_tags.append(f"[score={float(relevance_score):.4f}]")

            doc_entry = " ".join(meta_tags) + "\n" + content

            if used_chars + len(doc_entry) > char_budget:
                break

            formatted_lines.append(doc_entry)
            used_chars += len(doc_entry) + 2

        return "\n\n".join(formatted_lines), char_budget - used_chars

    @staticmethod
    def _format_kg_triples(kg_triples: List, char_budget: int) -> Tuple[str, int]:
        formatted_lines = []
        used_chars = 0
        for triple in kg_triples:
            triple_text = (str(triple) if triple is not None else "").strip()
            if not triple_text:
                continue
            if used_chars + len(triple_text) > char_budget:
                break
            formatted_lines.append(triple_text)
            used_chars += len(triple_text) + 1
        return "\n".join(formatted_lines), char_budget - used_chars

    def _invoke_generate(self, prompt: str) -> str:
        self.log_step("generate", "生成答案")
        llm_client = get_llm_client()
        try:
            response = llm_client.invoke(prompt)
            return response.content
        except Exception as e:
            self.logger.error(f"生成回答出错: {e}")
            return "抱歉，生成回答时出现错误。"

    def _stream_generate(self, llm_client, prompt, task_id):
        """流式生成，逐 chunk 推送 delta 事件。
         返回的是一个一个token(不是一个中文字符就是一个token )
        """
        accumulated_answer = ""
        try:
            for chunk in llm_client.stream(prompt):
                delta_text = getattr(chunk, "content", "") or ""
                if delta_text:
                    accumulated_answer += delta_text
                    push_sse_event(task_id, "delta", {"delta": delta_text})
        except Exception as e:
            self.logger.error(f"流式生成出错: {e}")
        return accumulated_answer

        # ★ 新增方法

    def _write_history(self, state: QueryGraphState):
        session_id = state["session_id"]
        rewritten_query = state.get("rewritten_query", "") or state.get("original_query", "")
        item_names = state.get("item_names") or []
        try:
            # 1. 写用户问题
            save_chat_message(
                session_id=session_id,
                role="user",
                text=state["original_query"],
                rewritten_query=rewritten_query,
                item_names=item_names,
            )
            # 2. AI回复（假的+真的）
            if state.get("answer"):
                save_chat_message(
                    session_id=session_id,
                    role="assistant",
                    text=state["answer"],  # 模型的输出
                    rewritten_query=rewritten_query,
                    item_names=item_names,
                )
        except Exception as e:
            self.logger.warning(f"写入历史记录失败: {e}")
