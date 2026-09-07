"""P0 智能化改造测试：意图路由 / 澄清循环 / 质量门控 / 答案层升级。

覆盖：
- IntentRouteNode：意图解析、chitchat 直答、非法输出降级
- 澄清循环：selected_item 结构化锁定、槽位文本/序数词命中、轮次超限降级
- AnswerGateNode：空结果兜底、低分兜底、分数平平提示
- 答案层：建议问法解析、低置信提示附加、对比模板注入
- 主图路由函数 + 端到端（闲聊直答 / 澄清命中→兜底）
"""

import pytest

from knowledge.processor.query_process.state import create_default_state
from knowledge.processor.query_process.config import QueryConfig

# ============================================================
# 通用假件
# ============================================================

class FakeCollection:
    """内存版 Mongo collection（够 pending_clarify 用）。"""

    def __init__(self):
        self.docs = {}

    def find_one(self, query):
        return self.docs.get(query.get("session_id"))

    def update_one(self, query, update, upsert=False):
        sid = query.get("session_id")
        self.docs[sid] = {**(self.docs.get(sid) or {}), **update.get("$set", {})}
        return type("R", (), {"modified_count": 1})()

    def delete_many(self, query):
        sid = query.get("session_id")
        removed = 1 if sid in self.docs else 0
        self.docs.pop(sid, None)
        return type("R", (), {"deleted_count": removed})()


class FakeMongoTool:
    def __init__(self):
        self.chat_message = FakeCollection()
        self.chat_session = FakeCollection()


@pytest.fixture()
def fake_mongo(monkeypatch):
    """把 mongo_history_util 与相关节点的 Mongo 访问全部指向内存假件。"""
    from knowledge.utils import mongo_history_util as mhu
    from knowledge.processor.query_process.nodes import item_name_confirm_node as inc_node

    tool = FakeMongoTool()
    monkeypatch.setattr(mhu, "get_history_mongo_tool", lambda: tool)
    # item_name_confirm_node 以 from-import 方式引用，需打在节点模块命名空间
    monkeypatch.setattr(inc_node, "get_pending_clarify", mhu.get_pending_clarify)
    monkeypatch.setattr(inc_node, "clear_pending_clarify", mhu.clear_pending_clarify)
    monkeypatch.setattr(inc_node, "get_recent_messages", lambda *a, **k: [])
    monkeypatch.setattr(inc_node, "update_message_item_names", lambda *a, **k: 1)
    return tool


# ============================================================
# P0-1 意图路由
# ============================================================

class TestIntentRouteNode:
    def _node(self):
        from knowledge.processor.query_process.nodes.intent_route_node import IntentRouteNode
        return IntentRouteNode(config=QueryConfig())

    def test_parse_valid_output(self):
        node = self._node()
        assert node._parse('{"intent": "chitchat", "reply": "您好"}') == ("chitchat", "您好")
        assert node._parse('```json\n{"intent": "meta", "reply": ""}\n```') == ("meta", "")

    def test_parse_invalid_output_returns_none(self):
        node = self._node()
        assert node._parse("") is None
        assert node._parse("不是JSON") is None
        assert node._parse('{"intent": "unknown_intent"}') is None

    def test_process_chitchat_sets_answer(self):
        node = self._node()
        node._route = lambda q, t: ("chitchat", "您好，请问想咨询哪款产品？")
        result = node.process({"original_query": "你好", "task_id": "t1"})
        assert result["intent"] == "chitchat"
        assert result["answer"] == "您好，请问想咨询哪款产品？"

    def test_process_product_qa_no_answer(self):
        node = self._node()
        node._route = lambda q, t: ("product_qa", "")
        result = node.process({"original_query": "RS-12怎么测电压", "task_id": "t1"})
        assert result == {"intent": "product_qa"}

    def test_route_llm_failure_falls_back(self, monkeypatch):
        from knowledge.processor.query_process.nodes import intent_route_node as ir_node

        class BoomClient:
            def invoke(self, messages):
                raise RuntimeError("llm down")

        monkeypatch.setattr(ir_node, "get_llm_client", lambda **k: BoomClient())
        monkeypatch.setattr(ir_node, "log_llm_call", lambda *a, **k: None)
        intent, reply = self._node()._route("RS-12怎么测电压", "t1")
        assert intent == "product_qa"
        assert reply == ""

    def test_disabled_routes_product_qa(self, monkeypatch):
        node = self._node()
        monkeypatch.setattr(node.config, "intent_route_enabled", False)
        assert node.process({"original_query": "你好"}) == {"intent": "product_qa"}


class TestGraphRoutingFunctions:
    def test_route_after_intent(self):
        from knowledge.processor.query_process.main_graph import route_after_intent
        assert route_after_intent({"answer": "hi"}) == "answer_output"
        assert route_after_intent({"answer": "", "intent": "meta"}) == "item_name_confirm"

    def test_route_after_item_confirm(self):
        from knowledge.processor.query_process.main_graph import route_after_item_confirm
        assert route_after_item_confirm({"answer": "问", "clarify_options": ["A"]}) == "clarify_output"
        assert route_after_item_confirm({"answer": "问", "clarify_options": []}) == "answer_output"
        assert route_after_item_confirm({"answer": ""}) == "multi_search"

    def test_route_search_paths(self):
        from knowledge.processor.query_process.main_graph import route_search_paths
        assert route_search_paths({"intent": "meta"}) == ["web_search_mcp"]
        assert route_search_paths({"intent": "troubleshoot"}) == ["search_embedding", "query_kg"]
        assert set(route_search_paths({"intent": "product_qa"})) == {
            "search_embedding", "search_embedding_hyde", "query_kg", "web_search_mcp"}
        assert set(route_search_paths({})) == {
            "search_embedding", "search_embedding_hyde", "query_kg", "web_search_mcp"}


# ============================================================
# P0-2 澄清循环
# ============================================================

OPTIONS = ["RS-12 万用表", "RS-13 万用表"]


class TestMatchOption:
    def _node(self):
        from knowledge.processor.query_process.nodes.item_name_confirm_node import ItemNameConfirmNode
        return ItemNameConfirmNode(config=QueryConfig())

    def test_exact_and_contains_match(self):
        node = self._node()
        assert node._match_option("我要问的是RS-12万用表", OPTIONS) == "RS-12 万用表"
        assert node._match_option("rs13", OPTIONS) == "RS-13 万用表"

    def test_ordinal_match(self):
        node = self._node()
        assert node._match_option("第一个", OPTIONS) == "RS-12 万用表"
        assert node._match_option("第2个", OPTIONS) == "RS-13 万用表"
        assert node._match_option("2", OPTIONS) == "RS-13 万用表"

    def test_no_match(self):
        node = self._node()
        assert node._match_option(" coffee maker ", OPTIONS) == ""
        assert node._match_option("", OPTIONS) == ""


class TestClarifyLoop:
    def _node(self):
        from knowledge.processor.query_process.nodes.item_name_confirm_node import ItemNameConfirmNode
        return ItemNameConfirmNode(config=QueryConfig())

    def test_selected_item_locks_without_llm(self, fake_mongo, monkeypatch):
        from knowledge.processor.query_process.nodes import item_name_confirm_node as inc_node
        from knowledge.domain.item_name import ItemNameExtractor

        def _never_call(*a, **k):
            raise AssertionError("命中结构化选择后不应再调用 LLM 提取")

        monkeypatch.setattr(ItemNameExtractor, "extract_item_name", _never_call)
        fake_mongo.chat_session.docs["s1"] = {
            "session_id": "s1", "options": OPTIONS, "turn": 1, "last_query": "怎么校准"}

        state = create_default_state(original_query="RS-12 万用表", session_id="s1",
                                     task_id="t1", selected_item="RS-12 万用表")
        self._node().process(state)

        assert state["item_names"] == ["RS-12 万用表"]
        assert "怎么校准" in state["rewritten_query"]
        assert state["clarify_options"] == []
        assert fake_mongo.chat_session.docs.get("s1") is None  # 槽位已清除

    def test_pending_text_match_locks(self, fake_mongo, monkeypatch):
        from knowledge.domain.item_name import ItemNameExtractor

        monkeypatch.setattr(ItemNameExtractor, "extract_item_name",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("不应调用LLM")))
        fake_mongo.chat_session.docs["s1"] = {
            "session_id": "s1", "options": OPTIONS, "turn": 1, "last_query": "怎么校准"}

        state = create_default_state(original_query="我问的是rs-12万用表", session_id="s1", task_id="t1")
        self._node().process(state)

        assert state["item_names"] == ["RS-12 万用表"]
        assert "怎么校准" in state["rewritten_query"]

    def test_options_path_sets_clarify_state(self, fake_mongo, monkeypatch):
        from knowledge.processor.query_process.nodes import item_name_confirm_node as inc_node

        monkeypatch.setattr(inc_node.ItemNameExtractor, "extract_item_name",
                            lambda self, q, h: {"item_names": ["万用表"], "rewritten_query": q})
        monkeypatch.setattr(inc_node.ItemNameAligner, "match_align_filter",
                            lambda self, names: ([], OPTIONS))

        state = create_default_state(original_query="万用表", session_id="s1", task_id="t1")
        self._node().process(state)

        assert state["clarify_options"] == OPTIONS
        assert state["clarify_turn"] == 1
        assert "1. RS-12 万用表" in state["answer"]
        # clarify_output 负责落库
        from knowledge.processor.query_process.nodes.clarify_output_node import ClarifyOutputNode
        ClarifyOutputNode(config=QueryConfig()).process(state)
        assert fake_mongo.chat_session.docs["s1"]["options"] == OPTIONS
        assert fake_mongo.chat_session.docs["s1"]["turn"] == 1

    def test_turn_limit_degrades(self, fake_mongo, monkeypatch):
        from knowledge.processor.query_process.nodes import item_name_confirm_node as inc_node

        monkeypatch.setattr(inc_node.ItemNameExtractor, "extract_item_name",
                            lambda self, q, h: {"item_names": [], "rewritten_query": q})
        fake_mongo.chat_session.docs["s1"] = {
            "session_id": "s1", "options": OPTIONS, "turn": 2, "last_query": "x"}

        state = create_default_state(original_query="就是那个", session_id="s1", task_id="t1")
        self._node().process(state)

        assert state["clarify_options"] == []
        assert "完整型号" in state["answer"]
        assert fake_mongo.chat_session.docs.get("s1") is None  # 降级后清槽位

    def test_confirmed_clears_pending(self, fake_mongo, monkeypatch):
        from knowledge.processor.query_process.nodes import item_name_confirm_node as inc_node

        monkeypatch.setattr(inc_node.ItemNameExtractor, "extract_item_name",
                            lambda self, q, h: {"item_names": ["RS-12 万用表"], "rewritten_query": q})
        monkeypatch.setattr(inc_node.ItemNameAligner, "match_align_filter",
                            lambda self, names: (["RS-12 万用表"], []))
        fake_mongo.chat_session.docs["s1"] = {
            "session_id": "s1", "options": OPTIONS, "turn": 1, "last_query": "x"}

        state = create_default_state(original_query="RS-12怎么测电压", session_id="s1", task_id="t1")
        self._node().process(state)

        assert state["item_names"] == ["RS-12 万用表"]
        assert fake_mongo.chat_session.docs.get("s1") is None


# ============================================================
# P0-3 质量门控
# ============================================================

class TestAnswerGateNode:
    def _node(self):
        from knowledge.processor.query_process.nodes.answer_gate_node import AnswerGateNode
        return AnswerGateNode(config=QueryConfig())

    def test_empty_docs_fallback(self):
        state = create_default_state(item_names=["RS-12 万用表"], reranked_docs=[])
        self._node().process(state)
        assert "没有找到" in state["answer"]
        assert "RS-12" in state["answer"]

    def test_low_score_fallback(self):
        docs = [{"content": "x", "score": -1.5}]
        state = create_default_state(reranked_docs=docs)
        self._node().process(state)
        assert "没有找到" in state["answer"]

    def test_score_none_fallback(self):
        docs = [{"content": "x", "score": None}]
        state = create_default_state(reranked_docs=docs)
        self._node().process(state)
        assert "没有找到" in state["answer"]

    def test_medium_score_sets_notice(self):
        docs = [{"content": "x", "score": 1.0}]
        state = create_default_state(reranked_docs=docs)
        result = self._node().process(state)
        assert state["answer_notice"]
        assert "仅供参考" in state["answer_notice"]
        assert "answer" not in result or not result.get("answer")

    def test_high_score_no_notice(self):
        docs = [{"content": "x", "score": 4.0}]
        state = create_default_state(reranked_docs=docs)
        result = self._node().process(state)
        assert state["answer_notice"] == ""
        assert result == {}

    def test_existing_answer_skips_gate(self):
        state = create_default_state(answer="已有答案", reranked_docs=[])
        result = self._node().process(state)
        assert result == {}
        assert state["answer"] == "已有答案"


# ============================================================
# P0-4 答案层升级
# ============================================================

class TestAnswerLayer:
    def test_parse_suggestions(self):
        from knowledge.processor.query_process.nodes.answer_output_node import AnswerOutputNode
        assert AnswerOutputNode._parse_suggestions('{"suggestions": ["如何校准？", " ", "换电池步骤", "额外"]}') == \
            ["如何校准？", "换电池步骤", "额外"]
        assert AnswerOutputNode._parse_suggestions("not json") == []
        assert AnswerOutputNode._parse_suggestions('{"suggestions": "no"}') == []

    def test_append_notice_non_stream(self):
        from knowledge.processor.query_process.nodes.answer_output_node import AnswerOutputNode
        node = AnswerOutputNode(config=QueryConfig())
        state = create_default_state(task_id="t1", is_stream=False,
                                     answer="答案正文", answer_notice="仅供参考提示")
        node._append_notice(state)
        assert state["answer"].endswith("仅供参考提示")
        assert "\n\n仅供参考提示" in state["answer"]

    def test_prompt_injects_compare_hint(self):
        from knowledge.processor.query_process.nodes.answer_output_node import AnswerOutputNode
        node = AnswerOutputNode(config=QueryConfig())
        state = create_default_state(intent="compare", item_names=["A", "B"],
                                     original_query="A和B哪个好", reranked_docs=[],
                                     history=[], kg_triples=[])
        prompt = node._build_prompt(state)
        assert "对比" in prompt
        assert "适用建议" in prompt

    def test_prompt_no_hint_for_normal_intent(self):
        from knowledge.processor.query_process.nodes.answer_output_node import AnswerOutputNode
        node = AnswerOutputNode(config=QueryConfig())
        state = create_default_state(intent="product_qa", item_names=["A"],
                                     original_query="A怎么用", reranked_docs=[],
                                     history=[], kg_triples=[])
        prompt = node._build_prompt(state)
        assert "适用建议" not in prompt

    def test_answer_prompt_has_citation_requirement(self):
        from knowledge.prompts.query.query_prompt import ANSWER_PROMPT
        assert "[1]" in ANSWER_PROMPT and "来源编号" in ANSWER_PROMPT


# ============================================================
# 端到端（不走真实 LLM / 检索）
# ============================================================

@pytest.fixture()
def patched_graph_nodes(monkeypatch):
    """把四路检索全部打桩为空结果，rerank 模型不加载。"""
    from knowledge.processor.query_process.nodes import (
        vector_search_node, hyde_search_node, mcp_search_node, kg_search_node,
        intent_route_node, answer_output_node,
    )
    monkeypatch.setattr(vector_search_node.VectorSearchNode, "process", lambda self, s: {})
    monkeypatch.setattr(hyde_search_node.HyDeSearchNode, "process", lambda self, s: {})
    monkeypatch.setattr(mcp_search_node.McpSearchNode, "process", lambda self, s: {})
    monkeypatch.setattr(kg_search_node.KnowledgeGraphSearchNode, "process", lambda self, s: {})
    monkeypatch.setattr(intent_route_node, "log_llm_call", lambda *a, **k: None)
    monkeypatch.setattr(answer_output_node, "save_chat_message", lambda *a, **k: "mid")
    return monkeypatch


class TestEndToEnd:
    def test_chitchat_skips_search(self, fake_mongo, patched_graph_nodes, monkeypatch):
        from knowledge.processor.query_process.main_graph import query_app
        from knowledge.processor.query_process.nodes import intent_route_node as ir_node

        class FakeClient:
            def invoke(self, messages):
                return type("R", (), {"content": '{"intent": "chitchat", "reply": "您好，请问想咨询哪款产品？"}'})()

        monkeypatch.setattr(ir_node, "get_llm_client", lambda **k: FakeClient())

        result = query_app.invoke(create_default_state(
            original_query="你好呀", session_id="s-e2e-1", task_id="t-e2e-1"))

        assert result["answer"] == "您好，请问想咨询哪款产品？"
        assert result["intent"] == "chitchat"
        assert result["reranked_docs"] == []  # 未触发任何检索

    def test_clarify_hit_then_gate_fallback(self, fake_mongo, patched_graph_nodes, monkeypatch):
        from knowledge.processor.query_process.main_graph import query_app
        from knowledge.processor.query_process.nodes import intent_route_node as ir_node
        from knowledge.domain.item_name import ItemNameExtractor

        class FakeClient:
            def invoke(self, messages):
                return type("R", (), {"content": '{"intent": "product_qa", "reply": ""}'})()

        monkeypatch.setattr(ir_node, "get_llm_client", lambda **k: FakeClient())
        monkeypatch.setattr(ItemNameExtractor, "extract_item_name",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("不应调用LLM")))
        fake_mongo.chat_session.docs["s-e2e-2"] = {
            "session_id": "s-e2e-2", "options": OPTIONS, "turn": 1,
            "last_query": "怎么校准"}

        result = query_app.invoke(create_default_state(
            original_query="第一个", session_id="s-e2e-2", task_id="t-e2e-2"))

        # 命中澄清候选 → 锁定商品 → 检索为空 → 门控坦诚兜底（而非强行作答）
        assert result["item_names"] == ["RS-12 万用表"]
        assert result["reranked_docs"] == []
        assert "没有找到" in result["answer"]
