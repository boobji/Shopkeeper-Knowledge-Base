"""回归测试：并行检索节点的失败路径与查询失败时的 SSE 行为。

背景：vector/hyde/mcp 三个节点在四路并行分支上，失败时曾 `return state`
（整个 state），导致与其他并行节点并发写 session_id 等键，
LangGraph 抛 InvalidUpdateError（INVALID_CONCURRENT_GRAPH_UPDATE），整轮查询中断。
"""

import json

import pytest


import knowledge.processor.query_process.nodes.hyde_search_node as hyde_mod
import knowledge.processor.query_process.nodes.vector_search_node as vector_mod
from knowledge.processor.query_process.nodes.hyde_search_node import HyDeSearchNode
from knowledge.processor.query_process.nodes.mcp_search_node import McpSearchNode
from knowledge.processor.query_process.nodes.vector_search_node import VectorSearchNode
from knowledge.utils.sse_util import get_sse_queue
from knowledge.utils.task_util import get_task_status

VALID_INPUTS = {"rewritten_query": "如何测量电压", "item_names": ["RS-12"]}


class TestParallelNodesReturnIncrementalUpdates:
    """失败/空结果时必须返回 {}（增量更新），而不是整个 state。"""

    def test_vector_missing_embedding_model_raises(self, monkeypatch):
        """嵌入模型不可用时应快速抛异常（任务失败），而非静默返回空结果。"""
        from knowledge.core.exceptions import EmbeddingError

        def raise_emb(*args, **kwargs):
            raise EmbeddingError("BGE-M3 加载失败")

        monkeypatch.setattr(vector_mod, "get_bge_m3_embedding_model", raise_emb)
        with pytest.raises(EmbeddingError):
            VectorSearchNode().process(dict(VALID_INPUTS))

    def test_vector_no_search_hits(self, monkeypatch):
        monkeypatch.setattr(vector_mod, "get_bge_m3_embedding_model", lambda: object())
        monkeypatch.setattr(vector_mod, "get_milvus_client", lambda: object())
        monkeypatch.setattr(
            vector_mod, "generate_hybrid_embeddings",
            lambda model, embedding_documents: {"dense": [[0.0]], "sparse": [{1: 1.0}]},
        )
        monkeypatch.setattr(vector_mod, "execute_hybrid_search_query", lambda **kwargs: [[]])
        assert VectorSearchNode().process(dict(VALID_INPUTS)) == {}

    def test_hyde_missing_embedding_model_raises(self, monkeypatch):
        from knowledge.core.exceptions import EmbeddingError

        def raise_emb(*args, **kwargs):
            raise EmbeddingError("BGE-M3 加载失败")

        monkeypatch.setattr(hyde_mod, "get_bge_m3_embedding_model", raise_emb)
        with pytest.raises(EmbeddingError):
            HyDeSearchNode().process(dict(VALID_INPUTS))

    def test_hyde_no_search_hits(self, monkeypatch):
        monkeypatch.setattr(hyde_mod, "get_bge_m3_embedding_model", lambda: object())
        monkeypatch.setattr(hyde_mod, "get_milvus_client", lambda: object())
        monkeypatch.setattr(
            hyde_mod, "generate_hybrid_embeddings",
            lambda model, embedding_documents: {"dense": [[0.0]], "sparse": [{1: 1.0}]},
        )
        monkeypatch.setattr(hyde_mod, "execute_hybrid_search_query", lambda *args, **kwargs: [[]])
        assert HyDeSearchNode().process(dict(VALID_INPUTS)) == {}

    def test_mcp_empty_result(self, monkeypatch):
        async def fake_search(self, query):
            return []

        monkeypatch.setattr(McpSearchNode, "_create_execute_web_search", fake_search)
        assert McpSearchNode().process(dict(VALID_INPUTS)) == {}

    def test_state_key_not_written_on_failure(self, monkeypatch):
        """返回 {} 时不得携带 session_id 等任何键（并发写会炸图）。"""
        monkeypatch.setattr(vector_mod, "get_bge_m3_embedding_model", lambda: None)
        result = VectorSearchNode().process({**VALID_INPUTS, "session_id": "s-1", "task_id": "t-1"})
        assert "session_id" not in result


class TestQueryServiceErrorEvent:
    def test_failed_query_marks_failed_and_pushes_error(self, monkeypatch):
        from knowledge.services import query_service as qs_mod

        def boom(state):
            raise RuntimeError("boom-检索失败")

        class FakeGraph:
            def invoke(self, state):
                boom(state)

        monkeypatch.setattr(qs_mod, "query_app", FakeGraph())


        service = qs_mod.QueryService()
        task_id = "test-task-error"
        service.submit_query(task_id, True)  # 创建 SSE 队列
        service.run_query_graph(task_id, "sess", "问题", True)

        assert get_task_status(task_id) == "failed"

        queue = get_sse_queue(task_id)
        assert queue is not None
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())

        error_events = [e for e in events if e["event"] == "error"]
        assert error_events, f"应推送 error 事件，实际事件: {[e['event'] for e in events]}"
        assert "boom-检索失败" in json.dumps(error_events[0]["data"], ensure_ascii=False)
