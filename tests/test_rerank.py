"""RerankNode 断崖截断与多源合并单测"""

import pytest

from knowledge.processor.query_process.nodes.rerank_node import RerankNode


@pytest.fixture()
def node():
    n = RerankNode()
    # 覆盖为确定值，避免受 knowledge/.env 中 RERANK_* 配置影响
    n.config.rerank_max_top_k = 10
    n.config.rerank_min_top_k = 3
    n.config.rerank_gap_ratio = 0.25
    n.config.rerank_gap_abs = 0.5
    return n


def _doc(score, cid="c"):
    return {"content": f"doc-{score}", "score": score, "chunk_id": cid}


class TestCliffCutoff:
    def test_empty(self, node):
        assert node._cliff_cutoff([]) == []

    def test_cliff_truncates(self, node):
        # 前三 0.9/0.85/0.8 紧凑，第 4 个 0.1 与 0.8 的落差 >= 0.5 → 截断在前 3
        docs = [_doc(0.9), _doc(0.85), _doc(0.8), _doc(0.1), _doc(0.05)]
        result = node._cliff_cutoff(docs)
        assert len(result) == 3

    def test_no_cliff_returns_all(self, node):
        docs = [_doc(0.9 - i * 0.05) for i in range(5)]
        assert len(node._cliff_cutoff(docs)) == 5

    def test_min_top_k_floor(self, node):
        # min_top_k=3：即使第 3、4 位之间有断崖，也保留到 lower_bound 之后才检查
        docs = [_doc(0.9), _doc(0.2), _doc(0.15), _doc(0.1)]
        result = node._cliff_cutoff(docs)
        assert len(result) == 3

    def test_none_scores_do_not_crash(self, node):
        docs = [_doc(0.9), {"content": "x", "score": None}, _doc(0.1)]
        assert len(node._cliff_cutoff(docs)) == 3


class TestMergeMultiSourceDocs:
    def test_local_and_web_merged(self, node):
        state = {
            "rrf_chunks": [
                {"chunk_id": "1", "title": "T", "content": "本地内容"},
                {"no_content": True},
                "not-dict",
            ],
            "web_search_docs": [
                {"url": "http://x", "title": "W", "snippet": "网络内容"},
            ],
        }
        merged = node._merge_multi_source_docs(state)
        assert len(merged) == 2
        assert merged[0]["source"] == "local"
        assert merged[0]["chunk_id"] == "1"
        assert merged[1]["source"] == "web"
        assert merged[1]["url"] == "http://x"
        assert merged[1]["content"] == "网络内容"

    def test_empty_state(self, node):
        assert node._merge_multi_source_docs({}) == []
