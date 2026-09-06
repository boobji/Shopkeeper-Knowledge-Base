"""RRF 融合节点单测"""

import pytest

from knowledge.processor.query_process.nodes.rrf_node import RrfNode


@pytest.fixture()
def node():
    return RrfNode()


def _doc(cid, text=""):
    return {"chunk_id": cid, "content": text or f"content-{cid}"}


class TestNormalizeInput:
    def test_unwraps_entity(self, node):
        raw = [{"entity": _doc("a")}, {"entity": _doc("b")}]
        assert node._normalize_input(raw) == [_doc("a"), _doc("b")]

    def test_skips_invalid(self, node):
        raw = [{"entity": _doc("a")}, "not-dict", {}, {"no_entity": 1}, None]
        assert node._normalize_input(raw) == [_doc("a")]

    def test_empty(self, node):
        assert node._normalize_input([]) == []


class TestRrfMerge:
    def test_multi_hit_ranks_higher(self, node):
        inputs = [
            ([_doc("a"), _doc("b")], 1.0),
            ([_doc("b")], 1.0),
        ]
        merged = node._rrf_merge(inputs, 60, 10)
        # b 命中两路，得分必然高于只命中一路的 a
        assert merged[0][0]["chunk_id"] == "b"
        assert merged[0][1] > merged[1][1]

    def test_weight_matters(self, node):
        single_low = [([_doc("a")], 0.7)]
        single_high = [([_doc("b")], 1.0)]
        merged = node._rrf_merge(single_low + single_high, 60, 10)
        assert [m[0]["chunk_id"] for m in merged] == ["b", "a"]

    def test_top_k_truncates(self, node):
        inputs = [([_doc(f"c{i}") for i in range(20)], 1.0)]
        assert len(node._rrf_merge(inputs, 60, 5)) == 5

    def test_top_k_zero_keeps_all(self, node):
        inputs = [([_doc(f"c{i}") for i in range(6)], 1.0)]
        assert len(node._rrf_merge(inputs, 60, 0)) == 6

    def test_missing_chunk_id_skipped(self, node):
        inputs = [([{"content": "no id"}, _doc("a")], 1.0)]
        merged = node._rrf_merge(inputs, 60, 10)
        assert [m[0]["chunk_id"] for m in merged] == ["a"]

    def test_first_occurrence_wins_for_payload(self, node):
        first = _doc("a", "first")
        second = _doc("a", "second")
        merged = node._rrf_merge([([first], 1.0), ([second], 1.0)], 60, 10)
        assert merged[0][0]["content"] == "first"


class TestProcess:
    def test_state_updated_and_kg_weighted(self, node):
        state = {
            "embedding_chunks": [{"entity": _doc("v1")}],
            "hyde_embedding_chunks": [{"entity": _doc("h1")}],
            "kg_chunks": [{"entity": _doc("k1")}],
        }
        result = node.process(state)
        assert len(result["rrf_chunks"]) == 3

    def test_empty_state(self, node):
        result = node.process({})
        assert result["rrf_chunks"] == []
