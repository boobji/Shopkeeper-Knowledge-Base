"""商品名归一化过滤与检索回退逻辑的单测。"""

from knowledge.domain.filters import (
    build_item_name_norm_expr,
    normalize_item_name,
    normalize_item_names,
)


class TestNormalize:
    def test_strips_whitespace_and_lowercases(self):
        assert normalize_item_name("  RS-12 数字万用表  ") == "rs-12数字万用表"
        assert normalize_item_name("RS-12　数字万用表") == "rs-12数字万用表"  # 全角空格
        assert normalize_item_name("Huawei MateStation B520") == "huaweimatestationb520"

    def test_empty(self):
        assert normalize_item_name("") == ""
        assert normalize_item_name(None) == ""

    def test_batch_dedup(self):
        assert normalize_item_names(["A B", "ab", "", None, "C"]) == ["ab", "c"]


class TestFilterExpr:
    def test_expr_uses_norm_field(self):
        expr = build_item_name_norm_expr(["RS-12 数字万用表"])
        assert expr == 'item_name_norm in ["rs-12数字万用表"]'

    def test_expr_multiple(self):
        expr = build_item_name_norm_expr(["RS-12", "华为 B520"])
        assert expr == 'item_name_norm in ["rs-12", "华为b520"]'

    def test_empty_names_no_filter(self):
        assert build_item_name_norm_expr([]) == ""
        assert build_item_name_norm_expr(["", "  "]) == ""


class TestVectorSearchFallback:
    def test_filtered_empty_retries_without_filter(self, monkeypatch):
        """带商品名过滤搜空时，必须回退一次不带过滤的检索。"""
        import knowledge.processor.query_process.nodes.vector_search_node as vm

        calls = []

        def fake_search(milvus_client, collection_name, search_requests, **kwargs):
            calls.append(search_requests[0].expr)
            if search_requests[0].expr:
                return [[]]
            return [[{"chunk_id": 1, "content": "c", "item_name": "x"}]]

        monkeypatch.setattr(vm, "get_milvus_client", lambda: object())
        monkeypatch.setattr(vm, "execute_hybrid_search_query", fake_search)
        node = vm.VectorSearchNode()
        result = node.process({"rewritten_query": "如何测电压", "item_names": ["RS-12"]})

        assert result["embedding_chunks"]
        # 第一次带归一化过滤，第二次不带
        assert calls[0] == 'item_name_norm in ["rs-12"]'
        assert not calls[1]
