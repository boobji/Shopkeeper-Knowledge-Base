"""Parent-Child 检索链路单测：回退链 + 子块命中反查父块 + 子块切分。"""

from knowledge.domain.retrieval import search_chunks
from knowledge.processor.import_process.nodes.child_chunk_node import ChildChunkNode


class FakeMilvus:
    """记录调用并按集合返回构造结果的最小 Milvus 桩。"""

    def __init__(self, results_by_collection):
        self.results_by_collection = results_by_collection
        self.search_calls = []      # (collection, expr)
        self.query_calls = []       # (collection, filter)

    def _search(self, collection_name, search_requests, **kwargs):
        expr = search_requests[0].expr
        self.search_calls.append((collection_name, expr))
        hits = self.results_by_collection.get((collection_name, expr), [[]])
        return hits if hits is not None else [[]]

    def query(self, collection_name, filter, output_fields):
        self.query_calls.append((collection_name, filter))
        return self._parent_rows.get(filter, []) if hasattr(self, "_parent_rows") else []


def _hit(entity):
    return {"id": None, "distance": 0.9, "entity": entity}


class TestSearchChunksFallbackChain:
    def _run(self, monkeypatch, results_by_collection, parent_rows=None, child_collection="kb_child_chunks"):
        import knowledge.domain.retrieval as rm

        fake = FakeMilvus(results_by_collection)
        fake._parent_rows = parent_rows or {}
        monkeypatch.setattr(rm, "execute_hybrid_search_query", fake._search)
        out = search_chunks(
            fake, [0.1], {1: 0.5}, item_names=["RS-12"], limit=5,
            chunks_collection="kb_chunks", child_collection=child_collection,
        )
        return fake, out

    def test_parent_collection_hit(self, monkeypatch):
        results = {
            ("kb_child_chunks", 'item_name_norm in ["rs-12"]'): [[]],          # 子块无数据
            ("kb_chunks", 'item_name_norm in ["rs-12"]'): [[_hit({"chunk_id": 7, "content": "父块"})]],
        }
        fake, out = self._run(monkeypatch, results)
        assert len(out) == 1
        assert out[0]["entity"]["chunk_id"] == 7
        # 调用顺序：子块(带过滤) → 子块(全库) → 父块(带过滤)
        assert [c[0] for c in fake.search_calls] == ["kb_child_chunks", "kb_child_chunks", "kb_chunks"]
        assert fake.search_calls[1][1] is None

    def test_child_hit_maps_to_parent(self, monkeypatch):
        child_entity = {"chunk_uid": "uid-1", "content": "子块文本", "title": "T", "item_name": "RS-12"}
        results = {
            ("kb_child_chunks", 'item_name_norm in ["rs-12"]'): [[_hit(child_entity)]],
        }
        parent_rows = {'chunk_uid in ["uid-1"]': [{"chunk_id": 42, "chunk_uid": "uid-1",
                                                   "content": "父块全文", "title": "T",
                                                   "item_name": "RS-12"}]}
        fake, out = self._run(monkeypatch, results, parent_rows)
        assert len(out) == 1
        assert out[0]["entity"]["chunk_id"] == 42
        assert out[0]["entity"]["content"] == "父块全文"
        assert fake.query_calls  # 发生过反查

    def test_filtered_empty_falls_back_to_unfiltered(self, monkeypatch):
        results = {
            ("kb_child_chunks", 'item_name_norm in ["rs-12"]'): [[]],
            ("kb_child_chunks", None): [[_hit({"chunk_uid": "u2", "content": "子块"})]],
            ("kb_chunks", None): [[_hit({"chunk_id": 9, "content": "父块"})]],
        }
        parent_rows = {'chunk_uid in ["u2"]': [{"chunk_id": 8, "chunk_uid": "u2",
                                                "content": "父块2", "title": "T", "item_name": "X"}]}
        fake, out = self._run(monkeypatch, results, parent_rows)
        assert out[0]["entity"]["chunk_id"] == 8
        # 依次尝试：子块过滤 → 子块全库 → 命中子块反查父块
        assert fake.search_calls[0][1] == 'item_name_norm in ["rs-12"]'
        assert fake.search_calls[1][1] is None

    def test_child_collection_missing_falls_to_parent(self, monkeypatch):
        import knowledge.domain.retrieval as rm

        def boom(*args, **kwargs):
            raise RuntimeError("collection not found")

        monkeypatch.setattr(rm, "execute_hybrid_search_query", boom)
        out = search_chunks(
            object(), [0.1], {1: 0.5}, item_names=["RS-12"], limit=5,
            chunks_collection="kb_chunks", child_collection="kb_child_chunks",
        )
        assert out == []

    def test_no_child_collection_skips_child_leg(self, monkeypatch):
        results = {
            ("kb_chunks", 'item_name_norm in ["rs-12"]'): [[_hit({"chunk_id": 1, "content": "父块"})]],
        }
        fake, out = self._run(monkeypatch, results, child_collection=None)
        assert out[0]["entity"]["chunk_id"] == 1
        assert all(c[0] == "kb_chunks" for c in fake.search_calls)


class TestChildChunkSplit:
    def _config(self, size=100, overlap=20):
        class C:
            child_chunk_size = size
            child_chunk_overlap = overlap
        return C()

    def test_long_content_split_with_metadata(self):
        node = ChildChunkNode()
        parent = {
            "chunk_uid": "u1", "content": "句子。" * 100,
            "title": "T", "parent_title": "P", "file_title": "F",
            "item_name": "RS-12", "context_prefix": "ctx",
        }
        children = node._build_children([parent], self._config())
        assert len(children) > 1
        assert all(c["parent_uid"] == "u1" for c in children)
        assert all(c["item_name_norm"] == "rs-12" for c in children)
        assert all(c["context_prefix"] == "ctx" for c in children)
        assert [c["part"] for c in children] == list(range(1, len(children) + 1))

    def test_short_content_single_child(self):
        node = ChildChunkNode()
        parent = {"chunk_uid": "u2", "content": "短内容", "title": "T", "parent_title": "P",
                  "file_title": "F", "item_name": "A"}
        children = node._build_children([parent], self._config())
        assert len(children) == 1
        assert children[0]["content"] == "短内容"

    def test_empty_content_skipped(self):
        node = ChildChunkNode()
        parent = {"chunk_uid": "u3", "content": "", "item_name": "A"}
        assert node._build_children([parent], self._config()) == []
