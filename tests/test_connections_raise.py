"""连接/模型层单测：失败必须抛异常（快速暴露），不再返回 None。"""

import pytest

from knowledge.core import connections
from knowledge.core.exceptions import LLMError, MilvusError, StorageError
from knowledge.utils import bge_rerank_util, llm_client_util


@pytest.fixture()
def _reset_singletons(monkeypatch):
    """隔离连接单例全局状态，避免测试间相互污染。"""
    monkeypatch.setattr(connections, "_milvus_client", None)


class TestConnectionsRaise:
    def test_milvus_ctor_failure_raises(self, monkeypatch, _reset_singletons):
        import pymilvus

        class Boom:
            def __init__(self, **kwargs):
                raise RuntimeError("连接不上")

        monkeypatch.setattr(pymilvus, "MilvusClient", Boom)
        with pytest.raises(MilvusError):
            connections.get_milvus_client()

    def test_milvus_success_is_cached(self, monkeypatch, _reset_singletons):
        import pymilvus

        made = []

        class FakeClient:
            def __init__(self, **kwargs):
                made.append(self)

        monkeypatch.setattr(pymilvus, "MilvusClient", FakeClient)
        c1 = connections.get_milvus_client()
        c2 = connections.get_milvus_client()
        assert c1 is c2 and len(made) == 1


class TestModelGettersRaise:
    def test_llm_client_ctor_failure_raises(self, monkeypatch):
        class Boom:
            def __init__(self, **kwargs):
                raise RuntimeError("bad key")

        monkeypatch.setattr(llm_client_util, "ChatOpenAI", Boom)
        monkeypatch.setitem(llm_client_util.cache_llm_client, ("x", 0.0, False), object())  # 不影响新 key
        with pytest.raises(LLMError):
            llm_client_util.get_llm_client(model_name="unreachable-model-x")

    def test_reranker_failure_raises(self, monkeypatch):
        class Boom:
            def __init__(self, **kwargs):
                raise RuntimeError("no weights")

        monkeypatch.setattr(bge_rerank_util, "FlagReranker", Boom)
        monkeypatch.setattr(bge_rerank_util, "_reranker_model", None)
        with pytest.raises(StorageError):
            bge_rerank_util.get_reranker_model()
