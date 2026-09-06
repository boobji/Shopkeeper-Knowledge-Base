import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Dict, Any, List, Tuple, Union
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.domain.filters import build_item_name_norm_expr
from knowledge.utils.bge_m3_embedding_util import get_bge_m3_embedding_model, generate_hybrid_embeddings
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query, get_milvus_client


class VectorSearchNode(BaseNode):
    name = "vector_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        # 1. 参数校验
        validated_query, validate_item_names = self._validate_query_inputs(state)

        # 2. 获取嵌入模型&milvus客户端（不可用会抛异常 → 任务失败，而非静默空结果）
        embedding_model = get_bge_m3_embedding_model()
        milvus_client = get_milvus_client()

        # 3. 对问题向量化(稀疏向量做了字典的处理) 注意：【generate_hybrid_embeddings】
        embedding_result = generate_hybrid_embeddings(embedding_model, embedding_documents=[validated_query])
        if not embedding_result:
            # 并行分支节点：失败/空结果必须返回增量更新，
            # 返回整个 state 会与其他并行节点并发写 session_id，触发 InvalidUpdateError
            return {}

        # 4. 构建过滤表达式（归一化字段，容忍空格/大小写差异）
        item_name_filter_expr = build_item_name_norm_expr(validate_item_names)

        # 5. 执行混合搜索请求；带过滤搜空时回退一次全库检索
        #（旧数据无归一化字段、或确认名与入库名不一致时，宁可多召回交给重排，不让该路空手而归）
        dense_vector = embedding_result['dense'][0]
        sparse_vector = embedding_result['sparse'][0]
        reps = self._hybrid_search(milvus_client, dense_vector, sparse_vector,
                                   item_name_filter_expr, limit=5)
        if not reps or not reps[0]:
            # 并行分支节点：失败/空结果必须返回增量更新，
            # 返回整个 state 会与其他并行节点并发写 session_id，触发 InvalidUpdateError
            return {}

        # 6. 更新state的embedding_chunks
        return {"embedding_chunks": reps[0]}

    def _validate_query_inputs(self, state: QueryGraphState) -> Tuple[str, List[str]]:

        # 1. 获取state的rewritten_query
        rewritten_query = state.get('rewritten_query', "")

        # 2. 获取state的item_names
        item_names = state.get('item_names', "")

        # 3. 校验
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)

        # 4. 返回
        return rewritten_query, item_names

    def _hybrid_search(self, milvus_client, dense_vector, sparse_vector,
                       item_name_filter_expr: str, limit: int):
        """执行混合检索；带过滤空结果时回退一次不带过滤的检索（表达式需要重建请求）。"""
        def _search(expr: str):
            reqs = create_hybrid_search_requests(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                expr=expr or None,
                limit=limit,
            )
            return execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self.config.chunks_collection,
                search_requests=reqs,
                norm_score=True,
                limit=limit,
                output_fields=["chunk_id", "content", "item_name"],
            )

        reps = _search(item_name_filter_expr)
        if (not reps or not reps[0]) and item_name_filter_expr:
            self.logger.warning("带商品名过滤检索为空，回退全库检索")
            reps = _search("")
        return reps
