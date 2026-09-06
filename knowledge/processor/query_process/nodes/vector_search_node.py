import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Dict, Any, List, Tuple, Union
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.domain.retrieval import search_chunks
from knowledge.utils.bge_m3_embedding_util import get_bge_m3_embedding_model, generate_hybrid_embeddings
from knowledge.utils.milvus_util import get_milvus_client


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

        # 4. 统一检索链路：子块(+商品过滤 → 全库) → 父块(+商品过滤 → 全库)
        reps = search_chunks(
            milvus_client,
            embedding_result['dense'][0],
            embedding_result['sparse'][0],
            item_names=validate_item_names,
            limit=self.config.embedding_search_limit,
            chunks_collection=self.config.chunks_collection,
            child_collection=self.config.child_chunks_collection if self.config.parent_child_enabled else None,
            logger_=self.logger,
        )
        if not reps:
            # 并行分支节点：失败/空结果必须返回增量更新，
            # 返回整个 state 会与其他并行节点并发写 session_id，触发 InvalidUpdateError
            return {}

        # 5. 更新state的embedding_chunks
        return {"embedding_chunks": reps}

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
