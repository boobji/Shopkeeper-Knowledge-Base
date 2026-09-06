import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import List, Tuple, Union, Any, Dict
from langchain_core.messages import SystemMessage, HumanMessage
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError

from knowledge.domain.filters import build_item_name_norm_expr
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.prompts.query.query_prompt import USER_HYDE_PROMPT_TEMPLATE
from knowledge.utils.milvus_util import get_milvus_client, create_hybrid_search_requests, execute_hybrid_search_query
from knowledge.utils.bge_m3_embedding_util import generate_hybrid_embeddings, get_bge_m3_embedding_model


class HyDeSearchNode(BaseNode):
    name = "hyde_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:

        # 1. 参数校验
        validated_query, validate_item_names = self._validate_query_inputs(state)

        # 2. 生成假设性文档
        hy_document = self._generate_hy_document(validated_query, validate_item_names)

        # 3. 获取嵌入模型 & milvus客户端（不可用会抛异常 → 任务失败，而非静默空结果）
        embedding_model = get_bge_m3_embedding_model()
        milvus_client = get_milvus_client()

        # 4. 假设性文档嵌入(注入问题+假设性文档)
        embedding_document = f"{validated_query}\n{hy_document}"
        embedding_result = generate_hybrid_embeddings(embedding_model, embedding_documents=[embedding_document])

        if not embedding_result:
            # 并行分支节点：失败/空结果必须返回增量更新（见 vector_search_node 说明）
            return {}

        # 5. 获取item_name的过滤表达式（归一化字段，容忍空格/大小写差异）
        item_name_filtered_expr = build_item_name_norm_expr(validate_item_names)

        # 6/7. 执行混合搜索请求；带过滤搜空时回退一次全库检索
        dense_vector = embedding_result['dense'][0]
        sparse_vector = embedding_result['sparse'][0]
        reps = self._hyde_search(milvus_client, dense_vector, sparse_vector, item_name_filtered_expr)

        if not reps or not reps[0]:
            # 并行分支节点：失败/空结果必须返回增量更新（见 vector_search_node 说明）
            return {}

        # 8. 只更新hyde_embedding_chunks
        return {"hyde_embedding_chunks": reps[0]}

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

    def _generate_hy_document(self, validated_query: str, validate_item_names: List[str]) -> str:

        # 1. 获取LLM客户端（失败会抛 LLMError → 任务失败）
        llm_client = get_llm_client()

        # 2. 获取系统提示词以及用户提示词
        user_prompt = USER_HYDE_PROMPT_TEMPLATE.format(item_hint=validate_item_names, rewritten_query=validated_query)
        system_prompt = f"您是一位{validate_item_names}的技术文档领域的专家，主要擅长编写技术文档、操作手册、文档规格说明"
        try:
            # 4. 获取AIMessage
            llm_response = llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            # 5. 获取内容
            llm_response_content = getattr(llm_response, 'content', "").strip()

            # 6. 判断是否存在
            if not llm_response_content:
                return ""

            return llm_response_content
        except Exception as e:
            self.logger.error(f"LLM调用失败:{str(e)}")
            return ""

    def _hyde_search(self, milvus_client, dense_vector, sparse_vector, item_name_filter_expr: str):
        """执行混合检索；带过滤空结果时回退一次不带过滤的检索（同 vector_search_node）。"""
        def _search(expr: str):
            reqs = create_hybrid_search_requests(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                expr=expr or None,
            )
            return execute_hybrid_search_query(milvus_client,
                                               collection_name=self.config.chunks_collection,
                                               search_requests=reqs,
                                               norm_score=True,
                                               output_fields=["chunk_id", "content", "item_name"])

        reps = _search(item_name_filter_expr)
        if (not reps or not reps[0]) and item_name_filter_expr:
            self.logger.warning("带商品名过滤检索为空，回退全库检索")
            reps = _search("")
        return reps
