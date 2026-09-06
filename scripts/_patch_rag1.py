# -*- coding: utf-8 -*-
"""一次性补丁脚本：hyde/kg 对齐/写入点接入归一化过滤。"""
import io


def edit(path, repls):
    src = io.open(path, encoding="utf-8").read()
    for old, new in repls:
        assert old in src, f"{path}: NOT FOUND -> {old[:90]!r}"
        src = src.replace(old, new)
    io.open(path, "w", encoding="utf-8", newline="").write(src)
    print("ok:", path)


# ---- hyde_search_node：归一化过滤 + 回退 ----
edit("knowledge/processor/query_process/nodes/hyde_search_node.py", [
    ('from knowledge.utils.llm_client_util import get_llm_client',
     'from knowledge.domain.filters import build_item_name_norm_expr\nfrom knowledge.utils.llm_client_util import get_llm_client'),
    ('''        # 5. 获取item_name的过滤表达式
        item_name_filtered_expr = self._item_name_filte_expr(validate_item_names)

        # 6. 创建混合搜索请求
        hybrid_search_requests = create_hybrid_search_requests(dense_vector=embedding_result['dense'][0],
                                                               sparse_vector=embedding_result['sparse'][0],
                                                               expr=item_name_filtered_expr)

        # 7. 执行混合搜索请求
        reps = execute_hybrid_search_query(milvus_client,
                                           collection_name=self.config.chunks_collection,
                                           search_requests=hybrid_search_requests,
                                           norm_score=True,
                                           output_fields=["chunk_id", "content", "item_name"])

        if not reps or not reps[0]:''',
     '''        # 5. 获取item_name的过滤表达式（归一化字段，容忍空格/大小写差异）
        item_name_filtered_expr = build_item_name_norm_expr(validate_item_names)

        # 6/7. 执行混合搜索请求；带过滤搜空时回退一次全库检索
        dense_vector = embedding_result['dense'][0]
        sparse_vector = embedding_result['sparse'][0]
        reps = self._hyde_search(milvus_client, dense_vector, sparse_vector, item_name_filtered_expr)

        if not reps or not reps[0]:'''),
    ('''    def _item_name_filte_expr(self, validate_item_names: List[str]) -> str:
        # filter = 'item_name in '"商品A", "商品B"'
        #  '"商品A", "商品B"'
        quoted = ", ".join(f'"{v}"' for v in validate_item_names)
        # filter = 'item_name in ["商品A", "商品B", "商品C"]'v   # 标量字段（动态字段）进行过滤
        return f" item_name in [{quoted}]"''',
     '''    def _hyde_search(self, milvus_client, dense_vector, sparse_vector, item_name_filter_expr: str):
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
        return reps'''),
])

# ---- kg_query：实体对齐过滤走归一化字段 + 回退 ----
edit("knowledge/domain/kg_query.py", [
    ('from knowledge.domain.llm_parse import strip_json_fence',
     'from knowledge.domain.filters import build_item_name_norm_expr\nfrom knowledge.domain.llm_parse import strip_json_fence'),
    ('''def _item_name_filter_expr(item_names: List[str]) -> str:
    quoted = ", ".join(f"'{item_name}'" for item_name in item_names)
    return f"item_name in [{quoted}]"''',
     '''def _item_name_filter_expr(item_names: List[str]) -> str:
    # 归一化字段过滤：容忍确认名与入库名的空格/大小写差异
    return build_item_name_norm_expr(item_names)'''),
    ('''        # 2. 创建混合搜索请求
        hybrid_search_requests = create_hybrid_search_requests(dense_vector=dense_vector,
                                                               sparse_vector=sparse_vector,
                                                               expr=item_name_filtered_expr, limit=5)
        # 3. 执行混合搜索请求
        reps = execute_hybrid_search_query(milvus_client=milvus_client,
                                           collection_name=_collection_name,
                                           search_requests=hybrid_search_requests,
                                           ranker_weights=(0.4, 0.6),
                                           norm_score=True,
                                           limit=5,
                                           output_fields=["source_chunk_id", "item_name", "context", "entity_name"],
                                           )''',
     '''        # 2/3. 创建并执行混合搜索请求；带过滤搜空时回退一次全库检索
        def _search(expr: str):
            reqs = create_hybrid_search_requests(dense_vector=dense_vector,
                                                 sparse_vector=sparse_vector,
                                                 expr=expr or None, limit=5)
            return execute_hybrid_search_query(milvus_client=milvus_client,
                                               collection_name=_collection_name,
                                               search_requests=reqs,
                                               ranker_weights=(0.4, 0.6),
                                               norm_score=True,
                                               limit=5,
                                               output_fields=["source_chunk_id", "item_name", "context", "entity_name"],
                                               )

        reps = _search(item_name_filtered_expr)
        if (not reps or not reps[0]) and item_name_filtered_expr:
            self._logger.warning("实体对齐带过滤检索为空，回退全库检索")
            reps = _search("")'''),
])

# ---- kg_writer：实体集合 schema + 记录补 item_name_norm ----
edit("knowledge/domain/kg_writer.py", [
    ('from knowledge.core.exceptions import EmbeddingError, MilvusError, Neo4jError',
     'from knowledge.core.exceptions import EmbeddingError, MilvusError, Neo4jError\nfrom knowledge.domain.filters import normalize_item_name'),
    ('''        schema.add_field("item_name", DataType.VARCHAR, max_length=65535)

        # 3. 构建索引''',
     '''        schema.add_field("item_name", DataType.VARCHAR, max_length=65535)
        schema.add_field("item_name_norm", DataType.VARCHAR, max_length=65535)

        # 3. 构建索引'''),
    ('''            record = {
                "entity_name": entity_name,
                "context": context,
                "item_name": item_name,
                "source_chunk_id": chunk_id,''',
     '''            record = {
                "entity_name": entity_name,
                "context": context,
                "item_name": item_name,
                "item_name_norm": normalize_item_name(item_name),
                "source_chunk_id": chunk_id,'''),
])

# ---- item_name_recognition_node：商品名集合写入补 item_name_norm ----
edit("knowledge/processor/import_process/nodes/item_name_recognition_node.py", [
    ('from knowledge.prompts.item_name_prompt import ITEM_NAME_SYSTEM_PROMPT,',
     'from knowledge.domain.filters import normalize_item_name\nfrom knowledge.prompts.item_name_prompt import ITEM_NAME_SYSTEM_PROMPT,'),
    ('''        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)''',
     '''        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name_norm", datatype=DataType.VARCHAR, max_length=65535)'''),
    ('''            # 4. 准备数据
            data = {
                "file_title": file_title,
                "item_name": item_name
            }''',
     '''            # 4. 准备数据
            data = {
                "file_title": file_title,
                "item_name": item_name,
                "item_name_norm": normalize_item_name(item_name),
            }'''),
])
