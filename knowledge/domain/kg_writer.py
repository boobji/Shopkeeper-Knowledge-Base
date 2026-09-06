"""知识图谱导入侧写入组件（从 knowledge_graph_node.py 拆出）。

与查询侧 domain/kg_query.py 的 Reader 模式对称：
─────────────────────────────────────────────────────────
  ProcessingStats      处理统计（日志/监控）
  Neo4jGraphWriter     Neo4j 写入：Chunk/Entity 节点 + 关系（幂等 MERGE）
  MilvusEntityWriter   实体名向量化写入 Milvus（实体对齐数据源）
  （主编排器 KnowledgeGraphNode 留在 knowledge_graph_node.py）
─────────────────────────────────────────────────────────
图结构常量与 Cypher 统一来自 domain/kg_schema。
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from pymilvus import MilvusClient, DataType

from knowledge.core.exceptions import EmbeddingError, MilvusError, Neo4jError
from knowledge.domain.kg_schema import (
    CYPHER_CLEAR_ITEM,
    CYPHER_LINK_ENTITY_TO_CHUNK,
    CYPHER_MERGE_CHUNK,
    CYPHER_MERGE_ENTITY_TEMPLATE,
    CYPHER_MERGE_RELATION_TEMPLATE,
)
from knowledge.utils.bge_m3_embedding_util import get_bge_m3_embedding_model


@dataclass
class ProcessingStats:
    """处理过程统计信息，用于日志和监控。"""

    total_chunks: int = 0
    processed_chunks: int = 0
    failed_chunks: int = 0
    total_entities: int = 0
    total_relations: int = 0
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"处理完成: {self.processed_chunks}/{self.total_chunks} 切片成功, "
            f"{self.failed_chunks} 失败, "
            f"共 {self.total_entities} 实体 / {self.total_relations} 关系"
        )


class Neo4jGraphWriter:
    """Neo4j 图谱写入器：幂等 MERGE，导入前可整商品清理。"""

    def __init__(self, database: str = ""):
        self._database = database
        self._logger = logging.getLogger(self.__class__.__name__)

    def clear(self, neo4j_driver, item_name: str) -> None:
        if not neo4j_driver:
            raise Neo4jError("Neo4j 驱动获取失败")

        try:
            with self._session(neo4j_driver) as session:
                session.execute_write(
                    lambda tx, name: tx.run(CYPHER_CLEAR_ITEM, item_name=name),
                    item_name,
                )
            self._logger.info(f"Neo4j 旧数据已清理: {item_name}")
        except Exception as e:
            raise Neo4jError(f"Neo4j 清理失败: {e}", cause=e)

    def insert(self, driver, entities, relations, chunk_id, item_name):
        """
        Neo4J的写入

        Args:
            driver: neo4j的驱动
            entities:  清洗后的实体
            relations: 清洗后的关系链
            chunk_id:  实体对应的chunk_id
            item_name: 文档对应LLM提取的商品名

        Returns:

        """
        # 1. 判断实体是否存在
        if not entities:
            raise ValueError("参数校验失败，实体列表为空")

        # 2.  判断驱动
        if not driver:
            raise Neo4jError("Neo4j 驱动获取失败")

        try:
            with self._session(driver) as session:
                session.execute_write(
                    self._write_graph_tx, entities, relations, chunk_id, item_name,
                )
            self._logger.info(f"Neo4j 写入: {len(entities)} 实体, {len(relations)} 关系")
        except Exception as e:
            raise Neo4jError(f"Neo4j 写入失败: {e}", cause=e)

    def _write_graph_tx(self, tx, entities, relations, chunk_id, item_name):

        # 1. 创建 Chunk 节点
        tx.run(CYPHER_MERGE_CHUNK, chunk_id=chunk_id, item_name=item_name)

        # 2. 创建实体节点 + 关联到 Chunk
        for entity in entities:
            name = entity.get("name")
            raw_label = entity.get("label")
            description = entity.get("description")

            # 动态格式化 Cypher，将安全标签注入(TODO )
            cypher_query = CYPHER_MERGE_ENTITY_TEMPLATE.format(label=raw_label)

            tx.run(cypher_query, name=name, description=description,
                   chunk_id=chunk_id, item_name=item_name)

            # 关联实体到 Chunk
            tx.run(CYPHER_LINK_ENTITY_TO_CHUNK,
                   name=name, chunk_id=chunk_id, item_name=item_name)

        # 3. 创建实体间关系
        for rel in relations:
            head = rel.get("head")
            tail = rel.get("tail")
            rel_type = rel.get("type")

            cypher = CYPHER_MERGE_RELATION_TEMPLATE.format(rel_type=rel_type)
            tx.run(cypher, head=head, tail=tail, item_name=item_name)

    def _session(self, driver):
        return driver.session(database=self._database)


class MilvusEntityWriter:
    """负责将实体向量化并写入 Milvus，作为查询侧实体对齐的数据源。"""

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.logger = logging.getLogger(self.__class__.__name__)

    def clear(self, milvus_client: MilvusClient, item_name: str):

        # 1. 清理 Milvus
        if not milvus_client:
            raise MilvusError("Milvus 客户端获取失败")

        collection_name = self.collection_name
        try:
            if milvus_client.has_collection(collection_name):
                milvus_client.delete(
                    collection_name=collection_name,
                    filter=f'item_name == "{item_name}"',
                )
                self.logger.info(f"Milvus 旧数据已清理: item_name={item_name}")
        except Exception as e:
            raise MilvusError(f"Milvus 清理失败: {e}", cause=e)

    def insert(self, milvus_client, entities: List[Dict], chunk_id: str, content: str, item_name: str) -> None:
        """对外唯一入口：将实体写入 Milvus。"""

        # 1. 判断实体是否存在
        if not entities:
            raise ValueError("参数校验失败，实体不存在")

        # 2. 获取去重后的实体名
        entities_names = list(dict.fromkeys(e["name"] for e in entities if e.get("name")))
        if not entities_names:
            raise ValueError("参数校验失败，无有效实体名")

        # 3. 获取嵌入模型
        bge_ef_model = get_bge_m3_embedding_model()

        if bge_ef_model is None:
            raise EmbeddingError("嵌入模型获取失败")

        # 4. 创建集合（不存在则创建）
        try:
            self._ensure_collection(milvus_client, self.collection_name)
        except Exception as e:
            raise MilvusError(f"Milvus 创建集合失败: {e}", cause=e)

        # 5. 嵌入向量化
        try:
            embedded_result = bge_ef_model.encode_documents(entities_names)
        except Exception as e:
            raise MilvusError(f"实体嵌入失败: {e}", cause=e)

        # 6. 构建记录
        records = self._build_records(entities_names, embedded_result, chunk_id, content, item_name)
        if not records:
            raise MilvusError("构建 Milvus 记录为空")

        # 7. 写入 Milvus
        try:
            milvus_client.insert(collection_name=self.collection_name, data=records)
            self.logger.info(f"Milvus 写入 {len(records)} 条实体向量")
        except Exception as e:
            raise MilvusError(f"Milvus 插入数据失败: {e}", cause=e)

    def _ensure_collection(self, client, collection_name: str) -> None:
        """集合不存在则创建（schema + 索引）。"""

        # 1. 判断集合是否已存在
        if client.has_collection(collection_name):
            return

        # 2. 构建 schema
        schema = client.create_schema(enable_dynamic_field=True)
        schema.add_field("ids", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("entity_name", DataType.VARCHAR, max_length=65535)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("source_chunk_id", DataType.VARCHAR, max_length=65535)
        schema.add_field("context", DataType.VARCHAR, max_length=65535)
        schema.add_field("item_name", DataType.VARCHAR, max_length=65535)

        # 3. 构建索引
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128},
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )

        # 4. 创建集合
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )

    @staticmethod
    def _build_records(
            entities_names: List[str],
            embedded_result: Dict[str, Any],
            chunk_id: str,
            content: str,
            item_name: str,
    ) -> List[Dict[str, Any]]:
        """组装插入记录。"""

        # 1. 校验嵌入结果
        if not embedded_result:
            raise ValueError("嵌入结果为空")

        # 2. 获取稠密向量和稀疏向量
        dense_vector_list = embedded_result.get("dense")
        sparse_matrix = embedded_result.get("sparse").tocsr()

        # 3. 校验向量是否存在
        if not dense_vector_list or sparse_matrix is None:
            raise ValueError("参数校验失败，向量不存在")

        # 4. 获取对应块的部分内容作为上下文
        context = content[:200]
        records: List[Dict] = []

        # 5. 遍历每一个实体名，构建记录
        for idx, entity_name in enumerate(entities_names):
            # 5.1 边界检查
            if idx >= len(dense_vector_list):
                break

            # 5.2 获取稠密向量
            dense = dense_vector_list[idx].tolist()

            # 5.3 解构稀疏向量（从 CSR 矩阵中提取当前实体的稀疏向量）
            start = sparse_matrix.indptr[idx]
            end = sparse_matrix.indptr[idx + 1]
            indices = sparse_matrix.indices[start:end].tolist()
            data = sparse_matrix.data[start:end].tolist()
            sparse_dict = dict(zip(indices, data))

            # 5.4 构建单条记录
            record = {
                "entity_name": entity_name,
                "context": context,
                "item_name": item_name,
                "source_chunk_id": chunk_id,
                "dense_vector": dense,
                "sparse_vector": sparse_dict,
            }

            records.append(record)

        return records
