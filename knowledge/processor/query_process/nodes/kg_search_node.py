"""node_query_kg — 知识图谱查询节点（主编排器）。

组件已拆分至 knowledge/domain/kg_query.py：
  EntityExtractor     LLM 实体抽取 + JSON 解析
  EntityAligner       Milvus ENTITY_NAME_COLLECTION 实体对齐
  Neo4jGraphReader    Neo4j 种子节点 / 一跳关系 / chunk 反查
  ChunkBackfiller     Milvus CHUNKS_COLLECTION chunk 回填
本文件只保留 LangGraph 节点编排逻辑。
"""

import logging
import re
from typing import Any, Dict, List, Tuple, Union

from knowledge.domain.kg_query import (
    ChunkBackfiller,
    EntityAligner,
    EntityExtractor,
    ItemEntityPair,
    Neo4jGraphReader,
    OneHopRelation,
    EntitySeedNode,
    _one_hop_relations_to_texts,
    build_item_entity_pairs,
)
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState

logger = logging.getLogger(__name__)


class KnowledgeGraphSearchNode(BaseNode):
    """
      知识图谱查询主编排器。

      职责：
      - 组装四个服务组件（Extractor / Aligner / GraphReader / Backfiller）
      - 按 pipeline 顺序编排调用

      Pipeline:
      ┌──────────┐   ┌──────────┐   ┌────────────┐   ┌──────────┐
         抽取实体  ──▶   对齐实体   ──▶    Neo4j查询   ──▶ 回填chunk
      └──────────┘   └──────────┘   └────────────┘   └──────────┘
      """

    name = "kg_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        # 1. 参数校验
        validated_query, validated_item_names = self._validate_inputs(state)

        # 2. 执行流水线
        kg_result: Dict[str, Any] = self._run_pipeline(validated_query, validated_item_names)

        # 3. 只更新state中的kg_chunks、kg_triples
        return {
            "kg_chunks": kg_result.get('kg_chunks'),
            "kg_triples": kg_result.get('kg_triples')
        }

    def _validate_inputs(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        # 1. 获取参数
        rewritten_query = state.get('rewritten_query', "")
        item_names = state.get('item_names', "")

        # 2. 校验
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)

        # 3. 从重写的问题中踢掉商品名(降噪以及无异议的查询)选择

        user_query = rewritten_query
        for name in item_names:
            if not name:
                continue
            pattern = r"\s*".join(re.escape(ch) for ch in name.replace(" ", ""))
            user_query = re.sub(pattern, "", user_query, flags=re.IGNORECASE)

        user_query = " ".join(user_query.split()).strip()
        # 4. 返回
        return user_query, item_names

    def _run_pipeline(self, validated_query: str, validated_item_names: List[str]) -> Dict[str, Any]:

        # 1. 初始化组件
        entity_extractor = EntityExtractor()
        entity_aligner = EntityAligner(collection_name=self.config.entity_name_collection)
        neo4g_graph_reader = Neo4jGraphReader(database=self.config.neo4j_database,
                                              kg_max_seed_candidates=self.config.kg_max_seed_candidates,
                                              kg_max_total_seeds=self.config.kg_max_total_seeds,
                                              kg_max_triples_per_seed=self.config.kg_max_triples_per_seed,
                                              kg_max_total_triples=self.config.kg_max_total_triples,
                                              kg_max_total_chunks=self.config.kg_max_total_chunks
                                              )
        chunk_back_filler = ChunkBackfiller(collection_name=self.config.chunks_collection)

        # 2. 各个组件执行各种的业务
        # 2.1 利用提取器组件、对齐器组件提取实体以及对齐后的实体（LLM+Milvus）
        entities_name = entity_extractor.extract(user_query=validated_query)
        entities_name_aligned: Dict[str, Any] = entity_aligner.align(entities_name, item_names=validated_item_names)
        # 获取所有对齐后的实体名(业务逻辑不使用)
        aligned_entities_name = entities_name_aligned.get('entities_aligned_name')
        # 获取所有对齐后的实体详情（结构信息细粒）
        aligned_entities_info = entities_name_aligned.get('entities_aligned_elements')
        # 构建商品名+实体名的pair对
        item_entity_pairs: List[ItemEntityPair] = build_item_entity_pairs(aligned_entities_info)

        # 2.2 利用Neo4J的读取器组件对Neo4J进行相关的查询(Neo4J)
        # a) 根商品名和实体名的pairs 查询种子节点
        seed_nodes: List[EntitySeedNode] = neo4g_graph_reader.find_seed_nodes(item_entity_pairs)
        # b) 根据种子节点查询一跳关系
        one_hop_relations: List[OneHopRelation] = neo4g_graph_reader.find_one_hop_relations(seed_nodes)
        # c)  根据种子节点(查询到的)以及一跳关系【种子节点/邻居节点】分别为其设置权重
        weighted_nodes: List[Dict[str, Any]] = neo4g_graph_reader.collect_node_weight(seed_nodes, one_hop_relations)
        # d) 根据带权重的节点反查chunk,并且基于权重给chunk排序（权重排【sum】降序/次数排降序/chunk_id升序）
        chunk_nodes_sorted: List[Dict[str, Any]] = neo4g_graph_reader.find_nodes_chunk_id(weighted_nodes)

        # 2.3 Milvus的操作(利用Chunk_Back_Filler回填器 进行反查chunk)
        kg_chunks = chunk_back_filler.back_fill(chunk_nodes_sorted)

        # 3. 将一跳关系转换成模型能够理解的真实图谱结构
        triples_docs = _one_hop_relations_to_texts(one_hop_relations)

        # 4. 汇总知识图谱节点的所有信息
        return {
            "kg_chunks": kg_chunks,  # 回填后的切片文本 → 送入 RRF
            "kg_triples": triples_docs,  # 关系文本描述 → 送入答案生成 prompt
            "kg_seed_nodes": seed_nodes,
            "kg_triples_raw": one_hop_relations,
            "kg_entities": entities_name,
            "kg_aligned_entities": aligned_entities_name,
            "kg_alignments": aligned_entities_info,
        }
