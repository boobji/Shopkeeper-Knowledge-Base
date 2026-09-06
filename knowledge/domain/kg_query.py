"""知识图谱查询侧组件（从 kg_search_node.py 拆出）。

组件结构（与导入侧 kg_writer_components 的 Writer 模式对称）:
─────────────────────────────────────────────────────────
  EntityExtractor     LLM 实体抽取 + JSON 解析
  EntityAligner       Milvus ENTITY_NAME_COLLECTION 实体对齐
  Neo4jGraphReader    Neo4j 种子节点 / 一跳关系 / chunk 反查
  ChunkBackfiller     Milvus CHUNKS_COLLECTION chunk 回填
  （主编排器 KnowledgeGraphSearchNode 留在 kg_search_node.py）
─────────────────────────────────────────────────────────
"""

import json
import logging
from json import JSONDecodeError
from typing import Any, Dict, List, Union

from pymilvus import MilvusClient
from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.domain.llm_parse import strip_json_fence
from knowledge.domain.kg_schema import (
    CYPHER_EXACT_SEEDS,
    CYPHER_FUZZY_SEEDS,
    CYPHER_LOOKUP_CHUNK,
    CYPHER_ONE_HOP_RELATIONS,
    DEFAULT_ENTITY_ALIGN_SCORE,
    MAX_ENTITY_NAME_LENGTH,
    NEIGHBOR_NODE_WEIGHT,
    SEED_NODE_WEIGHT,
)
from knowledge.prompts.query.query_prompt import ENTITY_EXTRACT_SYSTEM_PROMPT
from knowledge.utils.bge_m3_embedding_util import generate_hybrid_embeddings, get_bge_m3_embedding_model
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.utils.milvus_util import (
    create_hybrid_search_requests,
    execute_hybrid_search_query,
    fetch_chunks_by_chunk_ids,
    get_milvus_client,
)
from knowledge.utils.neo4j_util import get_neo4j_driver

logger = logging.getLogger(__name__)

# -------------------------------------------------
# 类型别名（Neo4j 查询结果结构）
# -------------------------------------------------
ItemEntityPair = Dict[str, Any]
EntitySeedNode = Dict[str, Any]
OneHopRelation = Dict[str, Any]


# -------------------------------------------------
# 工具函数（服务各个组件、不污染组件）
# -------------------------------------------------

def clean_parse_llm_content(llm_response_content: str) -> List[str]:
    """
    职责：清洗以及解析LLM输出
    Args:
        llm_response_content: LLM 原始输出

    Returns:
        List[str]:清洗后的实体名

    """
    # 1. 判断LLM输出内容是否为空
    if not llm_response_content:
        return []

    # 2. 清洗json代码围栏
    cleaned = strip_json_fence(llm_response_content)

    # 3. 反序列解析
    try:
        deserialized_result: Dict[str, Any] = json.loads(cleaned)
    except JSONDecodeError as e:
        logging.error(f"JSON 反序列失败，原因: {str(e)}")
        return []

    # 4. 获取提取的实体名
    entities_name = deserialized_result.get('entities', [])

    # 4.1 提取实体名的校验（为空）
    if not entities_name:
        return []
    # 4.2 提取实体名的校验（有效类型）
    if not isinstance(entities_name, list):
        return []

    # 4.3 遍历所有实体名
    seen = set()  # 集合
    entities_name_result = []

    for entity_name in entities_name:
        # 1. 判断是否为空
        if not entity_name:
            continue
        # 2. 判断是否有效类型
        if not isinstance(entity_name, str):
            continue

        # 3. 实体名是否过长
        truncated_entity_name = truncate_entity_name_length(entity_name)

        # 4. 去重保序【顺序：防御性】
        if truncated_entity_name not in seen:
            seen.add(truncated_entity_name)
            entities_name_result.append(truncated_entity_name)

    return entities_name_result


def truncate_entity_name_length(entity_name: str) -> str:
    name = entity_name.strip()
    return name[:MAX_ENTITY_NAME_LENGTH] if len(name) > MAX_ENTITY_NAME_LENGTH else name


def _item_name_filter_expr(item_names: List[str]) -> str:
    quoted = ", ".join(f"'{item_name}'" for item_name in item_names)
    return f"item_name in [{quoted}]"


def _clean_seed_rows(rows: List[Dict[str, Any]]) -> List[EntitySeedNode]:
    """
    职责：清洗查询种子节点的数据
    Args:
        rows:  查询到的结果记录

    Returns:
        干净的结果记录

    """

    if not rows:
        return []

    clean_seeds_result: List[EntitySeedNode] = []
    # 1. 遍历
    for row in rows:
        # 1.1 获取item_name
        item_name = row.get('item_name', '').strip()
        # 1.2 获取entity_name
        entity_name = row.get('name', '').strip()
        # 1.3 判断
        if not item_name or not entity_name:
            continue
        # 1.4 封装一下
        clean_seeds_result.append({
            "item_name": item_name,
            "entity_name": entity_name
        })
    # 2. 返回
    return clean_seeds_result


def _one_hop_relations_to_texts(triples: List[OneHopRelation]) -> List[str]:
    if not triples:
        return []
    docs: List[str] = []
    for tr in triples:
        it = (tr.get("item_name") or "").strip()
        h = (tr.get("head") or "").strip()
        r = (tr.get("rel") or "").strip()
        t = (tr.get("tail") or "").strip()
        if not (h and r and t):
            continue
        docs.append(f"[{it}] {h} -({r})-> {t}" if it else f"{h} -({r})-> {t}")
    return docs


class EntityExtractor:
    """
    实体提取器：
    责任： 利用LLM从查询问题中提取实体
    """

    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def extract(self, user_query: str) -> List[str]:
        """
         根据用户问题提取当前问题下的实体名
        Args:
            user_query:  用户问题

        Returns:
            List[str]: 提取后的实体名

        """

        # 1. 获取llm客户端（连接失败会抛 LLMError，由 BaseNode 包装为任务失败）
        llm_client = get_llm_client(response_format=True)

        # 2. 获取提示词
        entities_name_extract_system_prompt = ENTITY_EXTRACT_SYSTEM_PROMPT.format(
            MAX_ENTITY_NAME_LENGTH=MAX_ENTITY_NAME_LENGTH)

        # 4. 调用LLM
        try:
            # 4.1 发送请求
            llm_response = llm_client.invoke([
                SystemMessage(content=entities_name_extract_system_prompt),
                HumanMessage(content=f"用户问题:{user_query}")
            ])

            # 4.2 获取模型的结果
            llm_response_content = getattr(llm_response, 'content', "").strip()

            # 4.3 清洗&解析
            entities_name = clean_parse_llm_content(llm_response_content)
            return entities_name
        except Exception as e:
            self._logger.error(f"LLM 调用失败:{str(e)}")
            return []


class EntityAligner:
    """
     实体对齐器：
     责任： 根据LLM提取到的实体名 查询Milvus，获取真正的实体名（对齐后的实体名、能够查询neo4j(查询节点使用)）
    """

    def __init__(self, collection_name: str):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._collection_name = collection_name

    def align(self, entity_names: List[str], item_names: List[str]) -> Dict[str, Any]:
        """

        Args:
            entity_names:  LLM提取的实体名
            item_names: 商品名

        Returns:
         Dict[str,Any]:该字典准备封装两个key.
         第一个key:entities_aligned_name:[] 所有对齐后的实体名
         第二key:entities_aligned_elements[]: 所有对齐后的实体信息[source_id ,distance,origin,aligned,content]

        """

        fallback_result = {"entities_aligned_name": [], "entities_aligned_elements": []}
        # 1. 判断实体名是否有
        if not entity_names:
            return fallback_result

        # 2. 获取嵌入模型 / milvus客户端（失败会抛异常，快速暴露而非静默降级）
        embedding_model = get_bge_m3_embedding_model()
        milvus_client = get_milvus_client()

        # 3. 向量化实体名
        embedding_result = generate_hybrid_embeddings(embedding_model=embedding_model, embedding_documents=entity_names)

        # 5. 检验嵌入结果
        if embedding_result is None:
            self._logger.error("嵌入结果无法获取")
            return fallback_result
        # 5.1 获取嵌入后的稠密向量对象（二维数组）
        embedding_result_dense = embedding_result['dense']
        # 5.2 获取嵌入后的稀疏向量对象（二维数组）
        embedding_result_sparse = embedding_result['sparse']

        # 6. 获取item_name的表达式
        item_name_filtered_expr = _item_name_filter_expr(item_names)

        # 7. 遍历所有的实体名字
        aligned_entities_name: List[str] = []  # 存放所有实体的名字
        aligned_entity_elements: List[Dict[str, Any]] = []  # 存放所有实体的详细信息
        seen = set()

        for index, entity_name in enumerate(entity_names):
            # 7.1 对齐一个实体的
            align_one_result: List[Dict[str, Any]] = self._align_one(milvus_client,
                                                                     self._collection_name,
                                                                     item_name_filtered_expr,
                                                                     embedding_result_dense,
                                                                     embedding_result_sparse,
                                                                     index,
                                                                     entity_name)

            # 7.2 将商品对齐结果存储到最终结果中
            aligned_entity_elements.extend(align_one_result)

            # 7.3 遍历商品下的对齐结果
            for detail in align_one_result:
                # a) 获取对齐后的实体名
                aligned_name = detail.get("aligned")

                # b) 获取商品名
                item_name = detail.get("item_name")

                # c) 判断对齐名是否有
                if aligned_name:

                    # 去重 同名实体在不同商品下都保留
                    key = (item_name, aligned_name)
                    if key not in seen:
                        seen.add(key)
                        aligned_entities_name.append(aligned_name)

        self._logger.info(f"对齐后的实体个数 {len(aligned_entities_name)} 实体的名字：{aligned_entities_name}")

        return {
            "entities_aligned_name": aligned_entities_name,
            "entities_aligned_elements": aligned_entity_elements
        }

    def _align_one(self, milvus_client: MilvusClient,
                   _collection_name: str,
                   item_name_filtered_expr: str,
                   embedding_result_dense: List,
                   embedding_result_sparse: List,
                   index: int,
                   entity_name: str) -> List[Dict[str, Any]]:

        """
        对齐指定实体名
        Args:
            milvus_client:
            _collection_name:
            item_name_filtered_expr:
            embedding_result_dense:
            embedding_result_sparse:
            index:
        Returns:
        """
        dense_vector = embedding_result_dense[index]
        sparse_vector = embedding_result_sparse[index]

        # 1. 判断实体名的稠密和稀释向量
        if not dense_vector or not sparse_vector:
            return [{"original": entity_name, "aligned": "", "context": "", "reason": "vector values is not exist "}]

        # 2. 创建混合搜索请求
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
                                           )

        # 4. 解析结果
        hits = reps[0] if reps else []
        if not hits:
            return [{"original": entity_name, "aligned": "", "score": "", "reason": "no_hit"}]

        # 4.1  按 item_name 分组，每组取最高分
        best_by_item: Dict[str, Dict] = {}
        for hit in hits:
            # a) 获取实体
            entity = hit.get("entity")

            # b) 从实体中获取
            item_name = entity.get("item_name").strip()

            # c) 只保留每个 item_name 下的第一个（即最高分）
            if item_name not in best_by_item:
                best_by_item[item_name] = hit

        # 4.2 是否有最好的item_name
        if not best_by_item:
            return [{"original": entity_name, "aligned": "", "score": None, "reason": "no_valid_item_name"}]

        # 4.3  item_name 分组输出结果，过滤低于阈值的
        results: List[Dict[str, Any]] = []
        for item_name, best in best_by_item.items():
            # a) 获取最好的那一个分数
            score = best.get("distance")

            # b) 判断分数值
            if float(score) < float(DEFAULT_ENTITY_ALIGN_SCORE):
                continue

            # c) 获取实体信息
            ent = best.get("entity")

            # d) 将不同商品下最好的实体名添加到结果集中
            results.append({
                "original": entity_name,
                "aligned": ent.get("entity_name"),
                "score": score,
                "item_name": item_name,
                "source_chunk_id": ent.get("source_chunk_id"),
                "reason": "top1_per_item",
            })

        # 4.4 全部低于阈值时返回未命中
        if not results:
            return [{"original": entity_name, "aligned": "", "score": None, "reason": "all_below_threshold"}]

        return results


class Neo4jGraphReader:
    """
    职责：所有对Neo4j的读操作
    1. 种子节点的查询（1.1 精确查询 1.2 降级走模糊查询兜底 ）
    2. 查询种子节点一跳关系（双向：种子节点指向另外的节点，以及另外的节点指向种子节点） 保留完整的关系
    3. 根据所有的节点（种子节点以及邻居节点）方向查询chunk(item_name，id)
    4. 根据所有的chunk_id 查询milvus得到所有的chunk

    """

    def __init__(self,
                 database: str,
                 kg_max_seed_candidates: int,
                 kg_max_total_seeds: int,
                 kg_max_triples_per_seed: int,
                 kg_max_total_triples: int,
                 kg_max_total_chunks: int
                 ):
        self._database = database
        self._kg_max_seed_candidates = kg_max_seed_candidates
        self._kg_max_total_seeds = kg_max_total_seeds
        self.kg_max_triples_per_seed = kg_max_triples_per_seed
        self._kg_max_total_triples = kg_max_total_triples
        self._kg_max_total_chunks = kg_max_total_chunks
        self._logger = logging.getLogger(self.__class__.__name__)

    def _session(self):
        # 获取驱动（失败会抛 Neo4jError）
        neo4j_driver = get_neo4j_driver()
        return neo4j_driver.session(database=self._database)

    def find_seed_nodes(self, pairs: List[ItemEntityPair]) -> List[EntitySeedNode]:
        """
        职责：根据item_name 以及entity_name 查询种子节点
        策略：精确查询，只返回一条 模糊查询，返回三条
        Args:
            pairs:  _build_item_entity_pairs方法返回的商品名和实体名的pair对

        Returns:
            所有商品名下所有实体名对应的种子节点
        """
        # 1. pair对是否存在
        if not pairs:
            return []

        final_seeds_result: List[EntitySeedNode] = []

        # 2. 遍历所有pair对
        for pair in pairs:
            # 2.1 获取item_name
            item_name = pair.get('item_name', '').strip()
            # 2.2 获取entity_name
            entity_name = pair.get('entity_name', '').strip()
            # 2.3 过滤掉无效
            if not item_name or not entity_name:
                continue
            # 2.4 执行cypher语句（1) 精确查询 2）可能要模糊查询）
            try:
                with self._session() as session:
                    # 2.5 执行种子节点查询
                    candidates_seed_nodes = self._execute_seed_nodes(session, item_name, entity_name,
                                                                     self._kg_max_seed_candidates)

                    # 2.6 将查询到的种子节点加入到最终列表中
                    final_seeds_result.extend(candidates_seed_nodes)

                    # 2.7 截取种子节点的个数，防止下游查询关系的时候性能太差（作用不太大）
                    if len(final_seeds_result) > self._kg_max_total_seeds:
                        final_seeds_result = final_seeds_result[:self._kg_max_total_seeds]
                        break

            except Exception as e:
                self._logger.error(f"获取种子节点失败,原因 :{str(e)}")

        self._logger.info(f"获取种子节点 {len(final_seeds_result)} 个")
        return final_seeds_result

    def _execute_seed_nodes(self, session, item_name: str, entity_name: str, _kg_max_seed_candidates: int) -> List[
        EntitySeedNode]:
        """
         执行种子节点查询
        Args:
            session:  neo4j的驱动
            item_name: 商品名
            entity_name: 实体名
            _kg_max_seed_candidates: 单个商品留下的最大种子节点数

        Returns:
          List[EntitySeedNode] :找到的种子节点
        """

        # 1.精确查询
        exact_rows = session.execute_read(
            lambda tx: tx.run(
                CYPHER_EXACT_SEEDS, item_name=item_name, name=entity_name
            ).data()
        )
        if exact_rows:
            return _clean_seed_rows(exact_rows)

        # 2. 模糊查询
        fuzzy_rows = session.execute_read(
            lambda tx: tx.run(
                CYPHER_FUZZY_SEEDS, item_name=item_name, name=entity_name, limit=_kg_max_seed_candidates
            ).data()
        )
        return _clean_seed_rows(fuzzy_rows)

    def find_one_hop_relations(self, seed_nodes: List[EntitySeedNode]) -> List[OneHopRelation]:
        """
        职责： 根据种子节点查询一跳的关系（双向），并且过滤掉MENTIONED_IN 关系的节点
        注意：1. 去重（不允许同一条边出现多次）只能出现一次。 2.图谱中存储的节点和关系结构是什么 查询的时候一定要和存储的我结构保证一致 3. 邻居节点可以是你在一跳范围内指向的节点也可以别人在一跳范围内指向你的节点
        比如：A->B(类型：认识) A->B(类型：认识) B->A(类型：认识)

        Args:
            seed_nodes: find_seed_nodes:所有种子节点（所有商品的种子节点）

        Returns:
            List[OneHopRelation]:item_name/head/rel/tail

        """
        # 1. 判断种子节点
        if not seed_nodes:
            return []
        seen = set()
        one_hop_relations_final_result = []
        # 2. 遍历所有的种子节点
        for seed_node in seed_nodes:
            # 2.1 提取item_name
            item_name = seed_node.get('item_name', "")
            # 2.2 提取entity_name
            seed_name = seed_node.get('entity_name', "")
            # 2.3 判断是否都存在
            if not item_name or not seed_name:
                continue

            # 2.4 执行Cypher语句
            try:
                with self._session() as session:

                    # a) 查询种子节点的一跳关系
                    seed_one_hop_relations: List[OneHopRelation] = self._execute_one_hop_relations(session, item_name,
                                                                                                   seed_name,
                                                                                                   self.kg_max_triples_per_seed)
                    if not seed_one_hop_relations:
                        continue

                    # b) 遍历种子节点所有的关系
                    for seed_one_hop_relation in seed_one_hop_relations:
                        # b.1 获取头
                        head = seed_one_hop_relation.get('head')
                        # b.2 获取rel
                        rel = seed_one_hop_relation.get('rel')
                        # b.3 获取tail
                        tail = seed_one_hop_relation.get('tail')
                        # b.4 获取item_name
                        item_name = seed_one_hop_relation.get('item_name')

                        # b.4 去重（同一条边不能重复出现）同一个商品下，不运行有重复的 不同商品下不能叫重复的边
                        # 场景：A节点是种子节点 令居也是种子节点（A节点作为种子查询邻居节点的时候已经把他们的关系查找到了）所以当在以邻居节点为种子查询的时候，就会出现重复的边。因此要过滤掉
                        # 去重key
                        key = (item_name, head, rel, tail)

                        if key not in seen:
                            seen.add(key)
                            one_hop_relations_final_result.append(seed_one_hop_relation)

                    # c) 截取 种子节点的关系，防止超过LLM窗口阈值
                    if len(one_hop_relations_final_result) > self._kg_max_total_triples:
                        one_hop_relations_final_result = one_hop_relations_final_result[:self._kg_max_total_triples]
                        break

                    # d) 返回
            except Exception as e:
                self._logger.error(f"查询 {seed_name} 种子节点的一跳关系失败: {str(e)}")
                # 单个种子节点失败不影响已收集的关系，继续处理其余种子
                continue
        self._logger.info(f"查询 {len(seed_nodes)} 个种子节点对应的关系:{len(one_hop_relations_final_result)} 条")
        return one_hop_relations_final_result

    def _execute_one_hop_relations(self, session, item_name: str, seed_name: str, kg_max_triples_per_seed: int) -> List[
        OneHopRelation]:
        """

        Args:
            session: neo4j驱动
            item_name: 商品名
            seed_name: 种子节点名字
            kg_max_triples_per_seed:种子节点最大的关系数

        Returns:
            List[OneHopRelation]:种子节点的关系

        """
        # 1. 根据session执行查询方法
        one_hop_relations = session.execute_read(
            lambda tx: tx.run(
                CYPHER_ONE_HOP_RELATIONS, item_name=item_name, name=seed_name, limit=kg_max_triples_per_seed
            ).data()
        )

        # 2. 解析结构
        if not one_hop_relations:
            return []

        # 3. 遍历所有的一条关系
        one_hop_relations_result = []
        for one_hop_relation in one_hop_relations:
            # 3.1 提取head
            head = one_hop_relation.get('head', '').strip()
            # 3.2 提取rel
            rel = one_hop_relation.get('rel', '').strip()
            # 3.3 提取tail
            tail = one_hop_relation.get('tail', '').strip()

            # 3.4 判断是否存在关系链
            if not (head and rel and tail):
                continue

            # 3.5 将关系链添加到最终结果中
            one_hop_relations_result.append({
                "head": head,
                "rel": rel,
                "tail": tail,
                "item_name": item_name
            })
        return one_hop_relations_result

    def collect_node_weight(self, seed_nodes: List[EntitySeedNode], one_hop_relations: List[OneHopRelation]) -> \
            List[Dict[str, Any]]:
        """
         职责：
         为种子节点设置权重weight:高=2.0
         为邻居节点设置权重weigh:低=1.0
        Args:
            seed_nodes: 种子节点
            one_hop_relations:一跳关系

        Returns:
         所有节点带权重
        """

        # 1. 判断seed_nodes 是否存在
        if not seed_nodes:
            return []

        # 2. 判断 one_hop_relations 是否存在
        if not one_hop_relations:
            return []
        # 存放所有节点（种子节点和邻居节点的权重）
        weight_map: dict = {}
        seen = set()
        # 3. 遍历所有的种子节点
        for seed_node in seed_nodes:
            # 3.1 获取item_name
            item_name = seed_node.get('item_name')
            # 3.2 获取节点名
            seed_name = seed_node.get('entity_name')

            key = (item_name, seed_name)
            if key not in seen:
                seen.add(key)
                weight_map[key] = SEED_NODE_WEIGHT

        # 4. 遍历一跳三元组
        for one_hop_relation in one_hop_relations:

            # 4.1 获取head
            head = one_hop_relation.get('head')
            # 4.2 获取tail
            tail = one_hop_relation.get('tail')

            # 4.3 获取商品的名字
            item_name = one_hop_relation.get('item_name')

            # 4.4 为邻居节点赋值权重
            if head and (item_name, head) not in weight_map:
                weight_map[(item_name, head)] = NEIGHBOR_NODE_WEIGHT

            if tail and (item_name, tail) not in weight_map:
                weight_map[(item_name, tail)] = NEIGHBOR_NODE_WEIGHT

        return [{"item_name": it, "entity_name": en, "weight": w}
                for (it, en), w in weight_map.items()]

    def find_nodes_chunk_id(self, weighted_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        根据带权重的节点查询chunk_id
        并且基于权重和、次数、进行排序
        Args:
            weighted_nodes:带权重的节点

        Returns:
         查询到的chunk_id
        """

        # 1. 执行Cypher语句
        try:
            with self._session() as session:
                # 1.1 查询语句
                sorted_node_chunk_id = session.execute_read(lambda tx: tx.run(
                    CYPHER_LOOKUP_CHUNK, weighted_nodes=weighted_nodes, limit=self._kg_max_total_chunks
                ).data())

        except Exception as e:
            self._logger.error(f"反查chunk_id失败,原因:{str(e)}")
            return []

        hits = []
        # 2. 处理结果
        for chunk_row in sorted_node_chunk_id:
            chunk_id = chunk_row.get('chunk_id', "").strip()
            item_name = chunk_row.get('item_name', "").strip()
            score = chunk_row.get('score')

            if chunk_id and item_name:
                hits.append({
                    "id": None,
                    "distance": float(score or 0.0),
                    "entity": {"chunk_id": str(chunk_id), "item_name": str(item_name)}
                })

        return hits


def build_item_entity_pairs(aligned_entities_info: List[Dict[str, Any]]) -> List[ItemEntityPair]:
    """
    职责：从对齐后的实体详情中获取商品名+实体名的pair对
    去重：本质同一个商品名下的实体名只留一个, 不同商品名下的实体名都留
    Args:
        aligned_entities_info: 对齐后的实体详情列表

    Returns:
        商品+实体名的pairs

    """
    # 1. 判断对齐后的实体详情是否存在
    if not aligned_entities_info:
        return []

    seen = set()
    item_entity_pairs = []

    # 2. 遍历对齐后的实体详情
    for aligned_entity_info in aligned_entities_info:
        # 2.1 获取商品名item_name
        item_name = aligned_entity_info.get('item_name', "").strip()
        # 2.2 获取对齐后的实体名
        aligned_entity_name = aligned_entity_info.get('aligned', "").strip()
        # 2.3 商品名&实体名都存在
        if not (item_name and aligned_entity_name):
            continue
        # 2.4 去重
        key = (item_name, aligned_entity_name)
        if key not in seen:
            seen.add(key)
            item_entity_pairs.append({
                "item_name": item_name,
                "entity_name": aligned_entity_name
            })
    # 3. 返回
    return item_entity_pairs


class ChunkBackfiller:
    """

    职责：
    1、根据Neo4J返回的chunk信息（entity{"chunk_id"}） 获取到chunk_ids
    2、根据chunk_ids 查询milvus 获取到chunks(批量操作，返回的chunk没有所谓的顺序，只负责将chunk_id的对象给你)：导致根据分数降序的chunk_id就失去了作用
    3、构建一个chunk_id和chunk对象的字典映射表，将milvus返回的chunk_id和chunk对象存储进来
    4、遍历原有分数降序的chunk_id列表，然后在从该映射表中获取对应的chunk(保证原有根据分数降序的chunk_id)顺序能利用起来

    3、更新到state

    """

    def __init__(self, collection_name: str):
        self._collection_name = collection_name
        self.logger = logging.getLogger(self.__class__.__name__)

    def back_fill(self, chunk_nodes_sorted: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        职责：
        1. 获取所有chunk_id
        2. 根据批量的chunk_id查询milvus
        3. 构建chunk_id和chunk对象的映射表（内存操作 快且没有oom风险（batch））
        4. 遍历原有顺序的chunk_id列表(有分数顺序)，接着去到映射表中获取对应的chunk（有分数顺序）
        Args:
            chunk_nodes_sorted: 已经根据排序规则排好顺序的chunk节点

        Returns:

        """

        # 1. 判断chunk_nodes_sorted是否存在
        if not chunk_nodes_sorted:
            return []

        # 2. 获取chunk_ids
        chunk_ids: List[Union[str, int]] = self._collect_chunk_ids(chunk_nodes_sorted)

        # 3. 查询milvus 获取chunk对象
        try:
            chunks: List[Dict[str, Any]] = fetch_chunks_by_chunk_ids(
                collection_name=self._collection_name,
                chunk_ids=chunk_ids,
                output_fields=['chunk_id', 'content', 'title', 'item_name'],
                batch_size=30
            )
            if not chunks:
                return []
        except Exception as e:
            self.logger.error(f"根据chunk_id批量查询chunk对象失败：{str(e)}")
            return []

        # 4. 构建chunk_id和chunk对象的映射表
        chunk_id_map = {str(chunk.get('chunk_id')): chunk for chunk in chunks if chunk.get('chunk_id') is not None}

        # 5. 根据真实顺序的chunk_id从映射表查询真正的chunk对象
        return [{"entity": chunk_id_map.get(str(chunk_id))} for chunk_id in chunk_ids]

    def _collect_chunk_ids(self, chunk_nodes_sorted: List[Dict[str, Any]]) -> List[Union[str, int]]:
        """

        Args:
            chunk_nodes_sorted:

        Returns:

        """
        chunk_ids = []
        # 1. 遍历chunk_ids
        for chunk_node in chunk_nodes_sorted:

            # 2. 判断chunk_id
            if not chunk_node:
                continue

            # 3. 获取entity
            entity = chunk_node.get('entity', '')
            if not entity:
                continue

            # 4. 获取chunk_id
            chunk_id = entity.get('chunk_id')
            if not chunk_id:
                continue

            # 5. chunk_id转换
            chunk_id_str = str(chunk_id)
            try:
                chunk_id_int = int(chunk_id_str)
                chunk_ids.append(chunk_id_int)
            except (ValueError, TypeError):
                # 出现转换失败的异常很小
                chunk_ids.append(chunk_id_str)

        return chunk_ids
