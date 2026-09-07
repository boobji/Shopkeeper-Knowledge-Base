"""知识图谱导入节点（主编排器）。

写入组件已拆分至 knowledge/domain/kg_writer.py：
  ProcessingStats      处理统计
  Neo4jGraphWriter     Neo4j 幂等写入
  MilvusEntityWriter   实体向量化写入
本文件保留：LLM 图谱抽取（带重试）、实体/关系清洗、参数校验、并发编排。
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from json import JSONDecodeError
from typing import Dict, List, Any, Tuple, Set, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from pymilvus import MilvusClient
from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.config import ImportConfig
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.domain.kg_schema import (
    ALLOWED_ENTITY_LABELS,
    ALLOWED_RELATION_TYPES,
    DEFAULT_RELATION_TYPES,
    MAX_ENTITY_NAME_LENGTH,
)
from knowledge.domain.kg_writer import MilvusEntityWriter, Neo4jGraphWriter, ProcessingStats
from knowledge.domain.llm_parse import strip_json_fence
from knowledge.prompts.upload.import_prompt import KNOWLEDGE_GRAPH_SYSTEM_PROMPT
from knowledge.utils.milvus_util import get_milvus_client
from knowledge.utils.neo4j_util import get_neo4j_driver
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.utils.llm_call_logger import log_llm_call


class KnowledgeGraphNode(BaseNode):
    name = "knowledge_graph_node"

    def __init__(self, config: Optional[ImportConfig] = None):
        super().__init__(config)
        self._milvus_writer = MilvusEntityWriter(self.config.entity_name_collection)
        self._neo4j_writer = Neo4jGraphWriter(self.config.neo4j_database)

    def process(self, state: ImportGraphState) -> ImportGraphState:

        # 1. 参数校验
        validated_chunks, item_name = self._validate_get_inputs(state)

        # 2. 构建统计初始信息
        stats = ProcessingStats(total_chunks=len(validated_chunks))

        # 3. 获取
        # 3.1 获取milvus客户端
        milvus_client = get_milvus_client()
        neo4j_driver = get_neo4j_driver()

        # 4. 删除已经存在的数据（4.1 删除milvus中存储实体名字的记录（delete:item_name）：幂等性保证 4.2 删除neo4j的整个库下的所有节点以及关系）
        self._clean_exist_double_data(milvus_client, neo4j_driver, item_name)

        # 5. 批量处理（串行版本）
        # self._process_all_chunks_v1(stats, validated_chunks, milvus_client, neo4j_driver)
        # 5. 批量处理（多线程版本）
        self._process_chunks_concurrently(stats, validated_chunks, milvus_client, neo4j_driver,
                                          task_id=state.get('task_id') or '',
                                          task_dir=state.get('file_dir') or '')

        # 6. 简单的日志观察
        self.logger.info(stats.summary())

        # 7. 遵循节点契约：必须返回 state（此前遗漏 return，节点返回 None）
        return state

    def _clean_exist_double_data(self, milvus_client: MilvusClient, neo4j_driver,
                                 item_name: str):
        """
        删除milvus以及neo4j的对应文档的记录
        Args:
            milvus_client:
            neo4j_driver:
            item_name:

        Returns:

        """
        # 1. 导入前清理该 item_name 下的所有旧数据（Milvus）
        self._milvus_writer.clear(milvus_client, item_name)

        # 2. 导入前清理该 item_name 下的所有旧数据（Neo4J）
        self._neo4j_writer.clear(neo4j_driver, item_name)

    def _process_all_chunks_v1(self, stats: ProcessingStats,
                               validated_chunks: List[Dict[str, Any]],
                               milvus_client: MilvusClient,
                               neo4j_driver):
        """
        循环处理每一个chunk
        Args:
            validated_chunks:
            milvus_client:
            neo4j_driver:

        Returns:

        """

        # 1. 遍历所有的chunk
        for i, chunk in enumerate(validated_chunks):

            if not isinstance(chunk, dict):
                continue

            # 1.1 获取chunk的信息
            chunk_id = chunk.get('chunk_id')
            item_name = chunk.get('item_name')
            content = chunk.get('content')

            # 2. 处理单个chunk
            try:

                entities_count, relations_count = self._process_single_chunk(chunk_id,
                                                                             item_name,
                                                                             content,
                                                                             milvus_client,
                                                                             neo4j_driver)
                stats.processed_chunks += 1
                stats.total_entities += entities_count
                stats.total_relations += relations_count
                self.logger.info(f"成功处理完 {chunk_id} / {len(validated_chunks)}")
            except Exception as e:
                stats.failed_chunks += 1
                stats.errors.append(str(e))
                self.logger.error(f"处理失败 {chunk_id} / {len(validated_chunks)}")

    def _process_single_chunk(self, chunk_id: str,
                              item_name: str,
                              content: str,
                              milvus_client: MilvusClient,
                              neo4j_driver,
                              task_id: str = "", task_dir: str = "") -> Tuple[int, int]:

        llm_start = time.time()
        thread_name = threading.current_thread().name  # 获取线程名
        # 1. 调用模型提取chunk的实体、关系
        llm_response = self._extract_graph_with_retry(content, task_id=task_id, task_dir=task_dir,
                                                      chunk_id=chunk_id)
        llm_cost = time.time() - llm_start

        # 2. 解析并且清洗数据
        graph_result = self._parse_and_clean(llm_response)

        # 2.1 获取解析后的实体
        final_entities = graph_result.get('entities')
        # 2.2 获取解析后的关系
        final_relations = graph_result.get('relations')

        # 3. 写入
        # 3.1 将清洗后的实体名字（可能是多个）存储到milvus
        milvus_start = time.time()
        self._milvus_writer.insert(milvus_client, final_entities, chunk_id, content, item_name)
        milvus_cost = time.time() - milvus_start

        # 3.2 将清洗后的实体以及关系类型都存储到neo4j
        neo4j_start = time.time()
        self._neo4j_writer.insert(neo4j_driver, final_entities, final_relations, chunk_id, item_name)
        neo4j_cost = time.time() - neo4j_start

        total_cost = time.time() - llm_start
        # 4. 统计单块处理的时间信息
        self.logger.info(
            f"[{thread_name}] chunk={chunk_id} | "
            f"实体={len(final_entities)} 关系={len(final_relations)} | "
            f"LLM={llm_cost:.2f}s Milvus={milvus_cost:.2f}s Neo4j={neo4j_cost:.2f}s | "
            f"总计={total_cost:.2f}s"
        )

        return len(final_entities), len(final_relations)

    def _extract_graph_with_retry(self, content: str, task_id: str = "",
                                  task_dir: str = "", chunk_id: str = "") -> str:

        # 1. 获取LLM客户端
        llm_client = get_llm_client()
        if llm_client is None:
            raise ValueError("LLM客户端初始化失败")

        MAX_COUNT = 3
        last_error = None

        # 2.循环重试3次
        # TODO :将失败的异常原因给到模型
        for attempt in range(1, MAX_COUNT + 1):
            messages = [
                SystemMessage(content=KNOWLEDGE_GRAPH_SYSTEM_PROMPT),
                HumanMessage(content=f"切片信息\n\n{content}")
            ]
            t0 = time.perf_counter()
            try:
                # 2.1 调用模型
                llm_response = llm_client.invoke(messages)
                # 2.2 获取内容
                result = getattr(llm_response, 'content', '').strip()

                # 2.3 留档本次调用（成功与"空响应"都记录，空响应往往提示词有问题）
                log_llm_call('import_kg', messages=messages, response=llm_response,
                             task_id=task_id, task_dir=task_dir,
                             meta={'chunk_id': chunk_id, 'attempt': attempt,
                                   'empty_response': not result},
                             latency_ms=(time.perf_counter() - t0) * 1000)

                # 2.4 有内容
                if result:
                    return result
            except Exception as e:
                last_error = e

                # 2.5 失败同样留档
                log_llm_call('import_kg', messages=messages,
                             task_id=task_id, task_dir=task_dir,
                             meta={'chunk_id': chunk_id, 'attempt': attempt},
                             latency_ms=(time.perf_counter() - t0) * 1000, error=str(e))

                # 2.6 控制重试间隔
                if attempt < MAX_COUNT:
                    # 睡一会：间隔[固定间隔/指数退避]
                    delay = 0.5 * (2 ** (attempt - 1))
                    self.logger.warning(f"开始第{attempt}次重试，间隔：{delay:1.f}s")
                    time.sleep(delay)
        self.logger.error(f"已经进行了{MAX_COUNT}次重试，都失败原因：{str(last_error)}")

        # 3. 最终兜底
        return ""

    def _parse_and_clean(self, llm_response: str) -> Dict[str, Any]:
        """
        1.解析llm返回结果的json代码片段的围栏
        2.反序列化
        3.获取实体信息以及关系信息
        4.分别在清洗实体以及关系
        5. 清洗之后对应的实体和关系返回
        Args:
            llm_response: 模型的输出
        Returns:
             {
                "entities" :[{比较干净的实体名字:标签},{比较干净的实体名字:标签}]
                “relations” :[{比较干净的关系：“head”:"","tail":"","type":""},{比较干净的关系：“head”:"","tail":"","type":""}]
            }
        """

        # 1. 判断
        if not llm_response:
            raise ValueError("LLM提取chunk的图谱信息不存在")

        # 2. 清洗json代码块的围栏（```json ... ```）
        cleaned = strip_json_fence(llm_response)

        # 3. 反序列化
        try:
            parsed_llm_response: Dict[str, Any] = json.loads(cleaned)
        except JSONDecodeError as e:
            raise JSONDecodeError(f"反序列化失败 :{str(e)}")

        # 4. 获取信息
        # 4.1 获取实体信息
        entities = parsed_llm_response.get('entities', [])

        # 4.2 获取关系信息
        relations = parsed_llm_response.get('relations', [])

        # 5. 清洗实体
        cleaned_entities = self._clean_entities(entities)

        # 6. 获取清洗后的实体名
        cleaned_unique_entity_names = {entity.get('name') for entity in cleaned_entities}

        # 7. 清洗关系
        cleaned_relations = self._clean_relations(cleaned_unique_entity_names, relations)

        # 8. 构建返回字典
        return {"entities": cleaned_entities, "relations": cleaned_relations}

    def _clean_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        1. 清洗无效实体（实体名没有）
        2. 阶段过长的实体名（实体名太长）
        3. 实体的标签是否在白名单中
        4，去重（同名同标签的实体只能存在一份）
        5. 返回
        Args:
            entities: LLM中提取的实体信息

        Returns:
            合法干净的实体信息
        """

        unique_seen = set()
        clean_entities_result = []

        # 1. 遍历所有的实体信息
        for entity in entities:

            # 1.1 获取实体名
            entity_name = str(entity.get('name', '')).strip()

            # 1.2 校验名是否存在
            if not entity_name:
                continue

            # 1.3 截取实体名
            if len(entity_name) > MAX_ENTITY_NAME_LENGTH:
                entity_name = entity_name[:15]

            # 1.4 获取实体标签
            entity_label = str(entity.get('label', '')).strip()

            # 1.5 判断标签是否在定义的实体标签是否白名单中
            if entity_label not in ALLOWED_ENTITY_LABELS:
                continue

            # 1.6 定义去重key
            unique_key = (entity_name, entity_label)

            # 1.7 判断是否是同一个实体（实体名+标签）
            if unique_key in unique_seen:
                continue
            unique_seen.add(unique_key)

            # 1.8 构建返回数据结构
            clean_entities = {"name": entity_name, "label": entity_label}

            # 1.9 判断实体的描述
            entity_describe = str(entity.get('description', '')).strip()
            if entity_describe:
                clean_entities['description'] = entity_describe

            # 1.10 将清洗后的实体信息存储到列表
            clean_entities_result.append(clean_entities)

        # 2. 返回最终清洗后的实体列表
        return clean_entities_result

    def _clean_relations(self, cleaned_unique_entity_names: Set[str], relations: List[Dict[str, Any]]) -> List[
        Dict[str, Any]]:
        """
        清洗关系:
        1. 清洗关系的头尾节点是否不存在
        2. 截取头尾实体名过长
        3. 校验头尾实体名是否有效（悬空关系处理）
        4. 校验每一个关系的类型是否关系类型的白名单
        5. 返回

        Args:
            cleaned_unique_entity_names: 所有唯一的实体名集合
            relations:  LLM中提取的关系信息

        Returns:
            List[Dict[str,Any]] 合法干净的关系信息
        """
        clean_relations_result = []
        # 1. 遍历所有的关系
        for relation in relations:

            # 1.1 提取头（head）实体名
            head_entity_name = str(relation.get('head', '')).strip()

            # 1.2 提取尾 (tail) 实体名
            tail_entity_name = str(relation.get('tail', '')).strip()

            # 1.3 判断头尾实体是否有任意一个不存在
            if not head_entity_name or not tail_entity_name:
                continue

            # 1.4 判断头尾实体名是否超过阈值
            if len(head_entity_name) > MAX_ENTITY_NAME_LENGTH:
                head_entity_name = head_entity_name[:MAX_ENTITY_NAME_LENGTH]

            if len(tail_entity_name) > MAX_ENTITY_NAME_LENGTH:
                tail_entity_name = tail_entity_name[:MAX_ENTITY_NAME_LENGTH]

            # 1.5 判断头尾实体名是否有效
            if head_entity_name not in cleaned_unique_entity_names or tail_entity_name not in cleaned_unique_entity_names:
                continue

            # 1.6 获取关系类型
            relation_type = str(relation.get('type', '')).strip()

            # 1.7 判断关系类型是否在关系类型的白名单中
            if relation_type not in ALLOWED_RELATION_TYPES:
                # TODO 思路：反哺白名单
                relation_type = DEFAULT_RELATION_TYPES

            # 1.8 构建最终关系链的数据结构
            cleaned_relation = {"head": head_entity_name, "tail": tail_entity_name, "type": relation_type}

            # 1.9 将清洗后最终的关系链放到最终的结果中
            clean_relations_result.append(cleaned_relation)

        return clean_relations_result

    def _validate_get_inputs(self, state: ImportGraphState) -> Tuple[List[Dict[str, Any]], str]:
        self.log_step("step1", "知识图谱构建参数校验")

        # 1. 获取基础字段
        chunks = state.get("chunks") or []
        global_item_name = str(state.get("item_name", "")).strip()

        # 2. 校验整体 chunks 是否存在
        if not chunks:
            raise ValueError("待提取图谱的切块(chunks)不存在，跳过图谱构建。")

        # 3. 逐个校验 Chunk 的有效性
        validated_chunks = []
        for i, chunk in enumerate(chunks):

            # 3.1 chunk 是否是字典
            if not isinstance(chunk, dict):
                self.logger.warning(f"第 {i} 个 chunk 不是字典类型，已抛弃。")
                continue

            # 3.2 处理 chunk_id
            raw_id = chunk.get("chunk_id")
            chunk_id = str(raw_id).strip() if raw_id is not None else f"kg_chunk_temp_{i}"

            # 3.3 获取 content 内容
            content = str(chunk.get("content", "")).strip()
            if not content:
                self.logger.warning(f"Chunk {chunk_id} 缺少 content，已抛弃。")
                continue

            # 3.4 获取 item_name（chunk 级别优先，全局兜底）
            chunk_item = str(chunk.get("item_name", "")).strip() or global_item_name
            if not chunk_item:
                self.logger.warning(f"Chunk {chunk_id} 缺少 item_name 归属，已抛弃。")
                continue

            # 3.5 更新 chunk 字段
            chunk["chunk_id"] = chunk_id
            chunk["item_name"] = chunk_item
            chunk["content"] = content

            # 3.6 加入有效列表
            validated_chunks.append(chunk)

        # 4. 校验清洗后是否还有有效数据
        if not validated_chunks:
            raise ValueError(f"经过清洗后，没有任何有效的 chunk（{len(validated_chunks)}）可用于构建图谱。")

        self.logger.info(f"参数校验完成: 原始 {len(chunks)} 块 -> 有效 {len(validated_chunks)} 块。")

        return validated_chunks, global_item_name

    def _process_chunks_concurrently(self, stats: ProcessingStats, validated_chunks: List[Dict[str, Any]],
                                     milvus_client: MilvusClient, neo4j_driver,
                                     task_id: str = "", task_dir: str = ""):
        """
        多线程版本：
        多线程本质压榨CPU 和提高响应时间没有本质的关系
        Args:
            stats:
            validated_chunks:
            milvus_client:
            neo4j_driver:
            task_id:      任务 ID（用于 LLM 调用留档）
            task_dir:     任务目录（留档落到其下 llm_calls/）
        Returns:

        """

        with ThreadPoolExecutor(max_workers=4) as pool:
            # 1. 提交所有任务
            future_to_idx = {}
            for i, chunk in enumerate(validated_chunks):
                content = chunk.get("content")
                chunk_id = str(chunk.get("chunk_id"))
                item_name = chunk.get("item_name")

                # 像线程池中提交任务 返回任务对象
                future = pool.submit(
                    self._process_single_chunk,
                    chunk_id, item_name, content, milvus_client, neo4j_driver,
                    task_id, task_dir
                )
                future_to_idx[future] = (i, chunk_id)

            # 2. 收集结果（按完成顺序）（一定要让执行_process_chunks_concurrently方法的线程等所有任务做完）
            for future in as_completed(future_to_idx):
                idx, chunk_id = future_to_idx[future]
                try:

                    entity_count, relation_count = future.result()  # 任务的结果（_process_single_chunk 返回值）
                    stats.processed_chunks += 1
                    stats.total_entities += entity_count
                    stats.total_relations += relation_count
                except Exception as e:
                    stats.failed_chunks += 1
                    msg = f"切片 {chunk_id} 处理失败: {e}"
                    stats.errors.append(msg)
                    self.logger.error(msg)
