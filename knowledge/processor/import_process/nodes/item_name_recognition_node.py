import re
from typing import Tuple, List, Dict, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import DataType

from knowledge.processor.import_process.base import (BaseNode)
from knowledge.processor.import_process.exceptions import ValidationError, EmbeddingError, LLMError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.config import get_config
from knowledge.utils.bge_m3_embedding_util import get_bge_m3_embedding_model
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.utils.llm_call_logger import log_llm_call
from knowledge.domain.filters import normalize_item_name
from knowledge.prompts.item_name_prompt import ITEM_NAME_SYSTEM_PROMPT, \
    ITEM_NAME_USER_PROMPT_TEMPLATE
from knowledge.utils.milvus_util import get_milvus_client


class ItemNameRecognitionNode(BaseNode):

    name = "item_name_recognition_node"

    # 型号串：形如 RS-12 / UT890D / S7-1200 / B520 / KF-21B18
    MODEL_CODE_RE = re.compile(r"[A-Za-z]{1,}[\- ]?\d{2,}[A-Za-z0-9\-]*")

    def process(self, state: ImportGraphState,) -> ImportGraphState:
        # 1.参数校验
        chunks, file_title, config = self._validate_inputs(state)
        # 2.构建LLM上下文（结构优先采样，而非简单取前 K 块）
        item_name_context = self._prepare_item_name_context(chunks, config, file_title)

        # 调用LLM
        item_name = self._recognition_item_name_by_llm(
            file_title, item_name_context,
            task_id=state.get('task_id') or '', task_dir=state.get('file_dir') or '',
        )

        # 3.嵌入商品名
        dense_vector, sparse_vector = self._embedding_item_name(item_name)

        # 4.存储到milvus向量库
        self._save_milvus(dense_vector, sparse_vector, file_title, item_name, config)

        # 5.回填item_name信息
        self._fill_item_name(item_name, state, chunks, config)
        return state

    def _validate_inputs(self, state: ImportGraphState):
        self.log_step('step1','校验输入参数')
        config = get_config()
        # 1.获取state的file_title以及chunks
        file_title = state.get('file_title')
        chunks = state.get('chunks')

        # 2.判断提取到的参数
        if not file_title:
            raise ValidationError('文件标题为空',self.name)
        if not chunks or not isinstance(chunks,list):
            raise ValidationError('chunks为空或无效',self.name)
        item_name_chunk_k = config.item_name_chunk_k
        if not item_name_chunk_k or item_name_chunk_k <= 0:
            raise ValidationError('item_name_chunk_k为空或无效',self.name)

        self.logger.info(f'检测到文件:{file_title}对应的切片长度{len(chunks)}')
        # 3.返回
        return chunks, file_title, config

    def _prepare_item_name_context(self, chunks: Optional[List[Dict[str, Any]]], config,
                                   file_title: str = "") -> str:
        """构建商品名提取上下文 —— 结构优先采样。

        旧实现固定取"前 K 个切片"，而说明书的前几页恰恰是封面、目录、通用安全声明
        和厂商服务承诺（实测：海尔微波炉抽取时被"1+5 成套服务"干扰，扫描件首块常常
        是空白 OCR 结果）。这里改为按"含金量"挑片段：

            命中型号串 +2 / 落在真实章节下 +1 / 中文主体 +1 / 位置靠前略加权

        挑出来后再按原始顺序拼接，保持文档阅读顺序，避免 LLM 看到乱序内容。
        """
        self.log_step('step2', '构建商品名提取的上下文')
        if not chunks:
            return ''

        # 候选池放宽到 3 倍 K（或至少 15 块），避免前几块被占满时无米下锅
        pool_size = max(config.item_name_chunk_k * 3, 15)
        candidates = []
        for index, chunk in enumerate(chunks[:pool_size]):
            if not isinstance(chunk, dict):
                continue
            content = (chunk.get('content') or '').strip()
            if len(content) < 10:
                continue  # 空壳 OCR 片段

            title = (chunk.get('title') or '').strip()
            score = 0.0
            if self.MODEL_CODE_RE.search(content):
                score += 2.0                      # 型号串是最强的商品名信号
            if title and title != file_title:
                score += 1.0                      # 落在真实章节下的片段更有信息量
            if chunk.get('lang') != 'mixed':
                score += 1.0                      # 排除多语言/符号页，boilerplate 已由清洗节点剔除
            score -= index * 0.1                  # 同分时偏向前文
            candidates.append((score, index, content))

        if not candidates:  # 极端情况：全部不可用，退回原逻辑
            candidates = [(0.0, i, (c.get('content') or '').strip())
                          for i, c in enumerate(chunks[:config.item_name_chunk_k])
                          if isinstance(c, dict) and (c.get('content') or '').strip()]

        # 取分数最高的 K 个，再按文档顺序还原
        candidates.sort(key=lambda x: (-x[0], x[1]))
        picked = sorted(candidates[:config.item_name_chunk_k], key=lambda x: x[1])

        result, total = [], 0
        for score, index, content in picked:
            spices = f'[切片]-{index + 1}-{content}'
            total += len(spices)
            result.append(spices)
            if total > config.item_name_chunk_size:
                break

        return '\n\n'.join(result)[:config.item_name_chunk_size]

    def _recognition_item_name_by_llm(self, file_title: str, item_name_context: str,
                                      task_id: str = '', task_dir: str = '') -> str:
        self.log_step('step3', 'LLM识别商品名')
        # 1.实例化LLM客户端
        # 有意降级：导入场景下 LLM 不可用时回退文件标题，而不是让整个导入失败
        try:
            llm_client = get_llm_client()
        except LLMError as e:
            self.logger.error(f'LLM初始化失败({e})，安全回退到标题名：{file_title}')
            return file_title

        # 2.构建LLM提示词(格式化用户提示词模板)
        prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(file_title=file_title, context=item_name_context)
        # prompt_template = ChatPromptTemplate.from_template(ITEM_NAME_USER_PROMPT_TEMPLATE)
        # prompt_template.format_prompt()
        # prompt_template.invoke()

        # 3.调用模型（# str[] # promptvalue)
        messages = [SystemMessage(content=ITEM_NAME_SYSTEM_PROMPT), HumanMessage(content=prompt)]
        t0 = time.perf_counter()
        try:
            llm_response = llm_client.invoke(messages)

            # 获取模型输出内容
            item_name = getattr(llm_response, 'content', '').strip()

            # 3.1 留档本次调用（完整提示词 + 响应）
            log_llm_call('import_item_name', messages=messages, response=llm_response,
                         task_id=task_id, task_dir=task_dir,
                         latency_ms=(time.perf_counter() - t0) * 1000)

            # 判断
            if not item_name or item_name.upper() == 'UNKNOWN':
                self.logger.warning(f'LLM无法提取有效的商品名，安全回退到标题名：{file_title}')
                return file_title
            self.logger.info(f'提取到的商品名:{item_name}')
            return item_name
        except Exception as e:
            # 3.2 失败同样留档，便于排查是提示词问题还是服务问题
            log_llm_call('import_item_name', messages=messages, task_id=task_id, task_dir=task_dir,
                         latency_ms=(time.perf_counter() - t0) * 1000, error=str(e))
            self.logger.error(f'LLM调用失败，安全回退到标题名：{file_title}')
            return file_title

    def _embedding_item_name(self, item_name: str) -> Optional[Tuple[list, dict[Any, Any]]]:
        self.log_step('step4', '嵌入商品名')
        try:
            # 1.获取嵌入模型
            embedding_model = get_bge_m3_embedding_model()

            # 2.嵌入item_name
            embedding_result = embedding_model.encode_documents([item_name])

            # 3.获取稠密和稀疏向量
            dense_vector = embedding_result['dense'][0].tolist()

            # 注意：不同版本 BGE-M3 返回的 sparse 可能是 coo_array / csr_matrix / csr_array，
            # 统一转 COO 后按 row==0 取第 1 篇文档的 token，避免 'coo_array' has no attribute 'indices'
            sparse = embedding_result['sparse'].tocoo()
            row0 = sparse.row == 0
            tokenids = sparse.col[row0].tolist()
            weights = sparse.data[row0].tolist()
            sparse_vector = dict(zip(tokenids, weights))

            return dense_vector, sparse_vector
        except Exception as e:
            self.log_step(f'嵌入向量:{item_name}失败，原因是{str(e)}')
            raise EmbeddingError(f'嵌入向量:{item_name}失败，原因是{str(e)}',self.name)

    def _save_milvus(self, dense_vector: list[float], sparse_vector: dict[str,Any], file_title: str,
            item_name: str,config):
        self.log_step("step5", "保存到 Milvus")

        if not config.milvus_url or not config.item_name_collection:
            self.logger.warning("Milvus 配置不完整，跳过保存")
            return
        try:
            # 1. 获取 Milvus 客户端
            client = get_milvus_client()
            if client is None:
                return

            # 2. 获取集合名字
            collection_name = config.item_name_collection

            # 3. 检查并创建集合
            if not client.has_collection(collection_name=collection_name):
                self._create_item_name_collection(client, collection_name)

            # 4. 准备数据
            data = {
                "file_title": file_title,
                "item_name": item_name,
                "item_name_norm": normalize_item_name(item_name),
            }

            # 5. 构建稠密向量
            if dense_vector is not None:
                data["dense_vector"] = dense_vector

            # 6. 构建稀疏向量
            if sparse_vector is not None:
                data["sparse_vector"] = sparse_vector

            # 7. 插入数据
            result = client.insert(collection_name=collection_name, data=[data])
            self.logger.info(f"已保存到 Milvus，ID: {result['ids'][0]}")

        except Exception as e:
            self.logger.warning(f"Milvus 保存失败: {e}")

    def _create_item_name_collection(self, client, collection_name: str):
        """创建 item_name 集合"""
        self.logger.info(f"创建集合: {collection_name}")

        # 1. 定义字段
        schema = client.create_schema(enable_dynamic_fields=True)

        schema.add_field(field_name="ids", datatype=DataType.VARCHAR,
                         is_primary=True, auto_id=True, max_length=100)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name_norm", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 2. 创建索引
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP"
        )

        # 3. 创建集合
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )
        self.logger.info(f"集合 {collection_name} 创建成功")

    def _fill_item_name(self, item_name: str, state:ImportGraphState, chunks:list[dict[Any,Any]], config):
        for chunk in chunks:
            chunk["item_name"] = item_name # 方便下游大模型
        state["item_name"] = item_name # 方便程序员
