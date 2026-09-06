from typing import Tuple, List, Dict, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import DataType

from knowledge.processor.import_process.base import (BaseNode)
from knowledge.processor.import_process.exceptions import ValidationError, EmbeddingError, LLMError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.config import get_config
from knowledge.utils.bge_m3_embedding_util import get_bge_m3_embedding_model
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.domain.filters import normalize_item_name
from knowledge.prompts.item_name_prompt import ITEM_NAME_SYSTEM_PROMPT, \
    ITEM_NAME_USER_PROMPT_TEMPLATE
from knowledge.utils.milvus_util import get_milvus_client


class ItemNameRecognitionNode(BaseNode):

    name = "item_name_recognition_node"

    def process(self, state: ImportGraphState,) -> ImportGraphState:
        # 1.参数校验
        chunks, file_title, config = self._validate_inputs(state)
        # 2.构建LLM上下文
        item_name_context = self._prepare_item_name_context(chunks, config)

        # 调用LLM
        item_name = self._recognition_item_name_by_llm(file_title, item_name_context)

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

    def _prepare_item_name_context(self, chunks: Optional[List[Dict[str, Any]]], config):
        self.log_step('step2','构建商品名提取的上下文')
        result = []
        total = 0
        for index, chunk in enumerate(chunks[:config.item_name_chunk_k]):
            # 1.判断chunk的类型
            if not isinstance(chunk,dict):
                continue

            # 2.提取
            content = chunk.get('content')
            spices = f'[切片]-{index + 1}-{content}'

            # 3.计算长度
            total += len(spices)
            result.append(spices)

            # 4.判断收集到的长度是否超过阈值
            if total > config.item_name_chunk_size:
                break

        return '\n\n'.join(result)[:config.item_name_chunk_size]

    def _recognition_item_name_by_llm(self, file_title: str, item_name_context: str) -> str:
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
        try:
            llm_response = llm_client.invoke([
                SystemMessage(content=ITEM_NAME_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ])

            # 获取模型输出内容
            item_name = getattr(llm_response, 'content', '').strip()

            # 判断
            if not item_name or item_name.upper() == 'UNKNOWN':
                self.logger.warning(f'LLM无法提取有效的商品名，安全回退到标题名：{file_title}')
                return file_title
            self.logger.info(f'提取到的商品名:{item_name}')
            return item_name
        except Exception:
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
