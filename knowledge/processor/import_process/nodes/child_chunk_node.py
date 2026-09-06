"""子块切分入库节点（Parent-Child 索引的导入侧）。

父块（2000 字符）负责给 LLM 提供上下文，子块（~450 字符）负责被检索命中：
- 为每个父块生成稳定标识 chunk_uid（随父块一起入 kb_chunks）；
- 把父块正文切成子块，独立向量化后写入子块集合（默认 kb_child_chunks），
  子块记录 parent_uid，查询侧命中子块后按 uid 反查父块。

降级策略：子块是检索增强层，生成/入库失败只告警不阻断导入，
查询侧在子块集合为空时自动回退检索父块集合。
"""

import logging
import uuid
from typing import Any, Dict, List

from pymilvus import DataType, MilvusClient

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.domain.filters import normalize_item_name
from knowledge.utils.bge_m3_embedding_util import get_bge_m3_embedding_model
from knowledge.utils.milvus_util import get_milvus_client
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class ChildChunkNode(BaseNode):
    name = "child_chunk_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        config = get_config()

        # 0. 未启用直接跳过
        if not config.parent_child_enabled:
            self.logger.info("Parent-Child 索引未启用，跳过子块切分")
            return state

        chunks: List[Dict[str, Any]] = state.get('chunks') or []
        if not chunks:
            self.logger.warning("无父块可切分子块，跳过")
            return state

        try:
            # 1. 为父块生成稳定标识（随父块入库，供子块反查）
            for chunk in chunks:
                if not chunk.get('chunk_uid'):
                    chunk['chunk_uid'] = uuid.uuid4().hex

            # 2. 父块正文切子块
            children = self._build_children(chunks, config)
            self.logger.info(f"父块 {len(chunks)} 块切出子块 {len(children)} 块")
            if not children:
                return state

            # 3. 子块向量化
            self._embed_children(children, config)

            # 4. 子块入库（按 item_name 增量替换）
            milvus_client = get_milvus_client()
            self._ensure_collection(milvus_client, config.child_chunks_collection)
            self._delete_existing_items(milvus_client, config.child_chunks_collection, children)
            milvus_client.insert(collection_name=config.child_chunks_collection, data=children)
            self.logger.info(f"子块入库完成: {len(children)} 条 -> {config.child_chunks_collection}")
        except Exception as e:
            # 有意降级：子块是增强层，失败不影响父块检索（查询侧自动回退父块集合）
            self.logger.warning(f"子块切分/入库失败（查询将回退父块检索）: {e}")

        return state

    # ------------------------------------------------------------------
    def _build_children(self, chunks: List[Dict[str, Any]], config) -> List[Dict[str, Any]]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.child_chunk_size,
            chunk_overlap=config.child_chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
            keep_separator=False,
        )
        children: List[Dict[str, Any]] = []
        for chunk in chunks:
            content = chunk.get('content') or ''
            if not content:
                continue
            pieces = splitter.split_text(content)
            for index, piece in enumerate(pieces, 1):
                children.append({
                    "parent_uid": chunk['chunk_uid'],
                    "content": piece,
                    "title": chunk.get('title', ''),
                    "parent_title": chunk.get('parent_title', ''),
                    "file_title": chunk.get('file_title', ''),
                    "item_name": chunk.get('item_name', ''),
                    "item_name_norm": normalize_item_name(chunk.get('item_name', '')),
                    "context_prefix": chunk.get('context_prefix') or '',
                    "part": index,
                })
        return children

    def _embed_children(self, children: List[Dict[str, Any]], config) -> None:
        model = get_bge_m3_embedding_model()
        batch_size = 16
        for i in range(0, len(children), batch_size):
            batch = children[i:i + batch_size]
            # 子块嵌入带上父块的上下文前缀与章节标题，保持与父块向量空间一致
            texts = []
            for child in batch:
                prefix = child.get('context_prefix') or ''
                head = f"{prefix}\n{child['item_name']}\n{child['title']}" if prefix \
                    else f"{child['item_name']}\n{child['title']}"
                texts.append(f"{head}\n{child['content']}")
            result = model.encode_documents(documents=texts)
            csr = result['sparse'].tocsr()
            for index, child in enumerate(batch):
                child['dense_vector'] = result['dense'][index].tolist()
                start = csr.indptr[index]
                end = csr.indptr[index + 1]
                child['sparse_vector'] = dict(zip(
                    csr.indices[start:end].tolist(),
                    csr.data[start:end].tolist(),
                ))

    def _delete_existing_items(self, milvus_client: MilvusClient, collection_name: str,
                               children: List[Dict[str, Any]]):
        item_names = sorted({c.get("item_name") for c in children if c.get("item_name")})
        for name in item_names:
            try:
                milvus_client.delete(collection_name=collection_name, filter=f'item_name == "{name}"')
            except Exception as e:
                self.logger.warning(f"清理子块旧数据失败（继续写入）: {e}")

    def _ensure_collection(self, milvus_client: MilvusClient, collection_name: str, dim: int = 1024) -> None:
        if milvus_client.has_collection(collection_name):
            return
        schema = milvus_client.create_schema(enable_dynamic_field=True)
        schema.add_field("chunk_id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("content", DataType.VARCHAR, max_length=65535)
        schema.add_field("title", DataType.VARCHAR, max_length=65535)
        schema.add_field("parent_title", DataType.VARCHAR, max_length=65535)
        schema.add_field("file_title", DataType.VARCHAR, max_length=65535)
        schema.add_field("item_name", DataType.VARCHAR, max_length=65535)
        schema.add_field("item_name_norm", DataType.VARCHAR, max_length=65535)
        schema.add_field("parent_uid", DataType.VARCHAR, max_length=64)
        index = milvus_client.prepare_index_params(collection_name=collection_name)
        index.add_index(field_name="dense_vector", index_name="dense_vector_index",
                        index_type="AUTOINDEX", metric_type="COSINE")
        index.add_index(field_name="sparse_vector", index_name="sparse_vector_index",
                        index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
        milvus_client.create_collection(collection_name=collection_name, schema=schema, index_params=index)
