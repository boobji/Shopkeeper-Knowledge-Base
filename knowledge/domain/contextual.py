"""Contextual Retrieval：入库前为每个 chunk 生成上下文前缀。

方法来自 Anthropic 的 contextual retrieval 实践：切片丢失了"它出自哪份文档
的哪个章节"这一层语义，导致说明书类文档的章节性内容难以被正确命中。
在向量化之前，用 LLM 为每个 chunk 生成一句 50~120 字的定位说明
（"本段出自《XX 使用说明书》的『电池更换』章节，介绍了…"），拼在正文前
一起嵌入；原 content 不变，前缀单独存进 Milvus 的 context_prefix 字段备查。

失败策略：单个 chunk 生成失败不阻断导入，该 chunk 退化为无前缀（与旧行为一致）。
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from knowledge.utils.llm_client_util import get_llm_client

logger = logging.getLogger(__name__)

_CONTEXT_SYSTEM_PROMPT = (
    "你是文档检索系统的辅助助手。给定文档标题、章节标题和正文片段，"
    "请输出一句简短的上下文说明，介绍这个片段出自哪里、讲什么，"
    "帮助它在语义检索中被更准确地命中。"
    "只输出这一句话本身，不要任何前缀、引号、编号或解释。"
)

_CONTEXT_USER_TEMPLATE = (
    "文档标题：{file_title}\n"
    "章节标题：{title}\n"
    "正文片段：\n{content}"
)

_CONTEXT_MAX_CHARS = 150
_CONTENT_SAMPLE_CHARS = 800


def generate_chunk_context(llm_client, file_title: str, title: str, content: str) -> str:
    """为单个 chunk 生成上下文前缀，失败返回空串。"""
    try:
        response = llm_client.invoke([
            SystemMessage(content=_CONTEXT_SYSTEM_PROMPT),
            HumanMessage(content=_CONTEXT_USER_TEMPLATE.format(
                file_title=file_title,
                title=title,
                content=content[:_CONTENT_SAMPLE_CHARS],
            )),
        ])
        text = str(getattr(response, "content", "") or "").strip().strip('"“”「」')
        # 兜底清洗：截断超长输出
        return text[:_CONTEXT_MAX_CHARS]
    except Exception as e:
        logger.warning(f"生成上下文前缀失败（该 chunk 退化为无前缀）: {e}")
        return ""


def enrich_chunks_with_context(chunks: List[Dict[str, Any]], config) -> None:
    """就地 为每个 chunk 设置 context_prefix（并发生成，失败置空串）。"""
    llm_client = get_llm_client(timeout=config.contextual_timeout)
    file_title = ""
    for chunk in chunks:
        file_title = chunk.get("file_title") or file_title
        chunk["context_prefix"] = ""

    with ThreadPoolExecutor(max_workers=max(1, config.contextual_max_workers)) as pool:
        futures = {
            pool.submit(
                generate_chunk_context,
                llm_client,
                chunk.get("file_title") or file_title,
                chunk.get("title", ""),
                chunk.get("content", ""),
            ): chunk
            for chunk in chunks
        }
        done = 0
        for future in as_completed(futures):
            chunk = futures[future]
            chunk["context_prefix"] = future.result()
            done += 1
            if done % 5 == 0 or done == len(chunks):
                logger.info(f"上下文前缀生成进度: {done}/{len(chunks)}")
