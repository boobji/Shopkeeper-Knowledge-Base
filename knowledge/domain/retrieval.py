"""检索公共链路（查询侧向量/HyDE 与 eval 共用）。

Parent-Child 检索：优先检索子块集合（小块精度高），命中后按 chunk_uid
反查父块返回（大块上下文足）；子块集合不可用/为空时自动回退父块集合。

回退链（逐级兜底，尽量不让该路空手而归）：
  子块+商品过滤 → 子块全库 → 父块+商品过滤 → 父块全库
"""

import logging
from typing import Any, Dict, List, Optional

from knowledge.domain.filters import build_item_name_norm_expr
from knowledge.utils.milvus_util import (
    create_hybrid_search_requests,
    execute_hybrid_search_query,
)

logger = logging.getLogger(__name__)

_PARENT_OUTPUT_FIELDS = ["chunk_id", "content", "title", "parent_title", "file_title", "item_name"]
_CHILD_OUTPUT_FIELDS = ["chunk_uid", "content", "title", "item_name"]


def _search_once(milvus_client, collection_name: str, dense_vector, sparse_vector,
                 expr: Optional[str], limit: int, output_fields: List[str]):
    reqs = create_hybrid_search_requests(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        expr=expr or None,
        limit=limit,
    )
    return execute_hybrid_search_query(
        milvus_client=milvus_client,
        collection_name=collection_name,
        search_requests=reqs,
        norm_score=True,
        limit=limit,
        output_fields=output_fields,
    )


def _fetch_parents_by_uids(milvus_client, parent_collection: str, parent_uids: List[str],
                           batch_size: int = 50) -> Dict[str, Dict[str, Any]]:
    """按 chunk_uid 批量反查父块记录，返回 uid -> chunk 字典。"""
    result: Dict[str, Dict[str, Any]] = {}
    unique_uids = list(dict.fromkeys(parent_uids))
    for i in range(0, len(unique_uids), batch_size):
        batch = unique_uids[i:i + batch_size]
        uid_list = ", ".join(f'"{u}"' for u in batch)
        try:
            rows = milvus_client.query(
                collection_name=parent_collection,
                filter=f"chunk_uid in [{uid_list}]",
                output_fields=_PARENT_OUTPUT_FIELDS + ["chunk_uid"],
            )
        except Exception as e:
            logger.error(f"按 chunk_uid 反查父块失败: {e}")
            continue
        for row in rows or []:
            uid = row.get("chunk_uid")
            if uid:
                result[uid] = row
    return result


def _wrap_entity(record: Dict[str, Any]) -> Dict[str, Any]:
    """把父块记录包装成与 Milvus 命中一致的结构，下游 RRF/重排无需感知差异。"""
    return {"id": None, "distance": 1.0, "entity": record}


def search_chunks(milvus_client,
                  dense_vector,
                  sparse_vector,
                  item_names: List[str],
                  limit: int,
                  chunks_collection: str,
                  child_collection: Optional[str] = None,
                  logger_=None) -> List[Dict[str, Any]]:
    """向量/HyDE 路的统一检索入口，返回与原 Milvus 命中同构的结果列表。

    item_names 为当前问题确认出的商品名（可为空 = 不过滤）。
    """
    log = logger_ or logger
    expr = build_item_name_norm_expr(item_names)

    # 构造回退链：子块集合优先（存在才参与），父块集合兜底
    attempts = []
    if child_collection:
        attempts.append((child_collection, _CHILD_OUTPUT_FIELDS, "child"))
    attempts.append((chunks_collection, _PARENT_OUTPUT_FIELDS, "parent"))

    for collection_name, output_fields, kind in attempts:
        # a) 带商品过滤检索
        try:
            res = _search_once(milvus_client, collection_name, dense_vector, sparse_vector,
                               expr, limit, output_fields)
        except Exception as e:
            # 子块集合不存在等场景：跳过该级，走下一级
            log.warning(f"检索集合 {collection_name} 失败，尝试下一级回退: {e}")
            continue

        hits = res[0] if res else []

        # b) 带过滤为空 → 该集合内回退全库检索
        if not hits and expr:
            log.warning(f"集合 {collection_name} 带商品过滤检索为空，回退全库检索")
            res = _search_once(milvus_client, collection_name, dense_vector, sparse_vector,
                               None, limit, output_fields)
            hits = res[0] if res else []

        if not hits:
            continue

        # c) 子块命中 → 反查父块并按子块排名顺序返回
        if kind == "child":
            parent_uids = []
            for hit in hits:
                entity = hit.get("entity") or {}
                uid = entity.get("chunk_uid")
                if uid:
                    parent_uids.append(str(uid))
            if not parent_uids:
                continue
            uid_map = _fetch_parents_by_uids(milvus_client, chunks_collection, parent_uids)
            parents = []
            seen = set()
            for uid in parent_uids:
                record = uid_map.get(uid)
                if not record:
                    continue
                key = record.get("chunk_id")
                if key in seen:
                    continue
                seen.add(key)
                parents.append(_wrap_entity(record))
            if parents:
                return parents
            log.warning("子块命中但父块反查为空，回退父块集合检索")
            continue

        return hits

    return []
