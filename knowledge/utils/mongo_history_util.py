"""MongoDB 会话历史读写工具。

连接单例由 core.connections 统一管理（线程安全、懒加载）。
"""

import logging
from datetime import datetime
from typing import List, Dict, Any

from bson import ObjectId
from pymongo import DESCENDING

from knowledge.core.connections import get_history_mongo_tool, HistoryMongoTool  # noqa: F401 HistoryMongoTool 兼容别名

logger = logging.getLogger(__name__)


def clear_history(session_id: str) -> int:
    """清空指定会话的历史记录"""
    mongo_tool = get_history_mongo_tool()
    try:
        result = mongo_tool.chat_message.delete_many({"session_id": session_id})
        logger.info(f"Deleted {result.deleted_count} messages for session {session_id}")
        return result.deleted_count
    except Exception as e:
        logger.error(f"Error clearing history for session {session_id}: {e}")
        return 0


def save_chat_message(session_id: str, role: str, text: str, rewritten_query: str = "",
                      item_names: List[str] = None, message_id: str = None) -> str:
    """
    写入一条会话记录
    :param message_id: 主键（传入则按主键更新）
    :param rewritten_query: 重写后的查询
    :param session_id: 会话 ID
    :param role: 角色 (user/assistant)
    :param text: 对话内容
    :param item_names: 关联的商品名称 (可选、可多个)
    :return: 插入记录的 ObjectId 字符串
    """
    ts = datetime.now().timestamp()

    document = {
        "session_id": session_id,
        "role": role,
        "text": text,
        "rewritten_query": rewritten_query,
        "item_names": item_names,
        "ts": ts
    }

    mongo_tool = get_history_mongo_tool()
    if message_id:
        mongo_tool.chat_message.update_one({"_id": ObjectId(message_id)}, {"$set": document})
        return message_id
    result = mongo_tool.chat_message.insert_one(document)
    return str(result.inserted_id)


def update_message_item_names(ids: List[str], item_names: List[str]) -> int:
    """批量更新历史会话列表的 item_names（只补空，不覆盖已有值）"""
    mongo_tool = get_history_mongo_tool()
    try:
        result = mongo_tool.chat_message.update_many(
            {"_id": {"$in": [ObjectId(i) for i in ids]},
             "$or": [
                 {"item_names": {"$exists": False}},
                 {"item_names": []},
                 {"item_names": None}
             ]
             },
            {"$set": {"item_names": item_names}}
        )
        logger.info(f"Updated {result.modified_count} records to item_names: {item_names}")
        return result.modified_count
    except Exception as e:
        logger.error(f"Error updating history item_names: {e}")
        return 0


def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    查询最近 N 条对话记录 (返回原始字典格式)

    按时间倒序取最近 limit 条，再反转回时间正序（方便直接喂给 LLM）。
    """
    mongo_tool = get_history_mongo_tool()
    try:
        query = {"session_id": session_id}
        cursor = mongo_tool.chat_message.find(query).sort("ts", DESCENDING).limit(limit)
        messages = list(cursor)
        messages.reverse()
        return messages
    except Exception as e:
        logger.error(f"Error getting recent messages: {e}")
        return []
