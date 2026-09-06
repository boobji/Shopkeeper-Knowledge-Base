"""外部资源连接的统一单例管理（线程安全懒加载）。

所有客户端/驱动只在这里创建与缓存：
- 双重检查 + 线程锁：FastAPI BackgroundTasks 跑在线程池里，并发导入时
  无锁懒加载会竞态重复建连。
- 失败不缓存：创建失败返回 None，下次调用重试（与历史行为一致）。
- MinIO 建桶只在首次创建客户端时尝试一次，不做每次调用的 bucket_exists。
"""

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_singletons_lock = threading.RLock()

# ------------------------------------------------------------------
# Milvus
# ------------------------------------------------------------------
_milvus_client = None


def get_milvus_client():
    global _milvus_client
    if _milvus_client is not None:
        return _milvus_client
    with _singletons_lock:
        if _milvus_client is not None:
            return _milvus_client
        try:
            from pymilvus import MilvusClient

            milvus_uri = os.getenv('MILVUS_URL', 'http://127.0.0.1:19530')
            _milvus_client = MilvusClient(uri=milvus_uri)
            return _milvus_client
        except Exception as e:
            logger.error(f"Milvus 客户端创建失败: {e}")
            return None


# ------------------------------------------------------------------
# Neo4j
# ------------------------------------------------------------------
_neo4j_driver = None


def get_neo4j_driver():
    global _neo4j_driver
    if _neo4j_driver is not None:
        return _neo4j_driver
    with _singletons_lock:
        if _neo4j_driver is not None:
            return _neo4j_driver
        try:
            from neo4j import GraphDatabase

            uri = os.getenv("NEO4J_URI")
            username = os.getenv("NEO4J_USERNAME")
            password = os.getenv("NEO4J_PASSWORD")

            logger.info(f"正在初始化 Neo4j 驱动，连接 URI: {uri}")
            driver = GraphDatabase.driver(uri=uri, auth=(username, password))
            # 驱动默认懒连接，verify_connectivity 让账号密码错误/网络不通当场暴露
            driver.verify_connectivity()
            _neo4j_driver = driver
            logger.info("Neo4j 驱动初始化成功并已验证连接！")
            return _neo4j_driver
        except Exception as e:
            logger.error(f"初始化 Neo4j 驱动失败: {e}", exc_info=True)
            return None


# ------------------------------------------------------------------
# MongoDB（会话历史）
# ------------------------------------------------------------------
class HistoryMongoTool:
    """MongoDB 历史对话记录库连接（原生 PyMongo）。"""

    def __init__(self):
        try:
            from pymongo import MongoClient

            self.mongo_url = os.getenv("MONGO_URL")
            self.db_name = os.getenv("MONGO_DB_NAME")

            self.client = MongoClient(self.mongo_url)
            self.db = self.client[self.db_name]
            self.chat_message = self.db["chat_message"]

            # 加速按会话查询
            self.chat_message.create_index([("session_id", 1), ("ts", -1)])

            logger.info(f"Successfully connected to MongoDB: {self.db_name}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    def get_collection(self):
        return self.chat_message


_history_mongo_tool: Optional[HistoryMongoTool] = None


def get_history_mongo_tool() -> HistoryMongoTool:
    global _history_mongo_tool
    if _history_mongo_tool is not None:
        return _history_mongo_tool
    with _singletons_lock:
        if _history_mongo_tool is not None:
            return _history_mongo_tool
        _history_mongo_tool = HistoryMongoTool()
        return _history_mongo_tool


# ------------------------------------------------------------------
# MinIO
# ------------------------------------------------------------------
_minio_client = None


def get_minio_client():
    global _minio_client
    if _minio_client is not None:
        return _minio_client
    with _singletons_lock:
        if _minio_client is not None:
            return _minio_client
        try:
            from minio import Minio

            client = Minio(
                os.getenv("MINIO_ENDPOINT"),
                access_key=os.getenv("MINIO_ACCESS_KEY"),
                secret_key=os.getenv("MINIO_SECRET_KEY"),
                secure=os.getenv("MINIO_SECURE", "").strip().lower() in ("1", "true", "yes", "on"),
            )
            # 桶只在首次建连时确保存在；日常调用不再反复 bucket_exists
            bucket_name = os.getenv("MINIO_BUCKET_NAME")
            if not client.bucket_exists(bucket_name):
                client.make_bucket(bucket_name)
                logger.info(f"桶 {bucket_name} 已创建")
            else:
                logger.info(f"桶 {bucket_name} 已存在")
            _minio_client = client
            return _minio_client
        except Exception as e:
            logger.error(f"MinIO 客户端创建失败: {e}")
            return None
