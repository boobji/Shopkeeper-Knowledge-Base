"""Neo4j 驱动获取（单例由 core.connections 统一管理）。"""

from knowledge.core.connections import get_neo4j_driver  # noqa: F401 重新导出，保持调用方导入路径不变

__all__ = ["get_neo4j_driver"]
