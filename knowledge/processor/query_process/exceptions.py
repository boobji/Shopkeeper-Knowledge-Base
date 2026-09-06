"""查询流程自定义异常类

统一错误处理，提供更清晰的错误信息。
异常基类与消息格式在 core.exceptions.ProcessError 中统一定义。
"""

from knowledge.core.exceptions import (  # noqa: F401 存储类错误两侧共用 core 的统一定义
    EmbeddingError,
    MilvusError,
    Neo4jError,
    ProcessError,
    StorageError,
)


class QueryProcessError(ProcessError):
    """查询流程基础异常。"""
    pass


class StateFieldError(QueryProcessError):
    """状态字段错误。

    从 state 中获取必需字段缺失、为空或类型不符时抛出。

    Attributes:
        field_name: 缺失或无效的字段名称。
        expected_type: 期望的字段类型（可选）。
    """

    def __init__(
            self,
            node_name: str = "",
            field_name: str = "",
            expected_type: type = None,
            message: str = "",
            cause: Exception = None,
    ):
        self.field_name = field_name
        self.expected_type = expected_type
        if not message:
            message = f"状态字段 '{field_name}' 缺失或无效"
            if expected_type:
                message += f"，期望类型: {expected_type.__name__}"
        super().__init__(message, node_name=node_name, cause=cause)


class ConfigurationError(QueryProcessError):
    """配置错误。

    环境变量缺失或配置值无效时抛出。
    """
    pass


class SearchError(QueryProcessError):
    """搜索错误。

    向量搜索、混合搜索或网络搜索失败时抛出。
    """
    pass


class LLMError(QueryProcessError):
    """LLM 调用错误。

    API 调用失败、响应解析失败时抛出。
    """
    pass


class MongoDBError(StorageError):
    """MongoDB 存储错误。

    MongoDB 数据库操作失败时抛出。
    """
    pass


class ValidationError(QueryProcessError):
    """数据验证错误。

    输入数据不符合预期时抛出。
    """
    pass


class EntityAlignmentError(QueryProcessError):
    """实体对齐错误。

    知识图谱实体对齐过程失败时抛出。
    """
    pass


class RerankError(QueryProcessError):
    """重排序错误。

    文档重排序过程失败时抛出。
    """
    pass


class ItemNameConfirmError(QueryProcessError):
    """商品名称确认错误。

    商品名称识别或确认过程失败时抛出。
    """
    pass
