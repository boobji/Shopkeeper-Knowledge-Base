"""查询流程节点基类（共享 BaseNode 的查询侧配置）。"""

from knowledge.processor.base import BaseNode as _SharedBaseNode
from knowledge.processor.base import T, setup_logging  # noqa: F401 重新导出，保持节点导入路径不变
from knowledge.processor.query_process.config import get_config
from knowledge.processor.query_process.exceptions import QueryProcessError


class BaseNode(_SharedBaseNode):
    """查询流程节点基类（节点进度通过 SSE 推送）。"""

    log_domain = "query"
    push_progress = True
    process_error = QueryProcessError

    @classmethod
    def _default_config(cls):
        return get_config()
