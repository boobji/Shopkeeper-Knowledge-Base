"""导入流程节点基类（共享 BaseNode 的导入侧配置）。"""

from knowledge.processor.base import BaseNode as _SharedBaseNode
from knowledge.processor.base import T, setup_logging  # noqa: F401 重新导出，保持节点导入路径不变
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import ImportProcessError


class BaseNode(_SharedBaseNode):
    """导入流程节点基类。"""

    log_domain = "import"
    push_progress = False
    process_error = ImportProcessError

    @classmethod
    def _default_config(cls):
        return get_config()
