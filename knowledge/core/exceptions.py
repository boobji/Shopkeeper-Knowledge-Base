"""流程异常统一基类。

导入/查询两侧各自派生（ImportProcessError / QueryProcessError），
但共享同一套消息格式与公共基类，便于上层统一捕获。
"""


class ProcessError(Exception):
    """流程处理基础异常。"""

    def __init__(self, message: str, node_name: str = "", cause: Exception = None):
        self.node_name = node_name
        self.cause = cause
        super().__init__(message)

    def __str__(self):
        parts = []
        if self.node_name:
            parts.append(f"[{self.node_name}]")
        parts.append(super().__str__())
        if self.cause:
            parts.append(f"(原因: {self.cause})")
        return " ".join(parts)
