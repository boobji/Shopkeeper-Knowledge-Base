"""导入/查询流程共享的节点基类。

统一提供：日志命名、任务进度追踪（task_util）、可选的 SSE 进度推送、
异常包装。两侧差异通过类属性表达：
- log_domain:      日志命名空间（"import" / "query"）
- push_progress:   是否推送 SSE 节点进度（导入流程关闭）
- process_error:   节点异常统一包装成的流程异常类型
"""

from abc import ABC, abstractmethod
from typing import TypeVar

import logging

from knowledge.core.exceptions import ProcessError
from knowledge.core.logging import setup_logging
from knowledge.utils.sse_util import push_sse_event
from knowledge.utils.task_util import (
    add_done_task,
    add_running_task,
    get_done_task_list,
    get_running_task_list,
    get_task_status,
)

T = TypeVar("T")  # 泛型状态类型


class BaseNode(ABC):
    """LangGraph 节点基类，子类实现 process 方法。

    使用示例:
        class MyNode(BaseNode):
            name = "my_node"

            def process(self, state):
                return state

        # 作为 LangGraph 节点使用
        workflow.add_node("my_node", MyNode())
    """

    name: str = "base_node"
    log_domain: str = "node"
    push_progress: bool = False
    process_error: type = ProcessError

    def __init__(self, config=None):
        """初始化节点。

        Args:
            config: 配置对象，默认使用子类指定的全局配置。
        """
        self.config = config if config is not None else self._default_config()
        self.logger = logging.getLogger(f"{self.log_domain}.{self.name}")

    @classmethod
    def _default_config(cls):
        """子类侧基类覆盖：返回所属流程的全局配置单例。"""
        raise NotImplementedError

    def __call__(self, state: T) -> T:
        """节点执行入口（LangGraph 调用）。

        提供统一的日志输出、任务追踪、SSE 进度推送和异常包装。
        """
        task_id = state.get("task_id", "")
        is_stream = bool(state.get("is_stream", False))

        try:
            self.logger.info(f"--- {self.name} 开始 ---")
            if task_id:
                add_running_task(task_id, self.name)
                self._push_progress_if_needed(task_id, is_stream)

            result = self.process(state)

            self.logger.info(f"--- {self.name} 完成 ---")
            if task_id:
                add_done_task(task_id, self.name)
                self._push_progress_if_needed(task_id, is_stream)

            return result
        except Exception as e:
            self.logger.error(f"{self.name} 执行失败: {e}")
            raise self.process_error(
                message=str(e),
                node_name=self.name,
                cause=e
            )

    def _push_progress_if_needed(self, task_id: str, is_stream: bool):
        if not (task_id and self.push_progress and is_stream):
            return
        push_sse_event(task_id, "progress", {
            "status": get_task_status(task_id),
            "done_list": get_done_task_list(task_id),
            "running_list": get_running_task_list(task_id),
        })

    @abstractmethod
    def process(self, state: T) -> T:
        """节点核心处理逻辑，子类必须实现。"""
        raise NotImplementedError

    def log_step(self, step_name: str, message: str = ""):
        """记录步骤日志。"""
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)


__all__ = ["BaseNode", "T", "setup_logging"]
