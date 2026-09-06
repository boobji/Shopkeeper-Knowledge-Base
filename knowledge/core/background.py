"""后台任务执行工具。

统一用 daemon 线程承载长耗时的导入/查询任务：

- 为什么不用 Starlette BackgroundTasks：它跑在 anyio 的工作线程（非守护）里，
  uvicorn 优雅关闭时会等待后台任务跑完（"Waiting for background tasks to
  complete"）。导入/查询动辄几分钟且无超时，表现为进程 Ctrl+C 后无法结束。
- daemon 线程不会阻塞进程退出；代价是进程退出时未完成任务被直接丢弃
  （任务状态本来就在内存里，重启即失效，可接受）。
"""

import threading
from typing import Any, Callable


def run_in_daemon_thread(fn: Callable, *args: Any, **kwargs: Any) -> threading.Thread:
    """在 daemon 线程中执行 fn，立即返回线程对象。"""
    thread = threading.Thread(
        target=fn,
        args=args,
        kwargs=kwargs,
        name=f"kb-task-{getattr(fn, '__name__', 'worker')}",
        daemon=True,
    )
    thread.start()
    return thread
