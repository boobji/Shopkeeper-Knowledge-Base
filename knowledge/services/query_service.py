"""查询业务服务"""

import uuid
import logging
from typing import List, Dict, Any

# from knowledge.processor.query_process.main_graph import query_app
from knowledge.utils.task_util import update_task_status, get_task_result, \
    get_task_status, get_done_task_list, get_running_task_list, \
    TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED, TASK_STATUS_FAILED, set_task_result
from knowledge.processor.query_process.main_graph import query_app
from knowledge.utils.sse_util import SSEEvent, create_sse_queue, push_sse_event

# from knowledge.utils.sse_util import create_sse_queue, push_sse_event

logger = logging.getLogger(__name__)


class QueryService:

    def generate_session_id(self) -> str:
        return str(uuid.uuid4())

    def generate_task_id(self) -> str:
        return str(uuid.uuid4())

    def submit_query(self, task_id: str, is_stream: bool):
        """提交查询任务：更新状态 + 流式模式创建 SSE 队列。"""
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        if is_stream:
            create_sse_queue(task_id)

    def run_query_graph(self, task_id: str, session_id: str, user_query: str, is_stream: bool,
                        selected_item: str = None):
        """执行 LangGraph 查询流程。"""
        try:
            default_state = {
                "original_query": user_query,
                "session_id": session_id,
                "task_id": task_id,
                "is_stream": is_stream,
                "selected_item": selected_item or "",
            }
            query_app.invoke(default_state)
        except Exception as e:
            logger.error(f"查询流程执行失败: {e}", exc_info=True)
            update_task_status(task_id, TASK_STATUS_FAILED)
            set_task_result(task_id, "error", str(e))
            # 推送 error 事件让前端结束等待并展示错误，否则界面会一直停在生成中
            if is_stream:
                push_sse_event(task_id, SSEEvent.ERROR, {"error": str(e)})
        else:
            update_task_status(task_id, TASK_STATUS_COMPLETED)
        finally:
            if is_stream:
                push_sse_event(task_id, "progress", {
                    "status": get_task_status(task_id),
                    "done_list": get_done_task_list(task_id),
                    "running_list": get_running_task_list(task_id),
                })

    def get_answer(self, task_id: str) -> str:
        return get_task_result(task_id, "answer", "")

    def get_suggestions(self, task_id: str) -> List[str]:
        return get_task_result(task_id, "suggestions", [])

    def get_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        from knowledge.utils.mongo_history_util import get_recent_messages
        records = get_recent_messages(session_id, limit=limit)
        return [
            {
                "_id": str(r.get("_id", "")),
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "ts": r.get("ts"),
            }
            for r in records
        ]

    def clear_history(self, session_id: str) -> int:
        from knowledge.utils.mongo_history_util import clear_history
        return clear_history(session_id)
