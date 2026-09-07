"""LLM 调用留档：把每次真正发给大模型的完整内容与返回结果落盘。

为什么需要
----------
调试提示词、复盘检索 badcase、核对"模型到底看到了什么"，都需要一份与真实
请求一致的存档，而不是运行日志里被截断的摘要。本模块以"零侵入兜底"的
原则记录每次 LLM/VLM 调用：任何留档失败都不能影响主流程，任何大负载
（如 base64 图片）都不会原样落盘。

存储布局
--------
导入流程（有任务目录）::
    {file_dir}/llm_calls/0001_import_item_name_153012.json

问答流程（无任务目录，按日期+任务归档）::
    knowledge/temp_data/llm_logs/{YYYYMMDD}/{task_id}/0001_query_answer_153055.json

每个 JSON 文件包含：时间、阶段、模型、完整 messages/prompt、响应文本、
token 用量、耗时、错误信息。base64 图片数据统一替换为占位符。

用法
----
```python
from knowledge.utils.llm_call_logger import log_llm_call

t0 = time.perf_counter()
response = llm_client.invoke(messages)
log_llm_call(
    "import_item_name",
    messages=messages, response=response,
    task_id=task_id, task_dir=file_dir,
    latency_ms=(time.perf_counter() - t0) * 1000,
)
```

配置（.env）
------------
LLM_CALL_LOG_ENABLED  总开关，默认 true
LLM_CALL_LOG_DIR      留档根目录，默认 knowledge/temp_data/llm_logs
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------- 配置 ----------------

_ROOT_DIR = Path(__file__).resolve().parents[1] / "temp_data" / "llm_logs"
_B64_RE = re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/=\\-_]+")

_seq_lock = threading.Lock()
_seq_counters: Dict[str, int] = {}  # 目录 -> 已用序号（进程内缓存，避免反复扫描）

# LangChain 消息类型 -> 角色名的映射
_ROLE_MAP = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool", "function": "function"}


def _enabled() -> bool:
    return os.getenv("LLM_CALL_LOG_ENABLED", "true").strip().lower() not in ("0", "false", "no", "off")


def _default_model() -> str:
    return os.getenv("ITEM_MODEL", "qwen-flash")


# ---------------- 清洗与转换 ----------------

def _sanitize(value: Any) -> Any:
    """递归替换字符串里的 base64 图片数据（VLM 请求动辄数百 KB，不落盘）。"""
    if isinstance(value, str):
        return _B64_RE.sub(lambda m: f"[base64图片已省略 {len(m.group(0)) // 1024}KB]", value)
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    return value


def _content_to_str(content: Any) -> Any:
    """多模态 content（list[dict]）转成可读的结构化描述，纯文本原样返回。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                parts.append(str(block))
                continue
            btype = block.get("type")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "image_url":
                parts.append("[图片输入，base64已省略]")
            else:
                parts.append(f"[{btype} 输入已省略]")
        return "\n".join(parts)
    return str(content)


def _message_to_dict(message: Any) -> Dict[str, Any]:
    """把 LangChain Message / OpenAI dict / 纯字符串统一转成 {role, content}。"""
    if isinstance(message, str):
        return {"role": "user", "content": message}
    if isinstance(message, dict):
        return {"role": message.get("role", "user"),
                "content": _content_to_str(message.get("content", ""))}
    # LangChain BaseMessage
    role = _ROLE_MAP.get(getattr(message, "type", ""), getattr(message, "type", "user"))
    return {"role": role, "content": _content_to_str(getattr(message, "content", ""))}


def _extract_response_text(response: Any) -> str:
    if response is None:
        return ""
    text = getattr(response, "content", None)
    if text is None and isinstance(response, dict):
        text = response.get("content") or response.get("text")
    if text is None:
        text = str(response)
    return _content_to_str(text)


def _extract_usage(response: Any) -> Dict[str, Any]:
    """尽力提取 token 用量，取不到就返回空字典。"""
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        return {k: usage.get(k) for k in ("input_tokens", "output_tokens", "total_tokens")}
    return {}


# ---------------- 主入口 ----------------

def _next_seq(task_log_dir: Path) -> int:
    key = str(task_log_dir)
    with _seq_lock:
        current = _seq_counters.get(key, 0)
        # 进程重启后计数丢失，扫描目录续号
        if current == 0 and task_log_dir.exists():
            existing = [p for p in task_log_dir.glob("*.json") if p.name[:4].isdigit()]
            current = max((int(p.name[:4]) for p in existing), default=0)
        current += 1
        _seq_counters[key] = current
        return current


def _resolve_task_log_dir(task_id: str, task_dir: str) -> Tuple[Path, bool]:
    """导入流程优先写进任务目录（与 chunks.json 同处归档），问答流程按日期+任务归档。"""
    if task_dir:
        return Path(task_dir) / "llm_calls", True
    root = Path(os.getenv("LLM_CALL_LOG_DIR") or _ROOT_DIR)
    date_dir = datetime.now().strftime("%Y%m%d")
    return root / date_dir / (task_id or "adhoc"), False


def log_llm_call(
    stage: str,
    *,
    messages: Optional[List[Any]] = None,
    prompt: Optional[str] = None,
    response: Any = None,
    answer: Optional[str] = None,
    task_id: str = "",
    task_dir: str = "",
    model: str = "",
    meta: Optional[Dict[str, Any]] = None,
    latency_ms: Optional[float] = None,
    error: str = "",
) -> Optional[Path]:
    """记录一次 LLM/VLM 调用。任何异常都只告警不抛出，绝不影响主流程。

    Args:
        stage:      调用阶段标识，如 import_item_name / import_kg / query_answer
        messages:   LangChain 消息列表 / OpenAI 风格 dict 列表 / 字符串
        prompt:     单字符串提示词（与 messages 二选一）
        response:   模型返回对象（AIMessage 等），流式场景可传 None
        answer:     最终答案文本（流式场景在聚合后传入）
        task_id:    任务 ID
        task_dir:   任务目录（导入流程传 file_dir，日志落到其下 llm_calls/）
        model:      模型名，缺省读环境变量 ITEM_MODEL
        meta:       附加信息（如重试次数、chunk_id）
        latency_ms: 耗时毫秒
        error:      调用失败时的错误摘要

    Returns:
        落盘的 JSON 路径；留档关闭或失败时返回 None。
    """
    if not _enabled():
        return None
    try:
        task_log_dir, _in_task_dir = _resolve_task_log_dir(task_id, task_dir)
        seq = _next_seq(task_log_dir)
        now = datetime.now()
        stage_safe = re.sub(r"[^0-9A-Za-z_\-]", "_", stage or "call")
        suffix = "_error" if error else ""
        out_path = task_log_dir / f"{seq:04d}_{stage_safe}_{now.strftime('%H%M%S')}{suffix}.json"

        record: Dict[str, Any] = {
            "timestamp": now.isoformat(timespec="seconds"),
            "stage": stage,
            "task_id": task_id,
            "model": model or _default_model(),
            "latency_ms": round(latency_ms) if latency_ms else None,
        }
        if messages is not None:
            record["messages"] = [_sanitize(_message_to_dict(m)) for m in messages]
        elif prompt is not None:
            record["prompt"] = _sanitize(prompt)
        if response is not None:
            record["response"] = _extract_response_text(response)
            record["usage"] = _extract_usage(response)
        if answer is not None:
            record["answer"] = answer
        if meta:
            record["meta"] = _sanitize(meta)
        if error:
            record["error"] = error

        task_log_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path
    except Exception as e:  # 兜底：留档失败不影响主流程
        logger.warning(f"LLM 调用留档失败（已忽略）: {e}")
        return None
