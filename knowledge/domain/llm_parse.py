"""LLM 输出解析工具（导入/查询共用）。"""

import re

# 清洗 ```json ... ``` 代码围栏（历史实现散落三处，逐字符一致，统一于此）
_FENCE_HEAD = re.compile(r"^```(?:json)?\s*")
_FENCE_TAIL = re.compile(r"\s*```$")


def strip_json_fence(text: str) -> str:
    """去掉 LLM 输出前后的 ``` / ```json 代码围栏。"""
    if not text:
        return text
    cleaned = _FENCE_HEAD.sub("", text.strip())
    return _FENCE_TAIL.sub("", cleaned)
