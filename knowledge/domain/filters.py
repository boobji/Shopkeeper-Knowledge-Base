"""Milvus 标量过滤工具（导入/查询共用）。

商品名过滤的归一化：LLM 确认出的名称与入库写入的名称只要差一个空格/大小写，
精确匹配就会 0 命中（线上真实出现过）。因此入库时同时写入归一化字段
item_name_norm（去所有空白 + 小写），检索过滤一律走该字段。
"""

from typing import List


def normalize_item_name(name: str) -> str:
    """商品名归一化：去除所有空白字符并转小写。"""
    return "".join((name or "").split()).lower()


def normalize_item_names(item_names: List[str]) -> List[str]:
    """批量归一化，去掉空值并去重保序。"""
    result = []
    seen = set()
    for name in item_names or []:
        norm = normalize_item_name(name)
        if norm and norm not in seen:
            seen.add(norm)
            result.append(norm)
    return result


def _quote(values: List[str]) -> str:
    import json

    return ", ".join(json.dumps(v, ensure_ascii=False) for v in values)


def build_item_name_norm_expr(item_names: List[str]) -> str:
    """构建 item_name_norm 过滤表达式（空列表返回空串，表示不过滤）。"""
    norms = normalize_item_names(item_names)
    if not norms:
        return ""
    return f"item_name_norm in [{_quote(norms)}]"
