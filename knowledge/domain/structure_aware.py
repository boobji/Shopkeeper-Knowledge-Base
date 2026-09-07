"""结构感知：从 MinerU 中间产物还原 Markdown 标题层级。

为什么需要这个模块
------------------
大量真实说明书 PDF **没有文本层**（家电类实测占比 79%），必须靠 OCR 还原。
OCR 出来的 Markdown 往往丢失 `#` 标题标记，`document_split_node` 只认标题切块，
于是退化成按字数硬切，切片标题变成 `文件名-1 / -2 / -3`，语义边界消失。

MinerU 的中间产物 `*_content_list.json` 里保留了版面分析结果，其中
`text_level` 字段就是它识别出的**标题层级**（1~6），即使 Markdown 丢了 `#`，
这份结构化信息仍然在。本模块据此把标题还原回 Markdown。

用法
----
```python
from knowledge.domain.structure_aware import enhance_headings

md, stat = enhance_headings(md_content, file_dir, file_title)
# stat = {"applied": True, "headings": 18, "matched": 16, "reason": ""}
```

设计取舍
--------
* **只补 `#` 前缀，不重建正文**：md_content 可能已被 `md_img_node` 替换过图片链接，
  整体重建会丢失这些处理结果，因此只做行级标注。
* **两级匹配**：先整行精确匹配，再做去空白/标点的规范化匹配，容忍 OCR 的细微差异。
* **匹配不上的标题直接跳过**：宁可少补，也不把正文行误标成标题。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# content_list 中属于页面装饰、不是正文内容的类型
DECORATION_TYPES = {"footer", "page_number", "header"}
# 正文/标题/表格之外的类型（图片由 md_img_node 处理，此处不参与）
CONTENT_TEXT_KEY = "text"
TITLE_LEVEL_KEY = "text_level"
TABLE_BODY_KEY = "table_body"

# 归一化时丢弃的字符：全半角标点与空白，用于容忍 OCR 差异
_NORM_STRIP_RE = re.compile(r"[\s\u3000，。、；：？！（）【】《》“”‘’'\"()\[\]{}.,;:?!\-—_/\\|]")


def _normalize(text: str) -> str:
    """归一化：去空白与常见标点，用于模糊匹配标题行。"""
    return _NORM_STRIP_RE.sub("", text or "")


def find_content_list(file_dir: str, file_title: str) -> Optional[Path]:
    """定位 MinerU 生成的 *_content_list.json。

    约定路径：{file_dir}/{file_title}/auto/{file_title}_content_list.json
    找不到时在该目录内递归搜索兜底（不同 MinerU 版本落盘略有差异）。
    """
    if not file_dir:
        return None
    root = Path(file_dir)
    if not root.exists():
        return None

    direct = root / file_title / "auto" / f"{file_title}_content_list.json"
    if direct.exists():
        return direct

    candidates = sorted(root.rglob("*_content_list.json"))
    return candidates[0] if candidates else None


def load_items(path: Path) -> List[dict]:
    """加载 content_list，异常时返回空列表（不阻断导入流程）。"""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def extract_headings(items: List[dict], max_level: int = 6) -> List[Tuple[str, int]]:
    """从 content_list 中提取 (标题文本, 层级) 列表。

    判定依据：`text_level` 存在且为 1~max_level 的整数。
    页面装饰类型（页眉/页脚/页码）不参与。
    """
    headings: List[Tuple[str, int]] = []
    seen = set()
    for it in items:
        if not isinstance(it, dict):
            continue
        if it.get("type") in DECORATION_TYPES:
            continue
        level = it.get(TITLE_LEVEL_KEY)
        text = (it.get(CONTENT_TEXT_KEY) or "").strip()
        if not text or not isinstance(level, int):
            continue
        if level < 1 or level > max_level:
            continue
        key = (text, level)
        if key in seen:
            continue
        seen.add(key)
        headings.append((text, level))
    return headings


def _build_line_index(lines: List[str]) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """为正文行建立两级索引：精确行 -> 行号；归一化行 -> 行号。"""
    exact: Dict[str, List[int]] = {}
    norm: Dict[str, List[int]] = {}
    for i, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped:
            continue
        # 已经是标题的行不再参与匹配，避免重复加 #
        if re.match(r"^#{1,6}\s+", stripped):
            continue
        if len(stripped) > 60:  # 标题通常较短，超长行不作为候选
            continue
        exact.setdefault(stripped, []).append(i)
        norm.setdefault(_normalize(stripped), []).append(i)
    return exact, norm


def enhance_markdown(md_content: str, headings: List[Tuple[str, int]]) -> Tuple[str, int]:
    """把 content_list 中的标题层级还原到 Markdown 行首。

    Returns:
        (增强后的 Markdown, 成功补上标题的行数)
    """
    if not md_content or not headings:
        return md_content, 0

    lines = md_content.split("\n")
    exact, norm = _build_line_index(lines)
    used_lines = set()
    matched = 0

    for text, level in headings:
        hit = -1
        # 1) 精确行优先
        for i in exact.get(text, []):
            if i not in used_lines:
                hit = i
                break
        # 2) 退化到归一化匹配（容忍 OCR 的空格/标点差异）
        if hit < 0:
            for i in norm.get(_normalize(text), []):
                if i not in used_lines:
                    hit = i
                    break
        if hit < 0:
            continue  # 匹配不上就跳过，绝不强行标注
        used_lines.add(hit)
        lines[hit] = f"{'#' * level} {lines[hit].strip()}"
        matched += 1

    return "\n".join(lines), matched


def enhance_headings(
    md_content: str,
    file_dir: str,
    file_title: str,
    enabled: bool = True,
    hard_split_rate: float = 0.0,
    threshold: float = 0.30,
) -> Tuple[str, Dict[str, object]]:
    """对外主入口：结构缺失时还原标题层级。

    Args:
        md_content:    当前 Markdown 正文
        file_dir:      中间产物目录
        file_title:    文件名（不含扩展名）
        enabled:       总开关（对应 config.structure_aware_enabled）
        hard_split_rate: 当前硬切率（0~1），由调用方先评估传入
        threshold:     硬切率阈值，超过才认定为结构缺失

    Returns:
        (Markdown, 统计字典) —— 任何异常都不会抛出，只会返回原文。
    """
    stat: Dict[str, object] = {
        "applied": False,
        "headings": 0,
        "matched": 0,
        "reason": "",
    }
    if not enabled:
        stat["reason"] = "disabled"
        return md_content, stat
    if hard_split_rate < threshold:
        stat["reason"] = f"结构完好(hard_split={hard_split_rate:.0%})"
        return md_content, stat

    try:
        path = find_content_list(file_dir, file_title)
        if not path:
            stat["reason"] = "未找到 content_list.json"
            return md_content, stat
        items = load_items(path)
        headings = extract_headings(items)
        if not headings:
            stat["reason"] = "content_list 中无 text_level 标题"
            return md_content, stat

        enhanced, matched = enhance_markdown(md_content, headings)
        stat.update(applied=matched > 0, headings=len(headings), matched=matched,
                    reason="" if matched else "标题文本与正文行匹配失败")
        return enhanced, stat
    except Exception as e:  # 兜底：结构增强失败不能影响主流程
        stat["reason"] = f"异常: {type(e).__name__}"
        return md_content, stat


def estimate_hard_split_rate(md_content: str, file_title: str, heading_re: re.Pattern) -> float:
    """粗估 MarkDown 的结构质量：没有任何 `#` 标题时返回 1.0。

    仅用于在 _split_by_headings 之前做降级判断，不需要精确。
    """
    if not md_content:
        return 0.0
    heading_count = sum(1 for line in md_content.split("\n") if heading_re.match(line))
    lines = md_content.count("\n") + 1
    # 平均每 25 行至少应有 1 个标题；低于此认为结构稀疏
    expected = max(1, lines / 25)
    if heading_count == 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - heading_count / expected))
