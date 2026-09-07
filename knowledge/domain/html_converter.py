"""HTML → Markdown 转换（确定性规则，无 AI）。

覆盖说明书类文档的核心结构：标题层级、段落、列表、表格、图片、链接、代码块。
未识别的标签退化为纯文本。输出直接进入现有 md 导入流水线。
"""

import re
from typing import List

from bs4 import BeautifulSoup, NavigableString, Tag

_HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
_DROP_TAGS = {"script", "style", "noscript", "head", "meta", "link", "iframe"}


def _text_of(node: Tag) -> str:
    """取标签内纯文本，压缩空白。"""
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()


def _convert_table(table: Tag) -> str:
    """表格 → markdown 管道表（首行视为表头）。"""
    rows: List[List[str]] = []
    for tr in table.find_all("tr"):
        cells = [_text_of(td) for td in tr.find_all(["th", "td"])]
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    # 列数取最大值，缺的补空
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]

    lines = [
        "| " + " | ".join(rows[0]) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _convert_list(tag: Tag, ordered: bool, depth: int = 0) -> List[str]:
    """列表 → markdown 列表（支持一层嵌套，缩进表示）。"""
    lines: List[str] = []
    index = 0
    for item in tag.find_all("li", recursive=False):
        index += 1
        marker = f"{index}. " if ordered else "- "
        # li 内部：取直接文本 + 嵌套列表
        nested_lists = item.find_all(["ul", "ol"], recursive=False)
        for nl in nested_lists:
            nl.extract()
        text = _text_of(item)
        indent = "  " * depth
        if text:
            lines.append(f"{indent}{marker}{text}")
        for nl in nested_lists:
            lines.extend(_convert_list(nl, nl.name == "ol", depth + 1))
    return lines


def _convert_element(tag: Tag) -> List[str]:
    """把一个块级元素转成 markdown 行列表。"""
    name = (tag.name or "").lower()

    if name in _DROP_TAGS:
        return []
    if name in _HEADING_TAGS:
        text = _text_of(tag)
        return [f"{'#' * _HEADING_TAGS[name]} {text}"] if text else []
    if name == "p":
        text = _text_of(tag)
        return [text] if text else []
    if name in ("ul", "ol"):
        return _convert_list(tag, name == "ol")
    if name == "table":
        table_md = _convert_table(tag)
        return [table_md] if table_md else []
    if name == "pre":
        code = tag.get_text()
        return [f"```\n{code}\n```"] if code.strip() else []
    if name == "blockquote":
        inner = _convert_children(tag)
        return ["> " + line for line in inner]
    if name == "hr":
        return ["---"]
    if name in ("img",):
        src = (tag.get("src") or "").strip()
        alt = (tag.get("alt") or "").strip()
        return [f"![{alt}]({src})"] if src else []
    if name == "br":
        return [""]

    # div/section/article 等容器：递归处理子元素
    return _convert_children(tag)


def _convert_children(tag: Tag) -> List[str]:
    lines: List[str] = []
    for child in tag.children:
        if isinstance(child, NavigableString):
            text = re.sub(r"\s+", " ", str(child)).strip()
            if text:
                lines.append(text)
        elif isinstance(child, Tag):
            lines.extend(_convert_element(child))
    return lines


def html_to_markdown(html_text: str, title: str = "") -> str:
    """HTML 文本 → Markdown 文本。title 作为文档一级标题（html 无 h1 时兜底）。"""
    if not html_text or not html_text.strip():
        return ""

    soup = BeautifulSoup(html_text, "html.parser")
    for drop in soup.find_all(list(_DROP_TAGS)):
        drop.decompose()

    body = soup.body or soup
    lines = _convert_children(body)

    # 去掉连续空行
    result: List[str] = []
    prev_blank = True
    for line in lines:
        if not line:
            if prev_blank:
                continue
            prev_blank = True
        else:
            prev_blank = False
        result.append(line)

    md = "\n\n".join(result).strip()
    if title and not re.search(r"^#\s+", md, flags=re.MULTILINE):
        md = f"# {title}\n\n{md}"
    return md
