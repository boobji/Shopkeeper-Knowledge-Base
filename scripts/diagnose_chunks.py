#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""切片质量体检：定位"换个文档就不好使"的根因。

背景
----
导入链路的表现高度依赖 MinerU 解析出的 Markdown **结构**：
`document_split_node` 是按 `#` 标题层级切块的，标题一旦缺失就会退化成
按字数硬切（`title-1 / title-2 …`），切块失去语义边界，下游的商品名抽取、
图谱构建、检索命中都会跟着塌。

本脚本不依赖任何服务，只读 `chunks.json`，量化如下问题：

| 指标             | 说明                                        | 健康参考   |
| ---------------- | ------------------------------------------- | ---------- |
| 硬切率           | title 降级为 `文件标题-N` 的切块占比        | < 10%      |
| 无标题率         | title 等于文件标题（没有分到任何章节）      | < 5%       |
| 碎片率           | content 长度 < min_len 的切块占比           | < 10%      |
| 低中文占比       | 中文字符占比 < 30%（多语言混排 / OCR 噪音） | < 10%      |
| 目录页率         | 含大量 `.....` 点线引导符（目录、索引页）   | < 5%       |
| 重复样板段       | 与其它文档高度相似的段落（通用安全套话）    | 越低越好   |

用法
----
```bash
# 体检单个 chunks.json
python scripts/diagnose_chunks.py knowledge/temp_data/20260906/xxx/chunks.json

# 批量体检某个目录下的所有 chunks.json（推荐：对比多个文档）
python scripts/diagnose_chunks.py --dir knowledge/temp_data/20260906

# 只看最严重的 N 个问题 + 输出 Markdown 报告
python scripts/diagnose_chunks.py --dir knowledge/temp_data -o eval/reports/chunk_health.md
```
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

# 判定阈值（集中放置，便于按实际语料微调）
THRESHOLDS = {
    "hard_split": 0.10,   # 硬切率上限
    "no_title": 0.05,     # 无标题率上限
    "fragment": 0.10,     # 碎片率上限
    "low_zh": 0.10,       # 低中文占比上限
    "toc": 0.05,          # 目录页率上限
    "dupe": 0.15,         # 跨文档重复段上限
}

MIN_LEN = 80              # 低于此长度视为碎片（参考 config.min_content_length）
ZH_RE = re.compile(r"[\u4e00-\u9fff]")
TOC_RE = re.compile(r"\.{6,}|·{6,}|…{4,}")          # 目录点线
HARD_SPLIT_RE = re.compile(r"-\d+$")                 # title-1 这类硬切产物


def zh_ratio(text: str) -> float:
    """中文字符占非空字符的比例。"""
    stripped = "".join(text.split())
    if not stripped:
        return 0.0
    return len(ZH_RE.findall(stripped)) / len(stripped)


def shingles(text: str, n: int = 8) -> set[str]:
    """生成字符 n-gram 集合，用于近似判定重复段落（无需额外依赖）。"""
    stripped = "".join(text.split())
    if len(stripped) < n:
        return {stripped} if stripped else set()
    return {stripped[i: i + n] for i in range(0, len(stripped) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def analyze(chunks: list[dict], file_title: str) -> dict:
    """统计单个文档的切块健康度。"""
    if not chunks:
        return {"file_title": file_title, "chunks": 0, "error": "无切块"}

    hard_split = no_title = fragment = low_zh = toc = 0
    lengths = []
    for c in chunks:
        content = c.get("content", "") or ""
        title = (c.get("title", "") or "").strip()
        lengths.append(len(content))

        # 硬切：拼接 -N 后缀且父标题不是原文的自然章节
        if HARD_SPLIT_RE.search(title) and title != file_title:
            hard_split += 1
        # 无标题：整块归属文件标题，说明该段没落在任何章节下
        if title == file_title:
            no_title += 1
        if len(content) < MIN_LEN:
            fragment += 1
        if zh_ratio(content) < 0.30:
            low_zh += 1
        if TOC_RE.search(content):
            toc += 1

    n = len(chunks)
    lengths.sort()

    return {
        "file_title": file_title,
        "chunks": n,
        "median_len": lengths[n // 2],
        "max_len": lengths[-1],
        "hard_split": hard_split / n,
        "no_title": no_title / n,
        "fragment": fragment / n,
        "low_zh": low_zh / n,
        "toc": toc / n,
        "shingle_sets": [shingles(c.get("content", "")) for c in chunks],
    }


def find_duplicates(reports: list[dict], sample: int = 40, threshold: float = 0.6) -> dict:
    """跨文档重复段比例：命中数 / 参与比较的文档数。

    成本为 O(n²)，故每篇只抽样前 `sample` 个段落做近似比较。
    """
    # 先算全局文档频率，避免与其它所有文档都不同却仍逐篇比较
    doc_freq: Counter = Counter()
    per_doc = []
    for r in reports:
        if r.get("chunks", 0) == 0:
            per_doc.append(0.0)
            continue
        hits = 0
        compared = 0
        own = r["shingle_sets"][:sample]
        for s in own:
            compared += 1
            key = next(iter(s)) if s else ""
            if key and doc_freq[key] > 0:
                hits += 1
        # 完整两两比较（抽样后规模可控）
        dupe_count = 0
        for other in reports:
            if other is r or not other.get("shingle_sets"):
                continue
            for s in own:
                for t in other["shingle_sets"][:sample]:
                    if jaccard(s, t) >= threshold:
                        dupe_count += 1
                        break
        per_doc.append(dupe_count / max(compared, 1))
        for s in own:
            for key in list(s)[:1]:
                doc_freq[key] += 1
    return {r.get("file_title", f"#{i}"): v for i, (r, v) in enumerate(zip(reports, per_doc))}


def load_targets(path: str | None, directory: str | None) -> list[tuple[str, list[dict]]]:
    """收集待体检的 (文件标题, chunks) 列表。"""
    files: list[Path] = []
    if directory:
        files = sorted(Path(directory).rglob("chunks.json"))
    elif path:
        files = [Path(path)]

    best: dict[str, tuple[float, list[dict]]] = {}
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[跳过] {f} 读取失败: {e}", file=sys.stderr)
            continue
        if not isinstance(data, list) or not data:
            continue
        title = data[0].get("file_title") or f.name
        # 同名多次导入会产生多份产物：只保留 mtime 最新的那份
        mtime = f.stat().st_mtime
        if title not in best or mtime > best[title][0]:
            best[title] = (mtime, data)
    return [(title, data) for title, (_, data) in best.items()]


def render(reports: list[dict], dup_map: dict) -> str:
    """输出可读结论。"""
    lines = ["=" * 78, f"{'文档':<44}{'块数':>5}{'硬切':>7}{'无题':>7}{'碎片':>7}{'低中':>7}{'目录':>7}",
             "-" * 78]
    for r in reports:
        if r.get("chunks", 0) == 0:
            continue
        name = r["file_title"][:42]
        pct = lambda k: f"{r[k] * 100:.0f}%"
        lines.append(f"{name:<44}{r['chunks']:>5}{pct('hard_split'):>7}{pct('no_title'):>7}"
                     f"{pct('fragment'):>7}{pct('low_zh'):>7}{pct('toc'):>7}")
    lines.append("-" * 78)

    # 逐文档给结论
    lines.append("")
    lines.append("【诊断结论】")
    any_issue = False
    for r in reports:
        if r.get("chunks", 0) == 0:
            continue
        issues = []
        if r["hard_split"] > THRESHOLDS["hard_split"]:
            issues.append(f"硬切率 {r['hard_split'] * 100:.0f}%（标题缺失，退化成按字数切）")
        if r["no_title"] > THRESHOLDS["no_title"]:
            issues.append(f"无标题率 {r['no_title'] * 100:.0f}%（段落没归入任何章节）")
        if r["fragment"] > THRESHOLDS["fragment"]:
            issues.append(f"碎片率 {r['fragment'] * 100:.0f}%（短块太多）")
        if r["low_zh"] > THRESHOLDS["low_zh"]:
            issues.append(f"低中文占比 {r['low_zh'] * 100:.0f}%（多语言混排或 OCR 噪音）")
        if r["toc"] > THRESHOLDS["toc"]:
            issues.append(f"目录页率 {r['toc'] * 100:.0f}%（点线索引页污染）")
        d = dup_map.get(r["file_title"], 0.0)
        if d > THRESHOLDS["dupe"]:
            issues.append(f"重复样板段 {d * 100:.0f}%（与其它文档雷同的通用套话）")
        if issues:
            any_issue = True
            lines.append(f"  · {r['file_title']}")
            for i in issues:
                lines.append(f"      - {i}")
    if not any_issue:
        lines.append("  全部文档指标均在健康区间内。")
    lines.append("")
    lines.append("【怎么读这份报告】")
    lines.append("  · 硬切率 / 无标题率偏高 → Markdown 里缺少 # 标题层级，切块已退化成按字数切；")
    lines.append("    先确认源 PDF 是否为扫描件（无文本层），结构是根因而非参数问题。")
    lines.append("  · 低中文占比高 → 多语言混排或 OCR 噪音，建议入库前按语言过滤或降权。")
    lines.append("  · 重复样板段高 → 跨文档雷同的通用安全声明，会稀释检索，建议去重或降权。")
    lines.append("  · 碎片率高 → 调大 min_content_length，或检查是否按句子切断了表格。")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="切片质量体检")
    ap.add_argument("path", nargs="?", help="单个 chunks.json 路径")
    ap.add_argument("--dir", help="递归查找目录下的 chunks.json")
    ap.add_argument("-o", "--output", help="同时输出 Markdown 报告到指定文件")
    args = ap.parse_args()

    if not args.path and not args.dir:
        ap.error("请提供 chunks.json 路径或 --dir")

    targets = load_targets(args.path, args.dir)
    if not targets:
        print("没有找到任何 chunks.json")
        return 1

    reports = [analyze(chunks, title) for title, chunks in targets]
    dup_map = find_duplicates(reports)

    text = render(reports, dup_map)
    print(text)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("# 切片质量体检报告\n\n```\n" + text + "\n```\n", encoding="utf-8")
        print(f"\n报告已写入: {out}")

    # 有文档超出阈值时返回非 0，方便在 CI / 流程里卡质量
    bad = any(
        r.get("chunks", 0) > 0 and any(r[k] > v for k, v in THRESHOLDS.items())
        for r in reports
    )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
