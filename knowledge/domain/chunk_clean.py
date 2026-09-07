"""文档清洗与分析：剔除入库后会稀释检索的低质/重复切片。

解决的问题
----------
P0-2 **通用样板段**：不同商品说明书里的安全声明、免责条款几乎一字不差
（「本产品不打算由儿童或有体力、感官或精神缺陷的人使用…」）。它们语义相近、
跨文档重复，检索时会出现"问 A 商品召回 B 商品的安全声明"这类误命中。

P0-3 **语言与目录页噪音**：多语言混排说明书的外文页、纯目录页（点线索引）
同样占着向量空间却几乎没有信息量。

实现原则
--------
* 纯字符运算，**不调用任何模型/服务**，毫秒级完成；
* 任何异常都不阻断导入流程；
* **清洗后若切片为空会回退原文**，避免把整篇文档洗没导致下游报错。
"""

from __future__ import annotations

import heapq
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

# ---------------------------------------------------------------- 常量
ZH_RE = re.compile(r"[\u4e00-\u9fff]")
# 目录/索引页的点线引导符：  "放置 .......... 12"
TOC_RE = re.compile(r"(\.{6,}|·{6,}|…{4,}|-{6,})")
# 归一化时剔除：空白与常见标点（用于生成去重指纹）
_DEDUP_STRIP_RE = re.compile(
    r"[\s\u3000，。、；：？！（）【】《》“”‘’'\"()\[\]{}.,;:?!\-—_/\\|+*#@&%]"
)

NGRAM_N = 8          # 指纹用的字符 n-gram 长度
NGRAM_SAMPLE = 24    # 每段抽取多少条 n-gram 作为代表（保证两次运行采样一致）
MAX_SAMPLES = 3000   # 指纹库容量上限，超出后丢弃命中次数最低的条目


def zh_ratio(text: str) -> float:
    """中文字符占非空字符的比例。纯符号/外文页会很低。"""
    stripped = "".join(text.split())
    if not stripped:
        return 0.0
    return len(ZH_RE.findall(stripped)) / len(stripped)


def is_toc_page(text: str) -> bool:
    """判定目录/索引页：含连续点线引导符。"""
    return bool(TOC_RE.search(text or ""))


def _normalize_for_dedup(text: str) -> str:
    """归一化：去空白与标点，使"同一段话"的不同排版得到相同结果。"""
    return _DEDUP_STRIP_RE.sub("", text or "")


def _strip_digits_for_dedup(text: str) -> str:
    """进一步去数字：页码/规格值不同但句式相同的样板段也能对齐。"""
    return re.sub(r"\d+", "", _normalize_for_dedup(text))


def make_shingles(text: str, n: int = NGRAM_N) -> set:
    """生成字符 n-gram 集合。"""
    s = _strip_digits_for_dedup(text)
    if len(s) < n:
        return {s} if s else set()
    return {s[i: i + n] for i in range(0, len(s) - n + 1)}


def sample_shingles(shingles: Iterable[str], k: int = NGRAM_SAMPLE) -> List[str]:
    """确定性抽 k 条代表 n-gram。

    必须"确定性"——同段文字两次运行要得到相同代表集，否则跨文档比对永远失配。
    取字典序最小的 k 条（等价 k-min-wise hashing，可复现且稳定）。
    """
    return heapq.nsmallest(k, set(shingles))


def jaccard(a: set, b: set) -> float:
    """两段文字的近似相似度（基于抽样后的代表集）。"""
    if not a or not b:
        return 0.0
    inter = len(set(a) & set(b))
    if not inter:
        return 0.0
    return inter / len(set(a) | set(b))


# ---------------------------------------------------------------- 跨文档样板段指纹库
@dataclass
class BoilerplateStore:
    """跨文档样板段指纹库（落 JSON，体积小、可读、无需额外依赖）。"""

    path: Path
    docs: List[str] = field(default_factory=list)
    samples: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Optional[str], fallback_dir: Optional[str] = None) -> "BoilerplateStore":
        p = Path(path) if path else Path(fallback_dir or ".") / "boilerplate_fingerprints.json"
        try:
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                return cls(
                    path=p,
                    docs=list(data.get("docs", [])),
                    samples=list(data.get("samples", [])),
                )
        except Exception:
            pass  # 库损坏就当空库， safest
        return cls(path=p)

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps({"version": 1, "docs": self.docs, "samples": self.samples},
                           ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass  # 写库失败不影响本次导入

    def is_boilerplate(self, shingles: set, threshold: float, min_docs: int) -> bool:
        """该段是否在**其它**足够多的文档中出现过。"""
        for s in self.samples:
            seen_docs = set(s.get("d", []))
            if len(seen_docs) < min_docs:
                continue
            if jaccard(set(s.get("g", [])), shingles) >= threshold:
                return True
        return False

    def add(self, shingles: set, doc_title: str) -> None:
        """登记一段有效文本，供后续文档比对。"""
        reps = sample_shingles(shingles)
        if not reps:
            return
        # 先看能否合并进已有条目（避免库无限膨胀）
        for s in self.samples:
            if jaccard(set(s.get("g", [])), set(reps)) >= 0.85:
                docs = set(s.get("d", []))
                docs.add(doc_title)
                s["d"] = sorted(docs)
                s["n"] = len(docs)
                return
        self.samples.append({"g": reps, "d": [doc_title], "n": 1})
        if doc_title not in self.docs:
            self.docs.append(doc_title)
        # 超容：淘汰覆盖文档数最少的条目
        if len(self.samples) > MAX_SAMPLES:
            self.samples.sort(key=lambda x: x.get("n", 1), reverse=True)
            self.samples = self.samples[:MAX_SAMPLES]


# ---------------------------------------------------------------- 主清洗流程
@dataclass
class CleanResult:
    chunks: List[dict]
    stats: Dict[str, int]


def _content_of(chunk: dict) -> str:
    """取用于判定/去重的正文：优先 content，回退 body。"""
    return (chunk.get("content") or chunk.get("body") or "")


def clean_chunks(
    chunks: Sequence[dict],
    file_title: str,
    config: Any,
    file_dir: str = "",
) -> CleanResult:
    """按规则清洗切片。返回新列表，不修改传入对象。

    Args:
        chunks:     待清洗切片
        file_title: 文档标题（用于跨文档统计与日志）
        config:     ImportConfig
        file_dir:   中间产物目录（指纹库的候选落盘位置）

    Returns:
        CleanResult(chunks, stats)；清洗后为空时会**回退原始数据**。
    """
    stats = {
        "input": len(chunks),
        "drop_short": 0, "drop_toc": 0, "drop_non_zh": 0,
        "drop_in_doc_dup": 0, "drop_boilerplate": 0,
        "mixed_lang": 0, "kept": 0,
    }
    if not chunks:
        return CleanResult(list(chunks), stats)

    keep: List[dict] = []
    seen_in_doc: Dict[str, int] = {}
    store = BoilerplateStore.load(
        getattr(config, "boilerplate_store_path", "") or None,
        fallback_dir=file_dir,
    )

    for idx, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue
        content = _content_of(chunk)
        if not content.strip():
            stats["drop_short"] += 1
            continue

        # 1) 目录 / 索引页：全是点线引导符，无正文价值
        if getattr(config, "chunk_drop_toc", True) and is_toc_page(content):
            stats["drop_toc"] += 1
            continue

        # 2) 语言占比：过低说明是外文页或纯符号页
        ratio = zh_ratio(content)
        if ratio < getattr(config, "chunk_drop_zh_ratio", 0.05):
            stats["drop_non_zh"] += 1
            continue

        # 3) 文档内精确重复（同一段话在文档里出现多次）
        dedup_key = _normalize_for_dedup(content)
        if len(dedup_key) >= 20:
            if dedup_key in seen_in_doc:
                stats["drop_in_doc_dup"] += 1
                continue
            seen_in_doc[dedup_key] = idx

        # 4) 跨文档通用样板段
        shingles = make_shingles(content)
        if getattr(config, "chunk_boilerplate_enabled", True) and shingles:
            if store.is_boilerplate(
                shingles,
                threshold=getattr(config, "chunk_boilerplate_threshold", 0.60),
                min_docs=getattr(config, "chunk_boilerplate_min_docs", 2),
            ):
                stats["drop_boilerplate"] += 1
                continue

        new_chunk = dict(chunk)
        # 5) 低中文占比 → 标记 mixed，留给检索侧降权
        if ratio < getattr(config, "chunk_min_zh_ratio", 0.20):
            new_chunk["lang"] = "mixed"
            stats["mixed_lang"] += 1
        else:
            new_chunk["lang"] = "zh"
        keep.append(new_chunk)

        # 保留下来的才登记进指纹库：样板段不来污染库
        if getattr(config, "chunk_boilerplate_enabled", True) and len(dedup_key) >= 20:
            store.add(shingles, file_title)

    # 兜底：全被清洗掉说明阈值过激，保留原文并记录
    if not keep:
        stats["fallback"] = 1
        return CleanResult(list(chunks), stats)

    store.save()
    stats["kept"] = len(keep)
    return CleanResult(keep, stats)
