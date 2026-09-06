#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
掌柜智库 —— 检索效果评测脚本

评测对象：向量混合检索（与线上 VectorSearchNode 完全同一条链路），
         可选对比 "混合检索 + BGE-Reranker 精排" 两种模式。

评测集：eval/retrieval_eval_set.json（问题 -> 期望命中的 chunk 标题）

指标：
    Hit@k      期望标题（任一）出现在 top-k 的比例          （越高越好）
    Recall@k   命中的期望标题数 / 期望标题总数               （多答案题更有区分度）
    MRR@k      第一个命中结果排名倒数的均值                  （衡量"排得靠前不靠前"）
    nDCG@k     折损累积增益（二值相关度）                    （综合位置质量）

用法（在项目根目录执行）：
    python eval/run_eval.py                    # 默认 both：向量检索 + 重排对比
    python eval/run_eval.py --mode vector      # 只测向量检索
    python eval/run_eval.py --limit 20         # 重排召回池大小（默认 20）
    python eval/run_eval.py --self-test        # 无服务自检：用造的假结果验证指标计算
"""

import os
import sys
import json
import time
import socket
import argparse
import math
from pathlib import Path

# 保证直接用 `python eval/run_eval.py` 也能跑（项目根目录加入 sys.path）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EVAL_SET_PATH = Path(__file__).resolve().parent / "retrieval_eval_set.json"
REPORT_DIR = Path(__file__).resolve().parent / "reports"

KS = (1, 3, 5)          # 报告的 top-k 档位
RETRIEVE_TOP_K = 5      # 与线上 VectorSearchNode 一致


# ================================================================
#  指标计算（纯函数，可独立单测）
# ================================================================
def _norm_title(t: str) -> str:
    """标题归一化：去空白与 markdown 前缀，容忍 '电阻测量' vs '## 电阻测量' 的差异"""
    return "".join((t or "").replace("##", "").split()).lower()


def dcg_at_k(gains, k):
    """二值相关度下的 DCG@k"""
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def case_metrics(expected_titles, retrieved_titles, k):
    """
    单条用例指标。
    :param expected_titles: 期望命中的标题列表（多答案时为多个）
    :param retrieved_titles: 检索返回的标题列表（按排名序）
    :param k: 截断位置
    :return: dict(hit, recall, rr, ndcg)
    """
    expected = {_norm_title(t) for t in (expected_titles or [])}
    retrieved = [_norm_title(t) for t in (retrieved_titles or [])][:k]

    if not expected:
        return {"hit": None, "recall": None, "rr": None, "ndcg": None}

    gains = [1 if t in expected else 0 for t in retrieved]

    hit = 1 if any(gains) else 0
    recall = sum(gains) / len(expected)
    rr = 0.0
    for i, g in enumerate(gains):
        if g:
            rr = 1.0 / (i + 1)
            break
    ideal = sorted(gains, reverse=True)
    ndcg = dcg_at_k(gains, k) / dcg_at_k(ideal, k) if dcg_at_k(ideal, k) > 0 else 0.0

    return {"hit": hit, "recall": recall, "rr": rr, "ndcg": ndcg}


def evaluate_all(cases, retrieval_fn, k=5):
    """
    跑全部用例并汇总。
    :param retrieval_fn: function(question, item_name) -> List[str]  返回按排名序的标题列表
    """
    positives, negatives = [], []
    for case in cases:
        t0 = time.time()
        try:
            retrieved = retrieval_fn(case["question"], case.get("item_name") or "")
        except Exception as e:
            retrieved = []
            case["_error"] = f"{type(e).__name__}: {str(e)[:120]}"
        case["_latency"] = time.time() - t0
        case["_retrieved"] = retrieved
        if case.get("negative"):
            negatives.append(case)
        else:
            positives.append(case)

    # ---- 正例指标 ----
    summary = {}
    for kk in KS:
        hits, recalls, rrs, ndcgs = [], [], [], []
        for case in positives:
            m = case_metrics(case["expected_titles"], case["_retrieved"], kk)
            if m["hit"] is not None:
                hits.append(m["hit"])
                recalls.append(m["recall"])
                rrs.append(m["rr"])
                ndcgs.append(m["ndcg"])
        n = len(hits) or 1
        summary[f"Hit@{kk}"] = sum(hits) / n
        summary[f"Recall@{kk}"] = sum(recalls) / n
        if kk == 5:
            summary["MRR@5"] = sum(rrs) / n
            summary["nDCG@5"] = sum(ndcgs) / n

    return summary, positives, negatives


# ================================================================
#  检索实现（复用线上同一条链路：domain.retrieval.search_chunks）
# ================================================================
def check_milvus_reachable() -> tuple:
    """启动前先做 TCP 探测，给出清晰的服务未启动提示"""
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=PROJECT_ROOT / "knowledge" / ".env", override=True)
    uri = os.getenv("MILVUS_URL", "http://127.0.0.1:19530")
    s = uri.split("://", 1)[-1].rstrip("/")
    host, _, port = s.rpartition(":")
    try:
        with socket.create_connection((host or "127.0.0.1", int(port)), timeout=3):
            return True, uri
    except Exception:
        return False, uri


def build_retrieval_fn(mode: str, pool_limit: int, search_limit: int = None, use_child: bool = True):
    """
    构造 retrieval_fn(question, item_name) -> List[str 标题]
    mode=vector : 混合检索 top-search_limit（与线上 VectorSearchNode 同链路）
    mode=rerank : 混合检索 top-pool_limit -> BGE-Reranker 精排 -> top-5
    use_child   : 是否启用子块集合（Parent-Child）
    """
    from knowledge.utils.bge_m3_embedding_util import (
        get_bge_m3_embedding_model, generate_hybrid_embeddings)
    from knowledge.utils.milvus_util import get_milvus_client
    from knowledge.domain.retrieval import search_chunks
    from knowledge.processor.query_process.config import get_config

    config = get_config()
    search_limit = search_limit or RETRIEVE_TOP_K
    embedding_model = get_bge_m3_embedding_model()
    milvus_client = get_milvus_client()
    reranker = None
    if mode == "rerank":
        from knowledge.utils.bge_rerank_util import get_reranker_model
        reranker = get_reranker_model()

    def retrieve(question: str, item_name: str):
        # 1. 问题向量化（与线上完全一致的调用方式）
        emb = generate_hybrid_embeddings(embedding_model, embedding_documents=[question])
        if not emb:
            return []

        # 2. 统一检索链路：子块(+过滤→全库) → 父块(+过滤→全库)
        hits = search_chunks(
            milvus_client,
            emb["dense"][0],
            emb["sparse"][0],
            item_names=[item_name] if item_name else [],
            limit=pool_limit if mode == "rerank" else search_limit,
            chunks_collection=config.chunks_collection,
            child_collection=config.child_chunks_collection if (use_child and config.parent_child_enabled) else None,
        )
        if not hits:
            return []

        docs = []
        for hit in hits:
            entity = hit.get("entity", hit)  # 兼容不同返回结构
            docs.append({
                "title": entity.get("title", "") or "",
                "content": entity.get("content", "") or "",
            })

        # 3. rerank 模式：BGE-Reranker 精排（与线上 RerankNode 同一 compute_score）
        if mode == "rerank" and reranker is not None and docs:
            pairs = [(question, d["content"]) for d in docs]
            scores = reranker.compute_score(sentence_pairs=pairs)
            if not isinstance(scores, list):
                scores = [scores]
            docs = [d for _, d in sorted(
                zip(scores, docs), key=lambda x: x[0], reverse=True)]

        return [d["title"] for d in docs[:RETRIEVE_TOP_K]]

    return retrieve


# ================================================================
#  报告输出
# ================================================================
def print_case_details(positives, negatives, mode):
    print(f"\n{'=' * 76}")
    print(f"  逐条明细（{mode}）")
    print(f"{'=' * 76}")
    for case in positives:
        exp = {_norm_title(t) for t in case["expected_titles"]}
        ranks = [i + 1 for i, t in enumerate(case["_retrieved"][:5])
                 if _norm_title(t) in exp]
        flag = "✓" if ranks else "✗"
        rank_str = f"命中位次 {ranks}" if ranks else "未命中"
        err = f"  [异常:{case['_error']}]" if case.get("_error") else ""
        print(f"  {flag} [{case['id']:<4}] {case['category']:<5} "
              f"{rank_str:<12} {case['_latency'] * 1000:5.0f}ms{err} | {case['question'][:36]}")

    if negatives:
        print(f"\n  —— 负例（人工复核：top1 是否误导）——")
        for case in negatives:
            top1 = case["_retrieved"][0] if case["_retrieved"] else "(无结果)"
            print(f"  ? [{case['id']:<4}] top1={top1 or '(空)'} | {case['question'][:36]}")


def save_report(mode, summary, positives, negatives, args):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"eval_{mode}_{time.strftime('%Y%m%d_%H%M%S')}.md"
    lines = [
        f"# 检索效果评测报告（{mode}）",
        "",
        f"- 时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 评测集：{EVAL_SET_PATH.name}（正例 {len(positives)} 条，负例 {len(negatives)} 条）",
        f"- 检索池：top-{args.limit}，截断 k={RETRIEVE_TOP_K}",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v:.4f} |" for k, v in summary.items()]
    lines += ["", "## 逐条结果", "", "| id | 类别 | 命中位次 | 问题 |", "|---|---|---|---|"]
    for case in positives:
        exp = {_norm_title(t) for t in case["expected_titles"]}
        ranks = [i + 1 for i, t in enumerate(case["_retrieved"][:5]) if _norm_title(t) in exp]
        rank_str = "/".join(map(str, ranks)) if ranks else "未命中"
        lines.append(f"| {case['id']} | {case['category']} | {rank_str} | {case['question'][:40]} |")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ================================================================
#  网格对比：search_limit × 子块开关 × 模式
# ================================================================
def run_grid(cases, args):
    """跑一组检索配置并输出对比表。"""
    child_label = "无子块" if args.no_child else "含子块"
    configs = []
    for limit in (5, 10, 20):
        configs.append((f"vector limit={limit} ({child_label})", "vector", limit, not args.no_child))
    configs.append((f"rerank pool={args.limit} ({child_label})", "rerank", args.limit, not args.no_child))

    all_summaries = {}
    for label, mode, limit, use_child in configs:
        print(f"\n{'#' * 76}\n#  配置：{label}\n{'#' * 76}")
        retrieval_fn = build_retrieval_fn(mode, limit, search_limit=limit, use_child=use_child)
        summary, positives, negatives = evaluate_all(cases, retrieval_fn)
        all_summaries[label] = summary
        for k, v in summary.items():
            print(f"  {k:<10} {v:8.4f}")

    # 汇总对比表
    print(f"\n{'=' * 100}\n  网格对比（正例指标，越高越好）\n{'=' * 100}")
    metric_keys = ["Hit@1", "Hit@3", "Hit@5", "Recall@5", "MRR@5", "nDCG@5"]
    header = f"  {'配置':<28}" + "".join(f"{k:>12}" for k in metric_keys)
    print(header)
    print("  " + "-" * (28 + 12 * len(metric_keys)))
    for label, summary in all_summaries.items():
        row = f"  {label:<28}" + "".join(f"{summary.get(k, 0):>12.4f}" for k in metric_keys)
        print(row)

    if not args.no_report:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORT_DIR / f"eval_grid_{time.strftime('%Y%m%d_%H%M%S')}.md"
        lines = [
            "# 检索网格对比报告",
            "",
            f"- 时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- 评测集：{EVAL_SET_PATH.name}",
            "",
            "| 配置 | " + " | ".join(metric_keys) + " |",
            "|---|" + "---|" * len(metric_keys),
        ]
        for label, summary in all_summaries.items():
            lines.append("| " + label + " | " + " | ".join(f"{summary.get(k, 0):.4f}" for k in metric_keys) + " |")
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n网格报告已保存：{path}")


# ================================================================
#  自检：不依赖任何服务/模型，用造的结果验证指标计算
# ================================================================
def self_test():
    print("自检：用构造结果验证指标计算 ...")
    # 用例1：期望 [a]，返回 [x, a, b]
    m = case_metrics(["a"], ["x", "a", "b"], k=3)
    assert m["hit"] == 1 and abs(m["recall"] - 1.0) < 1e-9
    assert abs(m["rr"] - 0.5) < 1e-9, m["rr"]
    assert abs(m["ndcg"] - 1 / math.log2(3)) < 1e-9
    # 用例2：多答案 [a,b]，返回 [b,x,a]
    m = case_metrics(["a", "b"], ["b", "x", "a"], k=3)
    assert m["hit"] == 1 and abs(m["recall"] - 1.0) < 1e-9
    assert abs(m["rr"] - 1.0) < 1e-9
    assert abs(m["ndcg"] - (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))) < 1e-9
    # 用例3：全部未命中
    m = case_metrics(["a"], ["x", "y", "z"], k=3)
    assert m["hit"] == 0 and m["recall"] == 0 and m["rr"] == 0 and m["ndcg"] == 0
    # 用例4：标题归一化（'## 电阻测量' vs '电阻测量'）
    m = case_metrics(["电阻测量"], ["## 电阻测量"], k=1)
    assert m["hit"] == 1
    # 用例5：汇总函数
    cases = [
        {"id": "t1", "question": "q1", "expected_titles": ["a"], "negative": False},
        {"id": "t2", "question": "q2", "expected_titles": ["a"], "negative": False},
    ]
    for c in cases:
        c["_retrieved"] = ["a"]
        c["_latency"] = 0.0
    summary, pos, neg = evaluate_all(cases, lambda q, i: ["a"])
    assert abs(summary["Hit@1"] - 1.0) < 1e-9 and len(pos) == 2 and len(neg) == 0
    print("自检通过 ✓ 指标计算正确")


# ================================================================
#  主流程
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="检索效果评测")
    parser.add_argument("--mode", choices=["vector", "rerank", "both"], default="both",
                        help="vector=仅混合检索；rerank=仅精排；both=对比（默认）")
    parser.add_argument("--limit", type=int, default=20,
                        help="rerank 模式的召回池大小（默认 20）")
    parser.add_argument("--search-limit", type=int, default=None,
                        help="vector 模式的直接检索条数（默认 5，对齐线上）")
    parser.add_argument("--no-child", action="store_true",
                        help="禁用子块集合（对照 Parent-Child 的收益）")
    parser.add_argument("--grid", action="store_true",
                        help="网格对比：search_limit × 子块开关 × 模式，输出汇总对比表")
    parser.add_argument("--eval-set", type=str, default=str(EVAL_SET_PATH))
    parser.add_argument("--no-report", action="store_true", help="不写报告文件")
    parser.add_argument("--self-test", action="store_true", help="自检指标计算（无需服务）")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0

    with open(args.eval_set, encoding="utf-8") as f:
        eval_set = json.load(f)
    cases = eval_set["cases"]
    default_item = eval_set.get("_meta", {}).get("item_name", "")
    for c in cases:
        c.setdefault("item_name", default_item)
    n_neg = sum(1 for c in cases if c.get("negative"))
    print(f"评测集：{len(cases)} 条（正例 {len(cases) - n_neg}，负例 {n_neg}）")

    ok, uri = check_milvus_reachable()
    if not ok:
        print(f"\n[FAIL] Milvus 不可达：{uri}")
        print("  请先在项目根目录执行：")
        print("    docker compose up -d        # 启动外部服务")
        print("    docker compose ps           # 等待全部 healthy")
        return 1
    print(f"Milvus 已连通：{uri}")

    if args.grid:
        run_grid(cases, args)
        return 0

    modes = ["vector", "rerank"] if args.mode == "both" else [args.mode]
    all_summaries = {}
    report_paths = []
    for mode in modes:
        print(f"\n{'#' * 76}\n#  模式：{mode}\n{'#' * 76}")
        print("加载模型与客户端中（首次会下载/加载权重，请耐心等待）...")
        retrieval_fn = build_retrieval_fn(mode, args.limit,
                                          search_limit=args.search_limit,
                                          use_child=not args.no_child)
        summary, positives, negatives = evaluate_all(cases, retrieval_fn)
        all_summaries[mode] = summary

        print(f"\n{'=' * 76}\n  汇总指标（{mode}，正例 {len(positives)} 条）\n{'=' * 76}")
        for k, v in summary.items():
            print(f"  {k:<10} {v:8.4f}")

        print_case_details(positives, negatives, mode)
        if not args.no_report:
            report_paths.append(save_report(mode, summary, positives, negatives, args))

    if len(all_summaries) > 1:
        print(f"\n{'=' * 76}\n  模式对比（向量检索 vs +重排）\n{'=' * 76}")
        keys = list(next(iter(all_summaries.values())).keys())
        print(f"  {'指标':<10} {'vector':>10} {'rerank':>10} {'提升':>10}")
        for k in keys:
            v1 = all_summaries["vector"].get(k, 0)
            v2 = all_summaries["rerank"].get(k, 0)
            print(f"  {k:<10} {v1:>10.4f} {v2:>10.4f} {v2 - v1:>+10.4f}")

    if report_paths:
        print("\n报告已保存：")
        for p in report_paths:
            print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
