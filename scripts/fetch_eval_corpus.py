#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""下载评测语料：RealAppliance 家电说明书中文版（30 本 PDF）。

数据源：github.com/gaoyz1235/RealAppliance（真实家电产品手册，多语言）
        本脚本只抓取其中 `_ch` 目录下的中文版 `*_CH.pdf`。

版权提示（重要）：
    上游仓库未提供开源许可证，PDF 原文版权归各自厂商所有。
    因此本脚本下载的 PDF **仅用于本地评测**，默认落在被 .gitignore
    忽略的 `eval/corpus/` 目录，**请勿提交到公开仓库**。
    评测集本体（问题 + 期望答案）不受此限制，可正常公开。

通路说明（已实测）：
    raw.githubusercontent.com 在部分网络下不可达（HTTP 000）；
    ghfast.top 已失效。故采用双通路 + 自动回退：
      1) GitHub Contents API（Accept: application/vnd.github.raw）
      2) jsDelivr CDN（cdn.jsdelivr.net/gh/...）

用法：
    python scripts/fetch_eval_corpus.py                 # 下载全部 30 本
    python scripts/fetch_eval_corpus.py --limit 5       # 只下前 5 本（试跑）
    python scripts/fetch_eval_corpus.py --workers 8     # 调整并发
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------- 让 Windows 控制台也能输出中文 ----------
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

REPO = "gaoyz1235/RealAppliance"
BRANCH = "main"
PREFIX = "manuals_pdfs/"

# 品类名英 -> 中（用于生成可读的中文文件名）
CATEGORY_ZH = {
    "airfryer": "空气炸锅",
    "breadmachine": "面包机",
    "hotpot": "电火锅",
    "ricecooker": "电饭煲",
    "coffeemachine": "咖啡机",
    "blender": "搅拌机",
    "kettle": "电水壶",
    "microwave": "微波炉",
    "mixer": "搅拌器",
    "toaster": "烤面包机",
    "oven": "烤箱",
}

# 30 本中文手册（编号 -> 品类英文键）
MANUALS: list[tuple[str, str]] = [
    ("001", "airfryer"),
    ("002", "breadmachine"),
    ("003", "airfryer"),
    ("005", "hotpot"),
    ("010", "ricecooker"),
    ("011", "coffeemachine"),
    ("012", "ricecooker"),
    ("013", "airfryer"),
    ("014", "hotpot"),
    ("016", "hotpot"),
    ("017", "blender"),
    ("018", "kettle"),
    ("021", "kettle"),
    ("024", "microwave"),
    ("026", "coffeemachine"),
    ("031", "toaster"),
    ("032", "microwave"),
    ("033", "toaster"),
    ("034", "microwave"),
    ("035", "coffeemachine"),
    ("065", "mixer"),
    ("077", "airfryer"),
    ("083", "breadmachine"),
    ("084", "airfryer"),
    ("092", "microwave"),
    ("093", "ricecooker"),
    ("094", "blender"),
    ("096", "coffeemachine"),
    ("098", "microwave"),
    ("099", "oven"),
]

# 上游仓库中每个 PDF 的 Git blob SHA1，用于下载后校验文件完整性
# （来源：api.github.com/repos/gaoyz1235/RealAppliance/git/trees/main?recursive=1）
EXPECTED_SHA = {
    "001": "6f33b80bc14632850aeb28b5cdbbd3be39dd8c61",
    "002": "e71118294c9aadad4fa02ed6c761df5a22469c9e",
    "003": "ba8a13fe001555e7a8bdec822962ac4096660528",
    "005": "02eac767b491599317d6a5aabe3c1c9b3eed5363",
    "010": "c802bc18ed19d9e97f9973dbfe0aab17b48e2dca",
    "011": "94caa151fa61caa91a55fc8037471ab85df8ee4d",
    "012": "7a92e495d826c63547058f91980c292c077d311f",
    "013": "460e83c3d79d6f12dbeebb93c98a1511acbe10bb",
    "014": "2d1472e4e149022d2f18af9689a04b6bd340e72c",
    "016": "18242902ee7f96154d2392bf3974953b82b52100",
    "017": "8722dc88c5c9d3e0ba7a62a36111ed93dc2976ee",
    "018": "e0c073e2e9667feddb74a4181a948af09420b26f",
    "021": "aea99b71d39125f41e6111da3aaf4730475aabfd",
    "024": "58c7ac8107679aedc8e70a59d6167e5b02d21eb8",
    "026": "e5f4feea62743d730ed80495b61c7bac56ae27f5",
    "031": "a9db0002c3d524c4f834e05f9253f9a2e20388f6",
    "032": "5426525e73705ae82123f86f92de505b566b7b8a",
    "033": "01718893d31c28657b540b14e25f95a2d6df7370",
    "034": "2b9a457f0ab94a20aa347a937068ad96502381e8",
    "035": "4f01abce0936a13a7e8c0f7162eb462df49b0fc9",
    "065": "a48335e53b201208949ae7917a526514652e7706",
    "077": "b01d79c635d932a1750c2d99e1159da663f792d2",
    "083": "b89e04ef354faf18da30581a1f96dcafc88d4596",
    "084": "233f6422eb0ee0f924bc3525353ac9d816603a56",
    "092": "9896510cef85dd39cf677fb089c13102b09f7472",
    "093": "483908543541af8594fe23ca3a7713b638a5f0af",
    "094": "83d17c0fec8f3967e3320385f00e0a55b9e80be6",
    "096": "6880b24bcabb64bdb3981abbb77eb2de01ee98da",
    "098": "9f21785e4243a512e03d8644551c66f95f81f790",
    "099": "77bceafd071d1dc4cf196254b5a74c0d18cc871a",
}

# 超大扫描版说明书：超过 jsDelivr 的 20MB 单文件上限，且 GitHub Contents API
# 对其返回 500，只能通过 git 稀疏克隆获取（3GB 仓库，实测十几分钟仍未完成）。
# 因此**默认跳过**这两本；确实需要时用 --include-huge 开启。
HUGE_NO = {"013", "017"}

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ShopkeeperKB-corpus-fetcher/1.0"


def git_blob_sha1(data: bytes) -> str:
    """按 Git 对象格式计算 blob 的 SHA1（sha1('blob <len>\\0' + content)）。"""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def remote_path(no: str, cat: str) -> str:
    """仓库内的相对路径，如 manuals_pdfs/001_airfryer_ch/001_CH.pdf"""
    return f"{PREFIX}{no}_{cat}_ch/{no}_CH.pdf"


def url_candidates(rel: str) -> list[str]:
    """按优先级返回下载候选地址（失败自动回退下一个）。

    jsDelivr 放首位：实测 GitHub Contents API 在匿名访问时容易触发
    限流 / 500，而 jsDelivr 是 CDN，稳定且速度快。
    """
    return [
        f"https://cdn.jsdelivr.net/gh/{REPO}@{BRANCH}/{rel}",
        f"https://api.github.com/repos/{REPO}/contents/{rel}?ref={BRANCH}",
    ]


def _ssl_context() -> ssl.SSLContext:
    """构造 SSL 上下文。

    兼容性处理：Python 3.13 默认启用 VERIFY_X509_STRICT，某些企业/本机
    代理签发的证书缺少 Authority Key Identifier 扩展，会导致
    "Missing Authority Key Identifier" 报错。这里仅关闭严格模式，
    **仍保留证书链校验与主机名校验**，不是关闭验证。
    """
    ctx = ssl.create_default_context()
    try:
        ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    except Exception:  # pragma: no cover
        pass
    return ctx


_SSL_CTX = _ssl_context()


def _get(url: str, timeout: int, accept_raw: bool) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    if accept_raw:
        # Contents API 加这个 Accept 头会直接返回原始字节，否则返回 JSON
        req.add_header("Accept", "application/vnd.github.raw")
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
        return resp.read()


def download_one(no: str, cat: str, out_dir: Path, timeout: int, retries: int) -> dict:
    """下载单本手册，返回结果字典。"""
    rel = remote_path(no, cat)
    zh = CATEGORY_ZH.get(cat, cat)
    fname = f"{no}_{zh}.pdf"
    dest = out_dir / fname

    # 断点续传：已存在且校验通过就跳过（顺带校验，确保旧文件没损坏）
    if dest.exists() and dest.stat().st_size > 50 * 1024:
        head = dest.open("rb").read(4)
        if head == b"%PDF":
            digest = git_blob_sha1(dest.read_bytes())
            if digest == EXPECTED_SHA.get(no):
                return {
                    "no": no, "category": zh, "file": fname, "bytes": dest.stat().st_size,
                    "status": "skipped", "source": "sha1-ok",
                }

    last_err = ""
    for attempt in range(1, retries + 1):
        for url in url_candidates(rel):
            try:
                data = _get(url, timeout, accept_raw=url.startswith("https://api.github.com"))
                if not data.startswith(b"%PDF"):
                    last_err = f"返回内容非 PDF（{len(data)} 字节）"
                    continue
                if len(data) < 50 * 1024:
                    last_err = f"文件过小，疑似错误页（{len(data)} 字节）"
                    continue
                digest = git_blob_sha1(data)
                if digest != EXPECTED_SHA.get(no):
                    last_err = f"SHA1 不匹配（{digest[:8]} != {EXPECTED_SHA.get(no, '-')[:8]}）"
                    continue  # 视为本次下载失败，换通路/重试
                dest.write_bytes(data)
                return {
                    "no": no, "category": zh, "file": fname, "bytes": len(data),
                    "status": "ok",
                    "source": "jsdelivr" if "jsdelivr" in url else "github-api",
                }
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(1.5 * attempt)
    return {
        "no": no, "category": zh, "file": fname, "bytes": 0,
        "status": "failed", "source": last_err,
    }


def git_sparse_fetch(items: list[tuple[str, str]], out_dir: Path) -> list[dict]:
    """用 git 稀疏克隆补齐 HTTP 通路拿不到的超大文件。

    只 checkout 指定目录，避免下载整个 3GB 仓库；但 GitHub 侧仍需传输
    大量 tree 元数据，实测很慢，所以仅作为 --include-huge 的兜底手段。
    """
    print(f"\n[稀疏克隆] 尝试用 git 获取 {len(items)} 个超大文件（可能耗时很久）…")
    results: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="ra_sparse_") as td:
        repo = Path(td) / "repo"
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                 f"https://github.com/{REPO}.git", repo.name],
                cwd=td, check=True, timeout=3600,
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
            )
            dirs = [f"{PREFIX}{no}_{cat}_ch" for no, cat in items]
            subprocess.run(["git", "sparse-checkout", "set", *dirs],
                           cwd=repo, check=True, timeout=1800,
                           stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"[稀疏克隆] 失败：{e}")
            return [{"no": no, "category": CATEGORY_ZH.get(cat, cat),
                     "file": f"{no}_{CATEGORY_ZH.get(cat, cat)}.pdf", "bytes": 0,
                     "status": "failed", "source": "git 稀疏克隆失败"} for no, cat in items]

        for no, cat in items:
            zh = CATEGORY_ZH.get(cat, cat)
            src = repo / PREFIX / f"{no}_{cat}_ch" / f"{no}_CH.pdf"
            if src.exists():
                data = src.read_bytes()
                if git_blob_sha1(data) == EXPECTED_SHA.get(no):
                    dest = out_dir / f"{no}_{zh}.pdf"
                    shutil.copyfile(src, dest)
                    results.append({"no": no, "category": zh, "file": dest.name,
                                    "bytes": len(data), "status": "ok", "source": "git-sparse"})
                    continue
            results.append({"no": no, "category": zh, "file": f"{no}_{zh}.pdf",
                            "bytes": 0, "status": "failed", "source": "稀疏克隆未取到文件"})
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="下载 RealAppliance 中文家电说明书语料")
    ap.add_argument("-o", "--out", default="eval/corpus/zh_manuals", help="输出目录（相对项目根）")
    ap.add_argument("--limit", type=int, default=0, help="只下载前 N 本（0 = 全部）")
    ap.add_argument("--workers", type=int, default=6, help="并发下载线程数")
    ap.add_argument("--timeout", type=int, default=90, help="单次请求超时（秒）")
    ap.add_argument("--retries", type=int, default=3, help="每个文件的重试次数")
    ap.add_argument("--include-huge", action="store_true",
                    help="同时下载 013(32MB) / 017(59MB) 两本超大扫描版（很慢，可能失败）")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_dir = (root / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = MANUALS if args.limit <= 0 else MANUALS[: args.limit]
    if not args.include_huge:
        targets = [t for t in targets if t[0] not in HUGE_NO]
        print("已跳过超大扫描版 013 / 017（超 jsDelivr 20MB 上限），需要时加 --include-huge")
    total = len(targets)
    print(f"目标目录: {out_dir}")
    print(f"待下载: {total} 本（并发 {args.workers}）\n")

    results: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, no, cat, out_dir, args.timeout, args.retries): no
            for no, cat in targets
        }
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            done += 1
            flag = {"ok": "OK  ", "skipped": "SKIP", "failed": "FAIL"}[r["status"]]
            size = f"{r['bytes'] / 1024:.0f}KB" if r["bytes"] else "-"
            print(f"[{done:>2}/{total}] {flag} {r['file']:<24} {size:<8} {r['source']}")

    # 超大文件兜底：HTTP 通路失败时改用 git 稀疏克隆
    if args.include_huge:
        failed_nos = {r["no"] for r in results if r["status"] == "failed"}
        huge_failed = [(no, cat) for no, cat in targets if no in HUGE_NO and no in failed_nos]
        if huge_failed:
            fetched = git_sparse_fetch(huge_failed, out_dir)
            patch = {r["no"]: r for r in fetched}
            results = [patch.get(r["no"], r) if r["no"] in HUGE_NO else r for r in results]
            for r in fetched:
                print(f"[补齐] {r['file']}: {r['status']}")

    # 写清单（UTF-8 BOM，Excel 直接打开不乱码）
    results.sort(key=lambda x: x["no"])
    manifest = out_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["编号", "品类", "文件名", "大小(字节)", "状态", "来源/错误", "仓库原始路径"])
        for r in results:
            rel = remote_path(r["no"], next(c for n, c in MANUALS if n == r["no"]))
            w.writerow([r["no"], r["category"], r["file"], r["bytes"],
                        r["status"], r["source"], rel])

    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = [r for r in results if r["status"] == "failed"]
    total_bytes = sum(r["bytes"] for r in results)

    print(f"\n{'=' * 56}")
    print(f"完成: 新下载 {ok} 本，跳过(已存在) {skipped} 本，失败 {len(failed)} 本")
    print(f"总体积: {total_bytes / 1024 / 1024:.1f} MB")
    print(f"清单: {manifest}")
    if failed:
        print("\n失败明细（重跑本脚本即可自动续传）:")
        for r in failed:
            print(f"  - {r['file']}: {r['source']}")
    print("\n提醒: PDF 原文版权归厂商所有，请勿提交到公开仓库。")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
