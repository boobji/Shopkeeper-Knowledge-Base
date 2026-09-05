# -*- coding: utf-8 -*-
"""Reranker CPU 速度实测 + 区分度验证（20 条候选）"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from knowledge.utils.bge_rerank_util import get_reranker_model

query = "怎么测这块主板的短路问题？"

relevant = [
    "主板短路通常表现为通电后风扇转一下就停，可以用万用表蜂鸣档测量对地阻值。",
    "主板通电前先打各主供电电感的对地阻值，阻值偏低说明存在短路。",
    "使用数字万用表二极管档测量主板各供电测试点对地阻值，判断是否短路。",
    "维修主板短路常用烧机法，配合松香烟雾或热像仪定位发热点。",
    "检查主板是否有进水、腐蚀或元件击穿导致电源对地短路。",
]
irrelevant = [
    "今天中午去吃猪脚饭吧，这家店的卤味很不错。",
    "苹果发布新款手机，A系列芯片性能提升20%。",
    "成都下周天气转凉，记得添衣服。",
    "Python 的 asyncio 和多线程在 IO 密集型任务上性能差异明显。",
    "股票市场中涨停板策略需要注意炸板率和连板高度。",
    "新能源汽车的电池管理系统会影响冬季续航表现。",
    "这家咖啡店的手冲耶加雪菲风味明亮，酸度适中。",
    "装修时水电改造要注意走顶不走地，方便后期检修。",
    "孩子上小学前需要培养的三种习惯，家长越早知道越好。",
    "显卡驱动更新后游戏帧数提升明显，建议定期升级。",
    "主板的价格受品牌和芯片组影响较大，选购时注意扩展接口。",
    "电脑蓝屏可能是内存条接触不良，可以拔下来用橡皮擦拭金手指。",
    "机械键盘的轴体分为青轴红轴茶轴，手感各不相同。",
    "显示器色域覆盖 sRGB 99% 以上才能满足基础设计需求。",
    "路由器放在弱电箱里会导致 WiFi 信号衰减严重。",
]

docs = []
for i, d in enumerate(relevant):
    docs.append((f"相关{i + 1}", d))
for i, d in enumerate(irrelevant):
    docs.append((f"无关{i + 1}", d))

print("=" * 68)
print(f"查询: {query}")
print(f"候选文档: {len(docs)} 条（前 5 条相关，后 15 条无关）")
print("=" * 68)

t0 = time.time()
model = get_reranker_model()
load_time = time.time() - t0
print(f"\n模型加载耗时: {load_time:.1f}s")

pairs = [(query, d) for _, d in docs]

t0 = time.time()
scores = model.compute_score(pairs)
infer_time = time.time() - t0
print(f"{len(pairs)} 条推理耗时: {infer_time:.2f}s  (平均 {infer_time / len(pairs) * 1000:.0f} ms/条)")

ranked = sorted(zip([lb for lb, _ in docs], scores), key=lambda x: x[1], reverse=True)

print("\n[排序结果 Top 8]")
for i, (lb, s) in enumerate(ranked[:8], 1):
    flag = "对" if lb.startswith("相关") else "错"
    print(f"  {i:2d}. {lb:6s} score={s:+7.3f}   {flag}")

rel = [s for lb, s in ranked if lb.startswith("相关")]
irr = [s for lb, s in ranked if lb.startswith("无关")]
print(f"\n[区分度]")
print(f"  相关分数区间: {min(rel):+.3f} ~ {max(rel):+.3f}")
print(f"  无关分数区间: {min(irr):+.3f} ~ {max(irr):+.3f}")
print(f"  间隔 (最低相关 - 最高无关) = {min(rel) - max(irr):+.3f}")

top5_hit = sum(1 for lb, _ in ranked[:5] if lb.startswith("相关"))
print(f"  Top5 命中相关数: {top5_hit}/5")
print(f"\n  断崖位置: 第 {sum(1 for _, s in ranked if s > max(irr))} 条之后")

# ==================== 端到端：走真实 RerankNode 代码路径 ====================
print("\n" + "=" * 68)
print("端到端验证：RerankNode.process() 的实际截断结果")
print("=" * 68)

from knowledge.processor.query_process.base import setup_logging
from knowledge.processor.query_process.nodes.rerank_node import RerankNode

setup_logging()
node = RerankNode()
print(f"\n生效配置: gap_abs={node.config.rerank_gap_abs}, "
      f"gap_ratio={node.config.rerank_gap_ratio}, "
      f"min_top_k={node.config.rerank_min_top_k}, "
      f"max_top_k={node.config.rerank_max_top_k}")

state = {
    "rewritten_query": query,
    "rrf_chunks": [
        {"chunk_id": lb, "title": lb, "content": txt} for lb, txt in docs
    ],
}
kept = node.process(state).get("reranked_docs", [])
n_rel = sum(1 for d in kept if d["chunk_id"].startswith("相关"))

print(f"\n输入 {len(docs)} 条（5 条相关 / 15 条无关）")
print(f"截断后保留 {len(kept)} 条 → 相关 {n_rel} 条、无关 {len(kept) - n_rel} 条")
for i, d in enumerate(kept, 1):
    print(f"  {i:2d}. {d['chunk_id']:6s} score={d['score']:+7.3f}")

ok = (n_rel == 5 and len(kept) == 5)
print(f"\n结论: {'通过（5 条相关全召回且不混入噪声）' if ok else '未通过'}")
