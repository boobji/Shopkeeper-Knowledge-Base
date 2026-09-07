"""P0 通用化改造的单元测试：切片清洗 + 结构还原 + 商品名采样。

全部用例不依赖任何外部服务（Milvus / Neo4j / LLM），可直接运行：

    knowledge/.venv/Scripts/python.exe -m pytest tests/test_p0_generalization.py -q
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

from knowledge.domain.chunk_clean import (
    BoilerplateStore,
    clean_chunks,
    is_toc_page,
    jaccard,
    make_shingles,
    sample_shingles,
    zh_ratio,
)
from knowledge.domain.structure_aware import (
    enhance_headings,
    enhance_markdown,
    estimate_hard_split_rate,
    extract_headings,
)


def make_config(**overrides) -> SimpleNamespace:
    """构造 ImportConfig 的最小替身（测试只关心清洗相关字段）。"""
    base = dict(
        chunk_clean_enabled=True,
        chunk_min_zh_ratio=0.20,
        chunk_drop_zh_ratio=0.05,
        chunk_drop_toc=True,
        chunk_boilerplate_enabled=True,
        chunk_boilerplate_threshold=0.60,
        chunk_boilerplate_min_docs=2,
        boilerplate_store_path="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def chunk_with(content: str, title: str = "", **extra) -> dict:
    data = {"title": title, "content": content, "file_title": "测试文档"}
    data.update(extra)
    return data


# ------------------------------------------------------------------ 基础工具
def test_zh_ratio_basic():
    assert zh_ratio("如何使用 coffee maker 制作咖啡") > 0.0
    assert zh_ratio("Quick Guide to Operation") == 0.0      # 纯英文
    assert zh_ratio("200mA 250V ±0.5%") == 0.0              # 纯符号
    assert zh_ratio("") == 0.0


def test_is_toc_page_detects_dot_leaders():
    assert is_toc_page("微波炉的放置 .................... 12")
    assert is_toc_page("简明操作指南 ········· 4-5")
    assert not is_toc_page("将电源插头插入插座。")


def test_sample_shingles_is_deterministic():
    """两次采样必须一致，否则跨文档比对永远失配。"""
    text = "请勿在潮湿环境下使用本产品以免发生触电危险"
    a = sample_shingles(make_shingles(text))
    b = sample_shingles(make_shingles(text))
    assert a == b
    assert len(a) > 0


def test_jaccard_identical_text():
    s = make_shingles("安全注意事项：请勿拆解本产品")
    assert jaccard(s, s) == 1.0
    assert jaccard(s, make_shingles("完全不同的另一段文字内容")) < 0.3


# ------------------------------------------------------------------ P0-3 清洗规则
def test_clean_drops_toc_page():
    chunks = [
        chunk_with("使用前准备 ........................... 8"),
        chunk_with("将滤网对准卡槽推入，听到咔哒声即安装到位。"),
    ]
    result = clean_chunks(chunks, "doc1", make_config(), file_dir="")
    assert len(result.chunks) == 1
    assert result.stats["drop_toc"] == 1
    assert "咔哒声" in result.chunks[0]["content"]


def test_clean_drops_non_chinese_page():
    chunks = [
        chunk_with("This appliance is intended for household use only."),
        chunk_with("清洗水箱时应使用柔软的湿布擦拭。"),
    ]
    result = clean_chunks(chunks, "doc1", make_config(), file_dir="")
    assert len(result.chunks) == 1
    assert result.stats["drop_non_zh"] == 1


def test_clean_marks_mixed_language():
    """中文占比介于丢弃与正常之间 → 保留但标记 mixed。

    注意：40+ 字符的整句很常见，短序列里符号/单位占比高才会落进 mixed 区间。
    """
    content = "Press POWER button to start the brewing cycle 按下电源键。"
    result = clean_chunks([chunk_with(content)], "doc1", make_config(), file_dir="")
    assert len(result.chunks) == 1
    assert result.chunks[0]["lang"] == "mixed"
    # 同时确认它落在 (drop_zh_ratio, min_zh_ratio) 这个"保留但降权"的区间内
    assert 0.05 <= zh_ratio(content) < 0.20


def test_clean_removes_in_document_duplicates():
    # 归一后长度需 >= 20 才会判定重复：短句（如"请勿拆解本机。"）重复属正常表达，不能误杀
    same = "为确保使用安全请务必使用本型号专用原厂配件以免造成设备损坏或其他意外后果。"
    chunks = [chunk_with(same), chunk_with(same + "。"), chunk_with("清洗方法与日常保养事项的具体操作步骤说明。")]
    result = clean_chunks(chunks, "doc1", make_config(), file_dir="")
    contents = [c["content"] for c in result.chunks]
    assert len(contents) == 2
    assert result.stats["drop_in_doc_dup"] >= 1


def test_clean_keeps_repeated_short_sentences():
    """短句式重复是说明书的正常表达（如每节都提醒断电），不应被去重。"""
    short = "操作前请断电。"
    chunks = [chunk_with(short), chunk_with(short), chunk_with("清洗滤网请使用软布擦拭。")]
    result = clean_chunks(chunks, "doc1", make_config(), file_dir="")
    assert len(result.chunks) == 3
    assert result.stats["drop_in_doc_dup"] == 0


# ------------------------------------------------------------------ P0-2 跨文档样板段
def test_boilerplate_across_three_documents(tmp_path):
    """样板段要在 >=2 篇文档出现后才判定，故第三个文档才会被剔除。

    每篇文档各自的正文必须互不相同——否则它们之间也会互相判定为样板段。
    """
    boilerplate = "本产品不打算由儿童或有体力感官或精神缺陷的人使用除非有监护人监督指导。"
    own_map = {
        "docA": "本机配备不锈钢滤网可拆卸清洗维护请参照本章节图示步骤进行。",
        "docB": "蒸汽阀门需要每月除垢一次请按照说明书给出的操作流程执行。",
        "docC": "水箱请注入纯净水长期使用可减少水垢形成并提升加热效率。",
    }

    for name, own in own_map.items():
        r = clean_chunks([chunk_with(boilerplate), chunk_with(own)], name,
                         make_config(), file_dir=str(tmp_path))
        if name in ("docA", "docB"):
            assert len(r.chunks) == 2, f"{name} 不应误杀"

    # 第三篇：样板段命中被剔除，独有正文保留
    r3 = clean_chunks([chunk_with(boilerplate), chunk_with(own_map["docC"])], "docC",
                      make_config(), file_dir=str(tmp_path))
    assert len(r3.chunks) == 1
    assert r3.stats["drop_boilerplate"] == 1
    assert "水垢" in r3.chunks[0]["content"]


def test_boilerplate_disabled_keeps_everything(tmp_path):
    boilerplate = "本产品不打算由儿童或有体力感官或精神缺陷的人使用除非有监护人监督指导。"
    cfg = make_config(chunk_boilerplate_enabled=False)
    for name in ("docA", "docB", "docC"):
        r = clean_chunks([chunk_with(boilerplate)], name, cfg, file_dir=str(tmp_path))
        assert len(r.chunks) == 1


def test_clean_falls_back_when_everything_dropped(tmp_path):
    """阈值过激把整篇洗没时，必须回退原文而不是让下游拿到空列表。"""
    # zh_ratio 上限为 1.0，故阈值必须大于 1 才能把纯中文也判为"非中文"
    cfg = make_config(chunk_drop_zh_ratio=1.5)
    chunks = [chunk_with("正常的中文说明书内容应当被保留下来。")]
    result = clean_chunks(chunks, "doc1", cfg, file_dir=str(tmp_path))
    assert result.chunks == chunks
    assert result.stats.get("fallback") == 1


def test_clean_handles_empty_input():
    result = clean_chunks([], "doc1", make_config(), file_dir="")
    assert result.chunks == []
    assert result.stats["input"] == 0


def test_boilerplate_store_persists(tmp_path):
    store = BoilerplateStore.load(None, fallback_dir=str(tmp_path))
    shingles = make_shingles("这是第一段需要被记录的文本内容用于跨文档比对")
    store.add(shingles, "docX")
    store.save()

    reloaded = BoilerplateStore.load(None, fallback_dir=str(tmp_path))
    assert reloaded.docs == ["docX"]
    assert reloaded.is_boilerplate(shingles, threshold=0.60, min_docs=1)


# ------------------------------------------------------------------ P0-1 结构还原
def test_extract_headings_reads_text_level():
    items = [
        {"type": "text", "text": "使用说明书", "text_level": 1, "bbox": [0, 0, 10, 10]},
        {"type": "text", "text": "安全须知", "text_level": 2, "bbox": [0, 0, 10, 10]},
        {"type": "header", "text": "页眉品牌", "text_level": 1, "bbox": [0, 0, 10, 10]},
        {"type": "footer", "text": "28/09/2017", "bbox": [0, 0, 10, 10]},
        {"type": "text", "text": "请勿浸入水中", "bbox": [0, 0, 10, 10]},  # 无 level → 正文
    ]
    headings = extract_headings(items)
    assert headings == [("使用说明书", 1), ("安全须知", 2)]


def test_enhance_markdown_adds_heading_prefix():
    md = "使用说明书\n这是一段正文内容需要保留原样不被修改。\n安全须知\n请勿浸入水中。"
    headings = [("使用说明书", 1), ("安全须知", 2)]
    enhanced, matched = enhance_markdown(md, headings)
    assert matched == 2
    assert "# 使用说明书" in enhanced
    assert "## 安全须知" in enhanced
    assert "这是一段正文内容需要保留原样不被修改。" in enhanced


def test_enhance_markdown_tolerates_ocr_spacing():
    """OCR 带来的空格/标点差异应通过归一化匹配兜住。"""
    md = "注 意 事 项\n正文内容保持不变。"
    enhanced, matched = enhance_markdown(md, [("注意事项", 2)])
    assert matched == 1
    assert "## 注 意 事 项" in enhanced


def test_enhance_markdown_never_touches_body_on_miss():
    md = "正文里没有出现过任何标题字样。"
    enhanced, matched = enhance_markdown(md, [("不存在的标题", 1)])
    assert matched == 0
    assert enhanced == md


def test_enhance_markdown_skips_existing_headings():
    md = "## 已经是标题\n正文。"
    enhanced, matched = enhance_markdown(md, [("已经是标题", 1)])
    assert matched == 0, "已有 # 前缀的行不应重复标注"


def test_estimate_hard_split_rate():
    import re

    heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")
    assert estimate_hard_split_rate("一段没有标题的正文内容。", "doc", heading_re) == 1.0
    structured = "\n".join([f"## 标题{i}\n正文{i}" for i in range(30)])
    assert estimate_hard_split_rate(structured, "doc", heading_re) == 0.0


def test_enhance_headings_disabled_or_healthy_doc(tmp_path):
    md = "# 标题\n正文内容足够长用于测试。"
    enhanced, stat = enhance_headings(md, str(tmp_path), "doc",
                                      enabled=True, hard_split_rate=0.0, threshold=0.3)
    assert enhanced is md
    assert stat["reason"].startswith("结构完好")

    enhanced, stat = enhance_headings(md, str(tmp_path), "doc",
                                      enabled=False, hard_split_rate=1.0, threshold=0.3)
    assert stat["reason"] == "disabled"


def test_enhance_headings_end_to_end(tmp_path):
    """给定 content_list.json，应把丢失的标题还原回 Markdown。"""
    doc_dir = tmp_path / "doc" / "auto"
    doc_dir.mkdir(parents=True)
    items = [
        {"type": "text", "text": "安全须知", "text_level": 2, "bbox": [0, 0, 1, 1]},
        {"type": "text", "text": "请勿拆解本机。", "bbox": [0, 0, 1, 1]},
    ]
    (doc_dir / "doc_content_list.json").write_text(
        json.dumps(items, ensure_ascii=False), encoding="utf-8")

    md = "安全须知\n请勿拆解本机。"
    enhanced, stat = enhance_headings(md, str(tmp_path), "doc",
                                      enabled=True, hard_split_rate=1.0, threshold=0.3)
    assert stat["applied"] is True
    assert stat["matched"] == 1
    assert "## 安全须知" in enhanced


# ------------------------------------------------------------------ P0-4 商品名采样
def _make_node(cls):
    """手工构造节点实例：跳过 __init__（它会拉起 LLM/Milvus 等外部客户端），
    只补上 process 相关方法依赖的 logger。"""
    node = cls.__new__(cls)
    node.logger = logging.getLogger(f"test.{cls.name}")
    node.config = None
    return node


def test_item_name_sampling_prefers_model_code_chunk():
    pytest.importorskip("langchain_core.messages")
    from knowledge.processor.import_process.nodes.item_name_recognition_node import (
        ItemNameRecognitionNode,
    )

    node = _make_node(ItemNameRecognitionNode)
    cfg = SimpleNamespace(item_name_chunk_k=2, item_name_chunk_size=2000)

    chunks = [
        chunk_with("请勿在儿童可触及范围内使用本产品以免发生意外伤害事故。", lang="zh"),
        chunk_with("目录\n第一章 ........................ 4", lang="mixed"),
        chunk_with("本机型号为 KF-21B18，请核对铭牌上的型号是否一致。", lang="zh"),
    ]
    context = node._prepare_item_name_context(chunks, cfg, file_title="测试文档")

    # 型号片段必须被选入，且不能混入纯扫描 Blob 式的噪声内容
    assert "KF-21B18" in context
    assert "目录" not in context or context.index("KF-21B18") >= 0


def test_item_name_sampling_respects_order():
    pytest.importorskip("langchain_core.messages")
    from knowledge.processor.import_process.nodes.item_name_recognition_node import (
        ItemNameRecognitionNode,
    )

    node = _make_node(ItemNameRecognitionNode)
    cfg = SimpleNamespace(item_name_chunk_k=3, item_name_chunk_size=3000)
    chunks = [
        chunk_with("第一段含型号 RS-12 的文字内容足够长用于测试。", lang="zh"),
        chunk_with("第二段含型号 UT890D 的文字内容足够长用于测试。", lang="zh"),
    ]
    context = node._prepare_item_name_context(chunks, cfg, file_title="测试文档")
    assert context.index("RS-12") < context.index("UT890D"), "挑选后应按文档顺序还原"
