"""kg_search_node 中纯解析/清洗函数的单测"""

import pytest

from knowledge.processor.query_process.nodes.kg_search_node import (
    _build_item_entity_pairs,
    _clean_parse_llm_content,
    _clean_seed_rows,
    _item_name_filter_expr,
    truncate_entity_name_length,
)


class TestCleanParseLlmContent:
    def test_json_fence_stripped(self):
        raw = '```json\n{"entities": ["万用表", "蜂鸣档"]}\n```'
        assert _clean_parse_llm_content(raw) == ["万用表", "蜂鸣档"]

    def test_plain_json(self):
        raw = '{"entities": ["a"]}'
        assert _clean_parse_llm_content(raw) == ["a"]

    def test_invalid_json_returns_empty(self):
        assert _clean_parse_llm_content("不是json") == []

    def test_empty_and_missing_entities(self):
        assert _clean_parse_llm_content("") == []
        assert _clean_parse_llm_content('{"other": 1}') == []
        assert _clean_parse_llm_content('{"entities": "not-a-list"}') == []

    def test_dedup_and_skip_invalid_entries(self):
        raw = '{"entities": ["a", "", 123, "a"]}'
        assert _clean_parse_llm_content(raw) == ["a"]

    def test_long_entity_truncated(self):
        raw = '{"entities": ["' + "很" * 30 + '"]}'
        result = _clean_parse_llm_content(raw)
        assert len(result) == 1
        assert len(result[0]) == 15


class TestTruncateEntityNameLength:
    def test_short_unchanged(self):
        assert truncate_entity_name_length("  ab  ") == "ab"

    def test_long_truncated(self):
        assert len(truncate_entity_name_length("x" * 100)) == 15


class TestItemNameFilterExpr:
    def test_quoting(self):
        assert _item_name_filter_expr(["RS-12", "万用表"]) == "item_name in ['RS-12', '万用表']"

    def test_empty(self):
        assert _item_name_filter_expr([]) == "item_name in []"


class TestCleanSeedRows:
    def test_strips_and_filters(self):
        rows = [
            {"item_name": " RS-12 ", "name": " 蜂鸣档 "},
            {"item_name": "", "name": "x"},
            {"item_name": "RS-12", "name": ""},
            {},
        ]
        assert _clean_seed_rows(rows) == [{"item_name": "RS-12", "entity_name": "蜂鸣档"}]

    def test_empty(self):
        assert _clean_seed_rows([]) == []


class TestBuildItemEntityPairs:
    def test_dedup_same_item_keep_cross_item(self):
        info = [
            {"item_name": "A", "aligned": "x"},
            {"item_name": "A", "aligned": "x"},
            {"item_name": "A", "aligned": "y"},
            {"item_name": "B", "aligned": "x"},
            {"item_name": "", "aligned": "x"},
            {"item_name": "A", "aligned": ""},
        ]
        pairs = _build_item_entity_pairs(info)
        assert pairs == [
            {"item_name": "A", "entity_name": "x"},
            {"item_name": "A", "entity_name": "y"},
            {"item_name": "B", "entity_name": "x"},
        ]

    def test_empty(self):
        assert _build_item_entity_pairs([]) == []
