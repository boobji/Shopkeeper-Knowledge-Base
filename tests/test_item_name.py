"""domain/item_name 对齐逻辑与 domain/llm_parse 的单测。"""

import pytest

from knowledge.domain.item_name import (
    ItemNameAligner,
    ItemNameExtractor,
)
from knowledge.domain.llm_parse import strip_json_fence


def _match(name, score):
    return {"item_name": name, "score": score}


def _result(extracted, *matches):
    return {"extracted_name": extracted, "matches": list(matches)}


@pytest.fixture()
def aligner():
    return ItemNameAligner(collection_name="kb_item_names")


class TestStripJsonFence:
    def test_json_fence(self):
        assert strip_json_fence('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_plain_fence(self):
        assert strip_json_fence('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_no_fence(self):
        assert strip_json_fence('{"a": 1}') == '{"a": 1}'

    def test_empty(self):
        assert strip_json_fence("") == ""


class TestScoreAlign:
    def test_exact_match_goes_to_confirmed(self, aligner):
        results = [_result("RS-12", _match("RS-12 数字万用表", 0.9))]
        confirmed, options = aligner._item_name_score_align(results)
        assert confirmed == ["RS-12 数字万用表"]
        assert options == []

    def test_single_high_match_confirmed(self, aligner):
        results = [_result("RS12", _match("RS-12 数字万用表", 0.8))]
        confirmed, options = aligner._item_name_score_align(results)
        assert confirmed == ["RS-12 数字万用表"]

    def test_multiple_high_matches_go_to_options(self, aligner):
        results = [
            _result("万用表", _match("RS-12 数字万用表", 0.8), _match("RS-13 数字万用表", 0.75))
        ]
        confirmed, options = aligner._item_name_score_align(results)
        assert confirmed == []
        assert set(options) == {"RS-12 数字万用表", "RS-13 数字万用表"}

    def test_mid_score_goes_to_options(self, aligner):
        results = [
            _result("万用表", _match("RS-12 数字万用表", 0.65), _match("RS-13 数字万用表", 0.62))
        ]
        confirmed, options = aligner._item_name_score_align(results)
        assert confirmed == []
        assert set(options) == {"RS-12 数字万用表", "RS-13 数字万用表"}

    def test_below_mid_threshold_dropped(self, aligner):
        results = [_result("万用表", _match("RS-12 数字万用表", 0.5))]
        confirmed, options = aligner._item_name_score_align(results)
        assert confirmed == [] and options == []

    def test_options_capped_at_three(self, aligner):
        results = [
            _result(
                "万用表",
                _match("A", 0.65), _match("B", 0.64), _match("C", 0.63), _match("D", 0.62),
            )
        ]
        _, options = aligner._item_name_score_align(results)
        assert len(options) <= 3

    def test_no_duplicates_across_extractions(self, aligner):
        results = [
            _result("RS-12", _match("RS-12 数字万用表", 0.9)),
            _result("RS12万用表", _match("RS-12 数字万用表", 0.85)),
        ]
        confirmed, options = aligner._item_name_score_align(results)
        assert confirmed == ["RS-12 数字万用表"]
        assert options == []


class TestScoreFilter:
    def test_gap_filter_removes_outliers(self, aligner):
        confirmed = ["A", "B", "C"]
        results = [
            _result("x", _match("A", 0.92), _match("B", 0.88), _match("C", 0.66)),
        ]
        assert aligner._item_name_score_filter(confirmed, results) == ["A", "B"]

    def test_single_confirmed_not_filtered_by_caller(self, aligner):
        # match_align_filter 只在 len(confirmed) > 1 时调用过滤，这里直接验证过滤本身
        confirmed = ["A"]
        results = [_result("x", _match("A", 0.9))]
        assert aligner._item_name_score_filter(confirmed, results) == ["A"]


class TestMatchAlignFilter:
    def test_empty_input(self, aligner):
        assert aligner.match_align_filter([]) == ([], [])


class TestCleanParse:
    def test_valid_output(self):
        extractor = ItemNameExtractor()
        parsed = extractor._clean_parse('```json\n{"item_names": ["RS-12"], "rewritten_query": "如何测电压"}\n```')
        assert parsed == {"item_names": ["RS-12"], "rewritten_query": "如何测电压"}

    def test_non_list_item_names(self):
        extractor = ItemNameExtractor()
        parsed = extractor._clean_parse('{"item_names": "RS-12"}')
        assert parsed["item_names"] == []

    def test_invalid_json_raises(self):
        extractor = ItemNameExtractor()
        with pytest.raises(ValueError):
            extractor._clean_parse("不是json")
