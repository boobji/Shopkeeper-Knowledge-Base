"""商品名归一化过滤与检索回退逻辑的单测。"""

from knowledge.domain.filters import (
    build_item_name_norm_expr,
    normalize_item_name,
    normalize_item_names,
)


class TestNormalize:
    def test_strips_whitespace_and_lowercases(self):
        assert normalize_item_name("  RS-12 数字万用表  ") == "rs-12数字万用表"
        assert normalize_item_name("RS-12　数字万用表") == "rs-12数字万用表"  # 全角空格
        assert normalize_item_name("Huawei MateStation B520") == "huaweimatestationb520"

    def test_empty(self):
        assert normalize_item_name("") == ""
        assert normalize_item_name(None) == ""

    def test_batch_dedup(self):
        assert normalize_item_names(["A B", "ab", "", None, "C"]) == ["ab", "c"]


class TestFilterExpr:
    def test_expr_uses_norm_field(self):
        expr = build_item_name_norm_expr(["RS-12 数字万用表"])
        assert expr == 'item_name_norm in ["rs-12数字万用表"]'

    def test_expr_multiple(self):
        expr = build_item_name_norm_expr(["RS-12", "华为 B520"])
        assert expr == 'item_name_norm in ["rs-12", "华为b520"]'

    def test_empty_names_no_filter(self):
        assert build_item_name_norm_expr([]) == ""
        assert build_item_name_norm_expr(["", "  "]) == ""

