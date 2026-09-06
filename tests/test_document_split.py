"""DocumentSplitNode 切分/合并逻辑单测（纯函数，不依赖外部服务）"""

import pytest

from knowledge.processor.import_process.nodes.document_split_node import DocumentSplitNode


@pytest.fixture()
def node():
    return DocumentSplitNode()


class TestSplitByHeadings:
    def test_basic_two_levels(self, node):
        md = "# 商品介绍\n\n正文甲\n## 电池\n\n正文乙\n"
        sections = node._split_by_headings(md, "万用表")

        assert len(sections) == 2
        assert sections[0]["title"] == "# 商品介绍"
        assert "正文甲" in sections[0]["body"]
        # 二级标题的父标题应为一级行
        assert sections[1]["title"] == "## 电池"
        assert sections[1]["parent_title"] == "# 商品介绍"
        assert "正文乙" in sections[1]["body"]

    def test_no_heading_uses_file_title(self, node):
        sections = node._split_by_headings("只有一段正文\n", "手册A")
        assert len(sections) == 1
        assert sections[0]["title"] == "手册A"
        assert sections[0]["parent_title"] == "手册A"

    def test_heading_inside_code_fence_is_ignored(self, node):
        md = "# A\n\n```python\n# not a heading\n```\n\n尾巴\n"
        sections = node._split_by_headings(md, "F")
        assert len(sections) == 1
        assert sections[0]["title"] == "# A"
        assert "# not a heading" in sections[0]["body"]

    def test_sibling_resets_ancestor_chain(self, node):
        md = "# A\n## B\n## C\n"
        sections = node._split_by_headings(md, "F")
        assert [s["title"] for s in sections] == ["# A", "## B", "## C"]
        assert sections[2]["parent_title"] == "# A"


class TestSplitLongSection:
    def test_short_section_untouched(self, node):
        section = {"title": "T", "body": "短正文", "file_title": "F", "parent_title": "P"}
        result = node._split_long_section(section, 2000)
        assert result == [section]

    def test_long_section_split_with_part_suffix(self, node):
        section = {
            "title": "T",
            "body": "\n\n".join(["句子" * 30] * 20),  # 远超 max
            "file_title": "F",
            "parent_title": "P",
        }
        result = node._split_long_section(section, 500)
        assert len(result) > 1
        assert all(len(s["body"]) <= 500 for s in result)
        assert result[0]["title"].startswith("T-")
        assert result[0]["part"] == "1"

    def test_table_body_linearized(self, node):
        # 表格后带可分隔的正文段落，确保切分确实发生
        body = "<table><tr><td>x</td></tr></table>" + "\n\n".join(["句子。" * 20] * 30)
        section = {"title": "T", "body": body, "file_title": "F", "parent_title": "P"}
        result = node._split_long_section(section, 500)
        assert len(result) > 1
        # 表格被线性化，切出的片段不再包含原始 <table> 标签
        assert all("<table>" not in s["body"] for s in result)


class TestMergeShortSection:
    def test_short_sibling_merged(self, node):
        sections = [
            {"title": "P-1", "body": "短", "file_title": "F", "parent_title": "P"},
            {"title": "P-2", "body": "长" * 600, "file_title": "F", "parent_title": "P"},
        ]
        merged = node._merge_short_section(sections, 100)
        assert len(merged) == 1
        assert "短" in merged[0]["body"]
        assert merged[0]["part"] == 1

    def test_different_parent_not_merged(self, node):
        sections = [
            {"title": "P1", "body": "短", "file_title": "F", "parent_title": "P1"},
            {"title": "P2", "body": "短", "file_title": "F", "parent_title": "P2"},
        ]
        merged = node._merge_short_section(sections, 100)
        assert len(merged) == 2

    def test_long_first_section_kept_as_is(self, node):
        sections = [
            {"title": "P-1", "body": "长" * 600, "file_title": "F", "parent_title": "P"},
            {"title": "P-2", "body": "短", "file_title": "F", "parent_title": "P"},
        ]
        merged = node._merge_short_section(sections, 100)
        # 长的前段不合并，短的后段保留（无下一段可合）
        assert len(merged) == 2


class TestAssembleChunk:
    def test_content_prefixes_title(self, node):
        chunks = node._assemble_chunk(
            [{"title": "T", "body": "B", "file_title": "F", "parent_title": "P"}]
        )
        assert chunks[0]["content"] == "T\n\nB"
        assert "part" not in chunks[0]

    def test_part_carried_over(self, node):
        chunks = node._assemble_chunk(
            [{"title": "T", "body": "B", "file_title": "F", "parent_title": "P", "part": 2}]
        )
        assert chunks[0]["part"] == 2
