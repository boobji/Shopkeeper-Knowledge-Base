"""新格式支持（txt/html/docx）入口路由与转换节点的单测。"""

import pytest

from knowledge.processor.import_process.exceptions import (
    FileProcessingError,
    ValidationError,
)
from knowledge.processor.import_process.nodes.docx_to_md_node import DocxToMdNode
from knowledge.processor.import_process.nodes.entry_node import EntryNode
from knowledge.processor.import_process.nodes.html_to_md_node import HtmlToMdNode


def _make_file(tmp_path, name, content="x"):
    f = tmp_path / name
    f.write_text(content, encoding="utf-8")
    return f


class TestEntryRouting:
    def test_txt_routes_to_md_channel(self, tmp_path):
        f = _make_file(tmp_path, "说明.txt", "正文")
        state = EntryNode().process({"file_dir": str(tmp_path), "import_file_path": str(f)})
        assert state["is_md_read_enabled"] is True
        assert state["md_path"] == str(f)

    def test_html_and_htm_route_to_converter(self, tmp_path):
        for name in ("a.html", "a.htm"):
            f = _make_file(tmp_path, name, "<p>x</p>")
            state = EntryNode().process({"file_dir": str(tmp_path), "import_file_path": str(f)})
            assert state["is_html_enabled"] is True

    def test_docx_routes_to_converter(self, tmp_path):
        f = tmp_path / "手册.docx"
        f.write_bytes(b"PK")  # 内容不校验，只看后缀
        state = EntryNode().process({"file_dir": str(tmp_path), "import_file_path": str(f)})
        assert state["is_docx_enabled"] is True

    def test_unsupported_suffix_rejected(self, tmp_path):
        f = _make_file(tmp_path, "file.xlsx")
        with pytest.raises(ValidationError):
            EntryNode().process({"file_dir": str(tmp_path), "import_file_path": str(f)})

    def test_suffix_case_insensitive(self, tmp_path):
        f = _make_file(tmp_path, "MANUAL.TXT", "x")
        state = EntryNode().process({"file_dir": str(tmp_path), "import_file_path": str(f)})
        assert state["is_md_read_enabled"] is True


class TestHtmlToMdNode:
    def test_converts_and_sets_md_path(self, tmp_path):
        html = "<h1>万用表手册</h1><p>先装电池。</p><table><tr><th>量程</th></tr><tr><td>200V</td></tr></table>"
        f = _make_file(tmp_path, "manual.html", html)
        state = HtmlToMdNode().process({"file_dir": str(tmp_path), "import_file_path": str(f)})

        md_file = tmp_path / "manual.md"
        assert state["md_path"] == str(md_file)
        content = md_file.read_text(encoding="utf-8")
        assert "# 万用表手册" in content
        assert "先装电池。" in content
        assert "| 量程 |" in content

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileProcessingError):
            HtmlToMdNode().process(
                {"file_dir": str(tmp_path), "import_file_path": str(tmp_path / "nope.html")})

    def test_empty_body_raises(self, tmp_path):
        f = _make_file(tmp_path, "empty.html", "")
        with pytest.raises(FileProcessingError):
            HtmlToMdNode().process({"file_dir": str(tmp_path), "import_file_path": str(f)})


class TestDocxToMdNode:
    def test_missing_pandoc_gives_actionable_error(self, tmp_path, monkeypatch):
        import knowledge.processor.import_process.nodes.docx_to_md_node as mod

        f = tmp_path / "手册.docx"
        f.write_bytes(b"PK")
        monkeypatch.setattr(mod.shutil, "which", lambda name: None)
        with pytest.raises(FileProcessingError) as ei:
            DocxToMdNode().process({"file_dir": str(tmp_path), "import_file_path": str(f)})
        # 错误信息必须带安装指引
        assert "pandoc" in str(ei.value)

    def test_missing_docx_raises(self, tmp_path, monkeypatch):
        import knowledge.processor.import_process.nodes.docx_to_md_node as mod

        monkeypatch.setattr(mod.shutil, "which", lambda name: "pandoc")
        with pytest.raises(FileProcessingError):
            DocxToMdNode().process(
                {"file_dir": str(tmp_path), "import_file_path": str(tmp_path / "nope.docx")})
