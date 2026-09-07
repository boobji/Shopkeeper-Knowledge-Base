"""HTML → Markdown 转换器（domain/html_converter）单测。"""

from knowledge.domain.html_converter import html_to_markdown


class TestHtmlToMarkdown:
    def test_headings(self):
        md = html_to_markdown("<h1>标题一</h1><h2>标题二</h2><h3>三级</h3>")
        assert md == "# 标题一\n\n## 标题二\n\n### 三级"

    def test_paragraphs_and_links_kept(self):
        md = html_to_markdown('<p>第一段</p><p>第二段</p>')
        assert md == "第一段\n\n第二段"

    def test_unordered_list(self):
        md = html_to_markdown("<ul><li>甲</li><li>乙</li></ul>")
        assert md == "- 甲\n\n- 乙"

    def test_ordered_list(self):
        md = html_to_markdown("<ol><li>步骤一</li><li>步骤二</li></ol>")
        assert md == "1. 步骤一\n\n2. 步骤二"

    def test_nested_list_indent(self):
        md = html_to_markdown("<ul><li>父<ul><li>子</li></ul></li></ul>")
        assert "- 父" in md
        assert "  - 子" in md

    def test_table(self):
        html = """
        <table>
          <tr><th>量程</th><th>精度</th></tr>
          <tr><td>200V</td><td>±0.5%</td></tr>
          <tr><td>600V</td><td>±1.0%</td></tr>
        </table>
        """
        md = html_to_markdown(html)
        assert "| 量程 | 精度 |" in md
        assert "| --- | --- |" in md
        assert "| 200V | ±0.5% |" in md
        assert "| 600V | ±1.0% |" in md

    def test_image(self):
        md = html_to_markdown('<img src="images/a.png" alt="面板图">')
        assert "![面板图](images/a.png)" in md

    def test_code_block(self):
        md = html_to_markdown("<pre>print(1)</pre>")
        assert "```\nprint(1)\n```" in md

    def test_script_style_dropped(self):
        md = html_to_markdown(
            "<html><head><script>bad()</script><style>.x{}</style></head>"
            "<body><p>正文</p></body></html>"
        )
        assert md == "正文"

    def test_full_document_with_title_fallback(self):
        html = "<html><body><p>无标题文档</p></body></html>"
        md = html_to_markdown(html, title="产品说明")
        assert md.startswith("# 产品说明")
        assert "无标题文档" in md

    def test_title_not_duplicated_when_h1_exists(self):
        md = html_to_markdown("<h1>已有标题</h1><p>x</p>", title="产品说明")
        assert md.startswith("# 已有标题")

    def test_empty_input(self):
        assert html_to_markdown("") == ""
        assert html_to_markdown("   ") == ""

    def test_divs_recurse(self):
        md = html_to_markdown('<div><section><p>嵌套正文</p></section></div>')
        assert "嵌套正文" in md
