"""HTML → Markdown 转换节点。

读取上传的 .html/.htm 文件，用规则转换器（domain/html_converter，纯 bs4，
无 AI）转成 Markdown，写到 file_dir 下并把 md_path 交给后续流水线——
之后走与 .md 上传完全相同的链路（图片理解 → 切片 → …）。
"""

from pathlib import Path

from knowledge.domain.html_converter import html_to_markdown
from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import FileProcessingError
from knowledge.processor.import_process.state import ImportGraphState


class HtmlToMdNode(BaseNode):
    name = "html_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        self.log_step("step1", "HTML 转 Markdown")

        import_file_path = state.get('import_file_path', '')
        file_dir = state.get('file_dir', '')
        if not import_file_path or not file_dir:
            raise FileProcessingError("HTML 文件路径或输出目录缺失", self.name)

        html_path = Path(import_file_path)
        if not html_path.exists():
            raise FileProcessingError(f"HTML 文件不存在: {import_file_path}", self.name)

        try:
            html_text = html_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            raise FileProcessingError(f"HTML 文件读取失败: {e}", self.name)

        md_content = html_to_markdown(html_text, title=html_path.stem)
        if not md_content.strip():
            raise FileProcessingError("HTML 转换结果为空（文件可能没有正文内容）", self.name)

        md_path = Path(file_dir) / f"{html_path.stem}.md"
        md_path.write_text(md_content, encoding="utf-8")

        state['md_path'] = str(md_path)
        self.logger.info(f"HTML 已转换为 Markdown: {md_path}（{len(md_content)} 字符）")
        return state
