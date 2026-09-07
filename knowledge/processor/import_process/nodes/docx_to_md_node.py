"""Word (.docx) → Markdown 转换节点（pandoc 路线）。

调用外部 pandoc CLI 把 docx 转成 GFM（GitHub Flavored Markdown）：
- 标题层级/表格/列表精确转换（docx 是结构化格式，规则解析无损）；
- `--extract-media` 导出内嵌图片，扁平化归置到 <file_dir>/images/ 并重写
  md 内的图片链接，使后续 md_img_node 的 VLM 图片理解流程直接复用。

pandoc 是独立 CLI（非 Python 依赖）：未安装时抛出带安装提示的明确错误，
不影响 pdf/md/html 等其它格式。安装：winget install pandoc 或官网下载。
"""

import shutil
import subprocess
import uuid
from pathlib import Path

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import FileProcessingError
from knowledge.processor.import_process.state import ImportGraphState

_PANDOC_HINT = (
    "未检测到 pandoc，无法转换 Word 文档。请安装后重试："
    "winget install pandoc（或从 https://pandoc.org 安装），装完无需重启服务"
)

# md 中图片链接需要落在这个目录（md_img_node 按平铺目录扫描）
IMAGES_SUBDIR = "images"


class DocxToMdNode(BaseNode):
    name = "docx_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        self.log_step("step1", "Word(docx) 转 Markdown")

        import_file_path = state.get('import_file_path', '')
        file_dir = state.get('file_dir', '')
        if not import_file_path or not file_dir:
            raise FileProcessingError("Word 文件路径或输出目录缺失", self.name)

        docx_path = Path(import_file_path)
        if not docx_path.exists():
            raise FileProcessingError(f"Word 文件不存在: {import_file_path}", self.name)

        pandoc = shutil.which("pandoc")
        if not pandoc:
            raise FileProcessingError(_PANDOC_HINT, self.name)

        file_dir_path = Path(file_dir)
        extract_dir = file_dir_path / "images_extract"
        md_path = file_dir_path / f"{docx_path.stem}.md"

        # 1. pandoc 转换（cwd 限定在 file_dir，图片链接是相对路径，方便重写）
        cmd = [
            pandoc, str(docx_path),
            "-t", "gfm",
            "--extract-media", "images_extract",
            "-o", md_path.name,
        ]
        try:
            proc = subprocess.run(
                cmd, cwd=str(file_dir_path), capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=300,
            )
        except subprocess.TimeoutExpired as e:
            raise FileProcessingError("pandoc 转换超时（300s）", self.name, e)
        if proc.returncode != 0:
            raise FileProcessingError(f"pandoc 转换失败: {proc.stderr.strip()[:300]}", self.name)

        md_content = md_path.read_text(encoding="utf-8")

        # 2. 把提取出的图片扁平化到 images/ 并重写链接（md_img_node 只扫描平铺目录）
        md_content = self._flatten_images(extract_dir, file_dir_path / IMAGES_SUBDIR, md_content)

        # 3. 清理临时提取目录
        if extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)

        if not md_content.strip():
            raise FileProcessingError("Word 转换结果为空（文档可能没有正文内容）", self.name)

        md_path.write_text(md_content, encoding="utf-8")
        state['md_path'] = str(md_path)
        self.logger.info(f"Word 已转换为 Markdown: {md_path}（{len(md_content)} 字符）")
        return state

    def _flatten_images(self, extract_dir: Path, images_dir: Path, md_content: str) -> str:
        """把 images_extract 下的图片移到 images/，并重写 md 中的链接。"""
        if not extract_dir.exists():
            return md_content

        images_dir.mkdir(parents=True, exist_ok=True)
        image_exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

        # old_rel -> new_rel 映射
        link_map: dict = {}
        for src in sorted(extract_dir.rglob("*")):
            if not src.is_file() or src.suffix.lower() not in image_exts:
                continue
            target = images_dir / src.name
            # 重名消歧
            if target.exists():
                target = images_dir / f"{src.stem}_{uuid.uuid4().hex[:6]}{src.suffix}"
            shutil.move(str(src), str(target))
            old_rel = src.relative_to(extract_dir.parent).as_posix()
            new_rel = target.relative_to(extract_dir.parent).as_posix()
            link_map[old_rel] = new_rel
            # pandoc 可能用反斜杠或仅 media 相对路径引用，统一做宽松替换
            link_map.setdefault(src.name, new_rel)

        for old_rel, new_rel in link_map.items():
            md_content = md_content.replace(old_rel, new_rel)
        return md_content
