from pathlib import Path

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState


class EntryNode(BaseNode):
    """
    实体节点
    第一位
    对上传的文件类型做判断
    """
    name = 'entry'
    def process(self,state: ImportGraphState) -> ImportGraphState:
        """
        处理文件类型检测
        Args:
            state:ImportGraphState

        Returns:ImportGraphState处理之后的节点状态

        """
        # 1.获取导入文件路径
        self.log_step('step1', '[获取文件路径]')
        file_dir = state.get('file_dir')
        import_file_path = state.get('import_file_path')

        # 2.简单校验
        self.log_step('step2','[检测文件路径]')
        if not file_dir or not import_file_path:
            raise ValidationError('文件目录或者文件不存在',self.name)

        # 3.使用path对象操作文件
        path = Path(import_file_path).absolute()

        # 4.获取文件后缀
        suffix = path.suffix.lower()

        # 5.判断文件的后缀（pdf/md 直接入对应通道；html/docx 先经转换节点转成 md）
        if suffix == '.pdf':
            state['is_pdf_read_enabled'] = True
            state['pdf_path'] = import_file_path
        elif suffix in ('.md', '.txt'):
            # txt 无结构但切分器可兜底（按长度切），与 md 同通道
            state['is_md_read_enabled'] = True
            state['md_path'] = import_file_path
        elif suffix in ('.html', '.htm'):
            state['is_html_enabled'] = True
        elif suffix == '.docx':
            state['is_docx_enabled'] = True
        else:
            self.logger.debug(f'文件类型{suffix}不支持')
            raise ValidationError(f'文件类型{suffix}不支持（支持 pdf/md/txt/html/htm/docx）')

        # 6.获取文件的标题名
        file_title = path.stem.split('.')[0]
        state['file_title'] = file_title

        # 7.返回state
        return state
