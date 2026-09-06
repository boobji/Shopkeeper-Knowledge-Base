import time
import subprocess
import shutil
from pathlib import Path
from typing import Tuple
from subprocess import TimeoutExpired
import sys

from knowledge.processor.import_process.base import (BaseNode)
from knowledge.processor.import_process.exceptions import ValidationError,FileProcessingError,PdfConversionError
from knowledge.processor.import_process.state import ImportGraphState

class PdfToMdNode(BaseNode):
    """
        pdf_to_md节点
    """
    name = 'pdf_to_md_node'

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        pdf_to_md
        Args:
            state:

        Returns:

        """
        # 1.对参数校验
        import_file_path, file_dir_path = self._validate_state_inputs_path(state)

        # 2.利用mineru工具解析pdf为md
        processed_code = self._execute_mineru(import_file_path,file_dir_path)
        if processed_code != 0:
            raise PdfConversionError('mineru解析失败',self.name)

        # 3.获取md的path
        md_path = self._get_md_paths(import_file_path,file_dir_path)

        # 4.更新state 字典的md_path
        state['md_path'] = md_path

        # 5.返回state
        return state

    def _validate_state_inputs_path(self, state: ImportGraphState) -> Tuple[Path, Path]:
        """

        Args:
            state: 该节点收到的状态

        Returns:

        """
        self.log_step('step1','对状态的路径输入参数做校验')

        # 1.获取输入pdf文件路径
        import_file_path = state.get('import_file_path','')

        # 2.获取解析后的输出目录
        file_dir = state.get('file_dir','')

        # 3.校验输出的文件路径（非空判断）
        if not import_file_path:
            raise ValidationError('解析的文件不存在',self.name)

        # 4.用path标准化
        import_file_path_opj = Path(import_file_path)

        # 5.校验是一个真实路径
        if not import_file_path_opj.exists():
            raise FileProcessingError('解析的文件路径不存在',self.name)

        # 6.判断输出目录是否为空
        if not file_dir:
            # 默认目录做兜底
            file_dir = import_file_path_opj.parent

        # 7.标准化输出目录
        file_dir_path_obj = Path(file_dir)
        self.logger.info(f'上传文件的路径：{import_file_path}')
        self.logger.info(f'输出的目录：{file_dir}')
        file_dir_path_obj.mkdir(parents=True, exist_ok=True)

        # 8.返回 输出文件及输出目录的标准path
        return import_file_path_opj,file_dir_path_obj

    def _execute_mineru(self,import_file_path,file_dir_path):
        """

        Args:
            import_file_path: 解析的文件路径
            file_dir_path: 解析后的文件目录

        Returns:

        """
        # 执行命令
        self.log_step('step2','执行mineru解析pdf')
        # 自动创建输出目录
        file_dir_path.mkdir(parents=True, exist_ok=True)

        # 1.构建命令行(直接调用mineru命令,mineru没有__main__.py,不能使用python -m mineru)
        mineru_cmd = shutil.which('mineru')
        if not mineru_cmd:
            # 兼容venv场景:尝试在site-packages同级的Scripts目录查找
            scripts_dir = Path(sys.executable).parent
            candidate = scripts_dir / 'mineru.exe'
            if candidate.exists():
                mineru_cmd = str(candidate)
        if not mineru_cmd:
            raise FileProcessingError('未找到mineru命令,请确认已安装mineru',self.name)

        cmd=[
            mineru_cmd,
            '-b','pipeline',
            '-p',str(import_file_path),
            '-o',str(file_dir_path),
            '--source','local'
        ]
        self.logger.debug(f"执行命令：{' '.join(cmd)}")
        process_start_time = time.time()

        # 2.执行命令(子进程执行命令行)自动读取到主进程的环境变量
        proc = subprocess.Popen(args = cmd,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         errors='replace',
                         text=True,
                         encoding='utf-8',
                         bufsize=1
                         )

        # 3.获取日志信息
        if proc.stdout is not None:
            for line in proc.stdout:
                line_strip = line.strip()
                if line_strip:
                    self.logger.info(f"mineru输出：{line_strip}")

        # 4.等待子进程做完
        try:
            processed_code = proc.wait(timeout=300)  # 5分钟超时
        except TimeoutExpired:
            proc.kill()
            raise PdfConversionError("mineru执行超时", self.name)
        process_end_time = time.time()
        if processed_code == 0:
            self.logger.info(f'mineru成功解析pdf：{import_file_path.name}耗时{process_end_time - process_start_time:.3f}')
        else:
            self.logger.info(f'mineru解析pdf失败：{import_file_path.name}，退出码：{processed_code}')

        # 5.返回状态码
        return processed_code

    def _get_md_paths(self,import_file_path: Path,file_dir_path: Path):
        # file_dir_path:D:\Develop\Shopkeeper_Knowledge_Base\knowledge\processor\import_process\import_temp_dir
        file_name = import_file_path.stem
        md_path = file_dir_path / file_name / 'auto' / f'{file_name}.md'
        return str(md_path)
