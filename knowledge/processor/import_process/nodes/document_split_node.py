import os
import re
import json
from typing import Tuple, List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge.processor.import_process.base import (BaseNode, setup_logging)
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.config import get_config
from knowledge.utils.markdown_utils import MarkdownTableLinearizer


class DocumentSplitNode(BaseNode):
    name = "document_split_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 0.获取参数
        md_content, file_title, max_content_length, min_content_length = self._get_inputs(state)

        # 1.根据标题切割
        sections = self._split_by_headings(md_content, file_title)

        # 2.处理(多切少合)
        final_chunks = self._split_and_merge(sections, max_content_length, min_content_length)

        # 3.组装
        chunks = self._assemble_chunk(final_chunks)

        # 4.更新state（遵循 BaseNode 节点契约：返回更新后的 state 字典）
        state['chunks'] = chunks

        # 5.日志统计
        self._log_summary(md_content, chunks, max_content_length)

        # 6.备份
        self._backup_chunks(state, chunks)
        return state

    def _get_inputs(self, state: ImportGraphState) -> Tuple[str, str, int, int]:
        self.log_step('step1', '切分文档的参数校验及获取')
        config = get_config()
        # 1.获取md
        md_content = state.get('md_content')

        # 2.统一换行符
        if md_content:
            md_content = md_content.replace('\r\n', '\n').replace('\r', '\n')

        # 3.获取文件标题
        file_title = state.get('file_title')

        # 4.校验最小值
        if config.max_content_length <= 0 or config.min_content_length <= 0 or config.max_content_length <= config.min_content_length:
            raise ValidationError(f'切片长度校验失败')

        return md_content, file_title, config.max_content_length, config.min_content_length

    def _split_by_headings(self, md_content: str, file_title: str) -> List[dict]:
        """
        根据标题进行切分
        Args:
            md_content:
            file_title:

        Returns:

        """
        self.log_step('step2', '根据标题进行切割')
        # 0.定义变量
        in_fence = False
        body_lines = []
        sections = []
        current_level = 0
        current_title = ''
        hierarchy = [''] * 7

        # 1.定义正则表达式
        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")

        # 2.切分
        content_lines = md_content.split('\n')

        def _flush():
            """
            封装section对象
            Returns:

            """
            body = '\n'.join(body_lines)
            if current_title or body:
                parent_title = ''
                for i in range(current_level - 1, 0, -1):
                    if hierarchy[i]:
                        parent_title = hierarchy[i]
                        break
                if not parent_title:
                    parent_title = current_title if current_title else file_title
                sections.append({
                    'title': current_title if current_title else file_title,
                    'body': body,
                    'file_title': file_title,
                    'parent_title': parent_title,
                })

        for content_line in content_lines:
            # 判断是否存在代码块围栏
            if content_line.strip().startswith("```") or content_line.strip().startswith("~~~"):
                in_fence = not in_fence

            match = heading_re.match(content_line) if not in_fence else None
            if match:
                # 当前判断是标题
                _flush()
                level = len(match.group(1))
                current_level = level
                current_title = content_line
                hierarchy[level] = current_title
                # 存储当前遍历的标题
                for i in range(level + 1, 7):
                    hierarchy[i] = ''
                body_lines = []
            else:
                # 非标题内容逐行收集
                body_lines.append(content_line)
        _flush()
        return sections

    def _split_and_merge(self, sections: List[Dict[str, Any]], max_content_length: int, min_content_length: int):
        """

        Args:
            sections:
            max_content_length:
            min_content_length:

        Returns:

        """
        self.log_step('step3', '多切少合')
        # 1.切
        current_sections = []
        for section in sections:
            current_sections.extend(self._split_long_section(section, max_content_length))

        # 2.合
        final_sections = self._merge_short_section(current_sections, min_content_length)

        # 3.返回
        return final_sections

    def _split_long_section(self, section: Dict[str, Any], max_content_length: int):
        """
        满足条件才切
        Args:
            section:
            max_content_length:

        Returns:

        """
        self.log_step('step3.1', '多切')
        # 1.获取对象属性
        title = section.get('title')
        body = section.get('body')
        file_title = section.get('file_title')
        parent_title = section.get('parent_title')

        # 1.3判断表格
        if '<table>' in body:
            self.logger.info('检测到了表格数据。。正在处理')
            body = MarkdownTableLinearizer.process(body)

        # 1.5标题校验
        TITLE_MAX_LENGTH = 50
        if len(title) > TITLE_MAX_LENGTH:
            self.logger.warning(f'文件{file_title}对应的{title}过长')
            title = title[:TITLE_MAX_LENGTH]

        # 2.拼接title前缀
        title_prefix = f'{title}\n\n'

        # 3.计算总长
        total_length = len(title_prefix) + len(body)

        # 4.判断
        if total_length <= max_content_length:
            return [section]

        # 4.5计算body可用长度
        body_length = max_content_length - len(title_prefix)
        if body_length <= 0:
            return [section]

        # 5.切（定义langchain递归切分器'\n\n','\n','。','！'... 2.切分)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=body_length,
            chunk_overlap=0,
            # 优雅降级策略：优先按双换行切，再按单换行，最后按标点和空格
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
            keep_separator=False,
        )

        texts = text_splitter.split_text(body)

        if len(texts) <= 1:
            return [section]
        sub_section = []
        for index, text in enumerate(texts):
            sub_section.append({
                'title': title + '-' + f'{index + 1}',
                'body': text,
                'file_title': file_title,
                'parent_title': parent_title,
                'part': f'{index + 1}'
            })
        return sub_section

    def _merge_short_section(self, current_sections: List[Dict[str, Any]], min_content_length: int):
        """
        满足才合：将过短的相邻切片向前合并，避免产生碎片
        Args:
            current_sections:
            min_content_length:

        Returns:
            List[Dict]: 合并后的切片列表

        """
        self.log_step('step3.2', '少合')
        if not current_sections:
            return []

        final_sections = []
        current_section = current_sections[0]

        for next_section in current_sections[1:]:
            same_parent = (current_section["parent_title"] == next_section["parent_title"])
            if same_parent and len(current_section.get('body')) < min_content_length:
                # body的合并(更新当前的section的body)
                current_section['body'] = (
                        current_section.get('body').rstrip() + '\n\n' + next_section.get('body').lstrip()
                )
                # 更新current_title
                current_section['title'] = current_section['parent_title']
                current_section['part'] = 0
            else:
                # 1.将原来current_section进行封箱
                final_sections.append(current_section)
                # 更新next_section
                current_section = next_section
        final_sections.append(current_section)
        # 对所有section的part做处理(为每一个父标题设置对应的part计数器)
        part_counter = {}
        result = []
        for final_section in final_sections:
            if 'part' in final_section:
                # 获取同源
                parent_title = final_section.get('parent_title')
                part_counter[parent_title] = part_counter.get(parent_title, 0) + 1
                new_part = part_counter.get(parent_title)
                final_section['part'] = new_part
                final_section['title'] = final_section['parent_title'] + '-' + f'{new_part}'
            result.append(final_section)
        return result

    def _assemble_chunk(self, final_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        最终组合
        Args:
            final_chunks:

        Returns:

        """
        self.log_step("step_4", "chunk合并")
        chunks = []
        for chunk in final_chunks:
            # 1.获取chunk的信息
            title = chunk.get('title')
            file_title = chunk.get('file_title')
            parent_title = chunk.get('parent_title')
            body = chunk.get('body')
            content = f'{title}\n\n{body}'

            # 2.构建最终chunk对象
            assemble_chunk = {
                'title': title,
                'file_title': file_title,
                'parent_title': parent_title,
                'content': content,
            }

            # 2.判断part
            if 'part' in chunk:
                assemble_chunk['part'] = chunk.get('part')

            chunks.append(assemble_chunk)

        return chunks

    def _log_summary(self, raw_content: str, chunks: List[dict], max_length: int):
        self.log_step("step5", "输出统计")

        lines_count = raw_content.count("\n") + 1
        self.logger.info(f"原文档行数: {lines_count}")
        self.logger.info(f"最终切分章节数: {len(chunks)}")
        self.logger.info(f"最大切片长度: {max_length}")

        if chunks:
            self.logger.info("章节预览:")
            for i, sec in enumerate(chunks[:5]):
                title = sec.get("title", "")[:30]
                self.logger.info(f"  {i + 1}. {title}...")
            if len(chunks) > 5:
                self.logger.info(f"  ... 还有 {len(chunks) - 5} 个章节")

    def _backup_chunks(self, state: ImportGraphState, chunks: List[dict]):
        self.log_step("step6", "备份切片")

        # 优先使用 file_dir，兼容 local_dir (避免因为字段命名引发写入失败)
        local_dir = state.get("file_dir", state.get("local_dir", ""))
        if not local_dir:
            self.logger.debug("未设置 file_dir/local_dir，跳过备份")
            return

        try:
            os.makedirs(local_dir, exist_ok=True)
            output_path = os.path.join(local_dir, "chunks.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            self.logger.info(f"已备份到: {output_path}")
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")


if __name__ == '__main__':
    setup_logging()
    document_node = DocumentSpliterNode()
    file_path = r'D:\Develop\Shopkeeper_Knowledge_Base\knowledge\processor\import_process\import_temp_dir\万用表RS-12的使用\auto\万用表RS-12的使用_new.md'
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    state = {
        "file_title": "万用表的使用",
        "md_content": content,
        # 指向你想输出 JSON 的备份目录
        "file_dir": r'D:\Develop\Shopkeeper_Knowledge_Base\knowledge\processor\import_process\import_temp_dir\万用表RS-12的使用\auto'
    }
    document_node.process(state)
