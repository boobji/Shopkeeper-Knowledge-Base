import os
import re
import time
from pathlib import Path
from typing import Tuple, List, Deque
import base64

import logging

from openai import OpenAI

from knowledge.utils.minio_util import get_minio_client
from knowledge.processor.import_process.base import (BaseNode,setup_logging)
from knowledge.processor.import_process.exceptions import ValidationError, FileProcessingError, \
    ImageProcessingError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.config import get_config

class MarkDownImgNode(BaseNode):
    """
    处理md图片节点
    """
    name = 'md_img_node'

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """

        Args:
            state:上一个节点处理后的状态

        Returns:当前节点处理之后的状态（写回 md_content / images_context / images_summaries）

        """
        # 0.获取配置对象
        config = get_config()

        # 1. 处理文件路径
        md_content,md_path_obj,image_dir = self._get_img_md_content(state)
        if not image_dir.exists():
            self.logger.warning(f'文件{md_path_obj.name}暂无图片要处理')
            state['md_content'] = md_content
            state['images_context'] = []
            state['images_summaries'] = {}
            return state

        # 2. 扫描并处理图片（最复杂）
        target_images_context = self._scan_images_and_context(image_dir, md_content, config)

        # 3. 用VLM为图片生成摘要
        images_summaries = self._extract_img_summry(md_path_obj.stem, target_images_context, config)

        # 4. 图片上传minio（生成minio地址）（复合函数） 回写（图片描述和图片地址写回到md_content）
        new_md_content = self._upload_img_and_update_new_md(md_path_obj.stem, md_content, images_summaries, target_images_context, config)

        # 5.备份
        self._backup_new_md_file(original_md_path_str=md_path_obj, new_md_content=new_md_content)

        # 6.写回 state 并返回（遵循 BaseNode 节点契约：返回更新后的 state 字典）
        state['md_content'] = md_content
        state['images_context'] = target_images_context
        state['images_summaries'] = images_summaries
        state['new_md_content'] = new_md_content
        return state

    def _get_img_md_content(self, state: ImportGraphState) -> Tuple[str, Path, Path]:
        """

        Args:
            state:

        Returns:
            md_content:
            md_path_obj:
            image_dir:

        """
        self.log_step('step1','读取md内容及构建图片目录')
        # 1.获取md_path
        md_path = state.get('md_path', '')

        # 2.判断是否有内容
        if not md_path:
            raise ValidationError('md文件不存在',self.name)

        # 3.标准化处理
        md_path_obj = Path(md_path)

        # 4.判断路径是否有效
        if not md_path_obj.exists():
            raise FileProcessingError('md文件路径无效',self.name)

        # 5.读取文件内容
        with open(md_path_obj,'r',encoding='utf-8') as f:
            md_content = f.read() #全部读取

        # 6.构建图片目录
        image_dir = md_path_obj.parent / 'images'

        # 7.返回
        return md_content,md_path_obj,image_dir

    def _scan_images_and_context(self, image_dir: Path, md_content: str, config ) -> List[Tuple[str, str, Tuple[str, str, str]]]:
        """
        扫描处理图片
        返回所有有效图片的丰富信息（image_name, image_path, 图片的上下文（方便vlm理解））
        找上下文的策略通过max_char_number(max_total)
        最终获取上下文的策略是:
        1.先找到当前图片的最近一个标题(# ## ### #### ...)(定位到标题的位置以及标题的内容)
        2.从图片内容的上一行开始向上找一直找到最近的这个标题下一行
        3.根据开始索引和结束索引定位到两个索引间的内容
        4.利用段落和最大字符数选择从这个区域中最终留下多少
        Args:
            image_dir:
            md_content:
            config
        Returns:

        """
        self.log_step('step2',f'扫描图片目录{image_dir}')

        # 1.遍历图片文件目录
        target_images_context = []
        for img_name in os.listdir(image_dir):
            # 1.0获取文件后缀
            file_ext = os.path.splitext(img_name)[1]

            # 1.1如果文件后缀不是有效的
            if file_ext not in config.image_extensions:
                continue # 继续处理下一个图片文件

            # 1.2构建image_path转字符串
            img_path = str(image_dir / img_name)

            # 1.3构建图片上下文
            img_context = self._find_img_context_with_limit(md_content, img_name, config.img_content_length)
            if not img_context:
                self.logger.warning('md文件中暂未提取到可用图片')
                continue # 继续处理下一张

            # 1.4提取到当前图片上下文（同图多引用时，只取第一个）
            primary_img_context = img_context[0]

            # 1.5存储到列表中
            target_images_context.append((img_name, img_path, primary_img_context))

        self.logger.info(f'找到{len(target_images_context)}有效图片')
        return target_images_context

    def _find_img_context_with_limit(self, md_content: str, img_name: str, max_chars = 200) -> List[Tuple[str, str, str]]:
        """
        从md文档中提取上下文
        思路：使用正则查找图片在md的位置
        Args:
            md_content:
            img_name:
            max_chars:

        Returns:

        """
        # 1.定义正则的规则（从md找到图片）标准的图片在md中的语法结构：![图片描述](images/aaa.jpg'aaa')
        # 1.1第一部分
        # r:python不要在对正则中的字符做转义了
        # !md语法
        # [(需要正则转义):在正则中[代表的字符集(a-z A-Z 0-9+ /
        # . 任意字符
        # *(+)任意字符出现的数量0个或者多个(+)至上要有一个
        # ?(非贪婪模式:佛系)不加?就是贪婪模式(加班 积卵上)
        # ]
        # ()(需要正则转义):在正则中()代表捕获组
        # img_name=text.jpg :金标准:.后置的名字一定做escape处理
        re_pattern = re.compile( r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)")

        # 2.从md中定位图片位置（按行切分md内容，遍历每一行 看是否满足图片的正则规则）
        md_lines = md_content.split('\n')
        imgs_context = []
        for line_idx, line in enumerate(md_lines):
            # a.没有找到继续下一行
            if not re_pattern.search(line):
                continue# 继续下一行
            # b.找到了（索引）a.1定位这种图片最近的标题a.2提取上文内容a.3提取下文内容

            # 找图片的上文和索引
            # b.1（if re.match(r"^#{1,6}\s+", lines[i]):）
            head_title = '' # 初始标题
            head_index = -1 # 初始标题所以
            for i in range(line_idx-1, -1, -1):
                if re.match(r"^#{1,6}\s+", md_lines[i]):
                    head_title = md_lines[i]
                    head_index = i
                    break
            # b.2定义要截取上下文的索引
            pre_content_start_index = head_index + 1
            pre_content = md_lines[pre_content_start_index:line_idx]
            img_pre_context = self._extract_img_context_with_limit(pre_content, max_chars, direction = 'front')

            # 找图片的下文和索引
            section_index = len(md_lines)
            for i in range(line_idx + 1, section_index):
                if re.match(r"^#{1,6}\s+", md_lines[i]):
                    section_index = i
                    break
            # b.2定义要截取上下文的索引
            post_content_start_index = line_idx + 1
            post_content = md_lines[post_content_start_index:section_index]
            img_post_context = self._extract_img_context_with_limit(post_content, max_chars, direction='end')

            # 构建图片上下文信息
            imgs_context.append((head_title, img_pre_context, img_post_context)) # 实际上只有一个三元组对象，除非该图片在md中有多次引用

        return imgs_context

    def _extract_img_context_with_limit(self, extract_content: list, max_chars: int, direction: str) -> str:
        """
        提取图片到上下标题之间的内容
        如何从给定的内容中找段落
        策略:md中段落/n分割 行与行之间，行后都有两个空格
        Args:
            extract_content:
            max_chars:
            direction:

        Returns:

        """
        # 1.切分内容的段落
        current_paragraph = [] # 存储当前遍历到的内容
        final_paragraph = [] # 存储最终遍历到的段落（多个段落）

        # 2.遍历每一行
        for line in extract_content:
            clen_strip = line.strip()
            if not clen_strip: # 自然的段落分割
                if current_paragraph:
                    final_paragraph.append('\n'.join(current_paragraph))
                    current_paragraph = []
            else:
                if re.match(r"^!\[.*?\]\(.*?\)$", clen_strip): #遇到图片跳过
                    if current_paragraph:
                        final_paragraph.append('\n'.join(current_paragraph))
                        current_paragraph = []
                    continue
                current_paragraph.append(line)
        # 2.5处理最后一个段落
        if current_paragraph:
            final_paragraph.append('\n'.join(current_paragraph))

        # 2.6处理上文
        if direction == 'front':
            final_paragraph.reverse()

        # 3.判断长度
        total = 0
        selected = []
        for para in final_paragraph:
            para_len = len(para)
            if total + para_len > max_chars and selected:
                break
            selected.append(para)
            total += para_len

        # 4.返回selected
        # 4.1处理上文
        if direction == 'front':
            selected.reverse()
        return '\n\n'.join(selected)

    def _extract_img_summry(self, document_title: str, target_images_context: List[Tuple[str, str, Tuple[str, str, str]]], config) :
        """
        所有图片生成图片摘要VLM
        Args:
            document_title:
            target_images_context:
            config:

        Returns:

        """
        self.log_step('step3','提取图片摘要')
        summaries = {}
        request_timestamps: Deque[float] = Deque()
        # 1.构建客户端
        # 1.0 配置缺失守卫：避免 api_key/base/model 为空时静默失败、难以排查
        if not config.openai_api_key or not config.openai_api_base or not config.vl_model:
            logging.error(
                'VLM 配置缺失，请检查 .env 中的 OPENAI_API_KEY / OPENAI_API_BASE / VL_MODEL '
                f'(当前 key={"已填" if config.openai_api_key else "空"}, '
                f'base={"已填" if config.openai_api_base else "空"}, '
                f'model={"已填" if config.vl_model else "空"})'
            )
            return summaries
        try:
            client = OpenAI(
                api_key=config.openai_api_key,
                base_url=config.openai_api_base,
            )
        except Exception as e:
            logging.error(f'vlm客户端创建失败: {e}')
            return summaries
        # 2.发送请求
        for img_name, img_path, images_context in target_images_context:
            self._enforce_rate_limit(request_timestamps, config.requests_per_minute, 10)
            summary = self._get_img_summary(config, client, document_title, img_path, images_context)
            summaries[img_name] = summary

        # 3.通过映射表将每张表的摘要存储起来
        logging.info(f'生成{len(summaries)}图片摘要')
        return summaries

    def _enforce_rate_limit(
            self,
            request_timestamps: Deque[float],
            max_requests: int,
            window_seconds: int = 60
    ):
        """
        强制执行 API 请求速率限制。

        Args:
            request_timestamps (Deque[float]): 请求时间戳队列。
            max_requests (int): 窗口内最大请求数。
            window_seconds (int, optional): 时间窗口大小（秒）。
        """
        current_time = time.time()

        # 移除窗口外的时间戳
        while request_timestamps and current_time - request_timestamps[0] >= window_seconds:
            request_timestamps.popleft()

        # 达到上限则等待
        if len(request_timestamps) >= max_requests:
            sleep_duration = window_seconds - (current_time - request_timestamps[0])
            if sleep_duration > 0:
                self.logger.info(f"达到速率限制，暂停 {sleep_duration:.2f} 秒...")
                time.sleep(sleep_duration)

            current_time = time.time()
            while request_timestamps and current_time - request_timestamps[0] >= window_seconds:
                request_timestamps.popleft()

        request_timestamps.append(current_time)

    def _get_img_summary(self, config, client, document_title: str, img_path: str, images_context: Tuple[str, str, str]):
        # 1.解包构建上下文
        section_title, pre_content, post_content = images_context

        # 2.判断上下文
        context_parts = []
        if section_title:
            context_parts.append(section_title)
        if pre_content:
            context_parts.append(pre_content)
        if post_content:
            context_parts.append(post_content)
        final_context = '\n'.join(context_parts) if context_parts else '暂无可用上下文'

        # 3.读取图片文件
        local_img_content = ''
        try:
            with open(img_path, 'rb') as f:
                local_img_content = base64.b64encode(f.read()).decode('utf-8') # 将图片转为字符串
        except Exception as e:
            return '暂无图片'

        # 4.发送请求
        try:
            response = client.chat.completions.create(
                model=config.vl_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""任务：为Markdown文档中的图片生成一个简短的中文标题。
                            背景信息：
                                1. 所属文档标题："{document_title}"
                                2. 图片上下文：{final_context}
                                请结合图片视觉内容和上述上下文信息，用中文简要总结这张图片的内容，
                                生成一个精准的中文标题（不要包含"图片"二字）。""",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{local_img_content}"
                                }
                            }
                        ]
                    }
                ]
            )
            summary = response.choices[0].message.content.strip().replace("\n", " ")
            return summary
        except Exception as e:
            self.logger.warning(f"图片摘要生成失败 {img_path}: {e}")
            return "图片描述"

    def _upload_img_and_update_new_md(self, document_name, md_content, images_summaries, target_images_context, config):
        """
        上传图片替换md文件中的图片url和摘要
        Args:
            md_content: 
            images_summaries: 
            target_images_context: 

        Returns:

        """
        self.log_step('step5',"上传图片更新摘要")
        remote_urls = {}
        # 1.创建客户端
        minio_client = get_minio_client()
        if minio_client is None:
            self.logger.warning(f'无法将本地文件上传')

        # 2.遍历图片信息列表
        for img_name, img_path, _ in target_images_context:
            # 构建对象名
            object_name = f'{document_name}/{img_name}'
            # 上传
            try:
                minio_client.fput_object(
                    config.minio_bucket, object_name, img_path,
                )
                # 手动拼接远程地址
                remote_url = config.get_minio_base_url() + '/' + config.minio_bucket + '/' + object_name
                self.logger.info(f'{img_name} 上传成功')
                remote_urls[img_name] = remote_url
            except Exception as e:
                self.logger.warning(f'{img_name}上传失败')
                remote_urls[img_name] = 'http://minio_mock/' + document_name  + '/' + img_name
        self.logger.info(f'成功上传{len(remote_urls)}张图片')

        # 替换md中的内容
        new_md_content = md_content
        for img_name, images_summary in images_summaries.items():
            # 提取远程地址
            remote_url = remote_urls.get(img_name)
            if not remote_url:
                continue
            # 替换URL和摘要
            replace_pattern = re.compile(
                r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)",
                re.IGNORECASE,
            )
            new_md_content = replace_pattern.sub(f'![{images_summary}]({remote_url})',new_md_content)
        return new_md_content

    def _backup_new_md_file(
            self,
            original_md_path_str: Path,
            new_md_content: str
    ) -> str:
        """
        将处理后的 Markdown 内容写入新文件。

        Args:
            original_md_path_str (str): 原始文件路径。
            new_md_content (str): 新的 Markdown 内容。

        Returns:
            str: 新文件的绝对路径。
        """
        self.log_step("step_5", "备份新文件")

        original_path = Path(original_md_path_str)
        new_file_path = original_path.with_name(
            f"{original_path.stem}_new{original_path.suffix}"
        )

        try:
            with open(new_file_path, "w", encoding="utf-8") as f:
                f.write(new_md_content)
            self.logger.info(f"处理后的文件已备份至: {new_file_path}")
        except IOError as e:
            self.logger.error(f"写入新文件失败 {new_file_path}: {e}")
            raise ImageProcessingError(f"文件写入失败: {e}", node_name=self.name)

        return str(new_file_path)

if __name__ == "__main__":
    setup_logging()
    img_md_node = MarkDownImgNode()

    state = {
        'md_path':r'D:\Develop\Shopkeeper_Knowledge_Base\knowledge\processor\import_process\import_temp_dir\万用表RS-12的使用\auto\万用表RS-12的使用.md'
    }

    img_md_node.process(state)