"""

导入流程状态类型定义



定义完整的状态结构和辅助函数

"""

from typing import TypedDict, List, Dict, Tuple

import copy


class ImportGraphState(TypedDict, total=False):


    """

    导入流程图状态



    包含整个导入流程中传递的所有数据

    """

    # ==================== 任务标识 ====================

    task_id: str  # 任务 ID，用于任务追踪

    # ==================== 控制标志 ====================

    is_md_read_enabled: bool  # 是否启用 MD 读取

    is_pdf_read_enabled: bool  # 是否启用 PDF 读取

    is_html_enabled: bool  # 是否启用 HTML 转换

    is_docx_enabled: bool  # 是否启用 Word(docx) 转换

    # ==================== 路径信息 ====================

    import_file_path: str  # 导入文件路径

    file_dir: str  # 导入(出)文件目录

    pdf_path: str  # PDF 文件路径

    md_path: str  # 转换后Markdown 文件路径

    # ==================== 文件信息 ====================

    file_title: str  # 文件标题（不含扩展名）

    item_name: str  # 识别出的商品/产品名称

    # ==================== 处理中间数据 ====================

    md_content: str  # Markdown 文档内容

    chunks: List  # 文档切片列表

    structure_stat: Dict  # 结构增强统计（P0-1：标题还原情况）

    chunk_clean_stats: Dict  # 切片清洗统计（P0-2/P0-3：各类噪音剔除数量）

    images_context: List[Tuple[str, str, Tuple[str, str, str]]]  # 图片上下文列表，元素为 (img_name, img_path, (head_title, pre_context, post_context))

    images_summaries: Dict[str, str]  # 图片摘要映射，键为 img_name，值为 VLM 生成的中文标题

    new_md_content: List[str]

    # ==================== 默认状态 ====================


GRAPH_DEFAULT_STATE: ImportGraphState = {

    "task_id": "",

    "is_pdf_read_enabled": False,

    "is_md_read_enabled": False,

    "is_html_enabled": False,

    "is_docx_enabled": False,

    "file_dir": "",

    "import_file_path": "",

    "pdf_path": "",

    "md_path": "",

    "file_title": "",

    "md_content": "",

    "chunks": [],

    "structure_stat": {},

    "chunk_clean_stats": {},

    "images_context": [],

    "images_summaries": {},

    'new_md_content': [],

    "item_name": "",

}


def create_default_state(**overrides) -> ImportGraphState:
    """
    创建默认状态，支持覆盖

    Args:
        **overrides: 要覆盖的字段

    Returns:
        新的状态实例

    Examples:
        >>> state = create_default_state(task_id="task_001", local_file_path="doc.pdf")
    """
    state = copy.deepcopy(GRAPH_DEFAULT_STATE)
    state.update(overrides)
    return state


def get_default_state() -> ImportGraphState:
    """
    获取默认状态副本

    Returns:
        状态副本（避免全局污染）
    """
    return copy.deepcopy(GRAPH_DEFAULT_STATE)
