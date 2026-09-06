"""统一日志配置（唯一入口）。"""

import logging


def setup_logging(level: int = logging.INFO) -> None:
    """配置全局日志格式，两个服务入口共用。"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
