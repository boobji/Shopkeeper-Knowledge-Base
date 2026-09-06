"""FastAPI 应用工厂：导入/查询两个服务共用的创建逻辑。"""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from knowledge.core.paths import get_front_page_dir

logger = logging.getLogger(__name__)

# 启动即需要的关键配置（缺失时只告警不阻断，保持与历史懒加载行为一致）
_REQUIRED_ENV_KEYS = [
    "OPENAI_API_BASE",
    "OPENAI_API_KEY",
    "MODEL",
    "ITEM_MODEL",
]


def _check_env_on_startup() -> None:
    missing = [key for key in _REQUIRED_ENV_KEYS if not os.getenv(key)]
    if missing:
        logger.warning(f"以下环境变量未配置（检查 knowledge/.env）: {missing}")


def create_app(title: str, description: str, register_routes) -> FastAPI:
    """创建 FastAPI 实例：CORS、静态资源挂载、路由注册、启动自检。

    Args:
        title: 服务标题。
        description: 服务描述。
        register_routes: 路由注册函数，签名 register_routes(app) -> None。
    """

    async def lifespan(app: FastAPI):
        # 启动时做轻量自检（不主动建立重型连接，客户端仍为懒加载）
        _check_env_on_startup()
        yield

    app = FastAPI(title=title, description=description, lifespan=lifespan)

    # 跨域配置（浏览器前端页面使用）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 前端静态资源目录
    front_page_dir = get_front_page_dir()
    if front_page_dir and os.path.exists(front_page_dir):
        app.mount("/front", StaticFiles(directory=front_page_dir))

    # 注册路由
    register_routes(app)

    return app
