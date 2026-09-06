"""全局唯一的 .env 加载入口。

所有进程/脚本通过本模块一次性加载 `knowledge/.env`，
业务模块一律不得再自行调用 `load_dotenv`（历史上散落在 9+ 个文件、
策略互不相同，导致"谁先 import 谁生效"的隐式配置问题）。

加载时机：`knowledge/__init__.py` 在包导入时调用 `load_env()`，
因此任何 `import knowledge.xxx` 都保证环境变量已就绪。
"""

from pathlib import Path

from dotenv import load_dotenv

# knowledge/.env（core/ 的上一级）
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_env(override: bool = True) -> None:
    """加载 knowledge/.env。

    override=True：.env 优先于已存在的系统环境变量，防止机器上遗留的
    同名系统变量（如 OPENAI_API_KEY）被静默使用导致 401 一类的问题。
    """
    load_dotenv(dotenv_path=ENV_FILE, override=override)
