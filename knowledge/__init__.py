"""knowledge 包初始化：在这里完成 .env 的唯一一次加载。

任何 `import knowledge.xxx` 都会先触发本文件，
保证环境变量在所有业务模块读取之前就绪。
"""

from knowledge.core.config import load_env

load_env()
