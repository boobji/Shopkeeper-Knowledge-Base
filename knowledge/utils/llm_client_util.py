import json
import os
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 显式指向 knowledge/.env，避免受运行目录(CWD)影响；override=True 覆盖已存在的系统环境变量，
# 防止脚本静默使用错误的系统 OPENAI_API_KEY 导致 401
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_FILE, override=True)
cache_llm_client = {}
def get_llm_client(model_name: str = None, temperature: float = 0.0, response_format: bool = False) -> ChatOpenAI:
    """
    返回 LLM 客户端对象（带缓存）。
    缓存 key 为 (model_name, temperature, response_format)，三者任一不同都对应独立实例。
    注意：未显式传 model_name 时使用 ITEM_MODEL（历史行为，answer_output 等节点依赖此默认）。
    """

    model_name = model_name or os.getenv('ITEM_MODEL',"qwen-flash")
    api_key=os.getenv("OPENAI_API_KEY")
    base_url=os.getenv("OPENAI_API_BASE")

    # 缓存命中直接返回（temperature 必须参与 key，否则不同温度会复用同一实例）
    cache_key =(model_name, temperature, response_format) # 复合缓存key
    if cache_key in cache_llm_client:
        return cache_llm_client[cache_key]

    # 返回内容格式
    model_kwargs = {}
    if response_format:
        model_kwargs['response_format'] = {'type': 'json_object'}

    try:

        # 定义模型实例
        client = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            extra_body={"enable_thinking": False},
            model_kwargs=model_kwargs,
        )

        # 缓存同步数据
        cache_llm_client[cache_key] = client
        return client
    except Exception as e:
        logger.error('LLM的客户端创建失败', {str(e)})

if __name__ == "__main__":
    llm_client = get_llm_client()
    ai_message = llm_client.invoke('你好，请给我讲一个笑话，返回json格式：{\"key\":\"value\"}')
    print(ai_message.content)

    json_object = json.loads(ai_message.content)
    print(json_object)
