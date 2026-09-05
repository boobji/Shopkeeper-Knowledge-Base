from knowledge.processor.import_process.config import get_config
c = get_config()
print(c.openai_api_key[:6] + "...", c.openai_api_base, c.vl_model)
