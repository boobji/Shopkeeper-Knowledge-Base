"""BGE-Reranker 模型单例（线程安全懒加载）。"""

import logging
import os
import threading

from FlagEmbedding import FlagReranker

logger = logging.getLogger(__name__)

_reranker_model = None
_reranker_lock = threading.Lock()


def get_reranker_model() -> FlagReranker:
    """获取 Reranker 模型实例（单例模式）"""
    global _reranker_model
    if _reranker_model is not None:
        return _reranker_model
    with _reranker_lock:
        if _reranker_model is not None:
            return _reranker_model
        try:
            # 旧名 BGE_RERANKER_LARGE 仅作兼容保留（原名有误导，装的不一定是 large）
            model_path = os.getenv("BGE_RERANKER_PATH") or os.getenv("BGE_RERANKER_LARGE")
            device = os.getenv("BGE_RERANKER_DEVICE", "cpu")
            use_fp16 = os.getenv("BGE_RERANKER_FP16", "False").lower() == "true"

            logger.info(f"正在初始化 Reranker 模型，路径: {model_path}, 设备: {device}, fp16: {use_fp16}")

            _reranker_model = FlagReranker(
                model_name_or_path=model_path,
                device=device,
                use_fp16=use_fp16
            )

            logger.info("Reranker 模型初始化成功！")
        except Exception as e:
            logger.error(f"初始化 Reranker 模型失败: {e}", exc_info=True)
            return None
        return _reranker_model
