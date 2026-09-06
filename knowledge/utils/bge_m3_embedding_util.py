from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from typing import Optional, List
import os
import logging
import threading

from knowledge.core.exceptions import EmbeddingError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

bge_m3_ef: Optional[BGEM3EmbeddingFunction] = None
_bge_m3_lock = threading.Lock()


def get_bge_m3_embedding_model():
    global bge_m3_ef
    if bge_m3_ef is not None:
        return bge_m3_ef
    with _bge_m3_lock:
        return _load_bge_m3()


def _load_bge_m3():
    global bge_m3_ef
    if bge_m3_ef is not None:
        return bge_m3_ef
    # 1.获取参数
    model_name = os.getenv('BGE_M3_PATH', 'BAAI/bge_m3')
    device = os.getenv('BGE_DEVICE', 'cpu')
    # 环境变量是字符串，必须显式解析布尔值，避免 "0"/"false" 被当成真值
    use_fp16 = os.getenv('BGE_FP16', '').strip().lower() in ('1', 'true', 'yes', 'on')
    try:
        # 定义对象
        bge_m3_ef = BGEM3EmbeddingFunction(
            model_name=model_name,
            device=device,
            use_fp16=use_fp16
        )
    except Exception as e:
        logger.error(f"BGE-M3 嵌入模型加载失败: {e}")
        raise EmbeddingError(f"BGE-M3 嵌入模型加载失败: {e}", cause=e)

    return bge_m3_ef


def generate_hybrid_embeddings(embedding_model: BGEM3EmbeddingFunction, embedding_documents: List[str]):
    """
    为文本生成向量嵌入
    :param embedding_model: 嵌入模型(这里使用BGEM3)
    :param embedding_documents: 要生成嵌入的文本列表
    :return: 包含dense和sparse向量的字典
    """
    try:
        # 1. 生成嵌入
        embedding_result = embedding_model.encode_documents(embedding_documents)

        processed_sparse_result = []
        # 2. 遍历每一个文档
        for index in range(len(embedding_documents)):
            # 2.1 解构csr矩阵&获取稀疏向量
            # 注意：不同版本 BGE-M3 返回的 sparse 可能是 coo_array（无 .indptr/.indices），
            # 先统一转成 CSR 再按行索引，避免 'coo_array' object has no attribute 'indptr'
            csr_array = embedding_result['sparse'].tocsr()
            # a) 行索引
            ind_ptr = csr_array.indptr

            # b) 获取行索引的起始值
            start_ind_ptr = ind_ptr[index]
            end_ind_ptr = ind_ptr[index + 1]

            # c) 获取token_id
            token_id = csr_array.indices[start_ind_ptr:end_ind_ptr].tolist()

            # d) 获取权重
            weight = csr_array.data[start_ind_ptr:end_ind_ptr].tolist()

            # 2.2 获取稀疏向量
            sparse_vector = dict(zip(token_id, weight))

            processed_sparse_result.append(sparse_vector)

        # 3. 返回
        return {
            "dense": [den.tolist() for den in embedding_result["dense"]],
            "sparse": processed_sparse_result
        }
    except Exception:
        return None


if __name__ == "__main__":
    embedding_model = get_bge_m3_embedding_model()
    query = '我喜欢Python语言'
    result = embedding_model.encode_queries([query])
    # result = embedding_model.encode_documents([query])

    # 稠密向量
    dense = result['dense'][0].tolist()
    print(len(result['dense'][0].tolist()))

    # 稀疏向量（不同版本 BGE-M3 返回 coo_array / csr_matrix 格式不一，统一转 COO）
    sparse = result['sparse'].tocoo()
    row0 = sparse.row == 0
    tokenids = sparse.col[row0].tolist()
    weights = sparse.data[row0].tolist()
    sparse_vector = dict(zip(tokenids, weights))
