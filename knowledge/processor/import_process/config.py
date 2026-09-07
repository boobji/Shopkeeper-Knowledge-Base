"""
导入流程配置管理模块

集中管理所有配置项，支持环境变量覆盖
"""

from dataclasses import dataclass, field
from typing import Set, Optional
import os

# 环境变量由 knowledge/__init__.py → core.config.load_env() 统一加载，此处直接读取


@dataclass
class ImportConfig:
    """导入流程配置"""

    # ==================== 文档处理配置 ====================
    max_content_length: int = 2000  # 切片最大长度
    img_content_length: int = 200
    min_content_length: int = 500  # 合并短内容的最小长度
    overlap_sentences: int = 1  # 句子级切分时的重叠句数
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200"))
    )  # 长章节二次切分的相邻重叠字符数（0=关闭；建议为 max_content_length 的 10% 左右）

    # ==================== Parent-Child 索引 ====================
    parent_child_enabled: bool = field(
        default_factory=lambda: os.getenv("PARENT_CHILD_ENABLED", "1").strip().lower() in ("1", "true", "yes", "on")
    )  # 检索子块（小块高精度）、返回父块（大块足上下文）
    child_chunks_collection: str = field(
        default_factory=lambda: os.getenv("CHILD_CHUNKS_COLLECTION", "kb_child_chunks")
    )
    child_chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHILD_CHUNK_SIZE", "450"))
    )
    child_chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHILD_CHUNK_OVERLAP", "80"))
    )

    # ==================== Contextual Retrieval ====================
    contextual_retrieval_enabled: bool = field(
        default_factory=lambda: os.getenv("CONTEXTUAL_RETRIEVAL", "1").strip().lower() in ("1", "true", "yes", "on")
    )  # 入库前用 LLM 为每个 chunk 生成上下文前缀并参与向量化（Anthropic contextual retrieval）
    contextual_max_workers: int = field(
        default_factory=lambda: int(os.getenv("CONTEXTUAL_MAX_WORKERS", "4"))
    )  # 上下文生成的并发数
    contextual_timeout: float = field(
        default_factory=lambda: float(os.getenv("CONTEXTUAL_TIMEOUT", "30"))
    )  # 单次上下文生成的 LLM 超时（秒）
    item_name_chunk_k: int = 3  # 商品名识别时使用的切片数量
    item_name_chunk_size: int = 2500  # 商品名识别时使用的切片内容的长度

    # ==================== 结构感知分块（P0-1） ====================
    # 背景：79% 的真实说明书 PDF 无文本层，靠 OCR 还原时 Markdown 的 # 标题
    # 大量丢失，切块退化成按字数硬切。此开关启用后，标题稀疏时会改用 MinerU
    # 中间产物 *_content_list.json（含 type / bbox / page_idx）还原章节层级。
    structure_aware_enabled: bool = field(
        default_factory=lambda: os.getenv("STRUCTURE_AWARE_ENABLED", "1").strip().lower()
        in ("1", "true", "yes", "on")
    )
    structure_hard_split_threshold: float = field(
        default_factory=lambda: float(os.getenv("STRUCTURE_HARD_SPLIT_THRESHOLD", "0.30"))
    )  # 硬切率超过该值判定为"结构缺失"，启用 content_list 补结构

    # ==================== 切片清洗（P0-2 / P0-3） ====================
    chunk_clean_enabled: bool = field(
        default_factory=lambda: os.getenv("CHUNK_CLEAN_ENABLED", "1").strip().lower()
        in ("1", "true", "yes", "on")
    )
    # 中文占非空字符的比例低于此值 → 标 lang=mixed 降权（规格表符号多时会误伤，故可配）
    chunk_min_zh_ratio: float = field(
        default_factory=lambda: float(os.getenv("CHUNK_MIN_ZH_RATIO", "0.20"))
    )
    # 中文占比低于此值 → 判定外文页/纯符号页，直接丢弃
    chunk_drop_zh_ratio: float = field(
        default_factory=lambda: float(os.getenv("CHUNK_DROP_ZH_RATIO", "0.05"))
    )
    chunk_drop_toc: bool = field(
        default_factory=lambda: os.getenv("CHUNK_DROP_TOC", "1").strip().lower()
        in ("1", "true", "yes", "on")
    )  # 丢弃目录/索引页（连续点线引导符）
    chunk_min_keep_length: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_MIN_KEEP_LENGTH", "20"))
    )  # 清洗后低于该长度的片段直接丢弃
    # 通用安全声明等样板段：跨文档重复出现，会稀释检索，需去重降权
    chunk_boilerplate_enabled: bool = field(
        default_factory=lambda: os.getenv("CHUNK_BOILERPLATE_ENABLED", "1").strip().lower()
        in ("1", "true", "yes", "on")
    )
    chunk_boilerplate_threshold: float = field(
        default_factory=lambda: float(os.getenv("CHUNK_BOILERPLATE_THRESHOLD", "0.60"))
    )  # n-gram Jaccard 相似度阈值，超过即判定为同一段样板文字
    chunk_boilerplate_min_docs: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_BOILERPLATE_MIN_DOCS", "2"))
    )  # 至少在多少个不同文档中出现过才认定为通用样板段
    boilerplate_store_path: str = field(
        default_factory=lambda: os.getenv("BOILERPLATE_STORE_PATH", "")
    )  # 样板段指纹库路径，留空则落到 knowledge/temp_data/boilerplate_fingerprints.json

    image_extensions: Set[str] = field(
        default_factory=lambda: {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    )

    # ==================== LLM 配置 ====================
    openai_api_base: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_BASE", "")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    vl_model: str = field(
        default_factory=lambda: os.getenv("VL_MODEL", "")
    )
    item_model: str = field(
        default_factory=lambda: os.getenv("ITEM_MODEL", "")
    )
    default_model: str = field(
        default_factory=lambda: os.getenv("MODEL", "")
    )

    # ==================== Milvus 配置 ====================
    milvus_url: str = field(
        default_factory=lambda: os.getenv("MILVUS_URL", "")
    )
    chunks_collection: str = field(
        default_factory=lambda: os.getenv("CHUNKS_COLLECTION", "")
    )
    item_name_collection: str = field(
        default_factory=lambda: os.getenv("ITEM_NAME_COLLECTION", "")
    )
    entity_name_collection: str = field(
        default_factory=lambda: os.getenv("ENTITY_NAME_COLLECTION", "")
    )

    # ==================== Neo4j 配置 ====================
    neo4j_uri: str = field(
        default_factory=lambda: os.getenv("NEO4J_URI", "")
    )
    neo4j_username: str = field(
        default_factory=lambda: os.getenv("NEO4J_USERNAME", "")
    )
    neo4j_password: str = field(
        default_factory=lambda: os.getenv("NEO4J_PASSWORD", "")
    )
    neo4j_database: str = field(
        default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j")
    )

    # ==================== MinIO 配置 ====================
    minio_endpoint: str = field(
        default_factory=lambda: os.getenv("MINIO_ENDPOINT", "")
    )
    minio_access_key: str = field(
        default_factory=lambda: os.getenv("MINIO_ACCESS_KEY", "")
    )
    minio_secret_key: str = field(
        default_factory=lambda: os.getenv("MINIO_SECRET_KEY", "")
    )
    minio_bucket: str = field(
        default_factory=lambda: os.getenv("MINIO_BUCKET_NAME", "")
    )
    minio_secure: bool = False

    # ==================== 向量配置 ====================
    embedding_dim: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1024"))
    )
    embedding_batch_size: int = 5

    # ==================== 速率限制 ====================
    requests_per_minute: int = 20  # 图片总结 API 速率限制

    @classmethod
    def from_env(cls) -> "ImportConfig":
        """从环境变量加载配置"""
        return cls()

    def get_minio_base_url(self):
        base_protocol = 'http://' if self.minio_secure else ''
        return base_protocol + self.minio_endpoint


# ==================== 全局单例 ====================
_config: Optional[ImportConfig] = None


def get_config() -> ImportConfig:
    """获取配置单例"""
    global _config
    if _config is None:
        _config = ImportConfig.from_env()
    return _config
