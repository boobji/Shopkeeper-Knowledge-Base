"""
导入流程自定义异常类

统一错误处理，提供更清晰的错误信息。
异常基类与消息格式在 core.exceptions.ProcessError 中统一定义。
"""

from knowledge.core.exceptions import ProcessError


class ImportProcessError(ProcessError):
    """导入流程基础异常"""
    pass


class ConfigurationError(ImportProcessError):
    """配置错误：环境变量缺失或配置值无效"""
    pass


class FileProcessingError(ImportProcessError):
    """文件处理错误：文件不存在、格式错误、读写失败"""
    pass


class PdfConversionError(FileProcessingError):
    """PDF 转换错误：MinerU 转换失败"""
    pass


class ImageProcessingError(FileProcessingError):
    """图片处理错误：图片总结、上传失败"""
    pass


class DocumentSplitError(ImportProcessError):
    """文档切分错误：切分逻辑异常"""
    pass


class EmbeddingError(ImportProcessError):
    """向量化错误：模型调用失败、向量生成异常"""
    pass


class LLMError(ImportProcessError):
    """LLM 调用错误：API 调用失败、响应解析失败"""
    pass


class StorageError(ImportProcessError):
    """存储错误：数据库操作失败"""
    pass


class MilvusError(StorageError):
    """Milvus 存储错误"""
    pass


class Neo4jError(StorageError):
    """Neo4j 存储错误"""
    pass


class MinioError(StorageError):
    """MinIO 存储错误"""
    pass


class ValidationError(ImportProcessError):
    """数据验证错误：输入数据不符合预期"""
    pass
