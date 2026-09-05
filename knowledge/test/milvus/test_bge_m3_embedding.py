from pymilvus.model.hybrid import BGEM3EmbeddingFunction
bge_m3_ef = BGEM3EmbeddingFunction(
    model_name=r"D:\Develop\AiMenu\models\bge-m3", # 嵌入模型名字（如果本地有，把path填入）
    device='cpu',# 设备
    use_fp16=False # 半精度fp16（空间利用率低，只能在GPU上用） 单精度fp32
)

print(bge_m3_ef)