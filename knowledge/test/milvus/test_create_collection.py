from pymilvus import MilvusClient, DataType
from knowledge.utils.bge_m3_embedding_util import get_bge_m3_embedding_model

if __name__ == "__main__":
    # 定义milvus客户端对象
    client = MilvusClient(uri='http://127.0.0.1:19530')

    # 2.创建集合
    if client.has_collection(collection_name="test_collection"):
        client.drop_collection(collection_name="test_collection")

    # 创建schema
    schema = client.create_schema(enable_dynamic_field=True)

    # 主键字段
    schema.add_field(
        field_name="my_id", #字段名字
        datatype=DataType.INT64, # 值的类型
        is_primary=True, # 是否是主键
        auto_id=True, # 是否自增
    )

    # 向量字段
    schema.add_field(
        field_name="my_vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=1024
    )

    # 标量字段
    schema.add_field(
        field_name="my_varchar",
        datatype=DataType.VARCHAR,
        max_length=512
    )

    # 添加向量索引
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="my_vector",  # Name of the vector field to be indexed
        index_type="IVF_FLAT",  # Type of the index to create
        index_name="vector_index",  # Name of the index to create
        metric_type="COSINE",  # Metric type used to measure similarity
        params={
            "nlist": 64,  # Number of clusters for the index
        }  # Index building params
    )

    client.create_collection(collection_name='test_collection',
                             schema=schema, index_params=index_params)

    # 3.构建数据


    docs = [
        "Artificial intelligence was founded as an academic discipline in 1956.",
        "Alan Turing was the first person to conduct substantial research in AI.",
        "Born in Maida Vale, London, Turing was raised in southern England.",
    ]
    embedding_fn = get_bge_m3_embedding_model()
    vectors = embedding_fn.encode_documents(docs)

    # vectors 是 dict（含 'dense'/'sparse'/'colbert_vecs' 等键），必须按文档数 len(docs) 控制循环；
    # 向量要取第 i 条并调用 .tolist() 转成 list[float]
    data = [
        {"my_vector": vectors['dense'][i].tolist(), "my_varchar": docs[i], "subject": "history"}
        for i in range(len(docs))
    ]

    # 插入数据
    res = client.insert(collection_name="test_collection", data=data)

    # 搜索
    query_vectors = embedding_fn.encode_queries(["Who is Alan Turing?"])

    res1 = client.search(
        collection_name="test_collection",  # target collection
        data=[query_vectors['dense'][0].tolist()],  # query vectors
        limit=2,  # number of returned entities
        output_fields=["my_varchar"],
    )

    print(res1)