from pymongo import MongoClient
from pymongo.collection import Collection

# 1. 定义MongoDB的客户端
#    说明：MongoClient 构造是“懒连接”，不会立即联网；serverSelectionTimeoutMS 缩短等待，
#          让连接问题尽早暴露而不是卡 30s。
mongo_client = MongoClient("mongodb://127.0.0.1:27017", serverSelectionTimeoutMS=5000)

# 2. 创建库
db = mongo_client["my_db"]

# 3. 创建表
collection = db["students"]

# 主动发起一次真实连接（ping 会真正建连+选主节点），连接不上会立刻抛错，
# 避免误以为“构造成功=已连接”，把问题推迟到第一次 insert/find。
try:
    mongo_client.admin.command("ping")
    print("✅ MongoDB 连接成功")
except Exception as e:
    print("❌ MongoDB 连接失败：", e)
    raise


# print(mongo_client)
# print(db)
# print(collection)


def insert_document(connection: Collection):
    result = collection.insert_one({
        "name": "张三",
        "age": 20,
        "major": "计算机科学"
    })

    print(result)


def insert_documents(connection: Collection):
    results = collection.insert_many([
        {"name": "李四", "age": 22, "major": "软件工程"},
        {"name": "王五", "age": 21, "major": "计算机科学"},
    ])

    print(results)


def fetch_collection(connection: Collection):
    # 1. 查询全部
    # for document in collection.find():
    #     print(document['name'])

    # 2. 根据条件查询
    # for doc in collection.find({"major": "计算机科学"}):
    #     print(doc["name"], doc["age"])

    # 3. 查询单条记录
    # student = collection.find_one({"name": "张三"})
    # print(student)

    for doc in collection.find().sort("age", 1).limit(1):
        print(doc["name"], doc["age"])


def update_document():
    # 更新单条
    result = collection.update_one(
        {"name": "张三"},  # 查询条件
        {"$set": {"age": 21}}  # 更新操作
    )
    print(f"匹配 {result.matched_count} 条，修改 {result.modified_count} 条")

    # 更新多条
    result = collection.update_many(
        {"major": "计算机科学"},
        {"$set": {"status": "在读"}}
    )
    print(f"修改 {result.modified_count} 条")


def delete_document():
    # 删除单条
    result = collection.delete_one({"name": "王五"})
    print(f"删除 {result.deleted_count} 条")

    # 删除多条
    result = collection.delete_many({"age": {"$lt": 21}})
    print(f"删除 {result.deleted_count} 条")


if __name__ == '__main__':
    # insert_document(collection)

    # insert_documents(collection)

    # fetch_collection(collection)

    # update_document()

    delete_document()
