import os
import logging

from minio import Minio
from minio.error import S3Error
from dotenv import load_dotenv
load_dotenv()

def get_minio_client():
    # 1.实例化客户端
    try:
        client = Minio(os.getenv("MINIO_ENDPOINT"),
            access_key=os.getenv("MINIO_ACCESS_KEY"),
            secret_key=os.getenv("MINIO_SECRET_KEY"),
            secure=False,
        )

        # 2.桶是否存在
        bucket_name = os.getenv("MINIO_BUCKET_NAME")
        bucket_exists = client.bucket_exists(bucket_name)
        if not bucket_exists:
            client.make_bucket(bucket_name)
            logging.info(f"桶{bucket_name}已创建")
        else:
            logging.info(f"桶{bucket_name}已存在")
        return client
    except S3Error:
        logging.error('客户端创建失败')
        return None

if __name__ == "__main__":
    minio_client = get_minio_client()
    print(minio_client)
