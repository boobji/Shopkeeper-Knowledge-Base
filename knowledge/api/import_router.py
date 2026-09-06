import os.path

import uvicorn
from fastapi import FastAPI, File, UploadFile, Depends, BackgroundTasks
from fastapi.responses import FileResponse

from knowledge.core.app_factory import create_app
from knowledge.core.paths import get_front_page_dir
from knowledge.schema.upload_schema import UploadResponse
from knowledge.schema.task_schema import TaskStatusResponse
from knowledge.core.deps import get_task_service
from knowledge.core.deps import get_import_file_service
from knowledge.services.import_file_service import ImportFileService
from knowledge.services.task_service import TaskService
from knowledge.processor.import_process.base import setup_logging


def build_app() -> FastAPI:
    """创建导入服务实例"""
    return create_app(title="Import Service", description="知识库导入", register_routes=register_router)


def register_router(app: FastAPI):
    @app.get("/")
    def read_root():
        return {"Hello": "World"}

    # 1. 处理导入页面访问请求
    @app.get("/import")
    async def import_root():
        return FileResponse(path=os.path.join(get_front_page_dir(), "import.html"))

    # 2. 上传请求
    @app.post("/upload", response_model=UploadResponse)
    async def upload_file_endpoint(background_tasks: BackgroundTasks, file: UploadFile = File(...),
                                   service: ImportFileService = Depends(get_import_file_service)):
        # 1. 上传文件（本地/minio）
        task_id, file_dir, import_file_path = service.process_upload_file(file)

        # 2. 运行后台任务（跑graph的整个流程）
        background_tasks.add_task(service.run_import_graph, task_id, file_dir, import_file_path)

        # 3. 返回
        return UploadResponse(message="文件上传成功", task_id=task_id)


    @app.get("/status/{task_id}", response_model=TaskStatusResponse)
    async def get_status_endpoint(task_id:str,task_service:TaskService=Depends(get_task_service)):
        """
        根据任务id 查询任务的状态
        Returns:
        """
        task_info = task_service.get_task_info(task_id)
        return TaskStatusResponse(**task_info)


if __name__ == '__main__':
    """
    启动web服务器  (fastapi实例)
    uvicorn(性能高)
    """
    setup_logging()
    uvicorn.run(app=build_app(), port=8000, host="0.0.0.0")
