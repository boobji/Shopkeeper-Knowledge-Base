"""查询相关 Schema 定义"""

from typing import Optional, List
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="查询内容")
    session_id: Optional[str] = Field(None, description="会话ID，不传则自动生成")
    is_stream: bool = Field(False, description="是否流式返回")
    selected_item: Optional[str] = Field(
        None, description="用户点击澄清选项后回传的商品名（结构化，优先于文本解析）"
    )


class QueryResponse(BaseModel):
    message: str = Field(..., description="响应消息")
    session_id: str = Field(..., description="会话ID")
    answer: str = Field("", description="生成的答案")
    suggestions: List[str] = Field(default_factory=list, description="后续建议问法（最多3条）")


class StreamSubmitResponse(BaseModel):
    message: str = Field(..., description="响应消息")
    session_id: str = Field(..., description="会话ID")
    task_id: str = Field(..., description="任务ID，前端用此 ID 建立 SSE 连接")


class HistoryItem(BaseModel):
    id: str = Field("", alias="_id")
    session_id: str = ""
    role: str = ""
    text: str = ""
    rewritten_query: str = ""
    item_names: List[str] = Field(default_factory=list)
    ts: Optional[float] = None


class HistoryResponse(BaseModel):
    session_id: str
    items: List[HistoryItem]
