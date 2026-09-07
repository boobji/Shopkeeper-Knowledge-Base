"""LLM 调用留档工具的单元测试。"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from pathlib import Path

import pytest

os.environ.setdefault("LLM_CALL_LOG_DIR", "")  # 不影响其它测试

from knowledge.utils import llm_call_logger as m


def test_sanitize_replaces_base64():
    text = "前文 data:image/jpeg;base64," + "A" * 5000 + " 后文"
    out = m._sanitize(text)
    assert "A" * 100 not in out
    assert "[base64图片已省略" in out


def test_sanitize_nested_structures():
    payload = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "ok"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAABBBBCCCC"}},
    ]}]}
    out = m._sanitize(payload)
    assert "AAAA" not in json.dumps(out, ensure_ascii=False)


def test_message_to_dict_variants():
    assert m._message_to_dict("hi") == {"role": "user", "content": "hi"}
    msg = SimpleNamespace(type="human", content="问题")
    assert m._message_to_dict(msg) == {"role": "user", "content": "问题"}
    multimodal = SimpleNamespace(type="human", content=[
        {"type": "text", "text": "看图"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,xx"}},
    ])
    d = m._message_to_dict(multimodal)
    assert "[图片输入，base64已省略]" in d["content"]
    assert "看图" in d["content"]


def test_log_writes_file_in_task_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_CALL_LOG_ENABLED", "1")
    fake_resp = SimpleNamespace(
        content="数字万用表",
        usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    )
    out = m.log_llm_call(
        "import_item_name", prompt="完整提示词", response=fake_resp,
        task_id="t1", task_dir=str(tmp_path), latency_ms=123.4,
    )
    assert out is not None and out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["stage"] == "import_item_name"
    assert data["prompt"] == "完整提示词"
    assert data["response"] == "数字万用表"
    assert data["usage"]["total_tokens"] == 12
    assert data["latency_ms"] == 123


def test_log_disabled_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_CALL_LOG_ENABLED", "0")
    assert m.log_llm_call("x", prompt="p", task_dir=str(tmp_path)) is None


def test_log_never_raises(tmp_path, monkeypatch):
    # 制造故障：把 json.dumps 搞坏（返回不可序列化对象且 monkeypatch write 抛异常）
    monkeypatch.setenv("LLM_CALL_LOG_ENABLED", "1")
    monkeypatch.setattr(Path, "write_text", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    out = m.log_llm_call("x", prompt="p", task_dir=str(tmp_path))
    assert out is None  # 只告警不抛出


def test_sequence_increments_across_restarts(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_CALL_LOG_ENABLED", "1")
    for _ in range(3):
        m.log_llm_call("query_answer", prompt="p", task_id="t9", task_dir=str(tmp_path))
    m._seq_counters.pop(str(tmp_path / "llm_calls"), None)  # 模拟进程重启
    out = m.log_llm_call("query_answer", prompt="p", task_id="t9", task_dir=str(tmp_path))
    assert out.name.startswith("0004_")


def test_error_suffix_and_content(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_CALL_LOG_ENABLED", "1")
    out = m.log_llm_call("import_kg", prompt="p", task_dir=str(tmp_path),
                         error="超时", meta={"chunk_id": "c1", "attempt": 2})
    assert out.name.endswith("_error.json")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["error"] == "超时"
    assert data["meta"]["chunk_id"] == "c1"
