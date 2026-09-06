"""Contextual Retrieval（domain/contextual）单测。"""

from knowledge.domain.contextual import generate_chunk_context


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, content="本段出自《万用表手册》的电池更换章节。"):
        self.content = content
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return _FakeResponse(self.content)


class _BoomLLM:
    def invoke(self, messages):
        raise TimeoutError("llm timeout")


class TestGenerateChunkContext:
    def test_returns_clean_text(self):
        llm = _FakeLLM('  "本段介绍电池更换步骤。"  ')
        ctx = generate_chunk_context(llm, "万用表手册", "## 电池更换", "正文" * 100)
        assert ctx == "本段介绍电池更换步骤。"
        # 提示词包含文档标题与章节标题
        assert "万用表手册" in llm.calls[0][1].content
        assert "电池更换" in llm.calls[0][1].content

    def test_long_output_capped(self):
        llm = _FakeLLM("长" * 500)
        ctx = generate_chunk_context(llm, "T", "H", "C")
        assert len(ctx) <= 150

    def test_failure_returns_empty(self):
        assert generate_chunk_context(_BoomLLM(), "T", "H", "C") == ""
