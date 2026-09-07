"""切片清洗节点（P0-2 样板段去重 / P0-3 语言与目录页过滤）。

位置：document_split_node → **chunk_clean_node** → item_name_recognition_node

放在商品名抽取之前的原因很直接：`item_name_recognition_node` 取**前 K 个切片**
做抽取，而目录页、外文页、安全声明恰好都集中在文档前几页。先清洗再抽取，
商品名就不会被"海尔服务承诺""儿童请勿使用"这类内容带偏。
"""

from typing import Dict, Any

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.config import get_config
from knowledge.domain.chunk_clean import clean_chunks


class ChunkCleanNode(BaseNode):
    name = "chunk_clean_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        chunks = state.get("chunks") or []
        file_title = state.get("file_title", "")

        # 1. 参数校验（清洗是增强项，缺失时直接透传而不是中断导入）
        if not chunks:
            self.logger.warning("chunks 为空，跳过清洗")
            state["chunk_clean_stats"] = {"input": 0, "kept": 0, "reason": "chunks 为空"}
            return state

        config = get_config()
        if not getattr(config, "chunk_clean_enabled", True):
            self.logger.info("切片清洗未启用（CHUNK_CLEAN_ENABLED=0），保持原切片")
            state["chunk_clean_stats"] = {"disabled": 1, "input": len(chunks), "kept": len(chunks)}
            return state

        file_dir = state.get("file_dir", "")

        # 2. 执行清洗
        try:
            result = clean_chunks(chunks, file_title, config, file_dir)
        except Exception as e:
            # 清洗失败绝不影响主流程：原样保留
            self.logger.warning(f"清洗异常，保留原切片: {type(e).__name__}: {e}")
            state["chunk_clean_stats"] = {"error": str(e), "input": len(chunks), "kept": len(chunks)}
            return state

        stats: Dict[str, Any] = dict(result.stats)

        # 3. 回写 state（遵循节点契约）
        state["chunks"] = result.chunks
        state["chunk_clean_stats"] = stats

        # 4. 日志：一眼看清每类噪音剔了多少
        dropped = stats.get("input", 0) - stats.get("kept", 0)
        self.logger.info(
            f"切片清洗: {stats.get('input')} -> {stats.get('kept')} 块（剔除 {dropped}）"
        )
        detail = [
            f"目录页 {stats.get('drop_toc', 0)}",
            f"外文/符号页 {stats.get('drop_non_zh', 0)}",
            f"文档内重复 {stats.get('drop_in_doc_dup', 0)}",
            f"跨文档样板段 {stats.get('drop_boilerplate', 0)}",
        ]
        self.logger.info("  明细: " + " | ".join(detail))
        if stats.get("mixed_lang"):
            self.logger.info(f"  低中文占比切片 {stats['mixed_lang']} 块已标记 lang=mixed（检索降权）")
        if stats.get("fallback"):
            self.logger.warning("  清洗后切片为空，已回退原始切片（请检查阈值是否过激）")

        # 5. 同步备份清洗后的结果，方便用 diagnose_chunks.py 复核
        self._backup_cleaned(state, result.chunks)
        return state

    def _backup_cleaned(self, state: ImportGraphState, chunks: list):
        """把清洗结果落盘为 chunks_cleaned.json，便于事后体检。"""
        local_dir = state.get("file_dir", state.get("local_dir", ""))
        if not local_dir:
            return
        try:
            import json
            import os

            os.makedirs(local_dir, exist_ok=True)
            output_path = os.path.join(local_dir, "chunks_cleaned.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.debug(f"清洗结果备份失败: {e}")
