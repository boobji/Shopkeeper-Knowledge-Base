"""诊断工具：验证「查询进行中触发关闭 → 进程被非守护线程卡住」。

场景复现：流式查询在后台执行期间关闭服务（等价于查询中按 Ctrl+C），
10 秒后 dump 全线程栈。若主线程卡在 threading._shutdown 即为该问题实锤。
用法：python scripts/debug_shutdown.py --mid-query
"""

import os
import sys
import threading
import time
import traceback

import httpx
import uvicorn

from knowledge.api.query_router import build_app

BASE = "http://127.0.0.1:8001"


def watchdog(server: uvicorn.Server):
    while not server.started:
        time.sleep(0.2)
    print("[watchdog] started", flush=True)

    # 1. 发起一次流式查询（后台任务开始执行后立即返回）
    def fire_query():
        try:
            r = httpx.post(f"{BASE}/query", json={"query": "RS-12数字万用表如何测量电压？", "is_stream": True}, timeout=30)
            print(f"[watchdog] /query -> {r.status_code} {r.text[:120]}", flush=True)
        except Exception as e:
            print(f"[watchdog] /query failed: {e}", flush=True)

    threading.Thread(target=fire_query, daemon=True).start()
    time.sleep(8)  # 让查询跑到中途（LLM/检索进行中）

    # 2. 触发与 Ctrl+C 等价的优雅关闭
    server.should_exit = True
    print("[watchdog] should_exit=True (查询仍在后台执行)", flush=True)
    time.sleep(10)

    # 3. dump 所有线程栈
    print("\n===== thread stacks 10s after shutdown trigger =====", flush=True)
    for tid, frame in sys._current_frames().items():
        print(f"==== thread {tid} ====", flush=True)
        traceback.print_stack(frame, file=sys.stdout, flush=True)
    os._exit(42)


def main():
    mode = "--mid-query" in sys.argv
    config = uvicorn.Config(build_app(), host="0.0.0.0", port=8001, log_level="info")
    server = uvicorn.Server(config)
    threading.Thread(target=watchdog, args=(server,), daemon=True).start()
    server.run()


if __name__ == "__main__":
    main()
