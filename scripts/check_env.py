#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
掌柜智库 —— 环境自检脚本

用途：在启动应用前，一次性确认四个外部服务是否都已就绪、配置是否填对。
     避免"代码没问题但服务没起来"导致的无效排查。

用法：
    python scripts/check_env.py

退出码：
    0 = 全部通过
    1 = 至少一项失败（此时脚本会给出具体的修复建议）

说明：
    - 配置从 knowledge/.env 读取（与项目 config.py 同一份，无需重复配置）
    - 每个服务先做 TCP 端口探测（3 秒超时），再做协议级连通性验证
      这样"端口不通"和"端口通但认证失败"能被清晰区分开
"""

import os
import sys
import socket
import time
from pathlib import Path

# Windows 控制台默认可能是 GBK，强制 UTF-8 输出，避免中文乱码
try:
    if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------------------------------------------------------------- 基础路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / "knowledge" / ".env"

OK = "[ OK ]"
FAIL = "[FAIL]"
WARN = "[WARN]"
SKIP = "[SKIP]"


def load_env():
    """加载 knowledge/.env，返回 (是否成功, 提示信息)"""
    if not ENV_FILE.exists():
        return False, f"未找到配置文件 {ENV_FILE}"
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=ENV_FILE, override=True)
        return True, str(ENV_FILE)
    except ImportError:
        # 没有 python-dotenv 时手动解析，保证脚本本身可运行
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())
        return True, f"{ENV_FILE}（未装 python-dotenv，已手动解析）"


def check_port(host, port, timeout=3):
    """TCP 端口探测，返回 (是否可达, 耗时秒)"""
    t0 = time.time()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, time.time() - t0
    except Exception:
        return False, time.time() - t0


def parse_hostport(url, default_port):
    """
    从 URL 中解析 host 与 port。
    支持：http://host:port、mongodb://host:port、bolt://host:port、host:port
    """
    s = (url or "").strip()
    if "://" in s:
        s = s.split("://", 1)[1]
    s = s.rstrip("/").split("/")[0]
    if "@" in s:                      # 兼容 mongodb://user:pass@host:port
        s = s.rsplit("@", 1)[1]
    if ":" in s:
        host, _, port = s.rpartition(":")
        try:
            return host or "127.0.0.1", int(port)
        except ValueError:
            return host or "127.0.0.1", default_port
    return (s or "127.0.0.1"), default_port


def section(title):
    print()
    print("=" * 62)
    print(f"  {title}")
    print("=" * 62)


# ---------------------------------------------------------------- 各项检查
def check_milvus(results):
    url = os.getenv("MILVUS_URL", "http://127.0.0.1:19530")
    host, port = parse_hostport(url, 19530)
    reachable, cost = check_port(host, port)
    if not reachable:
        results.append((FAIL, "Milvus", f"{host}:{port} 端口不可达（{cost:.1f}s 超时）"))
        return

    try:
        from pymilvus import MilvusClient
        client = MilvusClient(uri=url, timeout=5)
        cols = client.list_collections()
        results.append((OK, "Milvus", f"{url}  已有集合 {len(cols)} 个"))
        if cols:
            preview = "、".join(cols[:5]) + ("..." if len(cols) > 5 else "")
            print(f"         集合列表：{preview}")
    except Exception as e:
        results.append((FAIL, "Milvus", f"端口通但连接失败：{type(e).__name__}: {str(e)[:120]}"))


def check_neo4j(results):
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    pwd = os.getenv("NEO4J_PASSWORD", "")
    host, port = parse_hostport(uri, 7687)
    reachable, cost = check_port(host, port)
    if not reachable:
        results.append((FAIL, "Neo4j", f"{host}:{port} 端口不可达（{cost:.1f}s 超时）"))
        return

    if not pwd:
        results.append((WARN, "Neo4j", "端口可达，但 NEO4J_PASSWORD 未配置，无法验证认证"))
        return

    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(uri, auth=(user, pwd), connection_timeout=5)
        driver.verify_connectivity()
        with driver.session() as s:
            n = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        driver.close()
        results.append((OK, "Neo4j", f"{uri}  认证通过，当前图库节点数 {n}"))
    except Exception as e:
        msg = str(e)[:120]
        if "authentication" in msg.lower() or "unauthorized" in msg.lower():
            results.append((FAIL, "Neo4j", f"认证失败，检查 NEO4J_USERNAME / NEO4J_PASSWORD：{msg}"))
        else:
            results.append((FAIL, "Neo4j", f"端口通但连接失败：{type(e).__name__}: {msg}"))


def check_mongodb(results):
    url = os.getenv("MONGO_URL", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB_NAME", "kb001")
    host, port = parse_hostport(url, 27017)
    reachable, cost = check_port(host, port)
    if not reachable:
        results.append((FAIL, "MongoDB", f"{host}:{port} 端口不可达（{cost:.1f}s 超时）"))
        return

    try:
        from pymongo import MongoClient
        c = MongoClient(url, serverSelectionTimeoutMS=5000)
        c.admin.command("ping")
        names = c.list_database_names()
        marked = "（已初始化）" if db_name in names else "（尚未写入数据，属正常）"
        results.append((OK, "MongoDB", f"{url}  库 {db_name} {marked}"))
        c.close()
    except Exception as e:
        results.append((FAIL, "MongoDB", f"端口通但连接失败：{type(e).__name__}: {str(e)[:120]}"))


def check_minio(results):
    endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    ak = os.getenv("MINIO_ACCESS_KEY", "")
    sk = os.getenv("MINIO_SECRET_KEY", "")
    bucket = os.getenv("MINIO_BUCKET_NAME", "")
    host, port = parse_hostport(endpoint, 9000)
    reachable, cost = check_port(host, port)
    if not reachable:
        results.append((FAIL, "MinIO", f"{host}:{port} 端口不可达（{cost:.1f}s 超时）"))
        return

    if not ak or not sk:
        results.append((WARN, "MinIO", "端口可达，但 MINIO_ACCESS_KEY / MINIO_SECRET_KEY 未配置"))
        return

    try:
        from minio import Minio
        client = Minio(endpoint, access_key=ak, secret_key=sk, secure=False)
        if not bucket:
            results.append((WARN, "MinIO", "连接正常，但 MINIO_BUCKET_NAME 未配置"))
            return
        exists = client.bucket_exists(bucket)
        if exists:
            results.append((OK, "MinIO", f"{endpoint}  桶 {bucket} 已就绪"))
        else:
            # 与 milvus_util / minio_util 行为一致：不存在则自动创建
            client.make_bucket(bucket)
            results.append((OK, "MinIO", f"{endpoint}  桶 {bucket} 不存在，已自动创建"))
    except Exception as e:
        results.append((FAIL, "MinIO", f"端口通但连接失败：{type(e).__name__}: {str(e)[:120]}"))


def check_required_config(results):
    """检查运行主流程必需的 LLM 配置"""
    missing = [
        k for k in ("OPENAI_API_KEY", "OPENAI_API_BASE")
        if not os.getenv(k)
    ]
    if missing:
        results.append((FAIL, "LLM 配置", f"缺少必填项：{'、'.join(missing)}"))
    else:
        key = os.getenv("OPENAI_API_KEY", "")
        bad = [c for c in key if ord(c) > 127]
        if bad:
            results.append((FAIL, "LLM 配置",
                            f"OPENAI_API_KEY 含 {len(bad)} 个非 ASCII 字符（多为复制时带入中文注释），"
                            f"会导致请求头编码失败"))
        else:
            results.append((OK, "LLM 配置", f"OPENAI_API_KEY 长度 {len(key)}，格式正常"))


# ---------------------------------------------------------------- 主流程
def main():
    print("掌柜智库 —— 运行环境自检")
    print("=" * 62)

    ok, info = load_env()
    if ok:
        print(f"配置文件：{info}")
    else:
        print(f"{WARN} {info}")
        print("       请执行：cp .env.example knowledge/.env 后填入配置")

    section("服务连通性检查")
    results = []
    for fn in (check_milvus, check_neo4j, check_mongodb, check_minio, check_required_config):
        try:
            fn(results)
        except ImportError as e:
            name = fn.__name__.replace("check_", "")
            results.append((SKIP, name, f"缺少依赖库，无法检查（pip install -r requirements.txt）：{e}"))
        except Exception as e:
            name = fn.__name__.replace("check_", "")
            results.append((FAIL, name, f"检查过程异常：{type(e).__name__}: {str(e)[:120]}"))

    # 输出汇总
    print()
    for status, name, detail in results:
        print(f"  {status} {name:<10} {detail}")

    failed = [r for r in results if r[0] == FAIL]
    warned = [r for r in results if r[0] == WARN]

    print()
    print("=" * 62)
    if not failed:
        print(f"  全部通过（{len(results) - len(warned)} 项正常"
              + (f"，{len(warned)} 项待确认" if warned else "") + "）")
        print("  可以启动服务：python -m knowledge.api.import_router")
        print("=" * 62)
        return 0

    print(f"  {len(failed)} 项失败，请按以下建议处理：")
    print("=" * 62)
    print()
    print("  1. 服务未启动？在项目根目录执行：")
    print("       docker compose up -d")
    print("       然后等待约 30 秒（Milvus 启动较慢），用 docker compose ps 确认全部 healthy")
    print()
    print("  2. 想复用已有配置启动（不必重填密码）：")
    print("       docker compose --env-file knowledge/.env up -d")
    print()
    print("  3. 端口通但认证失败？核对 knowledge/.env 中的账号密码")
    print("     与 docker-compose.yml 中的默认值是否一致")
    print()
    print("  4. 确认端口占用：docker compose ps")
    print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
