"""知识图谱领域层：导入/查询共用的 Schema 定义。

单一事实来源：
- 实体/关系白名单（导入侧写入时校验，查询侧无需重复定义）
- Cypher 语句（导入 Writer 与查询 Reader 操作的是同一套图结构，
  集中放置避免两侧结构漂移）
- 实体名长度、对齐阈值、节点权重等常量
"""

# ------------------------------------------------------------------
# 常量
# ------------------------------------------------------------------
MAX_ENTITY_NAME_LENGTH = 15          # 实体名最大长度（写入与查询两侧共用）
DEFAULT_ENTITY_ALIGN_SCORE = 0.5     # 查询侧实体对齐的最低分数阈值

# 查询侧图谱节点权重：种子节点高、邻居节点低
SEED_NODE_WEIGHT = 2.0
NEIGHBOR_NODE_WEIGHT = 1.0

# ------------------------------------------------------------------
# 白名单（导入侧写入时校验）
# ------------------------------------------------------------------
ALLOWED_ENTITY_LABELS = {
    "Device", "Part", "Operation", "Step",
    "Warning", "Condition", "Tool",
}

ALLOWED_RELATION_TYPES = {
    "HAS_OPERATION", "HAS_PART", "HAS_STEP", "USES_TOOL",
    "HAS_WARNING", "NEXT_STEP", "AFFECTS", "REQUIRES",
    "MENTIONED_IN", "RELATED_TO",
}
DEFAULT_RELATION_TYPES = "RELATED_TO"

# ------------------------------------------------------------------
# Cypher：导入侧 Writer
# ------------------------------------------------------------------
# Chunk 标签节点创建
CYPHER_MERGE_CHUNK = """
    MERGE (c:Chunk {id: $chunk_id, item_name: $item_name})
"""

# Entity 标签节点的创建
CYPHER_MERGE_ENTITY_TEMPLATE = """
    MERGE (n:Entity {{name: $name, item_name: $item_name}})
    ON CREATE SET
        n.source_chunk_id = $chunk_id,
        n.description     = $description
    ON MATCH SET
        n.description = CASE
            WHEN $description <> "" THEN $description
            ELSE coalesce(n.description, "")
        END
    SET n:`{label}`
"""

# Entity 关联 Chunk
CYPHER_LINK_ENTITY_TO_CHUNK = """
    MATCH (n:Entity {name: $name, item_name: $item_name})
    MATCH (c:Chunk  {id: $chunk_id, item_name: $item_name})
    MERGE (n)-[:MENTIONED_IN]->(c)
"""

# Entity 与 Entity 的关系
CYPHER_MERGE_RELATION_TEMPLATE = """
    MATCH (h:Entity {{name: $head, item_name: $item_name}})
    MATCH (t:Entity {{name: $tail, item_name: $item_name}})
    MERGE (h)-[:{rel_type}]->(t)
"""

# 清理某商品的全部图谱数据
CYPHER_CLEAR_ITEM = """
    MATCH (n {item_name: $item_name}) DETACH DELETE n
"""

# ------------------------------------------------------------------
# Cypher：查询侧 Reader
# ------------------------------------------------------------------
# 种子节点精确查询
CYPHER_EXACT_SEEDS = """
MATCH (n:Entity)
WHERE n.item_name=$item_name AND n.name=$name
RETURN  n.item_name as item_name,n.name as name
LIMIT 1
"""

# 种子节点模糊查询（精确未命中时降级兜底）
CYPHER_FUZZY_SEEDS = """
MATCH (n:Entity)
WHERE toLower(n.name) CONTAINS toLower($name)
      AND n.item_name = $item_name
RETURN n.name AS name, n.item_name AS item_name
LIMIT $limit
"""

# 查询种子节点的一跳关系（双向，过滤 MENTIONED_IN）
CYPHER_ONE_HOP_RELATIONS = """

MATCH (seed:Entity {name:$name,item_name:$item_name})-[r]-(nbr:Entity)

WHERE type(r) <> 'MENTIONED_IN' AND nbr.item_name=$item_name

RETURN
  CASE WHEN startNode(r)=seed  THEN  seed.name  ELSE nbr.name END AS head,
  type(r) as rel,
  CASE WHEN  startNode(r)=seed  THEN nbr.name ELSE seed.name END AS tail

limit $limit
"""

# 根据带权重的节点反查 chunk（按权重和、次数排序）
CYPHER_LOOKUP_CHUNK = """

UNWIND $weighted_nodes as n

MATCH (e:Entity{name:n.entity_name,item_name:n.item_name})-[r:MENTIONED_IN]->(c:Chunk{item_name:n.item_name})

WITH c,sum(n.weight) AS score, count(e) AS cnt

RETURN c.id AS chunk_id, c.item_name AS item_name, score, cnt

ORDER BY score DESC, cnt DESC,chunk_id DESC

LIMIT $limit

"""
