"""商品名领域层（查询侧组件，从 item_name_confirm_node.py 拆出）。

- ItemNameExtractor   LLM 从「用户问题 + 历史」提取商品名并重写问题
- ItemNameAligner     向量库对齐：检索 kb_item_names → 阈值分桶 → 分差过滤

阈值说明：与查询配置 item_name_* 系列字段含义对应，但历史行为为硬编码值
（max 每桶 3 个、分差 0.15 与配置默认值不一致），为不改变行为，此处以
常量固化；如需调参，先核对 .env 中对应配置的实际取值再接线。
"""

import json
import logging
from json import JSONDecodeError
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from knowledge.domain.llm_parse import strip_json_fence
from knowledge.prompts.query.query_prompt import ITEM_NAME_EXTRACT_TEMPLATE
from knowledge.utils.bge_m3_embedding_util import generate_hybrid_embeddings, get_bge_m3_embedding_model
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.utils.milvus_util import (
    create_hybrid_search_requests,
    execute_hybrid_search_query,
    get_milvus_client,
)

logger = logging.getLogger(__name__)

# 对齐阈值（历史硬编码值，语义见各方法 docstring）
HIGH_CONFIDENCE_SCORE = 0.7     # ≥ 该分数可进入 confirmed
MID_CONFIDENCE_SCORE = 0.6      # 未达 high 时，≥ 该分数可进入 options
SCORE_GAP_THRESHOLD = 0.15      # confirmed 内与最高分差值超过该值视为误判剔除
MAX_OPTIONS = 3                 # options / 单桶候选的最大个数


class ItemNameAligner:
    """
     主要职责：
     1. 查询向量数据库
     2. 评分对齐
     3. 分数差异过滤
    """

    def __init__(self, collection_name: str):
        self._collection_name = collection_name

    def match_align_filter(self, item_names: List[str]) -> Tuple[List[str], List[str]]:
        # 1. 查询向量数据库
        search_result: List[Dict[str, Any]] = self._match_vector(item_names)

        # 2. 评分对齐
        confirmed, options = self._item_name_score_align(search_result)

        # 3. 分数差异过滤
        if len(confirmed) > 1:
            confirmed = self._item_name_score_filter(confirmed, search_result)

        return confirmed, options

    def _match_vector(self, item_names: List[str]) -> List[Dict[str, Any]]:
        """
        职责：根据LLM提取的商品名，查询向量数据库
        Args:
            item_names:  LLM提取的商品名

        Returns:
             List[Dict[str, Any]]：每一个item_name下的查询结果
             Dict[str,Any]:{"extracted_name":"LLM提取出来的商品名字"，"matches":[{"item_name":"向量数据库的商品名","score":"结果分数值"}]}

        """
        # 1. 定义最终搜索结果
        search_results = []

        # 2. 获取milvus_client
        milvus_client = get_milvus_client()
        if milvus_client is None:
            return []

        # 3. 获取嵌入模型
        embedding_model = get_bge_m3_embedding_model()
        if embedding_model is None:
            logger.error("获取嵌入模型失败")

            return search_results

        # 4. 嵌入item_name获取稠密、稀疏向量
        hybrid_embedding_result = generate_hybrid_embeddings(embedding_model, item_names)

        # 4. 遍历LLM提取的所有商品名
        for index, extract_item_name in enumerate(item_names):
            # 混合向量检索
            # 4.1 创建混合检索的请求
            hybrid_search_requests = create_hybrid_search_requests(
                dense_vector=hybrid_embedding_result['dense'][index],
                sparse_vector=hybrid_embedding_result['sparse'][index],
            )

            # 4.2 执行混合检索的请求
            # (milvus集成bgem3嵌入模型只会对“稠密向量”进行L2的归一化：IP和COSINE【-1,1】相等 但是不会对稀疏向量进行归一化【权重】)
            # （WeightedRanker：属性：norm_score；权重融合排序器：对稠密向量检索的结果的分数值以及稀疏向量检索到的结果“分数值”进行归一化：为了统一最后在排序的时候，各个向量维度的结果用权重计算的时候，公平）---【0,1】
            hybrid_search_result = execute_hybrid_search_query(milvus_client,
                                                               collection_name=self._collection_name,
                                                               search_requests=hybrid_search_requests,
                                                               ranker_weights=(0.5, 0.5), norm_score=True,
                                                               output_fields=["item_name"])

            # 4.3 解析混合检索请求的结果对象
            item_name_search_result = {
                "extracted_name": extract_item_name,
                "matches": [
                    {"item_name": h["entity"]["item_name"], "score": h["distance"]}
                    for h in (hybrid_search_result[0] if hybrid_search_result else [])
                ]
            }
            # 4.4 将构建好的查询结果放入到最终搜索结果中
            search_results.append(item_name_search_result)
        return search_results

    def _item_name_score_align(self, search_results: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """
        主要职责：根据向量数据库检索到的商品名，放到对应的confirmed或者options

        Args:
            search_result:

        Returns:
            分数阈值的规则：confirm：0.7   options:0.6
            分数阈值作为放到confirmed或者options的条件。

            返回值：confirmed有，将confirmed中的商品名 传给下游四路检索
            返回值：options有，确认下一步，询问到底在咨询哪一款商品。
            返回值：confirmed没有 options没有，直接告诉没有找到具体的商品名
            返回值：confirmed有 options有，至少确定了一个商品名，没有必要让用户在次确认这个商品。

            注意：
            1. 如果像confirmed列表中添加某一次遍历向量数据库查询到的商品名时，发现confirmed已经有该商品名了。
            2. 如果像confirmed列表中添加某一次遍历向量数据库查询到的商品名时，发现confirmed已经有该商品名了。
            3. 如果confirmed中已经有某一个商品从向量数据库返回的某个对应的item_name，那么下一次从另外一个商品名中根据向量数据库中返回的同一个item_name 既不能加到confirmed（重复） 也不能加入options中
            4.如果options中已经有某一个商品从向量数据库返回的某个对应的item_name，那么下一次从另外一个商品名中根据向量数据库中返回的同一个item_name 不能加到options中（重复） 但是可以加入confirm中
            所以去重的方向是单向的
        """

        # 1. 定义两个容器
        confirmed = []
        options = []  # 条件 阈值0.6 最多只留下3个

        # 2. 遍历向量数据库查询到的所有LLM提取到商品名相关的相似性结果
        for item_name_search_result in search_results:

            # 2.0 获取LLM提取的商品名
            extracted_name = item_name_search_result.get('extracted_name')

            # 2.1 对某一给商品名下找到相似的item_name的分数值进行降序
            matches = sorted(item_name_search_result.get('matches'), key=lambda x: x['score'], reverse=True)

            # 2.2 获取matches中分数值比能进入到confirmed容器阈值大的对象获取到
            high = [m for m in matches if m.get('score') >= HIGH_CONFIDENCE_SCORE]

            # 询问是否能进入到confirmed中
            if high:
                # 3.1 准备找最精准的那一个
                extract = next((h for h in high if str(h['item_name']) == extracted_name), None)

                # 场景A:找到了(最准确)---情况很少见
                if extract:
                    picked = extract["item_name"]
                    # 重复的item_name confirmed中只留一份
                    if picked not in confirmed:
                        confirmed.append(picked)
                # 场景B:一般准确
                elif len(high) == 1:
                    picked = high[0]["item_name"]
                    if picked not in confirmed:
                        confirmed.append(picked)
                # 场景C:多个相似
                else:
                    # 如果没有找到精确的 & high中还有多个（options合适、confirmed中：选择放到某个容器。）
                    for h in high[:MAX_OPTIONS]:
                        picked = h.get('item_name')
                        if picked not in options and picked not in confirmed:
                            options.append(picked)

            # 4. 询问是否能进入到options中
            else:
                mid = [m for m in matches if
                       m['score'] >= MID_CONFIDENCE_SCORE and m.get('item_name') not in options and m.get('item_name') not in confirmed]

                if mid:
                    for m in mid[:MAX_OPTIONS]:
                        picked = m.get('item_name')
                        options.append(picked)

        # 最后返回
        return confirmed, options[:MAX_OPTIONS]

    def _item_name_score_filter(self, confirmed: List[str], search_results: List[Dict[str, Any]]):
        """
        item_names:有三个item_name
        item_name1:0.9 （最相似的（基准））
        item_name2:0.88（真实比对）
        item_name3:0.66（可能误判）
        分数差的阈值：0.15
        主要责任：将误判的item_name从confirmed剔除掉。留下真实的item_name
        Args:
            confirmed:
            search_results:
        Returns:

        """
        # 1. 定义字典容器（存储confirmed中item_name在向量数据库中的分数值）
        item_name_score = {}
        for search_result in search_results:
            # 1. 获取matches
            matches = search_result.get('matches')
            for m in matches:
                score = m.get('score')
                item_name = m.get('item_name')
                if item_name in confirmed:
                    item_name_score[item_name] = max(item_name_score.get(item_name) or 0, score)

        # 2. 对item_name_score进行排序
        sorted_item_name_score = sorted(item_name_score.items(), key=lambda x: x[1], reverse=True)

        # 3. 取出分数值最大的（问题询问的比较明确）
        max_item_name_score = sorted_item_name_score[0][1]
        return [name for name, score in item_name_score.items() if max_item_name_score - score <= SCORE_GAP_THRESHOLD]


class ItemNameExtractor:
    """
     基于用户的原始问题+【用户的历史对话】提取用户真正想问的商品名
     询问场景：（单级询问）请问RS12-万用表如何测量电阻--->LLM---->商品名：[RS12万用表,万用表测量电阻（假的）:但是有可能会进入到confirm中去]
     询问场景：（多级循环） 请问RS12-万用表和RS-13万用表分别如何测量电阻。---->>LLM---->商品名：[RS12-万用表,RS-13万用表]---confirm[RS12-万用表，,RS-13万用表]
     询问场景：（多级循环） 请问RS12-万用表和RS-13万用表分别如何测量电阻。---->>LLM---->商品名：[RS12-万用表,RS-13万用表，RS-DDD测量电阻]---confirm[RS12-万用表，,RS-13万用表,RS-DDD测量电阻:误判不能留]
    """

    def extract_item_name(self, original_query: str, history_text: str) -> Dict[str, Any]:
        """
        LLM根据用户原始问题提取商品名
        Args:
            original_query:
            history_text: 历史对话文本

        Returns:

        """

        result: Dict[str, Any] = {"item_names": [], "rewritten_query": original_query}

        # 1. 获取LLM客户端
        llm_client = get_llm_client(response_format=True)
        if llm_client is None:
            return result

        # 2. 定义提示词(用户级别的)
        human_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(history_text=history_text if history_text else "暂无上下文",
                                                         query=original_query)
        system_prompt = "你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"

        # 3. LLM调用
        llm_response = llm_client.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        llm_content = llm_response.content.strip()
        # 4. 判断LLM的输出
        if not llm_content.strip():
            return result
        try:
            # 5. 清洗和解析
            parsed_result = self._clean_parse(llm_content)
            result["rewritten_query"] = parsed_result.get("rewritten_query") or original_query
            result["item_names"] = parsed_result.get("item_names")

        except Exception as e:
            logger.error(f"清洗以及解析LLM的输出失败: {str(e)}")

        return result

    def _clean_parse(self, llm_response: str) -> Dict[str, Any]:

        # 1. 清洗json代码块围栏 + 反序列化
        try:
            parsed_llm_result: Dict[str, Any] = json.loads(strip_json_fence(llm_response))
        except JSONDecodeError as e:
            raise ValueError(f"JSON反序列LLM的输出失败：{str(e)}")

        # 2.1 清洗item_names
        raw_item_names = parsed_llm_result.get('item_names')
        if not isinstance(raw_item_names, list):
            clean_item_names = []
        else:
            clean_item_names = [raw_item for raw_item in raw_item_names if isinstance(raw_item, str) and raw_item.strip()]

        # 2.2 清洗rewritten_query
        raw_rewritten_query = parsed_llm_result.get('rewritten_query')
        clean_rewritten_query = "" if not isinstance(raw_rewritten_query, str) else raw_rewritten_query.strip()

        return {"item_names": clean_item_names, "rewritten_query": clean_rewritten_query}
