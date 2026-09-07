from langgraph.graph import StateGraph, END

from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.nodes.pdf_to_md_node import PdfToMdNode
from knowledge.processor.import_process.nodes.entry_node import EntryNode
from knowledge.processor.import_process.nodes.md_img_node import MarkDownImgNode
from knowledge.processor.import_process.nodes.document_split_node import DocumentSplitNode
from knowledge.processor.import_process.nodes.chunk_clean_node import ChunkCleanNode
from knowledge.processor.import_process.nodes.item_name_recognition_node import ItemNameRecognitionNode
from knowledge.processor.import_process.nodes.bge_embedding_chunks_node import BgeEmbeddingChunksNode
from knowledge.processor.import_process.nodes.import_milvus_node import ImportMilvusNode
from knowledge.processor.import_process.nodes.knowledge_graph_node import KnowledgeGraphNode
from knowledge.processor.import_process.nodes.child_chunk_node import ChildChunkNode
from knowledge.processor.import_process.nodes.docx_to_md_node import DocxToMdNode
from knowledge.processor.import_process.nodes.html_to_md_node import HtmlToMdNode


def import_router(state: ImportGraphState):
    if state.get('is_md_read_enabled'):
        return 'md_img_node'
    if state.get('is_pdf_read_enabled'):
        return 'pdf_to_md_node'
    if state.get('is_html_enabled'):
        return 'html_to_md_node'
    if state.get('is_docx_enabled'):
        return 'docx_to_md_node'
    return END


def create_import_graph() -> StateGraph:
    """
    定义导入业务的graph状态拓普谱(Langgraph构建流水线)整个流水线各个节点要读取或者写入的节点。
    Returns:

    """

    # 1.定义状态图
    graph_pipeline = StateGraph(ImportGraphState)  # type:ignore

    # 2.定义节点(入口、结束节点、自己需要添加的)
    # 2.1定义入口节点
    graph_pipeline.set_entry_point('entry_node')

    # 2.2添加剩下的节点
    nodes = {
        'entry_node': EntryNode(),
        'pdf_to_md_node': PdfToMdNode(),
        'html_to_md_node': HtmlToMdNode(),
        'docx_to_md_node': DocxToMdNode(),
        'md_img_node': MarkDownImgNode(),
        'document_split_node': DocumentSplitNode(),
        # P0-2/P0-3：清洗须在商品名抽取之前，否则"前 K 块"会被目录页/安全套话占满
        'chunk_clean_node': ChunkCleanNode(),
        'item_name_recognition_node': ItemNameRecognitionNode(),
        'bge_emdedding_node': BgeEmbeddingChunksNode(),
        'import_milvus_node': ImportMilvusNode(),
        'child_chunk_node': ChildChunkNode(),
        'knowledge_graph_node': KnowledgeGraphNode(),
    }
    for key, value in nodes.items():
        graph_pipeline.add_node(key, value)

    # 3.定义边(顺序边、条件边)
    # source: 路由开始节点
    # path: 路由函数
    # path_map: 路由函数的映射
    graph_pipeline.add_conditional_edges('entry_node',
                                         import_router,
                                         {
                                             'md_img_node': 'md_img_node',
                                             'pdf_to_md_node': 'pdf_to_md_node',
                                             'html_to_md_node': 'html_to_md_node',
                                             'docx_to_md_node': 'docx_to_md_node',
                                             END: END
                                         }
                                         )
    graph_pipeline.add_edge('pdf_to_md_node', 'md_img_node')
    graph_pipeline.add_edge('html_to_md_node', 'md_img_node')
    graph_pipeline.add_edge('docx_to_md_node', 'md_img_node')
    graph_pipeline.add_edge('md_img_node', 'document_split_node')
    graph_pipeline.add_edge('document_split_node', 'chunk_clean_node')
    graph_pipeline.add_edge('chunk_clean_node', 'item_name_recognition_node')
    graph_pipeline.add_edge('item_name_recognition_node', 'bge_emdedding_node')
    graph_pipeline.add_edge('bge_emdedding_node', 'import_milvus_node')
    graph_pipeline.add_edge('import_milvus_node', 'child_chunk_node')
    graph_pipeline.add_edge('child_chunk_node', 'knowledge_graph_node')
    graph_pipeline.add_edge('knowledge_graph_node', END)

    # 4.编译(编排)
    return graph_pipeline.compile()


import_graph_app = create_import_graph()
