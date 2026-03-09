import os

# 导入向量数据库相关依赖
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 导入项目内部工具/配置/模型
from utils.config_handler import chroma_conf  # 向量库配置（集合名、存储路径、分片参数等）
from model.factory import chat_model,embed_model  # 嵌入模型实例（文本转向量）

from utils.file_handler import (  # 文件处理工具
    pdf_loader,  # PDF文件加载器（转Document对象）
    txt_loader,  # TXT文件加载器（转Document对象）
    listdir_with_allowed_type,  # 获取指定目录下指定类型的文件列表
    get_file_md5_hex  # 计算文件的MD5值（用于去重）
)
from utils.logger_handler import logger  # 日志工具
from utils.path_tool import get_abs_path  # 路径工具（获取绝对路径）


class VectorStoreService():
    """向量库服务类：负责文档加载、分片、向量化入库、向量检索"""

    def __init__(self):
        """初始化：创建Chroma向量库实例 + 文本分片器实例"""
        # 初始化Chroma向量库（持久化存储、指定嵌入模型）
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],  # 向量库集合名称
            persist_directory=chroma_conf["persist_directory"],  # 向量库持久化目录
            embedding_function=embed_model,  # 文本转向量的嵌入模型
        )

        # 初始化递归字符分片器（处理长文本，保证分片上下文连续）
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],  # 每个分片的最大字符数
            chunk_overlap=chroma_conf["chunk_overlap"],  # 分片间重叠字符数（上下文衔接）
            separators=chroma_conf['separators'],  # 分片分隔符（换行/标点等）
            length_function=len,  # 文本长度计算方式（按字符数）
        )

    def get_retriever(self):
        """获取向量检索器（用于相似性查询）
        return: 检索器对象，可调用invoke(查询文本)获取相似结果
        """
        return self.vector_store.as_retriever(
            search_kwargs={"k": chroma_conf["k"]},  # 检索参数：返回top-k个最相似结果
        )

    def load_document(self):
        """核心方法：加载指定目录的PDF/TXT文件，去重后分片并入库"""

        # 嵌套函数：检查文件MD5是否已存在（判断是否处理过）
        def check_md5_hex(md5_for_check: str):
            # MD5存储文件不存在 → 创建空文件，返回未处理
            if not os.path.exists(get_abs_path(chroma_conf['md5_hex_store'])):
                with open(get_abs_path(chroma_conf['md5_hex_store']), 'w', encoding='utf-8') as f:
                    pass
                return False
            # 读取MD5文件，逐行对比
            with open(get_abs_path(chroma_conf['md5_hex_store']), 'r', encoding='utf-8') as fr:
                for line in fr.readlines():
                    if md5_for_check == line.strip():
                        return True  # MD5存在 → 已处理
                return False  # MD5不存在 → 未处理

        # 嵌套函数：保存文件MD5到本地（标记为已处理）
        def save_md5_hex(md5_for_save: str):
            with open(get_abs_path(chroma_conf['md5_hex_store']), 'a', encoding='utf-8') as f:
                f.write(md5_for_save + '\n')  # 追加写入MD5，换行分隔

        # 嵌套函数：根据文件类型加载为Document对象列表
        def get_file_documents(read_path: str) -> list[Document]:
            if read_path.endswith("txt"):
                return txt_loader(read_path)  # 加载TXT文件
            elif read_path.endswith("pdf"):
                return pdf_loader(read_path)  # 加载PDF文件
            else:
                return []  # 非目标类型返回空

        # 1. 获取指定目录下的PDF/TXT文件列表（绝对路径）
        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf['data_path']),  # 知识库文件目录
            tuple(chroma_conf['allow_knowledge_file_type'])  # 允许的文件类型（PDF/TXT）
        )

        # 2. 遍历文件，逐个处理
        for path in allowed_files_path:
            # 计算文件MD5，用于去重
            md5_hex = get_file_md5_hex(path)
            # MD5已存在 → 跳过该文件
            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]文件: {path} 已处理过,已存在在知识库内，跳过")
                continue

            try:
                # 3. 加载文件为Document对象
                documents: list[Document] = get_file_documents(path)
                if not documents:
                    logger.info(f"[加载知识库] {path} 无有效文本,跳过")
                    continue

                # 4. 文本分片（处理长文本）
                split_documents: list[Document] = self.spliter.split_documents(documents)
                if not split_documents:
                    logger.info(f"[加载知识库]文件: {path} 分片后没有有效内容,跳过")
                    continue

                # 5. 分片文档入库（转向量存储）
                self.vector_store.add_documents(split_documents)
                # 6. 保存MD5，标记为已处理
                save_md5_hex(md5_hex)
                logger.info(f"[加载知识库]文件: {path} 内容加载成功")

            except Exception as e:
                # 捕获异常，记录错误日志（含堆栈），继续处理下一个文件
                logger.error(f"[加载知识库]文件: {path} 加载失败,{str(e)}", exc_info=True)
                continue


# 测试代码：验证文档加载和向量检索功能
if __name__ == '__main__':
    # 实例化向量库服务
    vs = VectorStoreService()
    # 加载文档到向量库
    vs.load_document()
    # 获取检索器
    retriever = vs.get_retriever()
    # 检索与"迷路"相关的文本片段
    res = retriever.invoke("迷路")
    # 打印检索结果
    for r in res:
        print(r.page_content)  # 打印文本内容
        print("-" * 20)  # 分隔线，便于阅读