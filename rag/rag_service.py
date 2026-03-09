"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
核心逻辑：RAG（检索增强生成）→ 检索相关文档 → 拼接上下文+问题 → 大模型生成总结回复
"""
from langchain_core.documents import Document  # LangChain文档对象（存储文本+元数据）
from langchain_core.output_parsers import StrOutputParser  # 模型输出解析器（转为字符串）

# 导入项目内部依赖
from model.factory import chat_model  # 大语言模型实例（如ChatGLM/OpenAI等）
from rag.vector_store import VectorStoreService  # 向量库服务（用于检索相关文档）
from utils.prompt_loader import load_rag_prompts  # 加载RAG专用提示词模板
from langchain_core.prompts import PromptTemplate  # LangChain提示词模板类


def print_prompt(prompt):
    """辅助函数：打印拼接后的提示词（调试用）"""
    print("=" * 20)
    print(prompt.to_string())  # 输出提示词完整内容
    print("=" * 20)
    return prompt  # 返回prompt，保证链式调用不中断


class RagSummarizeService(object):
    """RAG总结服务类：实现「检索→拼接上下文→模型生成」的完整RAG流程"""

    def __init__(self):
        """初始化：加载向量检索器、提示词模板、大模型，构建调用链"""
        # 1. 初始化向量库服务（用于检索相关参考文档）
        self.vector_store = VectorStoreService()
        # 2. 获取向量检索器（相似性查询）
        self.retriever = self.vector_store.get_retriever()
        # 3. 加载RAG提示词模板（从配置/文件中读取）
        self.prompt_text = load_rag_prompts()
        # 4. 构建提示词模板对象（支持变量替换：input=问题，context=参考文档）
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        # 5. 加载大语言模型实例
        self.model = chat_model
        # 6. 初始化RAG调用链（提示词模板→打印→模型→输出解析）
        self.chain = self.__init_chain()

    def __init_chain(self):
        """私有方法：构建LangChain链式调用流程
        流程：提示词模板渲染 → 打印提示词 → 模型生成 → 输出转为字符串
        """
        chain = (
                self.prompt_template  # 第一步：渲染提示词（替换input和context变量）
                | print_prompt       # 第二步：打印渲染后的提示词（调试）
                | self.model         # 第三步：调用大模型生成回复
                | StrOutputParser()  # 第四步：将模型输出转为字符串
        )
        return chain

    def retriever_docs(self, query: str) -> list[Document]:
        """根据用户提问检索相关参考文档
        param query: 用户的问题字符串
        return: 相似的Document对象列表（包含文本内容+元数据）
        """
        return self.retriever.invoke(query)

    def rag_summarize(self, query: str) -> str:
        """核心方法：执行RAG总结流程，返回模型生成的总结回复
        param query: 用户的问题字符串
        return: 模型总结后的回复文本
        """
        # 1. 检索与问题相关的参考文档
        context_docs = self.retriever_docs(query)
        # 2. 拼接参考文档为上下文字符串（带编号+元数据，便于模型参考）
        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"[参考资料{counter}]: {doc.page_content}| 参考元数据: {doc.metadata}\n"
        # 3. 调用链式流程：传入问题+上下文，获取模型总结回复
        return self.chain.invoke(
            {
                "input": query,    # 用户提问
                "context": context # 检索到的参考文档上下文
            }
        )


# 测试代码：验证RAG总结服务功能
if __name__ == '__main__':
    # 实例化RAG总结服务
    rag_service = RagSummarizeService()
    # 调用总结方法，传入测试问题，打印结果
    print(rag_service.rag_summarize("小户型适合哪种扫地机器人？"))