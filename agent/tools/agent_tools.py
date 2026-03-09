import os
import random
from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
from utils.config_handler import agent_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

# 实例化 RAG 服务，用于后续操作向量库进行知识检索
rag = RagSummarizeService()

# 预设的模拟用户 ID 列表，用于提供随机测试数据
user_ids = ['1001', '1002', '1003', '1004', '1005', '1006', '1007', '1008', '1009', '1010']

# 预设的模拟月份列表，用于提供随机测试数据
month_arr = ['2025-01', '2025-02', '2025-03', '2025-04', '2025-05', '2025-06', '2025-07', '2025-08', '2025-09',
             '2025-10',
             '2025-11', '2025-12']

# 全局变量，用于缓存外部系统解析后的用户数据（避免每次查询都重新读取文件）
external_data = {}


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    """调用 RAG 服务，根据用户的查询 query 去知识库检索相关的文档并返回"""
    return rag.rag_summarize(query)


@tool(description="获取指定城市的天气，以消息字符串形式返回")
def get_weather(city: str) -> str:
    """模拟获取天气的接口，固定返回一段预设的天气信息"""
    return f"城市{city}天气为晴天，气温为25度，空气湿度为60%，南风1级，AQI21，最近6小时降雨概率极低。"


@tool(description="获取用户所在城市信息，以消息字符串形式返回")
def get_user_location() -> str:
    """模拟获取当前用户所在的城市，从城市列表中随机抽取一个返回"""
    return random.choice(
        ["北京", "上海", "广州", "深圳", "杭州", "西安", "武汉", "南京", "成都", "苏州", "无锡", "厦门", "福州", "青岛",
         "济南", "郑州", "太原", "武汉", ])


@tool(description="获取用户的ID，以春字符串形式返回")
def get_user_id() -> str:
    """模拟获取当前用户的 ID，从预设的 user_ids 列表中随机返回一个"""
    return random.choice(user_ids)


@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month() -> str:
    """模拟获取当前时间的月份，从预设的 month_arr 列表中随机返回一个"""
    return random.choice(month_arr)


def generate_external_data():
    """
    加载并解析外部数据（CSV/TXT）文件，将数据装载到全局字典 external_data 中。
    数据结构如下：
    {
        "user_id":{
            "month":{"特征“:xxx,"效率”:xxx,...},
            "month":{"特征“:xxx,"效率”:xxx,...},
            "month":{"特征“:xxx,"效率”:xxx,...},
        },
        "user_id":{
            "month":{"特征“:xxx,"效率”:xxx,...},
            "month":{"特征“:xxx,"效率”:xxx,...},
            "month":{"特征“:xxx,"效率”:xxx,...},
        },
        ...
    }
    :return: 无返回值，直接修改全局变量 external_data
    """
    # 判空：如果字典为空，说明是第一次调用，需要读取文件加载数据
    if not external_data:
        # 拼接并获取外部数据文件的绝对路径
        external_data_path = get_abs_path(agent_conf['external_data_path'])

        # 如果指定路径的文件不存在，抛出异常阻断程序
        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"[generate_external_data]外部数据文件不存在:{external_data_path}")

        # 以 UTF-8 编码打开文件
        with open(external_data_path, 'r', encoding='utf-8') as fr:
            # 跳过第一行（通常是表头），逐行遍历读取
            for line in fr.readlines()[1:]:
                # 去除换行符并用逗号分割为列表
                arr: list[str] = line.strip().split(',')

                # 提取数据内容，并清理多余的双引号
                user_id: str = arr[0].replace('"', '')
                feature: str = arr[1].replace('"', '')
                efficiency: str = arr[2].replace('"', '')
                consumables: str = arr[3].replace('"', '')
                comparison: str = arr[4].replace('"', '')
                time: str = arr[5].replace('"', '')

                # 如果当前处理的用户ID还不在全局字典中，为它初始化一个空字典
                if user_id not in external_data:
                    external_data[user_id] = {}

                # 将具体的指标字段挂载到对应的 user_id 和 month(time) 下
                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }


@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回，如果没有检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    """查询指定用户在特定月份的数据，给大模型工具调用"""
    # 调用数据生成函数，确保数据已经成功加载到内存（字典中已有数据时会直接跳过）
    generate_external_data()

    try:
        # 尝试返回目标用户在目标月份的数据结构字典
        return external_data[user_id][month]
    except KeyError:
        # 捕获字典键值不存在的异常（即没有该用户或没有该月数据），记录日志并返回空字符串
        logger.warning(f"[fetch_external_data]用户:{user_id} 在月份:{month} 没有使用记录")
        return ""


@tool(
    description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    """配合 Agent 的一个标记性工具，用于触发系统底层的上下文注入中间件"""
    return "fill_context_for_report已调用"


if __name__ == '__main__':
    # 之前报错留下的测试代码（如果后续要测试工具，记得使用 .invoke({"user_id": "1001", "month": "2025-01"})）
    # print(fetch_external_data("1001", "2025-01"))
    pass