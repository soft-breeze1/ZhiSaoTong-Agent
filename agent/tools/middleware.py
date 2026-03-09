from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from typing import Callable

from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from utils.logger_handler import logger
from utils.prompt_loader import load_report_prompts, load_system_prompts


@wrap_tool_call
def monitor_tool(
        request: ToolCallRequest,  # 工具调用请求的数据封装
        handler: Callable[[ToolCallRequest], ToolMessage | Command],  # 工具执行的函数本身
) -> ToolMessage | Command:  # 工具执行的监控软件

    # 自定义工具执行逻辑
    logger.info(f"[monitor_tool]工具调用:{request.tool_call['name']}")
    logger.info(f"[monitor_tool]传入参数:{request.tool_call['args']}")
    try:
        result = handler(request)
        logger.info(f"[monitor_tool]工具:{request.tool_call['name']}调用成功")

        # 添加报告标记,只要模型调用填写报告，则添加报告标记
        if request.tool_call['name'] == 'fill_context_for_report':
            request.runtime.context['report'] = True

        return result
    except Exception as e:
        logger.error(f"[monitor_tool]工具:{request.tool_call['name']}调用失败,原因:{str(e)}")
        raise e


@before_model
def log_before_model(
        state: AgentState,  # 整个Agent智能体中的状态记录
        runtime: Runtime,  # 记录了整个执行过程中的上下文信息
):  # 模型执行前输出日志
    logger.info(f'[log_before_model]即将调用模型，带有{len(state["messages"])}条信息')
    logger.debug(
        f"[log_before_model]当前消息记录:{type(state['messages'][-1]).__name__} | "
        f"内容:{state['messages'][-1].content.strip()}")

    return None


@dynamic_prompt  # 每次生成提示词之前，调用此函数
def report_prompt_switch(request: ModelRequest):  # 动态切换提示词
    is_report = request.runtime.context.get('report', False)
    if is_report:  # 返回报告生成提示词
        return load_report_prompts()
    return load_system_prompts()








