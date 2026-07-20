import copy
from typing import List

from langchain.agents import create_agent
from langchain.agents.middleware import ContextEditingMiddleware ,ClearToolUsesEdit, after_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.constants import END
from langgraph.runtime import Runtime

from src.config.configs import config_object
from src.graph.client_tools import mcp_client
from src.graph.state import State
from src.log import logger
from src.graph.configuration import Context
from src.graph.middlewares.simple_middlewares import inject_config_prompt, ToolStandardizationMiddleware
from src.graph.utils import load_model, load_sys_prompt


class BaseAgent:

    def __init__(self,
                 system_prompt_path: str,
                 runtime: Runtime[Context]):
        self.agent = None
        self.system_prompt_path = system_prompt_path
        self.model: BaseChatModel = load_model(runtime.context.llm_config)
        self.system_prompt = load_sys_prompt(self.system_prompt_path)

    def init_agent(self, response_class=None, tools=None, middlewares: list = None):
        if tools is None:
            tools = []
        if middlewares is None:
            middlewares = []
        middlewares.extend([inject_config_prompt,
                            ContextEditingMiddleware(
                                edits=[
                                    ClearToolUsesEdit(
                                        trigger=1000
                                    ),
                                ]
                            )])
        agent = create_agent(
            self.model,
            tools=tools,
            system_prompt=self.system_prompt,
            middleware=middlewares,
            response_format=response_class
        )
        return agent

    @staticmethod
    async def init_tools(url_mcp_servers: str, required_tools: list = None):
        tools_all = await mcp_client.get_tools(url=url_mcp_servers,
                                               transport=config_object.MCP.BASE_MCP_SERVER_TRANSPORT)
        if required_tools is None:
            return tools_all
        tools_temp = []
        for tool in tools_all:
            if tool.name in required_tools:
                tools_temp.append(tool)
        return tools_temp

    async def setup_agent(self,
                          response_class=None,
                          use_only_response_structured: bool = False,
                          url_mcp_servers: str = None,
                          middlewares: list = None,
                          required_tools: list = None):
        tools = []
        if middlewares is None:
            middlewares = []
        if url_mcp_servers is not None:
            tools = await self.init_tools(url_mcp_servers=url_mcp_servers, required_tools=required_tools)
        self.response_class = copy.deepcopy(response_class)
        if not use_only_response_structured and self.response_class is not None:
            @after_agent
            async def enforce_structured_wrapper(state: State, runtime: Runtime, **kwargs):
                config = kwargs.get('config')
                return await self.enforce_structured_output(state, runtime, config=config)

            middlewares.append(enforce_structured_wrapper)
            response_class = None

        self.agent = self.init_agent(response_class, tools, middlewares)
        return self.agent

    async def invoke(self, state, runtime, config=None):
        pass

    async def ainvoke(self, state, runtime, config=None):
        pass

    async def astream(self, state, runtime, config=None):
        pass

    @classmethod
    async def init_node(cls, state: State, config: RunnableConfig):
        pass

    async def enforce_structured_output(self, state: State, runtime: Runtime, config=None):
        messages: List[BaseMessage] = state["messages"]
        llm = load_model(runtime.context.llm_config).with_structured_output(self.response_class)
        response = await llm.ainvoke(messages, config=config)
        next_node = response.next_node.upper()
        next_node = END if next_node == 'END' else next_node

        response_json = response.model_dump()

        structured_output = {
            **response_json,
            "next_node": next_node
        }
        logger.info(f"--- NEXT NODE {next_node} FROM GeneralInfo---")
        return structured_output
