import asyncio

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from src.graph.configuration import Context
from src.config.configs import config_agents
from src.graph.nodes import supervisor_node
from src.graph.state import InputState, State

async def init_root_graph():
    builder = StateGraph(state_schema=State, input_schema=InputState, context_schema=Context)
    
    builder.add_node(config_agents.PROFILE_GRAPH.AGENT_SUPERVISOR, supervisor_node)
    builder.set_entry_point(config_agents.PROFILE_GRAPH.AGENT_SUPERVISOR)
    
    config_trace = RunnableConfig()
    root_graph = builder.compile(name=config_agents.PROFILE_GRAPH.GRAPH_NAME).with_config(config=config_trace)
    return root_graph

graph = asyncio.run(init_root_graph())
