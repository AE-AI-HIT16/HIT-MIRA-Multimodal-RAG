from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from src.graph.configuration import Context
from src.config.configs import config_agents
from src.graph.nodes import supervisor_node
from src.graph.state import InputState, State


def init_root_graph():
    """Dựng và biên dịch đồ thị gốc.

    Hàm này đồng bộ có chủ đích. Bản trước khai báo `async` rồi chạy
    `asyncio.run(...)` ngay lúc import, nên mọi tiến trình đang có vòng lặp
    sự kiện (FastAPI, notebook, hoặc chỉ là một script `asyncio.run`) đều
    chết ngay khi import với "asyncio.run() cannot be called from a running
    event loop". Thân hàm không `await` gì cả, nên bỏ `async` là đủ.
    """
    builder = StateGraph(state_schema=State, input_schema=InputState, context_schema=Context)

    builder.add_node(config_agents.PROFILE_GRAPH.AGENT_SUPERVISOR, supervisor_node)
    builder.set_entry_point(config_agents.PROFILE_GRAPH.AGENT_SUPERVISOR)

    config_trace = RunnableConfig()
    root_graph = builder.compile(name=config_agents.PROFILE_GRAPH.GRAPH_NAME).with_config(config=config_trace)
    return root_graph


graph = init_root_graph()
