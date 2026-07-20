from langgraph.runtime import Runtime


from src.graph.state import State
from src.graph.configuration import Context
from src.graph.agents.supervisor_agent import SupervisorAgent
from src.config.configs import config_prompts


from langchain_core.runnables import RunnableConfig

async def supervisor_node(state: State, runtime: Runtime[Context], config: RunnableConfig = None):
    system_prompt_path = config_prompts.PROFILE_GRAPH.PROMPT_SUPERVISOR_PATH
    agent = SupervisorAgent(system_prompt_path, runtime)
    response = await agent.ainvoke(state=state, runtime=runtime, config=config)
    return response
