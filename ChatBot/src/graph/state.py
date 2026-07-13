"""Define the state structures for the agent."""

from __future__ import annotations

from typing import Optional, List, Dict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field
from typing_extensions import Annotated
from src.config.configs import config_object


class InputState(BaseModel):
    """Defines the input state for the agent, representing a narrower interface to the outside world."""
    messages: Annotated[List[AnyMessage], add_messages] = Field(default=list)
    url_mcp_server: Optional[str] = config_object.MCP.BASE_MCP_SERVER_URL


class State(InputState):
    """Represents the complete state of the agent, extending InputState with additional attributes."""
    memories: Optional[str] = None
    next_node: Optional[str] = None
    current_node: Optional[str] = None
    attachments: Optional[list] = None