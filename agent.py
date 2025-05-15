from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import add_messages
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3:32b")

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]