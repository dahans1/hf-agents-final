from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_ollama import ChatOllama

tools = []
llm = ChatOllama(model="qwen3:32b")
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    textual_description_of_toools="""
    todo: add search_engine, and wiki_search
"""
    # system prompt provided to ensure agent answer is using the correct and expected format
    sys_message = f"You are a general AI assistant with provided tools:\n{textual_description_of_tools} \n I will ask you a question. Report your thoughts, and finish your answer with the following template: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."
    sys_msg = SystemMessage(content=sys_message)

    return {
        "messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]
    }

def build_graph():
    builder = StateGraph(AgentState)

    # Define nodes: these do the work
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    return builder.compile()