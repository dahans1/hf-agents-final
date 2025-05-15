from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_ollama import ChatOllama

from langchain_community.tools import DuckDuckGoSearchRun

def web_search(query: str) -> str:
    """
    Use this to search the web via DuckDuckGo
    """

    return DuckDuckGoSearchRun().invoke(query)

tools = [web_search]
llm = ChatOllama(model="qwen3:32b")
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    textual_description_of_tools="""
    web_search(query: str) -> str:
        Use this to search the web via DuckDuckGo
"""
    # system prompt provided to ensure agent answer is using the correct and expected format
    sys_message = f"You are a general AI assistant with provided tools:\n{textual_description_of_tools} \n I will ask you a question. Analyze the question and if you find an answer, your response should only have the following template: [YOUR FINAL ANSWER]. Do NOT include the opening and closing square brackets. Only include what's inside the square brackets. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."
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

# test agent for expected answer format
if __name__ == "__main__":
    question = "What country had the least number of athletes at the 1928 Summer Olympics? If there's a tie for a number of athletes, return the first in alphabetical order. Give the IOC country code as your answer."
    print(f"Answering question...{question}")

    graph = build_graph()
    messages = [HumanMessage(content=question)]
    messages = graph.invoke({"messages": messages})
    answer = messages["messages"][-1].content

    if "</think>" in answer:
        answer = answer.split("</think>", 1)[1].strip()
    else:
        answer = answer.strip()
    print(answer)