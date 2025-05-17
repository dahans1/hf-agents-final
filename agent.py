from typing import Annotated, TypedDict
import os
import base64
import io
import requests
import pandas as pd

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_ollama import ChatOllama

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import ArxivLoader, WikipediaLoader

from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL

from faster_whisper import WhisperModel
from pytubefix import YouTube
import torch

def web_search(query: str) -> str:
    """
    Search the web via DuckDuckGo
    """
    search_docs = DuckDuckGoSearchResults().invoke(query)
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"web_results": formatted_search_docs}

def wiki_search(query: str) -> str:
    """
    Search Wikapedia articles and return the most relevant article.
    """
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])
    return {"wiki_results": formatted_search_docs}

def arxiv_search(query: str) -> str:
    """
    Search arxiv articles and return a maximum of 3 results.
    """
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in search_docs
        ])
    return {"arvix_results": formatted_search_docs}

def analyze_excel(file_url: str) -> str:
    """
    Open an Excel file and read its contents

    Returns: the excel file in a readable format. Use it for analyzation.
    """
    try:
        resp = requests.get(file_url)
        resp.raise_for_status()

        excel = resp.content
        excel_file = io.BytesIO(excel)

        dfs = pd.read_excel(excel_file, sheet_name=None)
        texts = []
        for name, df in dfs.items():
            texts.append(f"### Sheet: {name}")
            texts.append(df.to_csv(index=False))
        
        return "\n\n".join(texts)
    except Exception as e:
        return f"Error analyzing Excel file: {str(e)}"

def analyze_code_file(file_url: str) -> str:
    """
    Open a Python file or other text files and convert it to text.

    Returns: the python file in text form. Use it for analyzation.
    """
    try:
        resp = requests.get(file_url, timeout=15)
        resp.raise_for_status()

        return resp.text
    except Exception as e:
        print(f"Failed to download file: {str(e)}")

    
def analyze_audio(audio_url: str) -> str:
    """
    Use an audio LLM that can interpret MP3 files to get the context
    """
    try:
        resp = requests.get(audio_url, timeout=15)
        resp.raise_for_status()
        
        audio = resp.content
        audio_file = io.BytesIO(audio)

        model = WhisperModel("large-v2")
        segments, _ = model.transcribe(audio_file)
        
        output = ""
        for segment in segments:
            output += segment.text + "\n"
        return output.strip()
    except Exception as e:
        return f"Error analyzing audio file: {str(e)}"

def analyze_youtube_audio(youtube_link: str) -> str:
    """
    Use this to analyze a YouTube video for audio purposes
    """
    try:
        yt = YouTube(youtube_link)
        stream = yt.streams.filter(only_audio=True).first()
        audio_path = stream.download(output_path="./yt_audio", filename="audio.mp3")

        model = WhisperModel("large-v2")
        segments, _ = model.transcribe(audio_path)

        output = ""
        for segment in segments:
            output += segment.text + "\n"
        return output.strip()
    except Exception as e:
        return f"Error analyzing youtube audio file: {str(e)}"

def analyze_image(img_url: str, question: str) -> str:
    """
    Use a vision LLM to view the image and answer the question for you
    """
    try:
        resp = requests.get(img_url, timeout=15)
        resp.raise_for_status()
        img_bytes = resp.content

        # infer extension
        _, ext = os.path.splitext(img_url)
        # fallback if no ext or query params present
        if not ext or "?" in ext:
            ext = ext.split("?")[0] or ".png"

        image_base64 = base64.b64encode(img_bytes).decode('utf-8')

        vision_llm = ChatOllama(model='qwen2.5vl:32b', temperature=0)
        message = [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": f"Describe the image in great detail. Then, answer the following question:\n {question}"
                    },
                    {
                        "type": "image",
                        "source_type": "base64",
                        "mime_type": f"image/{ext.lstrip('.')}",
                        "data": image_base64,
                    }
                ]
            )
        ]
        resp = vision_llm.invoke(message)
        return resp.content.strip()
    except Exception as e:
        return f"Error extracting text: {str(e)}"

python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)

tools = [
    analyze_audio,
    analyze_code_file,
    analyze_excel,
    analyze_image,
    analyze_youtube_audio,
    arxiv_search,
    web_search,
    wiki_search,
    repl_tool,
]
llm = ChatOllama(model="qwen3:32b", temperature=0)
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    # system prompt provided to ensure agent answer is using the correct and expected format
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        sys_message = f.read()
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
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    # question = "Hi, I\'m making a pie but I could use some help with my shopping list. I have everything I need for the crust, but I\'m not sure about the filling. I got the recipe from my friend Aditi, but she left it as a voice memo and the speaker on my phone is buzzing so I can\'t quite make out what she\'s saying. Could you please listen to the recipe and list all of the ingredients that my friend described? I only want the ingredients for the filling, as I have everything I need to make my favorite pie crust. I\'ve attached the recipe as Strawberry pie.mp3.\n\nIn your response, please only list the ingredients, not any measurements. So if the recipe calls for "a pinch of salt" or "two cups of ripe strawberries" the ingredients on the list would be "salt" and "ripe strawberries".\n\nPlease format your response as a comma separated list of ingredients. Also, please alphabetize the ingredients."
    file_url = "https://www.youtube.com/watch?v=1htKBjuUWec" #"https://agents-course-unit4-scoring.hf.space/files/99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3"
    print(analyze_youtube_audio(file_url))
    # api_url = "https://agents-course-unit4-scoring.hf.space"
    # questions_url = f"{api_url}/questions"
    # files_url = f"{api_url}/files"

    # response = requests.get(questions_url, timeout=15)
    # questions_data = response.json()

    # for item in questions_data:
    #     print(item)
    #     task_id = item.get("task_id")
    #     question_text = item.get("question")

    # # question = "What country had the least number of athletes at the 1928 Summer Olympics? If there's a tie for a number of athletes, return the first in alphabetical order. Give the IOC country code as your answer."
    # # print(f"Answering question...{question}")

    # # print(wiki_search("1928 Summer Olympics"))
    #     if task_id and question_text:
    #         file_url = f"{files_url}/{task_id}"
    #         if "application/json" in requests.get(file_url).headers.get("Content-Type"):
    #             file_url = None
    #         if file_url and '.mp3' in question_text:
    #             print(file_url)
    #             graph = build_graph()

                
    #             messages = [
    #                 HumanMessage(
    #                     content=f"{question_text}\n\n File URL: {file_url}"
    #                 )
    #             ]
    #             messages = graph.invoke({"messages": messages})
    #             answer = messages["messages"][-1].content

    #             print(answer)
    #             if "</think>" in answer:
    #                 answer = answer.split("</think>", 1)[1].strip()
    #             else:
    #                 answer = answer.strip()
    #             print(answer)
    #             break