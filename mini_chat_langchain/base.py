import requests

from langgraph.prebuilt import create_react_agent

from langchain_core.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel

from bs4 import BeautifulSoup

SYSTEM_PROMPT = """
You are an expert software engineer and technical writer.

When asked questions about LangGraph, you should research deeply into it.
Use the get_langgraph_docs_index tool to get an index of the LangGraph docs first,
then follow up with the get_request tool. Be persistent - if your first page does
not result in confident information, keep digging!

If you generate Python code to help answer a question, make sure it is correct, complete, and executable
without modification. Make sure that any generated code is contained in a properly formatted markdown code block.

You can use the following URLs with your "get_langgraph_docs_content" tool to help answer questions:

{langgraph_llms_txt}
"""


@tool
def get_langgraph_docs_content(url: str) -> str:
    """Sends a get request to a webpage and returns plain text
    extracted via BeautifulSoup."""
    res = requests.get(url).text
    soup = BeautifulSoup(res, features="html.parser")
    return soup.get_text()


def create_base_agent(model: BaseChatModel):
    langgraph_llms_txt = requests.get(
        "https://langchain-ai.github.io/langgraph/llms.txt"
    ).text
    return create_react_agent(
        model=model,
        tools=[get_langgraph_docs_content],
        prompt=SYSTEM_PROMPT.format(langgraph_llms_txt=langgraph_llms_txt),
    ).with_config(run_name="Base Agent")
