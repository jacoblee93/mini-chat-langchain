from typing import Optional

from e2b_code_interpreter import Sandbox

from langgraph_reflection import create_reflection_graph
from langgraph.graph import MessagesState

from langchain_core.language_models.chat_models import BaseChatModel
from langchain.chat_models import init_chat_model

from mini_chat_langchain.base import create_base_agent
from mini_chat_langchain.judge import create_judge_graph


def create_reflection_agent(*, sandbox: Sandbox, model: Optional[BaseChatModel] = None):
    if model is None:
        model = init_chat_model(
            model="anthropic:claude-3-5-haiku-latest",
        )
    judge = create_judge_graph(sandbox)
    return (
        create_reflection_graph(create_base_agent(model), judge, MessagesState)
        .compile()
        .with_config(run_name="Mini Chat LangChain")
    )
