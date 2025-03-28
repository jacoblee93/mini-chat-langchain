from typing import Optional

from e2b_code_interpreter import Sandbox

from langgraph_reflection import create_reflection_graph
from langgraph.graph import MessagesState

from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model

from mini_chat_langchain.base import create_base_agent
from mini_chat_langchain.judge import create_judge_graph


_GLOBAL_SANDBOX = None


def get_or_create_sandbox():
    global _GLOBAL_SANDBOX
    if _GLOBAL_SANDBOX is None:
        _GLOBAL_SANDBOX = Sandbox("OpenEvalsPython")
    return _GLOBAL_SANDBOX


def create_reflection_agent(
    config: RunnableConfig,
):
    configurable = config.get("configurable", {})
    sandbox = configurable.get("sandbox", None)
    model = configurable.get("model", None)
    if sandbox is None:
        sandbox = get_or_create_sandbox()
    if model is None:
        model = init_chat_model(
            model="anthropic:claude-3-5-haiku-latest",
            max_tokens=4096,
        )
    judge = create_judge_graph(sandbox)
    return (
        create_reflection_graph(create_base_agent(model), judge, MessagesState)
        .compile()
        .with_config(run_name="Mini Chat LangChain")
    )
