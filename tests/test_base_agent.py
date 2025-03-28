import pytest
from langsmith import testing as t

from langchain.chat_models import init_chat_model

from mini_chat_langchain.base import create_base_agent

from e2b_code_interpreter import Sandbox
from openevals.code.e2b.pyright import create_e2b_pyright_evaluator

@pytest.fixture
def sandbox():
    sandbox = Sandbox("OpenEvalsPython")
    yield sandbox


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "input",
    [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "How do I convert a LangChain agent to a LangGraph agent?"
                }
            ]
        }
    ],
)
def test_codegen(input, sandbox):
    model = init_chat_model(  
        model="anthropic:claude-3-5-haiku-latest",
    )
    agent = create_base_agent(model)
    outputs = agent.invoke(input)
    t.log_outputs(outputs)
    evaluator = create_e2b_pyright_evaluator(
        sandbox=sandbox,
        code_extraction_strategy="markdown_code_blocks",
    )
    eval_result = evaluator(outputs=outputs)
    print(eval_result)
