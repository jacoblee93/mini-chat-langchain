import pytest
from langsmith import testing as t

from mini_chat_langchain import create_reflection_agent

from e2b_code_interpreter import Sandbox
from openevals.code.e2b.pyright import create_e2b_pyright_evaluator


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "input",
    [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "How do I convert a LangChain agent to a LangGraph agent?",
                }
            ]
        }
    ],
)
def test_codegen(input):
    sandbox = Sandbox("OpenEvalsPython")
    agent = create_reflection_agent(sandbox=sandbox)
    outputs = agent.invoke(input)
    print(outputs["messages"][-1].content)
    t.log_outputs(outputs)
    evaluator = create_e2b_pyright_evaluator(
        sandbox=sandbox,
        code_extraction_strategy="markdown_code_blocks",
    )
    eval_result = evaluator(outputs=outputs)
    print(eval_result)
