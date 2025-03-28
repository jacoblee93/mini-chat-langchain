from e2b_code_interpreter import Sandbox

from langgraph.graph import StateGraph, MessagesState
from openevals.code.e2b.pyright import create_e2b_pyright_evaluator


def create_judge_graph(sandbox: Sandbox):
    def run_reflection(state: dict) -> dict | None:
        evaluator = create_e2b_pyright_evaluator(
            sandbox=sandbox,
            code_extraction_strategy="markdown_code_blocks",
        )

        result = evaluator(outputs=state)

        code_extraction_failed = result["metadata"] and result["metadata"].get(
            "code_extraction_failed"
        )

        if not result["score"] and not code_extraction_failed:
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": f"I ran pyright and found some problems with the code you generated: {result['comment']}\n\n"
                        "Try to fix it. Make sure to regenerate the entire code snippet. "
                        "If you are not sure what is wrong, search for more information by pulling more information "
                        "from the LangGraph docs.",
                    }
                ]
            }

    return (
        StateGraph(MessagesState)
        .add_node("run_reflection", run_reflection)
        .add_edge("__start__", "run_reflection")
        .compile()
    ).with_config(run_name="Judge Agent")
