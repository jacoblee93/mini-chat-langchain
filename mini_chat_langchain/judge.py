from e2b_code_interpreter import Sandbox

from langgraph.graph import StateGraph, MessagesState
from openevals.code.e2b.pyright import create_e2b_pyright_evaluator

def create_judge_graph(sandbox: Sandbox):
    def try_running(state: dict) -> dict | None:
        """Attempt to run and analyze the extracted Python code.

        Args:
            state: The current conversation state

        Returns:
            dict | None: Updated state with analysis results if code was found
        """
        evaluator = create_e2b_pyright_evaluator(
            sandbox=sandbox,
            code_extraction_strategy="markdown_code_blocks",
        )
        result = evaluator(outputs=state["messages"][-1].content)
        
        code_extraction_failed = result["metadata"] and result["metadata"].get("code_extraction_failed")

        if not result["score"] and not code_extraction_failed:
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": f"I ran pyright and found this: {result['comment']}\n\n"
                        "Try to fix it. Make sure to regenerate the entire code snippet. "
                        "If you are not sure what is wrong, search for more information using your tools "
                        "or ask a question rather than generating code",
                    }
                ]
            }

    return (
        StateGraph(MessagesState)
            .add_node("try_running", try_running)
            .add_edge("__start__", "try_running")
            .compile()
    )
