# Mini Chat LangChain

A minimal agentic implementation of [Chat LangChain](https://chat.langchain.com/) that can answer questions about LangGraph. Uses a small model and no indexing or vectorstores, just LangGraph's [LLMS.txt](https://langchain-ai.github.io/langgraph/llms.txt) file!

![](/static/img/mini-chat-langchain-studio.png)

To improve the correctness of generated code, Mini Chat LangChain verifies the correctness of generated code via a typechecking [OpenEvals](https://github.com/langchain-ai/openevals) evaluator used "in-the-loop". It first extracts generated code from the agent's output, then pushes it up to an [E2B](https://e2b.dev) sandbox that installs required packages and runs [`pyright`](https://github.com/microsoft/pyright) over it. If this check fails, it feeds the logs back to the original agent as a reflection step so that it can fetch more information about code structure as required.

## Installation

First, clone this repo:

```bash
git clone https://github.com/jacoblee93/mini-chat-langchain.git
cd mini-chat-langchain
```

First, copy the `.env.example` file to `.env`, then set the following environment variables:

Next, set the following environment variables:

```bash
export ANTHROPIC_API_KEY="YOUR_KEY_HERE"
export E2B_API_KEY="YOUR_KEY_HERE"
```

You can sign up and obtain an [Anthropic key here](https://www.anthropic.com/) and an [E2B key here](https://e2b.dev).

You can also set up [LangSmith tracing](https://smith.langchain.com) if desired by setting your API key and enabling tracing:

```bash
export LANGSMITH_API_KEY="YOUR_KEY_HERE"
export LANGSMITH_TRACING=true
```

This repo is set up to use [`uv`](https://docs.astral.sh/uv/). Run `uv sync` to install require deps:

```bash
uv sync

# Or, if you prefer not to use uv:
# pip install
```

## Trying it out

You can run `uv run langgraph dev` to open your graph in [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/).

If you do not want to run this project using LangGraph Studio, you will need to run the included agents by importing them as modules.

You can also run experiments with the agent by running the tests in the `tests/` folder.

`tests/test_base_agent.py` runs without the reflection step and grades its output using the typechecking evaluator, while `tests/test_agent.py` runs the full agent.

## Thank you!

This repo is meant to provide inspiration for how running evaluators "in-the-loop" as part of your agent can help improve performance. If you have questions or comments, please open an issue or reach out to us [@LangChainAI](https://x.com/langchainai) on X (formerly Twitter)!
