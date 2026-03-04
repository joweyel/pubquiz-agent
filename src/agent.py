from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from src.tools import vector_database_search_tool, web_search_tool

load_dotenv()

llm = init_chat_model(model="gpt-5.2", model_provider="openai")


system_prompt = """You are a pub quiz assistant competing in a quiz in Heidelberg, Baden-Württemberg, Germany.

## Answer Style
- Keep answers as short as possible, single words or short phrases preferred.
- Example: Q: "Who is the current German chancellor?" A: "Friedrich Merz"
- Never explain your reasoning in the final answer.

## Tool Usage
- For questions about German criminal law, Romeo and Juliet, The Gift of the Magi, or speeches by German government officials: use the database tool first.
- For all other factual or current-events questions: search the web.
- Combine multiple tools if needed.
- Verify that the answer you came to is absolutely correct.
- If no tool gives a result, use your own knowledge. If still unsure, make your best guess - never say "I don't know".
- Always validate the correctness of the answers.
- Numerical answers that are not full integers have to be represented as floating point values.

## Special Cases
- Fictional characters/settings: answer from an in-universe perspective. (Q: "When was Harry Potter born?" A: "31 July 1980")
- Geography: "largest" means by population unless stated otherwise.
- If multiple correct answers exist, give the most common one.
- Some questions require multiple steps, read carefully before answering.

Good performance will be rewarded with a tip. Thanks :)"""

tools = [
    vector_database_search_tool,
    web_search_tool,
]
agent = create_agent(llm, tools=tools, system_prompt=system_prompt)


def run_llm(query: str, chat_memory: list[dict[str, str]]) -> dict[str, Any]:
    """Run RAG pipeline to answer a question regarding different topics.

    Args:
        query: Question asked by the user.
        chat_memory: The entire chat history to have the whole chat context.

    Returns:
        Dictionary containing:
        - answer: The answer to the input query
        - sources: Sources where data was retrieved from.
    """

    messages = chat_memory + [{"role": "user", "content": query}]

    response = agent.invoke({"messages": messages})
    answer = response["messages"][-1].content

    # Extract sources from the answer
    sources: list = []
    tool_calls: list = []
    for msg in response["messages"]:
        # Save the tool calls to see what was used
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_calls.append(tool_call["name"])
        if isinstance(msg, ToolMessage) and hasattr(msg, "artifact"):
            artifact = msg.artifact
            if isinstance(artifact, list):
                for item in artifact:
                    # Tavily output parsing
                    if isinstance(item, dict) and "url" in item:
                        sources.append(item["url"])
                    # ChromaDB retrieval
                    elif hasattr(item, "metadata"):
                        if source := item.metadata.get("source"):
                            sources.append(source)

    return {"answer": answer, "sources": sources, "tools_used": tool_calls}
