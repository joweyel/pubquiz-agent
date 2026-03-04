from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.tools import tool
from langchain_tavily import TavilySearch


from src.ingestion import embeddings

load_dotenv()

vector_store = None


def get_vector_store() -> Chroma:
    global vector_store
    if vector_store is None:
        vector_store = Chroma(
            persist_directory="./PubDatabase/chroma", embedding_function=embeddings
        )
    return vector_store


@tool(response_format="content_and_artifact")
def vector_database_search_tool(query: str, k: int = 4):
    """Search the local knowledge base for questions about German criminal law,
    Romeo and Juliet, The Gift of the Magi, or speeches by German government officials.
    """
    # Get k similar documents
    retrieved_documents = get_vector_store().as_retriever().invoke(query, k=k)

    serialized_data = "\n\n".join(
        f"Source: {document.metadata.get('source', 'Unknown')}\n\nContent: {document.page_content}"
        for document in retrieved_documents
    )

    return serialized_data, retrieved_documents


# Search Tool
web_search_tool = TavilySearch(max_results=3)