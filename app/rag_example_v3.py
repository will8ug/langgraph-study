import os
from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


load_dotenv()

# model = ChatOpenAI(
#     model="qwen-turbo",
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )
model = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    max_retries=2,
)

loader = WebBaseLoader(
    web_path=("https://lilianweng.github.io/posts/2023-06-23-agent/"),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
    )
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
splitted_docs = text_splitter.split_documents(docs)
print(f"Splitted doc length: {len(splitted_docs)}\n")

vector_store = InMemoryVectorStore(embeddings)
doc_ids = vector_store.add_documents(splitted_docs)
print(f"The first 3 Doc IDs: {doc_ids[:3]}\n")


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    print(f"Retrieved docs length: {len(retrieved_docs)}\n")
    serialized_docs = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}") for doc in retrieved_docs
    )
    return serialized_docs, retrieved_docs

memory = MemorySaver()
config = {"configurable": {"thread_id": "agent001"}}

agent_executor = create_react_agent(
    model=model,
    tools=[retrieve],
    checkpointer=memory,
)

input_message = (
    "What is the standard method for Task Decomposition?\n\n"
    "Once you get the answer, please look up some common extensions of that method."
)
for event in agent_executor.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
    config=config
):
    event["messages"][-1].pretty_print()
