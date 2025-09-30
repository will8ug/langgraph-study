import os
from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph import graph
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition


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

def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    model_with_tools = model.bind_tools([retrieve])
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tools = ToolNode([retrieve])

def generate_answer(state: MessagesState):
    """Generate an answer."""
    recent_tool_messages = []
    for msg in reversed(state["messages"]):
        if msg.type == "tool":
            recent_tool_messages.append(msg)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    docs_content = "\n\n".join([doc.content for doc in tool_messages])
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message for message in state["messages"]
        if message.type in ("human", "system") or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    response = model.invoke(prompt)
    return {"messages": [response]}

graph_builder = StateGraph(MessagesState)
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate_answer)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate_answer")
graph_builder.add_edge("generate_answer", END)

graph = graph_builder.compile()

input_message = "Hello"
for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]}, stream_mode="values"
):
    step["messages"][-1].pretty_print()

input_message = "What is Task Decomposition?"
for chunk, _ in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]}, stream_mode="messages"
):
    print(chunk.content, end="", flush=True)
