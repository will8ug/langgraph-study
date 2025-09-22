import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()
# api_key = os.getenv("DASHSCOPE_API_KEY")
# print("API Key:", api_key)
#model_name = "qwen-plus" # for streaming
model_name = "qwen-turbo" # for non-streaming

# model = ChatOpenAI(
#     model="deepseek-chat",
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url="https://api.deepseek.com",
# )
model = ChatOpenAI(
    model=model_name,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def basic_model_call():
    # model.invoke()
    result = model.stream(
        [
            HumanMessage(content="Hi! I'm Will"),
            AIMessage(content="Hello Will! How can I assist you today?"),
            HumanMessage(content="What's my name?"),
        ]
    )
    for chunk in result:
        print(chunk.content, end="", flush=True)
    #print(result)

# basic_model_call()
print("\n")

prompt_template = ChatPromptTemplate.from_messages(
    [
        # SystemMessage(content="You talk like a pirate. Answer all questions to the best of your ability."),
        ("system", "You are a helpful assistant. Answer all questions to the best of your ability in {language}."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str
config = {"configurable": {"thread_id": "123"}}

def builtinMessageStateGraph():
    workflow = StateGraph(state_schema=MessagesState)

    def call_model(state: MessagesState):
        prompt = prompt_template.invoke(state)
        response = model.invoke(prompt)
        # response = model.invoke(state["messages"])
        return {"messages": response}

    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

def customStateGraph():
    workflow = StateGraph(state_schema=State)

    def call_model(state: State):
        prompt = prompt_template.invoke(state)
        response = model.invoke(prompt)
        return {"messages": [response]}

    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

def simple_test_with_graph():
    app = builtinMessageStateGraph()
    # input_messages = [HumanMessage("Introduce Hong Kong within 150 words")]
    input_messages = [HumanMessage("Hi! I'm Will")]
    output = app.invoke({"messages": input_messages}, config)
    # print(output)
    output["messages"][-1].pretty_print()

    output = app.invoke({"messages": [HumanMessage("What's my name?")]}, config)
    output["messages"][-1].pretty_print()

def prompt_test_with_graph_builtin():
    app = builtinMessageStateGraph()
    input_messages = [HumanMessage("Hi! I'm Bob")]
    output = app.invoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()

def prompt_test_with_graph():
    app = customStateGraph()
    input_messages = [HumanMessage("Hi! I'm Bob")]
    output = app.invoke({"messages": input_messages, "language": "Manderin"}, config)
    output["messages"][-1].pretty_print()

    output = app.invoke({"messages": [HumanMessage("What's my name?")]}, config)
    output["messages"][-1].pretty_print()

# simple_test_with_graph()
# prompt_test_with_graph_builtin()
prompt_test_with_graph()
