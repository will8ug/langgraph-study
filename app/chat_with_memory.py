import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()
#model_name = "qwen-plus" # for streaming
# model_name = "qwen-turbo" # for non-streaming

model = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)
# model = ChatOpenAI(
#     model=model_name,
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )

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
        ("system", "You are a helpful assistant. Answer all questions to the best of your ability in {language}."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str
config = {"configurable": {"thread_id": "123"}}

# trimmer = trim_messages(
#     max_tokens=50,
#     strategy="last",
#     token_counter=model,
#     include_system=True,
#     allow_partial=False,
#     start_on="human",
# )

def customStateGraph():
    workflow = StateGraph(state_schema=State)

    def call_model(state: State):
        # print(f"Messages before trimming: {len(state['messages'])}")
        # trimmed_messages = trimmer.invoke(state['messages'])
        # print(f"Messages after trimming: {len(trimmed_messages)}")
        # print("Remaining messages:")
        # for msg in trimmed_messages:
        #     print(f"  {type(msg).__name__}: {msg.content}")

        # prompt = prompt_template.invoke(
        #     {"messages": trimmed_messages, "language": state["language"]}
        # )
        prompt = prompt_template.invoke(state)
        response = model.invoke(prompt)
        return {"messages": [response]}

    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

def prompt_test_with_graph():
    app = customStateGraph()
    input_messages = [HumanMessage("Introduce China within 100 words?")]
    for chunk, metadata in app.stream(
        {"messages": input_messages, "language": "Manderin"},
        config,
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessage):
            print(chunk.content, end="", flush=True)

prompt_test_with_graph()
