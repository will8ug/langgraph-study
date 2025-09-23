import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint import memory
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

load_dotenv()

model = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

memory = MemorySaver()

search = TavilySearch(max_results=2)
tools = [search]

agent = create_react_agent(
    model=model,
    tools=tools,
    checkpointer=memory,
)

config = {"configurable": {"thread_id": "123"}}

input_message = {
    "role": "user",
    "content": "Hi, I'm Bob and I live in Guangzhou."
}
for step in agent.stream({"messages": [input_message]}, config, stream_mode="values"):
    step["messages"][-1].pretty_print()

input_message = {
    "role": "user",
    "content": "What's the weather where I live?"
}
for step in agent.stream({"messages": [input_message]}, config, stream_mode="values"):
    step["messages"][-1].pretty_print()
