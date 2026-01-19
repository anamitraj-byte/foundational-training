from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_tavily import TavilySearch


from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
)

tavily_search_tool = TavilySearch(
    max_results=5,
    topic="general",
)

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"



tools = [tavily_search_tool, get_weather]

agent = create_agent(
    model,
    tools,
    system_prompt="You are a helpful assistant. Be concise and accurate."
)


user_input = "What is the weather in Shimla?"

for step in agent.stream(
    {"messages": user_input},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()