from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool

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

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"

tools = [search, get_weather]

agent = create_agent(
    model,
    tools,
    system_prompt="You are a helpful assistant. Be concise and accurate."
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "what is the weather like in Shimla?"}]
})

print(result)