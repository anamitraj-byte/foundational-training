from dataclasses import dataclass
import os
import requests  # Add this import

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver

from dotenv import load_dotenv

load_dotenv()

# Define system prompt
SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location.

IMPORTANT: You MUST use the exact weather information returned by the get_weather_for_location tool. 
Do NOT make up or invent weather conditions. Always base your punny response on the ACTUAL tool output.
If the tool says it's sunny, your response must reflect sunny weather, not cloudy or rainy conditions."""

# Define context schema
@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

# Define tools
@tool
def get_weather_for_location(city: str) -> str:
    """Get current weather for a given city."""
    api_key = os.environ["WEATHER_API_KEY"]
    base_url = os.environ["WEATHER_URL"]

    params = {
        "key": api_key,
        "q": city,
        "aqi": "no"  # Optional: air quality data
    }
    
    try:
        # Use the current weather endpoint
        url = f"{base_url}/current.json"
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        temp_c = data["current"]["temp_c"]
        description = data["current"]["condition"]["text"]
        humidity = data["current"]["humidity"]
        
        return f"In {city}: {description}, {temp_c}°C, humidity {humidity}%"
    except requests.exceptions.RequestException as e:
        return f"Sorry, couldn't get weather for {city}. Error: {str(e)}"
    except KeyError as e:
        return f"Sorry, unexpected response format. Error: {str(e)}"

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"

# Configure model
model = init_chat_model(
    "llama-3.3-70b-versatile",
    model_provider="groq",
    temperature=0.5,
    timeout=10,
    max_tokens=1000
)

# Set up memory
checkpointer = InMemorySaver()

# Create agent WITHOUT structured output
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    checkpointer=checkpointer
)

# Run agent
print("Hi, I am a punny weather agent, how can I be of help?")
print("=" * 70)
print("Type 'exit' to leave the chat")

while True:
    config = {"configurable": {"thread_id": "1"}}
    inp = input("Enter prompt: ")
    if inp.lower() == 'exit':
        exit()
    
    response = agent.invoke(
        {"messages": [{"role": "user", "content": inp}]},
        config=config,
        context=Context(user_id="1")
    )

    # Get the last message from the agent
    print("Response:", response['messages'][-1].content)
    print("=" * 70)
