from smolagents import CodeAgent, InferenceClientModel, DuckDuckGoSearchTool
from dotenv import load_dotenv

load_dotenv()

# Initialize the model client
model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct") # or any other compatible model

# Define the tools the agent can use (e.g., a web search tool)
tools = [DuckDuckGoSearchTool()]

calcAgent = CodeAgent(tools=tools, model=model, additional_authorized_imports=['matplotlib', 'matplotlib.pyplot'])

calcResult = calcAgent.run("plot a beautiful and colorful bar graph with the revenue of Amazon in the last 5 years")