import os
import asyncio
import httpx
from typing import Any
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
# Initialize the MCP server
server = Server("huggingface-mcp")

# Initialize Hugging Face client
hf_client = OpenAI(
    base_url=os.environ["HUGGINGFACE_BASEURL"],
    api_key=os.environ.get("HF_TOKEN"),
)

# Serper API configuration
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
SERPER_URL = os.environ["SERPER_URL"]

# List of available models
AVAILABLE_MODELS = [
    "moonshotai/Kimi-K2-Instruct-0905",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-9b-it",
]


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available tools for calling Hugging Face models and web search."""
    return [
        Tool(
            name="call_hf_model",
            description=f"Call a Hugging Face model via their inference API. Available models: {', '.join(AVAILABLE_MODELS)}",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": f"The model to use. Options: {', '.join(AVAILABLE_MODELS)}",
                        "enum": AVAILABLE_MODELS,
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The prompt or question to send to the model",
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "Optional system prompt to guide the model's behavior",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum number of tokens to generate (default: 1000)",
                        "default": 1000,
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Sampling temperature between 0 and 2 (default: 0.7)",
                        "default": 0.7,
                    },
                },
                "required": ["model", "prompt"],
            },
        ),
        Tool(
            name="web_search",
            description="Search the web using Serper API and return relevant results",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="search_and_analyze",
            description="Search the web and use a Hugging Face model to analyze/summarize the results",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "model": {
                        "type": "string",
                        "description": f"The model to use for analysis. Options: {', '.join(AVAILABLE_MODELS)}",
                        "enum": AVAILABLE_MODELS,
                        "default": "meta-llama/Llama-3.3-70B-Instruct",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of search results to include (default: 5)",
                        "default": 5,
                    },
                    "analysis_type": {
                        "type": "string",
                        "description": "Type of analysis: 'summarize', 'answer', or 'compare'",
                        "enum": ["summarize", "answer", "compare"],
                        "default": "summarize",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="list_hf_models",
            description="List all available Hugging Face models that can be called",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


async def perform_web_search(query: str, num_results: int = 5) -> dict:
    """Perform web search using Serper API."""
    if not SERPER_API_KEY:
        raise ValueError("SERPER_API_KEY environment variable not set")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            SERPER_URL,
            headers={
                "X-API-KEY": SERPER_API_KEY,
                "Content-Type": "application/json",
            },
            json={"q": query, "num": num_results},
            timeout=10.0,
        )
        response.raise_for_status()
        return response.json()


def format_search_results(search_data: dict) -> str:
    """Format search results into readable text."""
    formatted = []
    
    # Add organic results
    if "organic" in search_data:
        formatted.append("Search Results:\n")
        for i, result in enumerate(search_data["organic"], 1):
            title = result.get("title", "No title")
            snippet = result.get("snippet", "No snippet")
            link = result.get("link", "")
            formatted.append(f"{i}. {title}\n   {snippet}\n   URL: {link}\n")
    
    # Add knowledge graph if available
    if "knowledgeGraph" in search_data:
        kg = search_data["knowledgeGraph"]
        formatted.append(f"\nKnowledge Graph:\n")
        formatted.append(f"Title: {kg.get('title', 'N/A')}\n")
        formatted.append(f"Description: {kg.get('description', 'N/A')}\n")
    
    return "\n".join(formatted)


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls for Hugging Face model inference and web search."""
    
    if name == "list_hf_models":
        model_list = "\n".join([f"- {model}" for model in AVAILABLE_MODELS])
        return [
            TextContent(
                type="text",
                text=f"Available Hugging Face models:\n{model_list}",
            )
        ]
    
    elif name == "web_search":
        query = arguments.get("query")
        num_results = arguments.get("num_results", 5)
        
        if not query:
            return [
                TextContent(
                    type="text",
                    text="Error: 'query' parameter is required",
                )
            ]
        
        try:
            search_data = await perform_web_search(query, num_results)
            formatted_results = format_search_results(search_data)
            
            return [
                TextContent(
                    type="text",
                    text=formatted_results,
                )
            ]
        
        except Exception as e:
            return [
                TextContent(
                    type="text",
                    text=f"Error performing web search: {str(e)}",
                )
            ]
    
    elif name == "search_and_analyze":
        query = arguments.get("query")
        model = arguments.get("model", "meta-llama/Llama-3.3-70B-Instruct")
        num_results = arguments.get("num_results", 5)
        analysis_type = arguments.get("analysis_type", "summarize")
        
        if not query:
            return [
                TextContent(
                    type="text",
                    text="Error: 'query' parameter is required",
                )
            ]
        
        try:
            # Perform web search
            search_data = await perform_web_search(query, num_results)
            search_results = format_search_results(search_data)
            
            # Create analysis prompt based on type
            if analysis_type == "summarize":
                analysis_prompt = f"Summarize the following search results for the query '{query}':\n\n{search_results}"
            elif analysis_type == "answer":
                analysis_prompt = f"Based on these search results, answer the question: {query}\n\nSearch results:\n{search_results}"
            else:  # compare
                analysis_prompt = f"Compare and contrast the information from these search results about '{query}':\n\n{search_results}"
            
            # Build messages for HF model
            messages = [
                {"role": "system", "content": "You are a helpful assistant that analyzes web search results and provides clear, accurate summaries."},
                {"role": "user", "content": analysis_prompt}
            ]
            
            # Call the Hugging Face model
            completion = hf_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
            )
            
            analysis = completion.choices[0].message.content
            
            return [
                TextContent(
                    type="text",
                    text=f"Query: {query}\nModel: {model}\n\n{'='*50}\nSEARCH RESULTS:\n{'='*50}\n\n{search_results}\n\n{'='*50}\nANALYSIS:\n{'='*50}\n\n{analysis}",
                )
            ]
        
        except Exception as e:
            return [
                TextContent(
                    type="text",
                    text=f"Error in search and analyze: {str(e)}",
                )
            ]
    
    elif name == "call_hf_model":
        model = arguments.get("model")
        prompt = arguments.get("prompt")
        system_prompt = arguments.get("system_prompt")
        max_tokens = arguments.get("max_tokens", 1000)
        temperature = arguments.get("temperature", 0.7)
        
        if not model or not prompt:
            return [
                TextContent(
                    type="text",
                    text="Error: Both 'model' and 'prompt' are required parameters",
                )
            ]
        
        if model not in AVAILABLE_MODELS:
            return [
                TextContent(
                    type="text",
                    text=f"Error: Model '{model}' not found. Available models: {', '.join(AVAILABLE_MODELS)}",
                )
            ]
        
        try:
            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Call the Hugging Face model
            completion = hf_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            response_text = completion.choices[0].message.content
            
            return [
                TextContent(
                    type="text",
                    text=f"Model: {model}\n\nResponse:\n{response_text}",
                )
            ]
        
        except Exception as e:
            return [
                TextContent(
                    type="text",
                    text=f"Error calling Hugging Face model: {str(e)}",
                )
            ]
    
    else:
        return [
            TextContent(
                type="text",
                text=f"Unknown tool: {name}",
            )
        ]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="huggingface-mcp",
                server_version="0.2.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())