from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from langchain.agents import create_agent
import os

load_dotenv()

# Initialize the model
model = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


# Define the file writer tool with better feedback
@tool
def write_to_file(content: str, filename: str = "research_summary.txt") -> str:
    """
    Write research summary to a file. YOU MUST USE THIS TOOL to save the summary.
    
    Args:
        content: The complete research summary content to write
        filename: Name of the file (default: research_summary.txt)
    
    Returns:
        Success message with filename and file path
    """
    try:
        filepath = os.path.abspath(filename)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"✓ Successfully wrote summary to {filename} at {filepath}"
    except Exception as e:
        return f"✗ Error writing to file: {str(e)}"

# Initialize Tavily search tool
tavily_search = TavilySearch(
    max_results=5,
    search_depth="advanced",
)

# Combine tools
tools = [tavily_search, write_to_file]

# System prompt that emphasizes tool usage
system_prompt = """You are a Research Assistant Agent. You MUST follow these steps exactly:

STEP 1: Use the 'tavily_search_results_json' tool to search for information about the topic.
        You may search multiple times with different queries to get comprehensive information.

STEP 2: Analyze and synthesize the search results into a comprehensive summary with:
   - Overview/Introduction
   - Main concepts or findings
   - Important details or statistics
   - Current trends or developments
   - Conclusion

STEP 3: **CRITICAL** - You MUST call the 'write_to_file' tool with your complete summary.
        DO NOT just say you will write it - actually call the write_to_file tool.
        Pass the entire formatted summary as the 'content' parameter.

STEP 4: After the file is written, confirm the file path from the tool's response.

Remember: You must ACTUALLY CALL the write_to_file tool, not just mention writing the file."""

# Create agent
agent_executor = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt
)

# Function to run research on a topic
def research_topic(topic: str) -> dict:
    """
    Research a given topic and generate a summary with key points.
    
    Args:
        topic: The topic to research
        
    Returns:
        Dictionary with summary and file_created status
    """
    prompt_text = f"""Research the topic: "{topic}"

IMPORTANT: You MUST complete all steps including:
1. Search for information (use tavily_search_results_json)
2. Create a comprehensive summary
3. ACTUALLY CALL write_to_file tool with the summary (don't just say you will)

The summary should include:
- Overview/Introduction
- Main concepts or findings  
- Important details or statistics
- Current trends or developments
- Conclusion

DO NOT finish until you have actually called the write_to_file tool and received confirmation."""
    
    print(f"\n{'='*60}")
    print(f"RESEARCHING TOPIC: {topic}")
    print(f"{'='*60}\n")

    final_message = ""
    
    # Stream the agent's response
    for step in agent_executor.stream(
        {"messages": [("user", prompt_text)]},
        stream_mode="values",
    ):
        msg = step["messages"][-1]
        msg.pretty_print()

        final_message = str(msg.content) if hasattr(msg, 'content') else str(msg)
    
    # Verify file was created
    if os.path.exists("research_summary.txt"):
        file_size = os.path.getsize("research_summary.txt")
        print(f"\n{'='*60}")
        print(f"✓ SUCCESS: File created at {os.path.abspath('research_summary.txt')}")
        print(f"✓ File size: {file_size} bytes")
        print(f"{'='*60}\n")
        return {
            "summary": final_message,
            "file_created": True,
            "file_path": os.path.abspath("research_summary.txt")
        }
    else:
        print(f"\n{'='*60}")
        print(f"✗ WARNING: File was not created!")
        print(f"The agent may have hallucinated the file creation.")
        print(f"{'='*60}\n")
        return {
            "summary": final_message,
            "file_created": False,
            "file_path": None
        }



# Main execution
if __name__ == "__main__":
    # Example usage
    topic = input("Enter research topic:")
    
    # Try the agent-based approach first
    result = research_topic(topic)
    
    print("\n" + "="*60)
    print("FINAL RESULT:")
    print("="*60)
    print(f"File Created: {result['file_created']}")
    if result['file_path']:
        print(f"File Path: {result['file_path']}")
    print("="*60)