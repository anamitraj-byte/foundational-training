from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from langchain.agents import create_agent
from tempfile import TemporaryDirectory
from langchain_community.agent_toolkits import FileManagementToolkit
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

# Set up working directory for file operations
working_directory = TemporaryDirectory()

# Initialize FileManagementToolkit
file_toolkit = FileManagementToolkit(
    root_dir=str(working_directory.name),
    selected_tools=["read_file", "write_file", "list_directory"],
)

# Get the file tools
file_tools = file_toolkit.get_tools()

# Initialize Tavily search tool with shorter results
tavily_search = TavilySearch(
    max_results=3,  # Reduced from 5 to 3 results
    search_depth="basic",  # Changed from "advanced" to "basic" for shorter snippets
    include_raw_content=False,  # Exclude raw HTML content
    include_answer=True,  # Get concise AI-generated answer
)

# Combine all tools (Tavily + File Management Tools)
tools = [tavily_search] + file_tools

# Updated system prompt
system_prompt = """You are a Research Assistant Agent. You MUST follow these steps exactly:

STEP 1: Use the 'tavily_search_results_json' tool to search for information about the topic.
        You may search multiple times with different queries to get comprehensive information.

STEP 2: Analyze and synthesize the search results into a comprehensive summary with:
   - Overview/Introduction
   - Main concepts or findings
   - Important details or statistics
   - Current trends or developments
   - Conclusion

STEP 3: **CRITICAL** - You MUST call the 'write_file' tool to save your complete summary.
        DO NOT just say you will write it - actually call the write_file tool.
        Parameters needed:
        - file_path: "research_summary.txt"
        - text: your complete formatted summary
        - append: false

STEP 4: After the file is written, you can optionally use 'list_directory' to verify the file exists.

Remember: You must ACTUALLY CALL the write_file tool, not just mention writing the file."""

# Create agent with all tools
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
3. ACTUALLY CALL write_file tool with the summary (don't just say you will)
   - Use file_path: "research_summary.txt"
   - Pass your complete summary as the 'text' parameter
   - Set append: false

The summary should include:
- Overview/Introduction
- Main concepts or findings  
- Important details or statistics
- Current trends or developments
- Conclusion

DO NOT finish until you have actually called the write_file tool and received confirmation."""
    
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
    
    # Check in the working directory for the file
    file_path = os.path.join(working_directory.name, "research_summary.txt")
    
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        print(f"\n{'='*60}")
        print(f"✓ SUCCESS: File created at {file_path}")
        print(f"✓ File size: {file_size} bytes")
        print(f"{'='*60}\n")
        return {
            "summary": final_message,
            "file_created": True,
            "file_path": file_path
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
    topic = input("Enter research topic: ")
    
    # Run the agent-based approach
    result = research_topic(topic)
    
    print("\n" + "="*60)
    print("FINAL RESULT:")
    print("="*60)
    print(f"File Created: {result['file_created']}")
    if result['file_path']:
        print(f"File Path: {result['file_path']}")
        # Optionally read and display the content
        if os.path.exists(result['file_path']):
            with open(result['file_path'], 'r', encoding='utf-8') as f:
                print(f"\nFile Contents Preview:")
                print("-" * 60)
                print(f.read()[:500] + "..." if len(f.read()) > 500 else f.read())
    print("="*60)
    

    working_directory.cleanup()