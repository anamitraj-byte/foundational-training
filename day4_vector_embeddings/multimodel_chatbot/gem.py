from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

MODEL_NAME = os.environ["GEMINI_MODEL"]
def chat_gemini(prompt):
    """Send message to Gemini and return response."""
    try:
        client = genai.Client()
        contents = build_contents(prompt)
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="text/plain",
                temperature=0.7,
            )
        )
        
        return response.text, None
    
    except Exception as e:
        return None, f"Error: {str(e)}"
    
def build_contents(new_user_input: str) -> list:
    """Convert chat history to API-compatible format."""
    contents = []
    
    # Add new user input
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=new_user_input)]
        )
    )
    
    return contents