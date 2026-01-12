from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os

def chat_groq(prompt: str):
    try:
        client = OpenAI(
            api_key=os.environ.get("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )

        response = client.responses.create(
            input=prompt,
            model="openai/gpt-oss-20b",
        )
        return response.output_text, None
    except Exception as e:
        return None, f"Error: {str(e)}"
