import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

BASE_URL = {"huggingface": os.environ["HUGGINGFACE_URL"], "groq": os.environ["GROQ_URL"]}
MODEL = {"huggingface": os.environ["HUGGINGFACE_MODEL"], "groq": os.environ["GROQ_MODEL"]}
HF_API = os.environ["HF_TOKEN"]
GROQ_API = os.environ["GROQ_API_KEY"]
API_KEY = {"huggingface": HF_API, "groq": GROQ_API}

def chat_open_source(prompt: str, provider: str):
    try:
        client = OpenAI(
            base_url=BASE_URL[provider],
            api_key=API_KEY[provider],
        )
        response = client.responses.create(
            input=prompt,
            model=MODEL[provider],
        )
        return response.output_text, None
    except Exception as e:
        return None, f"Error: {str(e)}"