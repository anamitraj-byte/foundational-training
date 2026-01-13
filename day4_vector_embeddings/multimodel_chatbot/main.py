from typing import Tuple
from gem import chat_gemini
from hf import chat_open_source
from dotenv import load_dotenv
import os

load_dotenv()

def chat(
    prompt: str,
    provider: str
) -> Tuple[str | None, str | None]:
    try:
        if provider == "huggingface" or provider == "groq":
            return chat_open_source(prompt, provider)
        elif provider == "gemini":
            return chat_gemini(prompt)
        else:
            return None, f"Unsupported provider: {provider}"
    except Exception as e:
        return None, str(e)

prompt = input("Enter prompt:")

provider = os.environ["MODEL_USE"]

response, error = chat(prompt, provider)

if not error:
    print(response)
else:
    print(error)