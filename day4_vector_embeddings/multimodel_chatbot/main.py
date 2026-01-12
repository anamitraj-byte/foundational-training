from typing import Tuple
from gem import chat_gemini
from groq import chat_groq
from hf import chat_huggingface
def chat(
    prompt: str,
    provider: str
) -> Tuple[str | None, str | None]:
    try:
        if provider == "groq":
            return chat_groq(prompt)
        elif provider == "huggingface":
            return chat_huggingface(prompt)
        elif provider == "gemini":
            return chat_gemini(prompt)
        else:
            return None, f"Unsupported provider: {provider}"
    except Exception as e:
        return None, str(e)

prompt = input("Enter prompt:")

provider = input("Enter provider: (groq/gemini/huggingface)").lower()

response, error = chat(prompt, provider)

if not error:
    print(response)
else:
    print(error)