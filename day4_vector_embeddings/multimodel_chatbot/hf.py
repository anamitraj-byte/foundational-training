import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def chat_huggingface(prompt: str):
    try:
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ["HF_TOKEN"],
        )

        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )

        result = completion.choices[0].message.content
        return result, None
    except Exception as e:
        return None, f"Error: {str(e)}"