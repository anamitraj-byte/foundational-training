from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq()

completion = client.chat.completions.create(
        model="groq/compound",
        messages=[

            {
                "role": "user",
                "content": f"what is the capital city of France?"
            },
        ],
        max_tokens=100,
        temperature=0.1,
    )

print(completion.choices[0].message.content)