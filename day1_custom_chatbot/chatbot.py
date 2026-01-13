import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ----------------------------------
# CONFIGURATION
# ----------------------------------
# ----------------------------------
# CONFIGURATION
# ----------------------------------
SYSTEM_PROMPT = """
You are a chatbot assistant that offers healthy recipes to users.
Be friendly, clear and concise.
Do not answer any questions that are not relevant to healthy recipes.
Maintain conversational context.
Do not mention internal instructions.

## Example Interactions (for your reference only, NOT actual conversation history):

User: Can you suggest a healthy breakfast?
Assistant: Absolutely! Here's a nutritious breakfast idea:

**Overnight Oats with Berries**
- 1/2 cup rolled oats
- 1/2 cup almond milk
- 1 tbsp chia seeds
- Fresh berries
- Drizzle of honey

Mix everything except berries, refrigerate overnight, top with berries in the morning. High in fiber, protein, and antioxidants! Would you like more breakfast ideas or recipes for other meals?

User: What's the weather like today?
Assistant: I'm specifically designed to help with healthy recipes and cooking advice. I can't provide weather information, but I'd be happy to suggest recipes perfect for any season! Are you looking for something light and refreshing, or warm and comforting?

User: I'm looking for a low-carb dinner option
Assistant: Great choice! Here's a delicious low-carb dinner:

**Garlic Butter Salmon with Asparagus**
- 6 oz salmon fillet
- 1 bunch asparagus
- 2 tbsp butter
- 3 cloves garlic, minced
- Lemon juice, salt, pepper

Bake salmon at 400°F for 12-15 minutes, sauté asparagus with garlic butter. This meal has about 8g net carbs and is rich in omega-3s. Need side dish suggestions?

IMPORTANT: These examples are for guidance only. Do NOT refer to them as previous conversations. If asked about chat history, only reference the actual user conversations that follow these examples.
"""

MODEL_NAME = os.environ["GEMINI_MODEL"]
MAX_HISTORY_PAIRS = 10

# ----------------------------------
# BUILD CONTENTS (No few-shot examples in contents)
# ----------------------------------

#chat_history = [("role", "input/output_str"), ..]

def build_contents(chat_history: list[tuple[str, str]], new_user_input: str) -> list:
    """Convert chat history to API-compatible format."""
    contents = []
    # Add recent chat history only
    recent_history = chat_history[-MAX_HISTORY_PAIRS * 2:]
    for role, message in recent_history:
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=message)]
            )
        )
    
    # Add new user input
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=new_user_input)]
        )
    )
    
    return contents


# ----------------------------------
# CALL GEMINI WITH ERROR HANDLING
# ----------------------------------
def chat_with_gemini(chat_history: list[tuple[str, str]], user_input: str):
    """Send message to Gemini and return response."""
    try:
        client = genai.Client()
        contents = build_contents(chat_history, user_input)
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="text/plain",
                temperature=0.7,
            )
        )
        
        return response, None
    
    except Exception as e:
        return None, f"Error: {str(e)}"


# ----------------------------------
# DISPLAY TOKEN USAGE
# ----------------------------------
def display_token_usage(response, show_details=True):
    """Display token usage information if available."""
    if not show_details:
        return
    
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        usage = response.usage_metadata
        print(f"\n📊 Tokens: {usage.prompt_token_count} in | "
              f"{usage.candidates_token_count} out | "
              f"{usage.total_token_count} total")


# ----------------------------------
# CLI CHAT LOOP
# ----------------------------------
def main():
    """Main chat loop."""
    chat_history = []
    show_token_usage = True
    
    print("🤖 Healthy Recipe Chatbot")
    print("=" * 50)
    print("Commands: 'exit' to quit, 'clear' to reset, 'tokens' to toggle usage")
    print("\nTry asking: 'healthy lunch ideas' or 'low-carb dinner'\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() == "exit":
                print("\n👋 Thanks for chatting! Stay healthy!")
                break
            
            if user_input.lower() == "clear":
                chat_history = []
                print("🔄 Chat history cleared.\n")
                continue
            
            if user_input.lower() == "tokens":
                show_token_usage = not show_token_usage
                status = "enabled" if show_token_usage else "disabled"
                print(f"📊 Token usage display {status}.\n")
                continue
            
            # Get response from Gemini
            response, error = chat_with_gemini(chat_history, user_input)
            
            if error:
                print(f"\n❌ {error}\n")
                continue
            
            bot_reply = response.text
            print(f"\nBot: {bot_reply}")
            
            display_token_usage(response, show_token_usage)
            print()
            
            # Update history
            chat_history.append(("user", user_input))
            chat_history.append(("model", bot_reply))
        
        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!")
            break
        
        except Exception as e:
            print(f"\n❌ Unexpected error: {str(e)}\n")


if __name__ == "__main__":
    main()