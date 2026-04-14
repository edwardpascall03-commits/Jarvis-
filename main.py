import anthropic
import os
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def chat(message):
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": message}]
    )
    return response.content[0].text

while True:
    user_input = input("You: ")
    print(f"Jarvis: {chat(user_input)}")
    print("Script started")