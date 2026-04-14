import anthropic
import os
import json
from dotenv import load_dotenv
from tools.memory import save_session, load_last_session

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def load_profile():
    with open("profile.json", "r") as f:
        profile = json.load(f)
    return f"You are Jarvis, a personal AI assistant for {profile['name']}. Here is everything you need to know about them: {json.dumps(profile, indent=2)}"

conversation_history = []

def chat(message):
    conversation_history.append({"role": "user", "content": message})
    
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=load_profile(),
        messages=conversation_history
    )
    
    reply = response.content[0].text
    conversation_history.append({"role": "assistant", "content": reply})
    return reply

print("Jarvis online. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["quit", "exit"]:
        save_session(conversation_history)
        print("Jarvis: Shutting down.")
        break
    if not user_input:
        continue
    print(f"Jarvis: {chat(user_input)}\n")