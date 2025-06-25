import json
import random
from model_loader import load_chat_model

with open("patterns.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

with open("label_map.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)

chat_model = load_chat_model()

def match_pattern(user_input):
    for intent in intents:
        for pattern in intent["patterns"]:
            if pattern.lower() in user_input.lower():
                return random.choice(intent["responses"])
    return None

def chat():
    print("مرحبًا! اكتب 'خروج' أو 'quit' لإنهاء المحادثة.")
    while True:
        user_input = input("أنت: ")
        if user_input.lower() in ["خروج", "quit", "exit"]:
            print("تشرفت بالدردشة معك! 👋")
            break

        response = match_pattern(user_input)

        if response is None:
            prediction = chat_model(user_input)[0]
            label_index = int(prediction['label'].split("_")[-1])
            bot_response = random.choice(label_map[str(label_index)])
            print("🤖 (نموذج الذكاء):", bot_response)
        else:
            print("🤖: " + response)

if __name__ == "__main__":
    chat()
