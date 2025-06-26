import json
import random
import os
from datetime import datetime
import tkinter as tk
from tkinter import scrolledtext
from model_loader import load_chat_model

# تحميل البيانات من كل ملفات patterns
patterns_dir = "patterns"
intents = []
for filename in os.listdir(patterns_dir):
    if filename.endswith(".json"):
        with open(os.path.join(patterns_dir, filename), "r", encoding="utf-8") as f:
            intents.extend(json.load(f))

with open("label_map.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)

chat_model = load_chat_model()

# إعداد ملفات الجلسة
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs("sessions", exist_ok=True)
os.makedirs("unmatched", exist_ok=True)

log_file_path = os.path.join("sessions", f"user_inputs_{timestamp}.txt")
unmatched_file_path = os.path.join("unmatched", f"unmatched_inputs_{timestamp}.txt")

def log_user_input(text):
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(text.strip() + "\n")

def log_unmatched_input(text):
    with open(unmatched_file_path, "a", encoding="utf-8") as f:
        f.write(text.strip() + "\n")

def match_pattern(user_input):
    for intent in intents:
        for pattern in intent["patterns"]:
            if pattern.lower() in user_input.lower():
                return random.choice(intent["responses"])
    return None

def handle_user_input():
    user_input = input_entry.get()
    if not user_input.strip():
        return

    chat_box.insert(tk.END, "👤 You: " + user_input + "\n")
    log_user_input(user_input)

    response = match_pattern(user_input)

    if response:
        chat_box.insert(tk.END, "🤖 Bot: " + response + "\n")
    else:
        prediction = chat_model(user_input)[0]
        label_index = int(prediction["label"].split("_")[-1])
        bot_response = random.choice(label_map[str(label_index)])
        chat_box.insert(tk.END, "🤖 Bot (AI): " + bot_response + "\n")
        log_unmatched_input(user_input)

    chat_box.see(tk.END)
    input_entry.delete(0, tk.END)

# إنشاء نافذة GUI
window = tk.Tk()
window.title("AI Chatbot")
window.geometry("500x600")
window.resizable(False, False)

chat_box = scrolledtext.ScrolledText(window, wrap=tk.WORD, font=("Arial", 12))
chat_box.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
chat_box.configure(state='normal')

input_frame = tk.Frame(window)
input_frame.pack(fill=tk.X, padx=10, pady=10)

input_entry = tk.Entry(input_frame, font=("Arial", 12))
input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
input_entry.bind("<Return>", lambda event: handle_user_input())

send_button = tk.Button(input_frame, text="Send", command=handle_user_input, font=("Arial", 12))
send_button.pack(side=tk.RIGHT)

chat_box.insert(tk.END, "🤖 Bot: مرحبًا بك! يمكنك البدء بالدردشة.\n")
window.mainloop()
