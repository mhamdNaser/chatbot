from transformers import pipeline

def load_chat_model():
    return pipeline(
        "text-classification",
        model="./trained_model",
        tokenizer="./trained_model"
    )
