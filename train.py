from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import json

with open("patterns.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

texts = []
labels = []
label_map = {}

for i, intent in enumerate(intents):
    label_map[i] = intent["responses"]
    for pattern in intent["patterns"]:
        texts.append(pattern)
        labels.append(i)

dataset = Dataset.from_dict({"text": texts, "label": labels})

tokenizer = BertTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function)

model = BertForSequenceClassification.from_pretrained(
    "aubmindlab/bert-base-arabertv02", 
    num_labels=len(label_map)
)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=1,
    save_steps=100,
    # evaluation_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

trainer.train()
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")

with open("label_map.json", "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False)
