import os
import json
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# إعداد المتغيرات
texts = []
labels = []
label_map = {}
label_index = 0

# قراءة كل ملفات JSON داخل مجلد patterns
patterns_dir = "patterns"
for filename in os.listdir(patterns_dir):
    if filename.endswith(".json"):
        filepath = os.path.join(patterns_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            intents = json.load(f)

            for intent in intents:
                # حفظ ردود كل نية (intent) في الماب
                label_map[label_index] = intent["responses"]
                for pattern in intent["patterns"]:
                    texts.append(pattern)
                    labels.append(label_index)
                label_index += 1

# إنشاء الداتا
dataset = Dataset.from_dict({"text": texts, "label": labels})

# تحميل التوكننايزر
tokenizer = BertTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")

# توكن
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function)

# تحميل الموديل
model = BertForSequenceClassification.from_pretrained(
    "aubmindlab/bert-base-arabertv02",
    num_labels=len(label_map)
)

# إعدادات التدريب
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=1,
    save_steps=100,
)

# تدريب
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

trainer.train()

# حفظ النموذج والتوكننايزر والماب
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")

with open("label_map.json", "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False)
