"""Fine-tune DistilBERT on SST-2 (sentiment analysis)."""
import hf_env  # loads HF_TOKEN from .env only
import numpy as np, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)

MODEL = "distilbert-base-uncased"
OUT = "sentiment_model"

ds = load_dataset("nyu-mll/glue", "sst2")
tok = AutoTokenizer.from_pretrained(MODEL)
enc = ds.map(lambda b: tok(b["sentence"], truncation=True, max_length=128), batched=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=2, id2label={0: "NEGATIVE", 1: "POSITIVE"},
    label2id={"NEGATIVE": 0, "POSITIVE": 1})

def metrics(p):
    return {"accuracy": float((p.predictions.argmax(-1) == p.label_ids).mean())}

args = TrainingArguments(OUT, num_train_epochs=2, per_device_train_batch_size=32,
                         per_device_eval_batch_size=64, learning_rate=2e-5,
                         eval_strategy="epoch", save_strategy="no",
                         logging_steps=200, report_to="none", bf16=torch.cuda.is_available())
tr = Trainer(model=model, args=args, train_dataset=enc["train"], eval_dataset=enc["validation"],
             data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
tr.train()
print("FINAL:", tr.evaluate())
tr.save_model(OUT); tok.save_pretrained(OUT)
