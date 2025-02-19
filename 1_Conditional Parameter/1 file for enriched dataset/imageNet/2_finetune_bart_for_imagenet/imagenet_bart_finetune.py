import os
import subprocess
import sys
from datetime import datetime
import json
from tqdm import tqdm
import torch
import gc
import wandb
import psutil
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    BartTokenizer,
    BartForSequenceClassification,
    Trainer,
    TrainingArguments,
    get_scheduler,
    GenerationConfig
)
from sklearn.metrics import precision_score, recall_score, f1_score
from huggingface_hub import login

# Environment Setup
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

required_packages = [
    "torch",
    "pandas",
    "datasets",
    "transformers",
    "scikit-learn",
    "tqdm",
    "wandb",
    "evaluate",
    "huggingface_hub"
]

for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        print(f"Package {pkg} is missing. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# Verify CUDA
print("Checking CUDA setup...")
assert torch.cuda.is_available(), "CUDA is not available. Please check your GPU setup."
print(f"Using GPU: {torch.cuda.get_device_name(0)}")


# Hugging Face Hub Login
huggingface_token = "---"  # Replace with your token
login(token=huggingface_token)


# Weights & Biases Login
wandb.login(key="----")
wandb.init(project="imagenet-classification", entity="---")


# Load Data from JSON Lines
data_path = "./Generated dataset/imagenet_descriptions_20241210_204654.json"

def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in tqdm(f, desc="Loading data"):
            if line.strip():
                item = json.loads(line.strip())
                combined_text = item['eva02_description'].strip() + " " + item['blip_description'].strip()
                label = item['true_label']
                data.append({'text': combined_text, 'label': label})
    return data

data = load_data(data_path)
gc.collect()
torch.cuda.empty_cache()

# Convert list of dicts to a Hugging Face Dataset
dataset = Dataset.from_list(data)
del data
gc.collect()
torch.cuda.empty_cache()

# Shuffle and split (80% train, 10% val, 10% test)
dataset = dataset.shuffle(seed=42)
total_len = len(dataset)
train_dataset = dataset.select(range(0, int(0.8 * total_len)))
val_dataset = dataset.select(range(int(0.8 * total_len), int(0.9 * total_len)))
test_dataset = dataset.select(range(int(0.9 * total_len), total_len))
del dataset
gc.collect()
torch.cuda.empty_cache()

# --------------------------
# Tokenization
# --------------------------
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.remove_columns(["text"])
val_dataset = val_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])

train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
gc.collect()
torch.cuda.empty_cache()


# Model Setup
num_labels = 1000  # ImageNet-1k classes
model = BartForSequenceClassification.from_pretrained('facebook/bart-base', num_labels=num_labels)
gc.collect()
torch.cuda.empty_cache()


# Training Arguments
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
output_dir = f"./bart-image-classification-{timestamp}"

training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,       # Increased batch size
    per_device_eval_batch_size=1,        # Reduce evaluation batch size
    num_train_epochs=5,                   # Increased epochs
    logging_steps=10,
    learning_rate=2e-5,                   # Lower learning rate for better convergence
    load_best_model_at_end=True,
    warmup_steps=500,                     # Add warmup steps
    report_to="wandb",
    run_name=f"bart_class_run_{timestamp}",
    gradient_checkpointing=True,
    fp16=True,                            # Enable mixed precision training
    lr_scheduler_type="linear",          # Linear scheduler for learning rate decay
    weight_decay=0.01,                   # Add weight decay for regularization
    gradient_accumulation_steps=4        # Simulate larger batch size
)

gc.collect()
torch.cuda.empty_cache()

# Log Memory Utility Function
def log_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory Usage: {mem_info.rss / (1024 * 1024):.2f} MB")


# Generation Configuration
generation_config = GenerationConfig(
    early_stopping=True,
    num_beams=4,
    no_repeat_ngram_size=3,
    forced_bos_token_id=0,
    forced_eos_token_id=2
)

generation_config.save_pretrained(output_dir)


# Trainer
optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=500, num_training_steps=len(train_dataset) // training_args.gradient_accumulation_steps * training_args.num_train_epochs)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizers=(optimizer, scheduler)
)
gc.collect()
torch.cuda.empty_cache()

# Training
log_memory()
trainer.train()
gc.collect()
torch.cuda.empty_cache()
log_memory()

# Evaluation on Test Set
log_memory()
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids

accuracy = (preds == labels).mean()
precision = precision_score(labels, preds, average='weighted')
recall = recall_score(labels, preds, average='weighted')
f1 = f1_score(labels, preds, average='weighted')

print("Test Set Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save Model and Tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Finish W&B run
wandb.finish()
gc.collect()
torch.cuda.empty_cache()


