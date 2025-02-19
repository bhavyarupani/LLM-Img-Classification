import os
import subprocess
import sys
import datetime
import torch

# Set environment variables for GPU usage
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("Checking CUDA setup...")
assert torch.cuda.is_available(), "CUDA is not available. Please check your GPU setup."
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Install required packages if not already installed
for package in ["wandb", "scikit-learn", "accelerate", "datasets", "transformers", "torch"]:
    try:
        __import__(package)
    except ImportError:
        print(f"{package} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import pandas as pd
from datasets import Dataset
from transformers import BartTokenizer, BartForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import wandb

# Get timestamp for saving model
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("Checking CUDA setup...")
assert torch.cuda.is_available(), "CUDA is not available. Please check your GPU setup."
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

wandb.login(key="")
wandb.init(project="", entity="")


train_data_path = "./Generated dataset/cifar10_train_20250209_203407.json"
test_data_path  = "./Generated dataset/cifar10_test_20250209_203407.json"

train_df = pd.read_json(train_data_path, lines=True)
test_df = pd.read_json(test_data_path, lines=True)

train_df['input_text'] = train_df.apply(lambda row: str(row[0]).strip() + " " + str(row[1]).strip(), axis=1)
test_df['input_text'] = test_df.apply(lambda row: str(row[0]).strip() + " " + str(row[1]).strip(), axis=1)

label2id = {label: i for i, label in enumerate(train_df['true_label'].unique())}
train_df['label'] = train_df['true_label'].map(label2id)
test_df['label'] = test_df['true_label'].map(label2id)

train_dataset_full = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForSequenceClassification.from_pretrained('facebook/bart-base', num_labels=len(label2id))

model.gradient_checkpointing_enable()

def tokenize_function(examples):
    return tokenizer(examples['input_text'], truncation=True, padding='max_length', max_length=128)

tokenized_train_dataset_full = train_dataset_full.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

split_data = tokenized_train_dataset_full.train_test_split(test_size=0.15, seed=42)
train_dataset = split_data["train"]
eval_dataset = split_data["test"]

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
tokenized_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = preds.argmax(-1)
    labels = p.label_ids
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    
    # Log metrics to wandb
    wandb.log({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": classification_report(labels, preds, output_dict=True)
    })
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

training_args = TrainingArguments(
    output_dir=f'./results_{timestamp}',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    fp16=True,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=f'./logs_{timestamp}',
    logging_steps=10,
    metric_for_best_model='accuracy',
    dataloader_num_workers=4,
    report_to="wandb"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()

eval_results = trainer.evaluate(eval_dataset)
print(f"Evaluation Results: {eval_results}")

test_results = trainer.evaluate(tokenized_test_dataset)
print(f"Test Results: {test_results}")

wandb.log({
    "test_accuracy": test_results["eval_accuracy"],
    "test_loss": test_results["eval_loss"]
})

# Save the fine-tuned model and tokenizer
save_path = f'./fine_tuned_bart_cifar100_{timestamp}'
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model saved at: {save_path}")

# Finish wandb logging
wandb.finish()
