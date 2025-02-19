import os
import subprocess
import sys
import datetime

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
import torch
from sklearn.model_selection import train_test_split
import wandb

# Get timestamp for saving model
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Log in to wandb
wandb.login(key="---")

# Initialize wandb
wandb.init(project="cifar100-bart-classification", entity="bhavyarupani")

# Load the dataset
data_path = './Generated dataset/cifar100_descriptions/cifar100_descriptions_20241021_141553.json'  # Update with actual dataset path
df = pd.read_json(data_path, lines=True)

# Prepare the dataset for training
df['input_text'] = df['swin_description'] + " " + df['blip_description']  # Concatenate descriptions

# Map labels to integers
label2id = {label: i for i, label in enumerate(df['true_label'].unique())}
df['label'] = df['true_label'].map(label2id)

# Create a Dataset object
dataset = Dataset.from_pandas(df)

# Load tokenizer and model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForSequenceClassification.from_pretrained('facebook/bart-base', num_labels=len(label2id))

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['input_text'], truncation=True, padding='max_length', max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Split dataset into train (70%), validation (15%), and test (15%) using Hugging Face's `train_test_split`
split_data = tokenized_dataset.train_test_split(test_size=0.3, seed=42)
train_dataset = split_data["train"]
temp_data = split_data["test"]

# Further split temp_data into validation (15%) and test (15%)
split_temp = temp_data.train_test_split(test_size=0.5, seed=42)
eval_dataset = split_temp["train"]
test_dataset = split_temp["test"]

# Set format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])


# Compute metrics function
def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = preds.argmax(-1)  # Get predicted class indices
    labels = p.label_ids
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')

    # Log metrics to W&B
    wandb.log({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": classification_report(labels, preds, output_dict=True)
    })

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

# Training arguments
training_args = TrainingArguments(
    output_dir=f'./results_{timestamp}',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    fp16=torch.cuda.is_available(),
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=f'./logs_{timestamp}',
    logging_steps=10,
    metric_for_best_model='accuracy',
    dataloader_num_workers=4,
    report_to="wandb"
)

# Initialize Trainer with Early Stopping
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Train the model
trainer.train()

# Evaluate on validation set
eval_results = trainer.evaluate()
print(f"Validation Results: {eval_results}")

# Evaluate on test set
test_results = trainer.evaluate(test_dataset)
print(f"Test Results: {test_results}")

# Log test results to wandb
wandb.log({
    "test_accuracy": test_results["eval_accuracy"],
    "test_loss": test_results["eval_loss"]
})

# Save the fine-tuned model and tokenizer with timestamp
save_path = f'./fine_tuned_bart_cifar100_{timestamp}'
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model saved at: {save_path}")

# Finish wandb logging
wandb.finish()
