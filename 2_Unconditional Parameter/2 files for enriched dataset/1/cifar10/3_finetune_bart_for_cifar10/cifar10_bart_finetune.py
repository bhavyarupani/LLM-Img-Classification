import os
import json
import datetime
import random
import torch
import wandb
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    BartTokenizer,
    BartForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------------
# Environment Setup
# -------------------------------------------------------------------------
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("Checking CUDA setup...")
assert torch.cuda.is_available(), "CUDA is not available. Please check your GPU setup."
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Login to Weights & Biases
wandb.login(key="")
wandb.init(project="", entity="")


train_data_path = "./Generated dataset/cifar10_train_descriptions_20250208_191623.json"
test_data_path  = "./Generated dataset/cifar10_test_descriptions_20250208_191623.json"
dataset_name = os.path.basename(train_data_path).split("_")[0]

embedder = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
_global_ref = embedder.encode("universal semantic anchor", normalize_embeddings=True)

def compute_semantics_1(strA: str, strB: str) -> dict:
    embA = embedder.encode(strA, normalize_embeddings=True)
    embB = embedder.encode(strB, normalize_embeddings=True)
    
    # Basic metrics
    blended_score = ((np.mean(embA) + np.mean(embB)) / 2) * 1000
    dotA = np.dot(embA, _global_ref) * 1000
    dotB = np.dot(embB, _global_ref) * 1000
    global_score = (dotA + dotB) / 2
    cosine_sim = float(util.cos_sim(embA, embB)[0][0])
    
    # Additional metrics
    euclidean_distance = float(np.linalg.norm(embA - embB))
    manhattan_distance = float(np.sum(np.abs(embA - embB)))
    dot_product = float(np.dot(embA, embB))
    if np.std(embA) > 0 and np.std(embB) > 0:
        correlation = float(np.corrcoef(embA, embB)[0, 1])
    else:
        correlation = 0.0

    return {
        "blended_score": blended_score,
        "global_score": global_score,
        "cosine_similarity": cosine_sim,
        "euclidean_distance": euclidean_distance,
        "manhattan_distance": manhattan_distance,
        "dot_product": dot_product,
        "correlation": correlation
    }


def load_data(path: str, mode: str = "train") -> pd.DataFrame:
    records = []
    # Simple prompt templates (train vs. test)
    train_template = (
        "Observation A: {a}\nObservation B: {b}\n"
        "Additional Semantics: {extra}\n"
        "Analyze and describe the context."
    )
    test_template = (
        "Evaluation:\nData1: {a}\nData2: {b}\n"
        "Calculated Additional Semantics: {extra}\n"
    )
    template = train_template if mode == "train" else test_template

    with open(path, 'r') as f:
        for line in tqdm(f, desc=f"Processing {mode} data from {path}", dynamic_ncols=True):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            fields = list(record.values())
            if len(fields) < 3:
                continue

            strA = str(fields[0]).strip()
            strB = str(fields[1]).strip()
            true_label = str(fields[2]).strip()
            
            # Compute extra semantic metrics for prompt text
            extra_data = compute_semantics_1(strA, strB)
            extra_string = (
                f"blended_score={int(extra_data['blended_score'])}, "
                f"global_score={int(extra_data['global_score'])}, "
                f"cosine_similarity={extra_data['cosine_similarity']:.4f}, "
                f"euclidean_distance={extra_data['euclidean_distance']:.4f}, "
                f"manhattan_distance={extra_data['manhattan_distance']:.4f}, "
                f"dot_product={extra_data['dot_product']:.4f}, "
                f"correlation={extra_data['correlation']:.4f}"
            )
            prompt = template.format(a=strA, b=strB, extra=extra_string)

            records.append({
                "input_text": prompt,
                "labels": true_label
            })
    df = pd.DataFrame(records)
    return df

train_df = load_data(train_data_path, mode="train")
test_df  = load_data(test_data_path, mode="test")

all_label_strings = sorted(set(train_df["labels"].unique()) | set(test_df["labels"].unique()))
label_to_id = {label_str: i for i, label_str in enumerate(all_label_strings)}

train_df["labels"] = train_df["labels"].map(label_to_id)
test_df["labels"]  = test_df["labels"].map(label_to_id)

dataset_train = Dataset.from_pandas(train_df)
dataset_test  = Dataset.from_pandas(test_df)

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

def tokenize_fn(examples):
    return tokenizer(
        examples["input_text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

dataset_train = dataset_train.map(tokenize_fn, batched=True)
dataset_test  = dataset_test.map(tokenize_fn, batched=True)

dataset_train = dataset_train.remove_columns(["input_text"])
dataset_test  = dataset_test.remove_columns(["input_text"])

train_split, val_split = train_test_split(dataset_train, test_size=0.15, random_state=42)
dataset_train = Dataset.from_dict(train_split)
dataset_val   = Dataset.from_dict(val_split)

num_labels = len(label_to_id)
model = BartForSequenceClassification.from_pretrained(
    "facebook/bart-base",
    num_labels=num_labels
)
model.cuda()  # If you want to ensure it’s on GPU explicitly

def compute_metrics(p):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(logits, axis=1)
    labels = p.label_ids
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def custom_data_collator(features):
    # features is a list of dicts
    batch = {
        "input_ids": torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
        "attention_mask": torch.tensor([f["attention_mask"] for f in features], dtype=torch.long),
        "labels": torch.tensor([f["labels"] for f in features], dtype=torch.long),
    }
    return batch


timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"./results_bart_classification_{dataset_name}_{timestamp}"

training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="steps",      # Evaluate more frequently for debugging
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    num_train_epochs=3,              # Adjust as needed
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_steps=100,
    fp16=True,
    learning_rate=2e-5,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    report_to="wandb",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    compute_metrics=compute_metrics,
    data_collator=custom_data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()

test_metrics = trainer.evaluate(dataset_test)
print("Test Accuracy:", test_metrics.get("eval_accuracy"))
print("Test Precision:", test_metrics.get("eval_precision"))
print("Test Recall:", test_metrics.get("eval_recall"))
print("Test F1 Score:", test_metrics.get("eval_f1"))


model_save_path = f"./fine_tuned_bart_{dataset_name}_{timestamp}"
print(f"Model will be saved at: {model_save_path}")

os.makedirs(model_save_path, exist_ok=True)
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

wandb.finish()
