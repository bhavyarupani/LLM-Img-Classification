import os
import json
import datetime
import torch
import torch.nn as nn
import wandb
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    BartTokenizer,
    BartForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    BartConfig,
    TrainerCallback
)
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("Checking CUDA setup...")
assert torch.cuda.is_available(), "CUDA is not available. Please check your GPU setup."
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

wandb.login(key="")  
wandb.init(project="", entity="")


imagenet_data_path = "./Generated dataset/imagenet_train_descriptions_20250208_192003.json"
dataset_name = os.path.basename(imagenet_data_path).split("_")[0]

with open("./imagenet-simple-labels.json", "r") as f:
    imagenet_labels = json.load(f)


embedder = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
_global_ref = embedder.encode("universal semantic anchor", normalize_embeddings=True)


def compute_semantics_1(strA: str, strB: str) -> dict:
    """
    Compute multiple similarity/distance metrics between two texts.
    """
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
    """
    Loads JSON records, computes semantic metrics for each record,
    and builds a prompt including these metrics.
    """
    records = []
    # Define prompt templates for train and test.
    train_template = (
        "Observation A: {a}\nObservation B: {b}\n"
        "Additional Semantics: {extra}\n"
        "Analyze and describe the context."
    )
    test_template = (
        "Evaluation:\nData1: {a}\nData2: {b}\n"
        "Calculated Additional Semantics: {extra}"
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

            # First two fields are observations; third is true label (string)
            strA = str(fields[0]).strip()
            strB = str(fields[1]).strip()
            true_label = str(fields[2]).strip()

            # Compute semantic metrics (only using compute_semantics_1)
            sem_data = compute_semantics_1(strA, strB)
            extra_string = (
                f"blended_score={int(sem_data['blended_score'])}, "
                f"global_score={int(sem_data['global_score'])}, "
                f"cosine_similarity={sem_data['cosine_similarity']:.4f}, "
                f"euclidean_distance={sem_data['euclidean_distance']:.4f}, "
                f"manhattan_distance={sem_data['manhattan_distance']:.4f}, "
                f"dot_product={sem_data['dot_product']:.4f}, "
                f"correlation={sem_data['correlation']:.4f}"
            )
            prompt = template.format(a=strA, b=strB, extra=extra_string)

            records.append({
                "input_text": prompt,
                "labels": true_label,      # True label as a string (will be mapped later)
                "semantic": sem_data       # Extra semantic metrics for logging/analysis
            })
    return pd.DataFrame(records)


df = load_data(imagenet_data_path, mode="train")
unique_labels = sorted(df["labels"].unique())
label_to_id = {label: i for i, label in enumerate(unique_labels)}
df["labels"] = df["labels"].map(label_to_id)

# Split into train (70%), validation (15%), and test (15%)
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

dataset_train = Dataset.from_pandas(train_df)
dataset_val = Dataset.from_pandas(val_df)
dataset_test = Dataset.from_pandas(test_df)


tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

def tokenize_fn(examples):
    tokens = tokenizer(
        examples["input_text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )
    # Store extra semantic metrics (for logging/analysis)
    tokens["semantic"] = examples["semantic"]
    return tokens

dataset_train = dataset_train.map(tokenize_fn, batched=True)
dataset_val = dataset_val.map(tokenize_fn, batched=True)
dataset_test = dataset_test.map(tokenize_fn, batched=True)

dataset_train = dataset_train.remove_columns(["input_text"])
dataset_val = dataset_val.remove_columns(["input_text"])
dataset_test = dataset_test.remove_columns(["input_text"])


num_labels = len(label_to_id)
config = BartConfig.from_pretrained('facebook/bart-base', num_labels=num_labels)
model = BartForSequenceClassification.from_pretrained('facebook/bart-base', config=config)
model.gradient_checkpointing_enable()

def compute_metrics(p):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(logits, axis=1)
    labels = p.label_ids
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="weighted")
    recall = recall_score(labels, preds, average="weighted")
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def data_collator(features):
    batch = {k: torch.tensor([f[k] for f in features]) for k in features[0]}
    return batch


class MemoryCleanupCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        print("GPU cache cleared at epoch end.")
        return control


print("\n=== Phase 1: Training Classification Head (Encoder Frozen) ===")
for param in model.model.parameters():
    param.requires_grad = False
print("Encoder frozen; training only the classification head.")

phase1_output_dir = "./results/phase1_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
training_args_phase1 = TrainingArguments(
    output_dir=phase1_output_dir,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=3,  # Adjust epochs as needed
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,
    warmup_steps=500,
    label_smoothing_factor=0.1,
    fp16=True,
    learning_rate=3e-6,
    weight_decay=0.005,
    lr_scheduler_type="cosine",
    report_to="wandb",
    logging_steps=10,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True,
)

trainer_phase1 = Trainer(
    model=model,
    args=training_args_phase1,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5), MemoryCleanupCallback()]
)

trainer_phase1.train()

print("\n=== Phase 2: Fine-Tuning Encoder (Unfreezing Selected Layers) ===")
num_encoder_layers = len(model.model.encoder.layers)
for i, layer in enumerate(model.model.encoder.layers):
    if i >= num_encoder_layers - 2:
        for param in layer.parameters():
            param.requires_grad = True
print("Unfroze the last two encoder layers.")

phase2_output_dir = "./results/phase2_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
training_args_phase2 = TrainingArguments(
    output_dir=phase2_output_dir,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=5,  # Adjust epochs as needed
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,
    warmup_steps=500,
    label_smoothing_factor=0.1,
    fp16=True,
    learning_rate=3e-6,
    weight_decay=0.005,
    lr_scheduler_type="cosine",
    report_to="wandb",
    logging_steps=10,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True,
)

trainer_phase2 = Trainer(
    model=model,
    args=training_args_phase2,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5), MemoryCleanupCallback()]
)

trainer_phase2.train()

print("\n=== Evaluation on Test Set ===")
test_trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir='./results_test',
        per_device_eval_batch_size=32,
    ),
    eval_dataset=dataset_test,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)
results = test_trainer.evaluate()
print("Test Accuracy:", results.get("eval_accuracy"))


timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_save_path = f'./fine_tuned_bart_{dataset_name}_{timestamp}'
print(f"Model will be saved at: {model_save_path}")

os.makedirs(model_save_path, exist_ok=True)
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

wandb.finish()
