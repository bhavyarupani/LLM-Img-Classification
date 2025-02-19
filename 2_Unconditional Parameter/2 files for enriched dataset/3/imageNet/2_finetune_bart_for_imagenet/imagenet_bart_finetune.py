import os
import json
import datetime
import random
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
    BartConfig
)
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("Checking CUDA setup...")
assert torch.cuda.is_available(), "CUDA is not available. Please check your GPU setup."
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

wandb.login(key="")
wandb.init(project="", entity="")

train_data_path = "./Generated dataset/cifar100_descriptions/cifar100_train_descriptions_20250208_191331.json"
test_data_path = "./Generated dataset/cifar100_descriptions/cifar100_test_descriptions_20250208_191331.json"
dataset_name = os.path.basename(train_data_path).split("_")[0]

embedder = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
_global_ref = embedder.encode("universal semantic anchor", normalize_embeddings=True)


def compute_semantics_1(strA: str, strB: str) -> dict:
    embA = embedder.encode(strA, normalize_embeddings=True)
    embB = embedder.encode(strB, normalize_embeddings=True)

    blended_score = ((np.mean(embA) + np.mean(embB)) / 2) * 1000

    dotA = np.dot(embA, _global_ref) * 1000
    dotB = np.dot(embB, _global_ref) * 1000
    global_score = (dotA + dotB) / 2

    sim_score = float(util.cos_sim(embA, embB)[0][0])

    return {
        "blended_score": blended_score,
        "global_score": global_score,
        "similarity_score": sim_score
    }

def compute_semantics_2(extra_data: dict, strC: str, is_training: bool) -> int:
    a = 10 
    b = 2 

    primary_value = extra_data["blended_score"]
    secondary_value = extra_data["global_score"]

    try:
        seed_component = int(strC)
    except ValueError:
        seed_component = 0

    coefficient = (a ** 4) + seed_component + 1

    processed_primary = primary_value * coefficient
    processed_secondary = secondary_value * a
    combined = (processed_primary + processed_secondary) // (b * a)

    # added a tiny amount of noise in training mode
    if is_training and random.random() < 0.0:
        combined += random.choice([-1, 1])

    return combined
    

def load_data(path: str, mode: str = "train"):

    records = []
    metrics = set()

    train_template = (
        "Observation A: {a}\nObservation B: {b}\n"
        "Semantics -> {extra}\n"
        "Semantics => {m}\n"
        "Analyze and describe the context."
    )
    test_template = (
        "Evaluation:\nData1: {a}\nData2: {b}\n"
        "Semantics: {extra}\n"
        "Semantics => {m}"
    )
    template = train_template if mode == "train" else test_template
    is_train = (mode == "train")

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
            strC = str(fields[2]).strip()

            extra_data = compute_semantics_1(strA, strB)
            semantic = compute_semantics_2(extra_data, strC, is_train)

            record["sementic"] = semantic

            if len(fields) < 4:
                fields.append(semantic)
            else:
                fields[3] = semantic

            metrics.add(fields[3])

            extra_string = (
                f"blended_score={int(extra_data['blended_score'])}, "
                f"global_score={int(extra_data['global_score'])}, "
                f"similarity={extra_data['similarity_score']:.4f}"
            )
            prompt = template.format(a=strA, b=strB, extra=extra_string, m=semantic)

            records.append({
                "input_text": prompt,
                "labels": fields[3],
                "semantic_metric": semantic
            })

    df = pd.DataFrame(records)
    return df, metrics


class BartSequenceClassification(BartForSequenceClassification):
    def __init__(self, config: BartConfig, regression_loss_weight: float = 0.2):
        super().__init__(config)
        self.regression_loss_weight = regression_loss_weight
        self.regressor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1)
        )
        self.mse_loss = nn.MSELoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None, regression_target=None):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        pooled = outputs.encoder_last_hidden_state[:, 0, :]
        regression_pred = self.regressor(pooled).squeeze(-1)

        if regression_target is not None:
            regression_loss = self.mse_loss(regression_pred.float(), regression_target.float())
            outputs.loss = (outputs.loss if outputs.loss is not None else 0) + self.regression_loss_weight * regression_loss
            outputs.regression_loss = regression_loss

        outputs.regression_pred = regression_pred
        return outputs

train_df, train_label = load_data(train_data_path, mode="train")
test_df, test_label = load_data(test_data_path, mode="test")

all_label = sorted(set(train_label))
label_to_id = {m: i for i, m in enumerate(all_label)}

train_df["labels"] = train_df["labels"].map(label_to_id)
test_df["labels"] = test_df["labels"].map(label_to_id)

dataset_train = Dataset.from_pandas(train_df)
dataset_test = Dataset.from_pandas(test_df)

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

def tokenize_fn(examples):
    tokens = tokenizer(
        examples["input_text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )
    tokens["regression_target"] = examples["semantic_metric"]
    return tokens

dataset_train = dataset_train.map(tokenize_fn, batched=True)
dataset_test = dataset_test.map(tokenize_fn, batched=True)
dataset_train = dataset_train.remove_columns(["input_text"])
dataset_test = dataset_test.remove_columns(["input_text"])
train_split, val_split = train_test_split(dataset_train, test_size=0.15, random_state=42)
dataset_train = Dataset.from_dict(train_split)
dataset_val = Dataset.from_dict(val_split)

num_labels = len(label_to_id)
config = BartConfig.from_pretrained('facebook/bart-base', num_labels=num_labels)

model = BartSequenceClassification.from_pretrained(
    'facebook/bart-base',
    config=config,
    regression_loss_weight=0.2
)

model.gradient_checkpointing_enable()

def compute_metrics(p):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(logits, axis=1)
    labels = p.label_ids
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="weighted")
    recall = recall_score(labels, preds, average="weighted")
    f1 = f1_score(labels, preds, average="weighted")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def data_collator(features):
    batch = {}
    for k in features[0]:
        batch[k] = torch.tensor([f[k] for f in features])
    return batch

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"cifar100_bart_finetuning_{timestamp}"
model_save_path = f"./models/{model_name}"
checkpoint_dir = f"./results/{model_name}/checkpoints"

os.makedirs(model_save_path, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=checkpoint_dir,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        num_train_epochs=10.5,
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
    ),
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

trainer.train()

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
print("Test Precision:", results.get("eval_precision"))
print("Test Recall:", results.get("eval_recall"))
print("Test F1 Score:", results.get("eval_f1"))

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_save_path = f'./fine_tuned_bart_{dataset_name}_{timestamp}'
print(f"Model Saved at: {model_save_path}")

model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

wandb.finish()
