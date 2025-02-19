import os
import torch
import gc
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import BartTokenizer, BartForSequenceClassification
from datasets import Dataset
import json

output_dir = "./bart-image-classification-20241216-002600"  # Base folder for checkpoints

def get_latest_checkpoint(folder):
    print("Finding latest checkpoint...")
    checkpoints = [f for f in os.listdir(folder) if f.startswith("checkpoint-")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    return os.path.join(folder, checkpoints[-1]) if checkpoints else folder

latest_checkpoint = get_latest_checkpoint(output_dir)
print(f"Loading model from: {latest_checkpoint}")

# --------------------------
# Load Pretrained Model and Tokenizer
# --------------------------
print("Loading tokenizer...")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")  # Use base tokenizer

print("Loading model...")
model = BartForSequenceClassification.from_pretrained(latest_checkpoint)
model.eval()  # Set model to evaluation mode

print("Model and tokenizer loaded successfully.")
gc.collect()
torch.cuda.empty_cache()

# --------------------------
# Load Test Data
# --------------------------
data_path = "./Generated dataset/imagenet_descriptions_20241210_204654.json"

def load_data(file_path):
    data = []
    print("Loading test data...")
    with open(file_path, 'r') as f:
        for line in tqdm(f, desc="Reading data", unit=" lines"):
            if line.strip():
                item = json.loads(line.strip())
                combined_text = item['eva02_description'].strip() + " " + item['blip_description'].strip()
                label = item['true_label']
                data.append({'text': combined_text, 'label': label})
    return data

# Load and prepare dataset
data = load_data(data_path)
print("Converting to HuggingFace dataset...")
dataset = Dataset.from_list(data)
del data
gc.collect()
torch.cuda.empty_cache()

# Shuffle and split test data
print("Shuffling and splitting test data...")
dataset = dataset.shuffle(seed=42)
test_dataset = dataset.select(range(int(0.9 * len(dataset)), len(dataset)))
del dataset
gc.collect()
torch.cuda.empty_cache()

# --------------------------
# Tokenize Test Data
# --------------------------
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

print("Tokenizing test dataset...")
test_dataset = test_dataset.map(tokenize_function, batched=True, desc="Tokenizing")
test_dataset = test_dataset.remove_columns(["text"])
test_dataset = test_dataset.rename_column("label", "labels")
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
gc.collect()
torch.cuda.empty_cache()

# --------------------------
# Evaluation on Test Set
# --------------------------
def evaluate(model, test_dataset):
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8)
    all_preds = []
    all_labels = []

    print("Starting evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", unit=" batches"):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    print("Calculating metrics...")
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return accuracy, precision, recall, f1

# Run Evaluation
accuracy, precision, recall, f1 = evaluate(model, test_dataset)
print("Test Set Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

gc.collect()
torch.cuda.empty_cache()

