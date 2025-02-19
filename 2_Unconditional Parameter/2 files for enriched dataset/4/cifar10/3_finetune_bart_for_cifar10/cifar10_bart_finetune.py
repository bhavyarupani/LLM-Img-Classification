import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BartTokenizer, BartModel, AdamW, get_linear_schedule_with_warmup
from sentence_transformers import SentenceTransformer
import wandb

# ----------------- Environment Setup -----------------
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("Checking CUDA setup...")
assert torch.cuda.is_available(), "CUDA is not available. Please check your GPU setup."
device = torch.device("cuda")
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

wandb.login(key="")
wandb.init(project="", entity="")

# File Paths
train_data_path = "./Generated dataset/cifar10_train_descriptions_20250208_191623.json"
test_data_path  = "./Generated dataset/cifar10_test_descriptions_20250208_191623.json"
dataset_name = os.path.basename(train_data_path).split("_")[0]

# ----------------- Label Mapping -----------------
label2id = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9
}
id2label = {v: k for k, v in label2id.items()}

# ----------------- Dataset Definition -----------------
class CIFAR10DualDescriptionDataset(Dataset):
    """
    Each example contains:
      - 'beit_text': Structured description (e.g., "This is a photo of a cat.")
      - 'blip_text': Generated caption from the captioner.
    The structured text is tokenized for BART and the raw caption text is returned
    for semantic embedding.
    """
    def __init__(self, file_path, tokenizer, label2id, max_length=128):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    beit_text = data['beit_description'].strip()
                    blip_text = data['blip_description'].strip()
                    label = label2id[data['true_label']]
                    self.examples.append({
                        "beit_text": beit_text,
                        "blip_text": blip_text,
                        "label": label
                    })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        # Tokenize the structured description for BART
        beit_enc = self.tokenizer(
            example["beit_text"],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        beit_enc = {key: val.squeeze(0) for key, val in beit_enc.items()}
        label = torch.tensor(example["label"], dtype=torch.long)
        # Return the raw caption text for semantic encoding.
        return {
            "beit_input_ids": beit_enc["input_ids"],
            "beit_attention_mask": beit_enc["attention_mask"],
            "blip_raw": example["blip_text"],
            "labels": label
        }

# ----------------- Custom Model with Semantic Embeddings & Auxiliary Loss -----------------
class DualInputBartClassifierSemanticAux(nn.Module):

    def __init__(self, bart_model_name, num_labels, sem_model_name='paraphrase-MiniLM-L6-v2', aux_loss_weight=0.5):
        super(DualInputBartClassifierSemanticAux, self).__init__()
        self.bart = BartModel.from_pretrained(bart_model_name)
        self.tokenizer = BartTokenizer.from_pretrained(bart_model_name)
        # Load SentenceTransformer as a fixed semantic encoder.
        self.semantic_model = SentenceTransformer(sem_model_name)
        self.semantic_model.to(device)
        self.semantic_model.eval()  # Freeze the semantic model

        hidden_size = self.bart.config.d_model  # e.g., 768 for bart-base

        # SentenceTransformer 'paraphrase-MiniLM-L6-v2' produces 384-dim embeddings.
        sem_dim = 384
        # Project semantic embeddings to match BART hidden size.
        self.sem_proj = nn.Linear(sem_dim, hidden_size)
        # Fusion classifier: concatenates BART representation with projected semantic features.
        self.fusion_classifier = nn.Linear(hidden_size * 2, num_labels)
        # Auxiliary classifier solely on the semantic branch.
        self.aux_classifier = nn.Linear(hidden_size, num_labels)
        self.aux_loss_weight = aux_loss_weight

    def forward(self, beit_input_ids, beit_attention_mask, blip_raw, labels=None):
        # Encode the structured description using BART's encoder.
        bart_outputs = self.bart.encoder(input_ids=beit_input_ids, attention_mask=beit_attention_mask)
        beit_repr = bart_outputs.last_hidden_state[:, 0, :]  # use the first token representation

        # Compute semantic embeddings from the raw caption text.
        # SentenceTransformer expects a list of strings.
        with torch.no_grad():
            sem_emb = self.semantic_model.encode(blip_raw, convert_to_tensor=True, device=beit_input_ids.device)
        # Project semantic embeddings to match BART's hidden size.
        sem_proj = self.sem_proj(sem_emb)  # shape: (batch, hidden_size)

        # Fusion representation: concatenate structured and semantic representations.
        fusion_repr = torch.cat((beit_repr, sem_proj), dim=1)  # shape: (batch, hidden_size*2)
        fusion_logits = self.fusion_classifier(fusion_repr)
        # Auxiliary logits computed solely from the semantic branch.
        aux_logits = self.aux_classifier(sem_proj)

        loss = None
        if labels is not None:
            fusion_loss = F.cross_entropy(fusion_logits, labels)
            aux_loss = F.cross_entropy(aux_logits, labels)
            loss = fusion_loss + self.aux_loss_weight * aux_loss

        return {"loss": loss, "fusion_logits": fusion_logits, "aux_logits": aux_logits}

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = DualInputBartClassifierSemanticAux('facebook/bart-base', num_labels=len(label2id), aux_loss_weight=0.5)
model.to(device)

train_dataset_full = CIFAR10DualDescriptionDataset(train_data_path, tokenizer, label2id, max_length=128)
test_dataset = CIFAR10DualDescriptionDataset(test_data_path, tokenizer, label2id, max_length=128)

train_size = int(0.9 * len(train_dataset_full))
val_size = len(train_dataset_full) - train_size
train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])
print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_epochs = 3
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        beit_input_ids = batch["beit_input_ids"].to(device)
        beit_attention_mask = batch["beit_attention_mask"].to(device)
        # 'blip_raw' remains as a list of strings.
        blip_raw = batch["blip_raw"]
        labels = batch["labels"].to(device)

        outputs = model(
            beit_input_ids=beit_input_ids,
            beit_attention_mask=beit_attention_mask,
            blip_raw=blip_raw,
            labels=labels
        )
        loss = outputs["loss"]
        logits = outputs["fusion_logits"]
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            beit_input_ids = batch["beit_input_ids"].to(device)
            beit_attention_mask = batch["beit_attention_mask"].to(device)
            blip_raw = batch["blip_raw"]
            labels = batch["labels"].to(device)

            outputs = model(
                beit_input_ids=beit_input_ids,
                beit_attention_mask=beit_attention_mask,
                blip_raw=blip_raw,
                labels=labels
            )
            loss = outputs["loss"]
            logits = outputs["fusion_logits"]
            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

# ----------------- Main Training Loop -----------------
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
    val_loss, val_acc = evaluate(model, val_loader, device)

    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"  Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")
    print(f"  Val Loss:   {val_loss:.4f} | Val Accuracy:   {val_acc:.4f}")

    # Log metrics and classifier bias (if available) to wandb.
    if hasattr(model, "fusion_classifier") and hasattr(model.fusion_classifier, "bias"):
        bias_vals = model.fusion_classifier.bias.detach().cpu().numpy()
        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "classifier_bias": wandb.Histogram(bias_vals)
        })
    else:
        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })

# ----------------- Final Test Evaluation -----------------
test_loss, test_acc = evaluate(model, test_loader, device)
print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
wandb.log({"test_loss": test_loss, "test_accuracy": test_acc})

# ----------------- Save the Model -----------------
save_path = f"./{dataset_name}_bart_semantic_aux_model"
os.makedirs(save_path, exist_ok=True)
torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
tokenizer.save_pretrained(save_path)
print(f"Model and tokenizer saved to {save_path}")
