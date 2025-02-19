import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
import torchvision
from transformers import AutoModelForImageClassification, AdamW, get_scheduler
import wandb

wandb.init(project="")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)  # Further reduced batch size
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)  # Further reduced batch size

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {'GPU' if device.type == 'cuda' else 'CPU'}")

model_name = 'microsoft/beit-large-patch16-224'
model = AutoModelForImageClassification.from_pretrained(model_name, num_labels=10, ignore_mismatched_sizes=True)  # CIFAR-10 has 10 classes

checkpoint = torch.load('./checkpoints/pretrained_cifar10_head_for_microsoft-beit-large-patch16-224.state_dict', map_location=device)
model.load_state_dict(checkpoint, strict=False) 

model.classifier = nn.Linear(model.config.hidden_size, 10)
model.to(device)  # Move model to the appropriate device

for name, param in model.named_parameters():
    if 'classifier' not in name:
        param.requires_grad = False

optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 5
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

accumulation_steps = 8  # Increased accumulation steps to reduce memory usage per step
scaler = torch.cuda.amp.GradScaler()  # For mixed precision training
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    optimizer.zero_grad()
    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")):
        images, labels = images.to(device), labels.to(device)  # Move data to the appropriate device
        
        with torch.cuda.amp.autocast():
            outputs = model(images).logits
            loss = nn.CrossEntropyLoss()(outputs, labels)
        
        loss = loss / accumulation_steps
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()  # Clear cache periodically
        
        running_loss += loss.item() * accumulation_steps

        # Log the loss to wandb
        wandb.log({"batch_loss": loss.item() * accumulation_steps, "epoch": epoch + 1})

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss}")
    wandb.log({"epoch_loss": epoch_loss, "epoch": epoch + 1})

    # Save model checkpoint
    torch.save(model.state_dict(), f'./checkpoint_epoch_{epoch+1}.pth')
    wandb.save(f'./checkpoint_epoch_{epoch+1}.pth')

# Unfreeze some layers for further fine-tuning
for name, param in model.named_parameters():
    if 'beit.encoder.layer' in name:
        param.requires_grad = True

# Fine-tune with unfreezing layers
optimizer = AdamW(model.parameters(), lr=1e-5)
num_epochs = 4  # Additional epochs
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    optimizer.zero_grad()
    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")):
        images, labels = images.to(device), labels.to(device)  # Move data to the appropriate device
        
        with torch.cuda.amp.autocast():
            outputs = model(images).logits
            loss = nn.CrossEntropyLoss()(outputs, labels)
        
        loss = loss / accumulation_steps
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()  # Clear cache periodically
        
        running_loss += loss.item() * accumulation_steps

        # Log the loss to wandb
        wandb.log({"batch_loss": loss.item() * accumulation_steps, "epoch": epoch + 1})

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss}")
    wandb.log({"epoch_loss": epoch_loss, "epoch": epoch + 1})

    # Save model checkpoint
    torch.save(model.state_dict(), f'./checkpoint_epoch_{epoch+1+3}.pth')
    wandb.save(f'./checkpoint_epoch_{epoch+1+3}.pth')

model_save_path = './fine_tuned_beit_cifar10.pth'
torch.save(model.state_dict(), model_save_path)
wandb.save(model_save_path)

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for i, (images, labels) in enumerate(tqdm(test_loader, desc="Evaluating", unit="batch")):
        images, labels = images.to(device), labels.to(device)  # Move data to the appropriate device
        
        with torch.cuda.amp.autocast():
            outputs = model(images).logits
        
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Clear cache to free up memory
        if i % 10 == 0:
            torch.cuda.empty_cache()

accuracy = 100 * sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
print(f'Accuracy of the model on the CIFAR-10 test images: {accuracy:.2f}%')
wandb.log({"accuracy": accuracy})

conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
wandb.log({"confusion_matrix": wandb.Image(plt)})

class_report = classification_report(all_labels, all_preds, target_names=test_dataset.classes, output_dict=True)
df_class_report = pd.DataFrame(class_report).transpose()
print(df_class_report)
wandb.log({"classification_report": df_class_report})

def imshow(img, title):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(10, 5))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()
    wandb.log({"sample_predictions": wandb.Image(plt, caption=title)})

dataiter = iter(test_loader)
images, labels = dataiter.next()
images, labels = images.to(device), labels.to(device)

with torch.cuda.amp.autocast():
    outputs = model(images).logits

_, predicted = torch.max(outputs, 1)

imshow(torchvision.utils.make_grid(images.cpu()), title=[f'Pred: {test_dataset.classes[pred]}, True: {test_dataset.classes[label]}' for pred, label in zip(predicted, labels)])

final_model_save_path = './final_fine_tuned_beit_cifar10_new.pth'
torch.save(model.state_dict(), final_model_save_path)
wandb.save(final_model_save_path)