import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from transformers import AutoModelForImageClassification, BlipProcessor, BlipForConditionalGeneration
import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime

# Define the transformation for the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure the image is resized to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load CIFAR-10 training and test datasets separately
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Get CIFAR-10 class labels
classes = train_dataset.classes  

# Create DataLoaders for train and test datasets
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

# Check if CUDA is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {'GPU' if device.type == 'cuda' else 'CPU'}")

# Load the pretrained BEiT model
model_name = 'microsoft/beit-large-patch16-224'
model = AutoModelForImageClassification.from_pretrained(model_name, num_labels=10, ignore_mismatched_sizes=True)

# Modify the classifier to match CIFAR-10 classes
model.classifier = nn.Linear(model.config.hidden_size, 10)

# Load the checkpoint
checkpoint_path = './final_fine_tuned_beit_cifar10_new.pth'
try:
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
except RuntimeError as e:
    print(f"Error loading model checkpoint: {e}")
    exit()

# Move model to the appropriate device
model.to(device)
model.eval()

# Load BLIP model and processor
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

# Create output directory
output_dir = 'Generated dataset'
os.makedirs(output_dir, exist_ok=True)

# Generate unique output file paths
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
train_output_path = os.path.join(output_dir, f'cifar10_train_descriptions_{timestamp}.json')
test_output_path = os.path.join(output_dir, f'cifar10_test_descriptions_{timestamp}.json')

# Function to process dataset
def process_dataset(data_loader, output_path, dataset_type):
    data = []
    pbar = tqdm(data_loader, desc=f'Processing {dataset_type} images')

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # Predict the class with the BEiT model
        with torch.no_grad():
            outputs = model(images).logits
            _, predicted_class_indices = torch.max(outputs, 1)
            predicted_classes = [classes[idx] for idx in predicted_class_indices]

        # Create descriptions using the BEiT predictions and BLIP
        for image, predicted_class, label in zip(images, predicted_classes, labels):
            beit_description = f"This is a photo of a {predicted_class}."

            # Generate BLIP description
            image_pil = transforms.ToPILImage()(image.cpu()).convert('RGB')
            inputs_blip = blip_processor(text= beit_description, images=image_pil, return_tensors="pt").to(device)
            out_blip = blip_model.generate(**inputs_blip)
            blip_description = blip_processor.decode(out_blip[0], skip_special_tokens=True)

            # Collect data for the dataset
            data.append({
                'beit_description': beit_description,
                'blip_description': blip_description,
                'true_label': classes[label]
            })

    # Save dataset to JSON file
    df = pd.DataFrame(data)
    df.to_json(output_path, orient='records', lines=True)
    print(f"{dataset_type.capitalize()} dataset saved at {output_path}")

# Process and save training dataset
process_dataset(train_loader, train_output_path, "train")

# Process and save test dataset
process_dataset(test_loader, test_output_path, "test")
