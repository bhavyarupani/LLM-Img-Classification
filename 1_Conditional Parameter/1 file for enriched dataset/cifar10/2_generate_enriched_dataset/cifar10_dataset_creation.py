import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from transformers import AutoModelForImageClassification, BlipProcessor, BlipForConditionalGeneration
import pandas as pd
import os
from tqdm import tqdm 

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
classes = dataset.classes  # Get CIFAR-10 class names
test_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {'GPU' if device.type == 'cuda' else 'CPU'}")

model_name = "microsoft/beit-large-patch16-224"
model = AutoModelForImageClassification.from_pretrained(model_name, num_labels=10, ignore_mismatched_sizes=True)

# Modify classifier to match CIFAR-10 classes
model.classifier = nn.Linear(model.config.hidden_size, 10)

# Load the fine-tuned checkpoint
checkpoint_path = "./final_fine_tuned_beit_cifar10_new.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))

# Move BEiT model to the device
model.to(device)
model.eval()  # Set BEiT to evaluation mode

# Load BLIP model and processor
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

# Directory where the dataset will be saved
output_dir = "Generated dataset"
os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

# Initialize list to store data
data = []

# Process images using tqdm for script execution
for images, labels in tqdm(test_loader, desc="Processing Images", dynamic_ncols=True):
    images = images.to(device)
    labels = labels.to(device)

    # Predict the class using the BEiT model
    with torch.no_grad():
        outputs = model(images).logits
        _, predicted_class_indices = torch.max(outputs, 1)
        predicted_classes = [classes[idx] for idx in predicted_class_indices]

    # Generate descriptions using BEiT predictions and BLIP
    for image, predicted_class, label in zip(images, predicted_classes, labels):
        beit_description = f"This is a photo of a {predicted_class}."

        # FIX: Use PIL image directly, no need for Image.fromarray()
        image_pil = transforms.ToPILImage()(image.cpu()).convert("RGB")  # Corrected
        inputs_blip = blip_processor(text=beit_description, images=image_pil, return_tensors="pt").to(device)
        
        # Generate BLIP description
        with torch.no_grad():
            out_blip = blip_model.generate(**inputs_blip)
        
        blip_description = blip_processor.decode(out_blip[0], skip_special_tokens=True)

        # Collect data
        data.append({
            "beit_description": beit_description,
            "blip_description": blip_description,
            "true_label": classes[label]
        })

# Convert collected data to a DataFrame
df = pd.DataFrame(data)

# Save dataset as a JSON file
output_path = os.path.join(output_dir, "cifar10_descriptions_full.json")
df.to_json(output_path, orient="records", lines=True)

print(f"Dataset with BEiT & BLIP descriptions saved at: {output_path}")
