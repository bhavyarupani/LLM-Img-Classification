import os
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, BlipProcessor, BlipForConditionalGeneration
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login
from datetime import datetime

# Log in to Hugging Face Hub
login(token="")

# Load the CIFAR-100 dataset from the uoft-cs repository with streaming
print("Loading datasets...")
try:
    train_dataset = load_dataset("uoft-cs/cifar100", split="train", streaming=True)
    test_dataset = load_dataset("uoft-cs/cifar100", split="test", streaming=True)
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

# Initialize the image processor and model
processor = AutoImageProcessor.from_pretrained("MazenAmria/swin-base-finetuned-cifar100")
model = AutoModelForImageClassification.from_pretrained("MazenAmria/swin-base-finetuned-cifar100")

# Move the model to the appropriate device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Initialize BLIP model and processor
print("Loading BLIP model and processor...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

# Create output directory
output_dir = 'Generated dataset/cifar100_descriptions'
os.makedirs(output_dir, exist_ok=True)

# Generate a unique output path with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(output_dir, f'cifar100_descriptions_{timestamp}.json')
print(f"Output path: {output_path}")

# Initialize a file in append mode to store the JSON dataset
with open(output_path, 'a') as f_output:
    # Process the training dataset
    print("Processing training dataset...")
    pbar = tqdm(total=50000, desc="Processing training images", leave=False)

    for example in train_dataset:
        image = example['img']
        label = example['fine_label']

        # Ensure the image is in RGB format (convert if grayscale)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess the image
        inputs = processor(images=image, return_tensors="pt").to(device)

        # Predict the class with the Swin model
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class_idx = outputs.logits.argmax(dim=-1).item()
            class_name = train_dataset.features['fine_label'].int2str(predicted_class_idx).split(',')[0]

        # Create Swin description                         
        swin_description = f"This is a photo of a {class_name}."

        # Generate BLIP description
        inputs_blip = blip_processor(text=swin_description, images=image, return_tensors="pt").to(device)
        out_blip = blip_model.generate(**inputs_blip)
        blip_description = blip_processor.decode(out_blip[0], skip_special_tokens=True)

        # Prepare the data entry
        data_entry = {
            'swin_description': swin_description,
            'blip_description': blip_description,
            'true_label': label
        }

        # Write the entry to the JSON file in line format
        f_output.write(f'{pd.Series(data_entry).to_json()}\n')
        pbar.update(1)

    pbar.close()

    # Process the test dataset
    print("Processing test dataset...")
    pbar = tqdm(total=10000, desc="Processing test images", leave=False)

    for example in test_dataset:
        image = example['img']
        label = example['fine_label']

        # Ensure the image is in RGB format (convert if grayscale)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess the image
        inputs = processor(images=image, return_tensors="pt").to(device)

        # Predict the class with the Swin model
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class_idx = outputs.logits.argmax(dim=-1).item()
            class_name = test_dataset.features['fine_label'].int2str(predicted_class_idx).split(',')[0]

        # Create Swin description
        swin_description = f"This is a photo of a {class_name}."

        # Generate BLIP description
        inputs_blip = blip_processor(text=swin_description, images=image, return_tensors="pt").to(device)
        out_blip = blip_model.generate(**inputs_blip)
        blip_description = blip_processor.decode(out_blip[0], skip_special_tokens=True)

        # Prepare the data entry
        data_entry = {
            'swin_description': swin_description,
            'blip_description': blip_description,
            'true_label': label
        }

        # Write the entry to the JSON file in line format
        f_output.write(f'{pd.Series(data_entry).to_json()}\n')
        pbar.update(1)

    pbar.close()

print(f'Dataset with Swin and BLIP descriptions created and saved at {output_path}.')