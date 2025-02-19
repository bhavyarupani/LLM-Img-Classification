import os
import sys
import subprocess

# Set environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Adjust if needed

# Check CUDA availability
import torch
print("Checking CUDA setup...")
assert torch.cuda.is_available(), "CUDA is not available. Please check your GPU setup."
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Required packages
required_packages = [
    "tqdm",
    "timm",
    "wandb",
    "pillow",
    "datasets",
    "torchvision",
    "transformers",
    "scikit-learn",
    "huggingface_hub"
]

# Install missing packages
missing_packages = []
for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        missing_packages.append(pkg)

if missing_packages:
    print(f"Some required packages are missing: {missing_packages}")
    print("Installing them...")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)

# Re-import after installation
import torch
from PIL import Image
import timm
from tqdm import tqdm
from datasets import load_dataset
from timm.data import resolve_model_data_config, create_transform
import wandb
from transformers import BlipProcessor, BlipForConditionalGeneration
from huggingface_hub import login
import pandas as pd
from datetime import datetime

# Log into Hugging Face Hub
huggingface_token = "---"  # Replace with your token
login(token=huggingface_token)

# W&B login
wandb_key = "---"  # Replace with your W&B key
wandb.login(key=wandb_key)
wandb.init(project="imagenet-classification", entity="--")

print("Loading ImageNet-1k datasets...")
try:
    train_dataset = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True, trust_remote_code=True)
except Exception as e:
    print(f"Error loading datasets: {e}")
    raise

# Total images in dataset
total_images = 1281167 + 50000
print(f"Total images: {total_images}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {'GPU' if device.type == 'cuda' else 'CPU'}")

print("Initializing EVA-02 model and transforms...")
try:
    # EVA-02 model from timm
    model = timm.create_model('eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=True)
    model = model.to(device)
    model.eval()

    # Transforms
    data_config = resolve_model_data_config(model)
    transform = create_transform(**data_config, is_training=False)
except Exception as e:
    print(f"Error initializing EVA-02 model: {e}")
    raise

print("Loading BLIP model and processor...")
try:
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
except Exception as e:
    print(f"Error loading BLIP model: {e}")
    raise

# Directory where the dataset will be saved
output_dir = 'Generated dataset'
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(output_dir, f'imagenet_descriptions_{timestamp}.json')
print(f"Output path: {output_path}")

with open(output_path, 'a') as f_output:
    pbar = tqdm(total=total_images, desc="Processing images", leave=False)

    for example in train_dataset:
        image = example['image']
        label = example['label']

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Transform and classify with EVA-02 model
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            predicted_class_idx = output.argmax(dim=-1).item()
            class_name = train_dataset.features['label'].int2str(predicted_class_idx).split(',')[0]

        eva02_description = f"This is a photo of a {class_name}."

        # Generate BLIP description
        inputs_blip = blip_processor(text=eva02_description, images=image, return_tensors="pt").to(device)
        out_blip = blip_model.generate(**inputs_blip)
        blip_description = blip_processor.decode(out_blip[0], skip_special_tokens=True)

        data_entry = {
            'eva02_description': eva02_description,
            'blip_description': blip_description,
            'true_label': label
        }

        f_output.write(f'{pd.Series(data_entry).to_json()}\n')

        pbar.update(1)

pbar.close()
print(f"Dataset with EVA-02 and BLIP descriptions created and saved at {output_path}.")