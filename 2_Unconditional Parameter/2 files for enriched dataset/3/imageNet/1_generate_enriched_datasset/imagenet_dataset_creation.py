import os
import sys
import subprocess
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

# Set environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Adjust if needed

# Check CUDA availability
print("Checking CUDA setup...")
assert torch.cuda.is_available(), "CUDA is not available. Please check your GPU setup."
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Required packages
required_packages = [
    "tqdm", "timm", "wandb", "pillow", "datasets", "torchvision", "transformers", "scikit-learn", "huggingface_hub"
]

# Install missing packages
missing_packages = [pkg for pkg in required_packages if not subprocess.call([sys.executable, "-m", "pip", "show", pkg], stdout=subprocess.DEVNULL)]
if missing_packages:
    print(f"Installing missing packages: {missing_packages}")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)

# Log into Hugging Face Hub
huggingface_token = "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"  # Replace with your token
login(token=huggingface_token)

# W&B login
wandb_key = "3fefb189f1095897698330024d268bc868773005"  # Replace with your W&B key
wandb.login(key=wandb_key)
wandb.init(project="imagenet-classification", entity="bhavyarupani")

print("Loading ImageNet-1k datasets...")
try:
    train_dataset = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True, trust_remote_code=True)
    test_dataset = load_dataset("ILSVRC/imagenet-1k", split="test", streaming=True, trust_remote_code=True)
except Exception as e:
    print(f"Error loading datasets: {e}")
    raise

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {'GPU' if device.type == 'cuda' else 'CPU'}")

print("Initializing EVA-02 model and transforms...")
try:
    # EVA-02 model from timm
    model = timm.create_model('eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=True).to(device)
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

# Generate unique output file names
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
train_output_path = os.path.join(output_dir, f'imagenet_train_descriptions_{timestamp}.json')
test_output_path = os.path.join(output_dir, f'imagenet_test_descriptions_{timestamp}.json')

# Function to process dataset and save to file
def process_dataset(dataset, output_path, dataset_type, total_samples):
    print(f"Processing {dataset_type} dataset...")

    with open(output_path, 'a') as f_output:
        pbar = tqdm(total=total_samples, desc=f"Processing {dataset_type} images", leave=False)

        for example in dataset:
            image = example['image']
            label = example['label']

            if image.mode != "RGB":
                image = image.convert("RGB")

            # Transform and classify with EVA-02 model
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                predicted_class_idx = output.argmax(dim=-1).item()
                class_name = dataset.features['label'].int2str(predicted_class_idx).split(',')[0]

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
        print(f"{dataset_type.capitalize()} dataset saved at {output_path}")

# Process and save training dataset
process_dataset(train_dataset, train_output_path, "training", total_samples=1281167)

# Process and save test dataset
process_dataset(test_dataset, test_output_path, "test", total_samples=50000)