import os
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, BlipProcessor, BlipForConditionalGeneration
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login
from datetime import datetime


login(token="")


print("Loading datasets...")
try:
    train_dataset = load_dataset("uoft-cs/cifar100", split="train", streaming=True)
    test_dataset = load_dataset("uoft-cs/cifar100", split="test", streaming=True)
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

processor = AutoImageProcessor.from_pretrained("MazenAmria/swin-base-finetuned-cifar100")
model = AutoModelForImageClassification.from_pretrained("MazenAmria/swin-base-finetuned-cifar100")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print("Loading BLIP model and processor...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

output_dir = 'Generated dataset/cifar100_descriptions'
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
train_output_path = os.path.join(output_dir, f'cifar100_train_descriptions_{timestamp}.json')
test_output_path = os.path.join(output_dir, f'cifar100_test_descriptions_{timestamp}.json')

print(f"Train output path: {train_output_path}")
print(f"Test output path: {test_output_path}")

with open(train_output_path, 'a') as f_train:
    print("Processing training dataset...")
    pbar = tqdm(total=50000, desc="Processing training images", leave=False)

    for example in train_dataset:
        image = example['img']
        label = example['fine_label']

        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class_idx = outputs.logits.argmax(dim=-1).item()
            class_name = train_dataset.features['fine_label'].int2str(predicted_class_idx).split(',')[0]

        swin_description = f"This is a photo of a {class_name}."

        inputs_blip = blip_processor(text=swin_description, images=image, return_tensors="pt").to(device)
        out_blip = blip_model.generate(**inputs_blip)
        blip_description = blip_processor.decode(out_blip[0], skip_special_tokens=True)

        data_entry = {
            'swin_description': swin_description,
            'blip_description': blip_description,
            'true_label': label
        }

        f_train.write(f'{pd.Series(data_entry).to_json()}\n')
        pbar.update(1)
    pbar.close()

with open(test_output_path, 'a') as f_test:
    print("Processing test dataset...")
    pbar = tqdm(total=10000, desc="Processing test images", leave=False)

    for example in test_dataset:
        image = example['img']
        label = example['fine_label']

        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class_idx = outputs.logits.argmax(dim=-1).item()
            class_name = test_dataset.features['fine_label'].int2str(predicted_class_idx).split(',')[0]

        swin_description = f"This is a photo of a {class_name}."

        inputs_blip = blip_processor(text=swin_description, images=image, return_tensors="pt").to(device)
        out_blip = blip_model.generate(**inputs_blip)
        blip_description = blip_processor.decode(out_blip[0], skip_special_tokens=True)

        data_entry = {
            'swin_description': swin_description,
            'blip_description': blip_description,
            'true_label': label
        }

        f_test.write(f'{pd.Series(data_entry).to_json()}\n')
        pbar.update(1)
    pbar.close()

print(f'Dataset with Swin and BLIP descriptions created and saved at:\n- {train_output_path}\n- {test_output_path}')
