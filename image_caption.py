import os

project_dir = os.path.dirname(os.path.abspath(__file__))
hf_cache_dir = os.path.join(project_dir, ".hf_cache")
os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
os.environ["HF_HOME"] = hf_cache_dir

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

def generate_caption(image_path):
    model_name = "Salesforce/blip-image-captioning-large"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_length=50)

    return processor.decode(output[0], skip_special_tokens=True)
