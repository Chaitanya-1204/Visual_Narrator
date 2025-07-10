import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
import os, json, random
from torchvision import transforms
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from utils.all_utils import build_json_data
import logging

logging.basicConfig(
    filename="vit_opt350M.log",
    filemode="a",
    format="\n%(asctime)s | %(levelname)s\n%(message)s\n" + "-"*80,
    level=logging.INFO
)

# Get root directory of project
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CocoCaptionDataset(Dataset):
    
    def __init__(self, json_path, image_dir, tokenizer, transform=None, max_length=64):
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
        
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx):
        
        # Get the item from the dataset
        item = self.data[idx]
        # Load the image
        image = Image.open(os.path.join(self.image_dir, item["file_name"])).convert("RGB")
        
        # Applying transformations
        if self.transform: 
            image = self.transform(image)
           
        # Tokenizing the caption    
        caption = item['caption']
        tokenized = self.tokenizer(caption, padding="max_length", 
                                   truncation=True,
                                   max_length=self.max_length, 
                                   return_tensors="pt")
        

        assert hasattr(image, "shape") and image.shape == (3, 224, 224), f"Image shape mismatch: {getattr(image, 'shape', None)}"
        assert tokenized["input_ids"].shape == (1, self.max_length), f"Input ID shape: {tokenized['input_ids'].shape}"
        
        
        return {
            "pixel_values": image,
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "caption": caption
        }
        
def get_dataloaders(decoder_model_name):
    
    # Geting the JSON files
    get_json_file()
    
    # Defining the image transformations
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Loading the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Creating the datasets
    train_dataset = CocoCaptionDataset(
        json_path="train_data.json",
        image_dir="coco/images/train2014",
        tokenizer=tokenizer,
        transform=image_transform
    )

    val_dataset = CocoCaptionDataset(
        json_path="val_data.json",
        image_dir="coco/images/val2014",
        tokenizer=tokenizer,
        transform=image_transform  
    )
    
    # Splitting the validation dataset into validation and test set
    generator = torch.Generator().manual_seed(42)

    val_size = len(val_dataset) // 2
    test_size = len(val_dataset) - val_size
    val_subset, test_subset = random_split(val_dataset, [val_size, test_size] , generator=generator)
    
    
    # Creating the dataloaders
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=32, 
                                  shuffle=True, 
                                  num_workers=4, 
                                  pin_memory=True, 
                                  persistent_workers=True)
    val_dataloader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    test_dataloader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    return train_dataloader, val_dataloader, test_dataloader , tokenizer

def get_json_file():
    
    annotations_train_file_path = "coco/annotations/captions_train2014.json"  
    annotations_val_file_path = "coco/annotations/captions_val2014.json"

    train_output_path = "train_data.json"
    val_output_path = "val_data.json"

    if os.path.exists(train_output_path):
        logging.info("Training Data Exists")
    else:
        build_json_data(annotations_train_file_path , train_output_path)

    if os.path.exists(val_output_path):
        logging.info("Validation Data Exists")
    else:
        build_json_data(annotations_val_file_path , val_output_path)