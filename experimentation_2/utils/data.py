import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
import  json, random

from torchvision import transforms
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from torchvision.transforms.functional import InterpolationMode

from utils.all_utils import build_json_data
from utils.randomaugument import RandomAugment


class CocoCaptionDataset(Dataset):
    
    def __init__(self, json_path, image_dir ,prompt = "" , transform=None, max_length=64):
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
        self.image_dir = image_dir
        self.prompt = prompt
        
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

        return {
            "image": image,
            "caption": caption
        }
        
def get_dataloaders(config , min_scale = 0.5):
    
    logging.info("Preparing dataloaders with image size %d", config['image_size'])
    # Geting the JSON files
    get_json_file()
    
    # Defining the image transformations
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    transform_train = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_size'],scale=(min_scale, 1.0),interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
    ])   
    
    transform_test = transforms.Compose([
        transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])  
       
    # Creating the datasets
    train_dataset = CocoCaptionDataset(
        json_path="train_data.json",
        image_dir="coco/images/train2014",
        prompt = config["prompt"],
        transform=transform_train
    )
    logging.info("Train dataset size: %d", len(train_dataset))

    val_dataset = CocoCaptionDataset(
        json_path="val_data.json",
        image_dir="coco/images/val2014",
        transform=transform_test
    )
    logging.info("Validation dataset size: %d", len(val_dataset))
    
    # Splitting the validation dataset into validation and test set
    generator = torch.Generator().manual_seed(42)

    val_size = len(val_dataset) // 2
    test_size = len(val_dataset) - val_size
    val_subset, test_subset = random_split(val_dataset, [val_size, test_size] , generator=generator)
    logging.info("Validation subset size: %d", len(val_subset))
    logging.info("Test subset size: %d", len(test_subset))
    
    # Creating the dataloaders
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=16, 
                                  shuffle=True, 
                                  num_workers=4, 
                                  pin_memory=True, 
                                  persistent_workers=True)
    
    val_dataloader = DataLoader(val_subset, 
                                batch_size=16, 
                                shuffle=False, 
                                num_workers=4, 
                                pin_memory=True, 
                                persistent_workers=True)
    
    test_dataloader = DataLoader(test_subset, 
                                 batch_size=16, 
                                 shuffle=False, 
                                 num_workers=4, 
                                 pin_memory=True, 
                                 persistent_workers=True)
    
    logging.info("Dataloaders created and ready.")
    return train_dataloader, val_dataloader, test_dataloader 

def get_json_file():
    
    annotations_train_file_path = "coco/annotations/captions_train2014.json"  
    annotations_val_file_path = "coco/annotations/captions_val2014.json"

    train_output_path = "train_data.json"
    val_output_path = "val_data.json"

    if os.path.exists(train_output_path):
        logging.info("Training data already exists at %s", train_output_path)
    else:
        build_json_data(annotations_train_file_path , train_output_path)
        logging.info("Built and saved data to %s", train_output_path)

    if os.path.exists(val_output_path):
        logging.info("Validation data already exists at %s", val_output_path)
    else:
        build_json_data(annotations_val_file_path , val_output_path)
        logging.info("Built and saved data to %s", val_output_path)