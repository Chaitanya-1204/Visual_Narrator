import sys
import os

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.bleu_score import corpus_bleu
from pycocoevalcap.cider.cider import Cider

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json 
from collections import defaultdict
import torch 
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
import logging
from transformers import AutoTokenizer
import random

import math


logging.basicConfig(
    filename="vit_opt350M.log",
    filemode="a",
    format="\n%(asctime)s | %(levelname)s\n%(message)s\n" + "-"*80,
    level=logging.INFO
)


def build_json_data(file_path , output_file):
    
    """ 
        This functions reads the COCO annotations file and for each captions associates with its image.
    
    """
    
    with open(file_path , "r") as f:
        data = json.load(f)
    
    caption_dict = defaultdict(list)
    for ann in data["annotations"]:
        caption_dict[ann["image_id"]].append(ann["caption"])
        
    simplified_data = []
    for img in data["images"]:
        image_id = img["id"]
        captions = caption_dict.get(image_id, [])
        for cap in captions:
            entry = {
                "image_id": image_id,
                "file_name": img["file_name"],
                "height": img["height"],
                "width": img["width"],
                "caption": cap 
            }
            simplified_data.append(entry)
        
    with open(output_file, "w") as f:
        json.dump(simplified_data, f, indent=2)
    
    logging.info("Saved data to %s Total length : %d", output_file, len(simplified_data))
    
    
def train_one_epoch(model, dataloader, optimizer, epoch ,  device):
    
    """ 
        This function trains the model for one epoch.
        
        Args:            
            model: The model to be trained.
            dataloader: The dataloader for the training data.
            optimizer: The optimizer for the model.
            epoch : Epoch number 
            device: The device to run the model on (CPU or GPU).
            
            
    """
    
    model.train()
    total_loss = 0.0
    progress = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress:
        # Getting the batch data and moving it to the device
        image = batch['image']
        image = image.to(device)
        caption = batch['caption']

        with torch.amp.autocast("cuda" , dtype=torch.bfloat16):
            loss = model(image , caption)
        
        # Backward pass and optimization with scaler
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
 
    

        # Accumulating the loss
        total_loss += loss.item()
        # Update tqdm with current loss
        progress.set_postfix(loss=loss.item())

    # Calculating the average loss for the epoch
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, device  , config):
    """ 
        This function evaluates the model on the validation set or test set.
        
        Args:
            model: The model to be evaluated.
            dataloader: The dataloader for the validation or test data.
            device: The device to run the model on (CPU or GPU).
    """
    
    model.eval()
    total_loss = 0.0
    # Disabling gradient calculation for evaluation
    with torch.no_grad():
    
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Getting the batch data and moving it to the device
            image = batch['image']
            image = image.to(device)
            
            caption = batch['caption']
            
            # Forward pass
            loss  = model(image , caption)
            
            # Accumulating the loss
            total_loss += loss.item()

    # Calculating the average loss for the evaluation
    avg_loss = total_loss / len(dataloader)
    return avg_loss


# Function to generate and log predicted/original captions for the first batch
def generate_captions(model, data_loader, device, config):
    model.eval()
    data = next(iter(data_loader))
    image = data['image'].to(device)
    captions_gt = data['caption']

    # Generate predictions
    predictions = model.generate(
        image,
        sample=False,
        num_beams=config['num_beams'],
        max_length=config['max_length'],
        min_length=config['min_length']
    )

    smoothie = SmoothingFunction().method4
    bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores = [], [], [], []
    gts, res = {}, {}
    references, hypotheses = [], []

    for i, (pred, gt) in enumerate(zip(predictions, captions_gt)):
        logging.info("Image %d\nOriginal: %s\nPredicted: %s", i, gt, pred)
        reference = [gt.lower().split()]
        hypothesis = pred.lower().split()

        # For BLEU
        references.append(reference)
        hypotheses.append(hypothesis)

        bleu1_scores.append(sentence_bleu(reference, hypothesis, weights=(1, 0, 0, 0), smoothing_function=smoothie))
        bleu2_scores.append(sentence_bleu(reference, hypothesis, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie))
        bleu3_scores.append(sentence_bleu(reference, hypothesis, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie))
        bleu4_scores.append(sentence_bleu(reference, hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie))

        # For CIDEr
        gts[str(i)] = [gt]
        res[str(i)] = [pred]

    avg_bleu1 = sum(bleu1_scores) / len(bleu1_scores) if bleu1_scores else 0.0
    avg_bleu2 = sum(bleu2_scores) / len(bleu2_scores) if bleu2_scores else 0.0
    avg_bleu3 = sum(bleu3_scores) / len(bleu3_scores) if bleu3_scores else 0.0
    avg_bleu4 = sum(bleu4_scores) / len(bleu4_scores) if bleu4_scores else 0.0

    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)

    logging.info("Average BLEU-1: %.4f", avg_bleu1)
    logging.info("Average BLEU-2: %.4f", avg_bleu2)
    logging.info("Average BLEU-3: %.4f", avg_bleu3)
    logging.info("Average BLEU-4: %.4f", avg_bleu4)
    logging.info("Average CIDEr: %.4f", cider_score)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total:,}")
    logging.info(f"Trainable parameters: {trainable:,}")

def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr