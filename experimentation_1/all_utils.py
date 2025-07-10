import json 
from collections import defaultdict
import torch 
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
import logging
from transformers import AutoTokenizer
import random
# Metrics imports
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from pycocoevalcap.cider.cider import Cider

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
    
    
def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    
    """ 
        This function trains the model for one epoch.
        
        Args:            
            model: The model to be trained.
            dataloader: The dataloader for the training data.
            optimizer: The optimizer for the model.
            scheduler: The learning rate scheduler.
            device: The device to run the model on (CPU or GPU).
            scaler: The GradScaler for mixed precision training
            
    """
    
    model.train()
    total_loss = 0.0
    progress = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress:
        # Getting the batch data and moving it to the device
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        
        loss, logits = model(images=pixel_values, input_ids=input_ids, attention_mask=attention_mask, device=device)

        # Backward pass and optimization with scaler
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
 
        scheduler.step()

        # Accumulating the loss
        total_loss += loss.item()
        # Update tqdm with current loss
        progress.set_postfix(loss=loss.item())

    # Calculating the average loss for the epoch
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, device):
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
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            loss , logits = model(images=pixel_values, input_ids=input_ids, attention_mask=attention_mask , device = device)
            
            # Accumulating the loss
            total_loss += loss.item()

    # Calculating the average loss for the evaluation
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def train_model(model, train_dataloader, val_dataloader, device, epochs, lr , tokenizer):
    
    """
        This function trains the model for a specified number of epochs and evaluates it on the validation set after each epoch.
        
        Args:
            model: The model to be trained.
            train_dataloader: The dataloader for the training data.
            val_dataloader: The dataloader for the validation data.
            device: The device to run the model on (CPU or GPU).
            epochs: The number of epochs to train the model.
            lr: The learning rate for the optimizer.    
    
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    num_training_steps = len(train_dataloader) * epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )


    patience = 3  # Early stopping patience
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    
    # log_caption_predictions(model, val_dataloader.dataset, device, tokenizer=tokenizer)

    for epoch in range(epochs):
        logging.info("=" * 80)
        logging.info(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_one_epoch(model, train_dataloader, optimizer, scheduler, device)
        val_loss = evaluate(model, val_dataloader, device)
        logging.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # log_caption_predictions(model, val_dataloader.dataset, device, tokenizer=tokenizer)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "vit_opt350M_best_model.pt")
            logging.info(">> Best model saved.")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logging.info(">> Early stopping triggered.")
                break
    logging.info("=" * 80)
    
def log_caption_predictions(model, val_dataset, device, tokenizer, num_samples=5 , max_length = 64):
    model.eval()
    indices = random.sample(range(len(val_dataset)), num_samples)
    logging.info("Sample Caption Predictions (Ground Truth | Predicted):")
    predictions = []
    references = []

    for idx in indices:
        sample = val_dataset[idx]
        image = sample["pixel_values"].unsqueeze(0).to(device)
        gt_caption = sample["caption"]

        with torch.no_grad():
            context = model.encoder(image, device)
            context = model.decoder.project_context(context)
            
            input_ids = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(device)

            for _ in range(max_length) :
                text_embeds = model.decoder.model.model.decoder.embed_tokens(input_ids.to(device))
                 
                output = context 
                for block in model.decoder.cross_attn_blocks:
                    attn_out, _ = block["attn"](query=output, key=text_embeds, value=text_embeds)
                    output = block["attn_ln"](output + attn_out)

                    ffn_out = block["ffn"](output)
                    output = block["ffn_ln"](output + ffn_out)
                
                full_input = torch.cat([output, text_embeds], dim=1)
                attention_mask = torch.ones(full_input.shape[:2], dtype=torch.long).to(device)
                outputs = model.decoder.model(inputs_embeds=full_input, attention_mask=attention_mask)
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)  # (1, 1)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
                # Stop if <eos>
                if next_token.item() == tokenizer.eos_token_id:
                    break
            pred_caption = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            
                    
            
        predictions.append(pred_caption)
        references.append(gt_caption)
        logging.info(f"True: {gt_caption} || Pred: {pred_caption}")

    avg_bleu, _ = compute_metrics(predictions, references)
    logging.info(f"BLEU Score (avg): {avg_bleu:.4f}")

# Metric computation function
def compute_metrics(predictions, references):
    smoothie = SmoothingFunction().method4
    bleu_scores = [sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie) for pred, ref in zip(predictions, references)]

    # Prepare for CIDEr
    # cider_scorer = Cider()
    # gts = {i: [references[i]] for i in range(len(references))}
    # res = {i: [predictions[i]] for i in range(len(predictions))}
    # cider_score, _ = cider_scorer.compute_score(gts, res)

    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    # return avg_bleu, cider_score
    return avg_bleu, None
