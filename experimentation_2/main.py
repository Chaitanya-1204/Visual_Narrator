import os 
from ruamel.yaml import YAML
import random
import torch 

import logging
import matplotlib.pyplot as plt

# Set up logging 
logging.basicConfig(
    filename='logs.log',
    filemode='w', 
    format='%(asctime)s - %(levelname)s - %(message)s\n' + '-'*80,
    level=logging.INFO
)

from model.blip import create_model
from utils.all_utils import cosine_lr_schedule , train_one_epoch , evaluate , generate_captions , count_parameters
from utils.data import get_dataloaders


def main(device, config):
    
    # import transformers
    # print("Transformers version:", transformers.__version__)
    # return 0
        # Fixing the seed for reproducibiltiy 
    
    seed = config.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logging.info("Seed set for reproducibility.")
    
    # Dataset 
    
    logging.info("Loading datasets...")
    train_loader , val_loader , test_loader = get_dataloaders(config)
    
    
    # Model 
    
    model  = create_model(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                           vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                           prompt=config['prompt'])
    model = model.to(device)
    
    
    count_parameters(model)


    logging.info("Model created and moved to device: %s", device)

    optimizer = torch.optim.AdamW(model.parameters() , lr = config['init_lr'] , weight_decay=config['weight_decay'])
    
    logging.info("Starting training for %d epochs.", config['max_epoch'])

    best_eval_loss = float('inf')
    early_stopping_patience = config.get('early_stopping_patience', 3)
    no_improvement_counter = 0
    best_model_weights = None

    # For plotting losses
    train_losses = []
    val_losses = []

    generate_captions(model , test_loader , device , config)
    
    
    for epoch in range(0 , config['max_epoch']):
        
        logging.info("Epoch %d/%d", epoch + 1, config['max_epoch'])
        cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
        logging.info("Learning rate at epoch %d: %.8f", epoch + 1, optimizer.param_groups[0]['lr'])
        
        train_loss = train_one_epoch(model , train_loader , optimizer , epoch , device)
        train_losses.append(train_loss)
        
        eval_loss = evaluate(model , val_loader , device , config)
        val_losses.append(eval_loss)
       
        
        logging.info("Epoch %d: |  Train Loss = %.4f  |  Val Loss = %.4f", epoch + 1, train_loss, eval_loss)
        
      
        
        generate_captions(model , test_loader , device , config)
        
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            no_improvement_counter = 0
            best_model_weights = model.state_dict()
            torch.save({
                'model_state_dict': best_model_weights,
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_eval_loss
            }, "model.pt")
            logging.info("Best model checkpoint saved to model.pt")
        else:
            no_improvement_counter += 1
            logging.info("No improvement for %d epoch(s)", no_improvement_counter)

        if no_improvement_counter >= early_stopping_patience:
            logging.info("Early stopping triggered after %d epochs with no improvement.", early_stopping_patience)
            break
    
    # Plot and save losses after training/early stopping
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    logging.info("Loss plot saved to loss_plot.png")



if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    yaml = YAML()
    
    with open("config.yaml" , "r") as f:
        config = yaml.load(f)
    
    logging.info("Configuration loaded and training initialized.")
    main(device , config)    