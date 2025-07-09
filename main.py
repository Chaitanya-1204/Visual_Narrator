# Setting up environment for Hugging Face cache
import os 
os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), ".hf_cache")

import logging

logging.basicConfig(
    filename="vit_opt350M.log",
    filemode="a",
    format="\n%(asctime)s | %(levelname)s\n%(message)s\n" + "-"*80,
    level=logging.INFO
)

# Importing necessary libraries
from model import ImageCaptioningModel, ViTEncoder, OptDecoder , count_parameters
from data import get_dataloaders
from all_utils import train_model , evaluate

import torch

# Hyperparameters
EPOCHS = 20
LEARNING_RATE = 3e-5
decoder_model_name = "facebook/opt-350m"

# Setting up the device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Setting up the random seed for reproducibility
torch.manual_seed(42)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(42)
    

# Building Dataloader 

train_dataloader, val_dataloader , test_dataloader , tokenizer = get_dataloaders(decoder_model_name)
logging.info("Dataloaders created")

# Loading the model components
encoder = ViTEncoder()
logging.info("Encoder Loaded")

decoder = OptDecoder()
logging.info("Decoder Loaded")

model = ImageCaptioningModel(encoder=encoder, decoder=decoder)


logging.info("Model Created")

count_parameters(model)


# Training the model
train_model(model, train_dataloader, val_dataloader, device, EPOCHS, LEARNING_RATE , tokenizer)

logging.info("Training Complete")

test_loss = evaluate(model , test_dataloader , device )
logging.info(f"Testing Loss : {test_loss}")

