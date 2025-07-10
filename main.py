import os 
from ruamel.yaml import YAML
import random

import torch 

from model.blip import create_model
def main(device, config):
    
    # Fixing the seed for reproducibiltiy 
    
    seed = config.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Dataset 
    
    
    
    
    # Model 
    
    model = create_model()

if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    yaml = YAML()
    
    with open("config.yaml" , "r") as f:
        config = yaml.load(f)
    
    main(device , config)    