import torch 
import torch.nn as nn
import os
from torch.hub import download_url_to_file
from urllib.parse import urlparse
from timm.models.hub import download_cached_file
from transformers import BertTokenizer


from vit import VisionTransformer , interpolate_pos_embed


def create_model(pretrained=None, **kwargs):
    model = BLIP(**kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained)
    return model

def load_checkpoint(model, path_or_url):
    
    if urlparse(path_or_url).scheme in ('http', 'https'):
        
        # get the model name 
        # filename = path_or_url.split("/")[-1]
        # # cached_file_path = os.path.join("/pretrained_model", filename)
        # # os.makedirs(cached_file , exist_ok=True)
        
        cached_file = download_cached_file(path_or_url, check_hash = False , progress = True)
        checkpoint = torch.load(cached_file, map_location="cpu")
        
    elif os.path.isfile(path_or_url):
        checkpoint = torch.load(path_or_url, map_location='cpu')
        
    else:
        raise FileNotFoundError(f"Invalid path or URL: {path_or_url}")

    state_dict = checkpoint['model']
    
    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder) 
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                                         model.visual_encoder_m)    
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape!=model.state_dict()[key].shape:
                del state_dict[key]
    
    
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded weights from {path_or_url}")
    return model , msg

def create_vit(vit , image_size , use_grad_checkpointing = False , ckpt_layer = 0 ,drop_path_rate = 0):
    
    vision_width = 768 
    
    visual_encoder = VisionTransformer(
        img_size = image_size,
        patch_size = 16,
        embed_dim = vision_width ,
        depth = 12,
        num_heads = 12 , 
        use_grad_checkpointing = use_grad_checkpointing,
        ckpt_layer = ckpt_layer,
        drop_path_rate = 0 , 
        
    )
    
    return visual_encoder , vision_width
     
    
def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer
    

class BLIP(nn.Module):
    
    def __init__(self , 
                image_size , 
                vit = "base" , 
                vit_grad_ckpt = False , 
                vit_ckpt_layer = 0 ,
                prompt = "A picture of "):
        
        
        super(BLIP , self).__init__()
        
        self.visual_encoder , vision_width = create_vit(vit , image_size , vit_grad_ckpt , vit_ckpt_layer)
        
        self.tokenizer = init_tokenizer()
        