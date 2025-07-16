import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch 
import torch.nn as nn

from torch.hub import download_url_to_file
from urllib.parse import urlparse
from timm.models.hub import download_cached_file
from transformers import BertTokenizer


from model.vit import VisionTransformer , interpolate_pos_embed
from model.bert import BertConfig , BertModel , BertLMHeadModel


def create_model(pretrained=None, **kwargs):
    model = BLIP(**kwargs)
    if pretrained:
        model , msg = load_checkpoint(model, pretrained)
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
                prompt = "A picture of ",
                bert_config = "bert_config.json"):
        
        
        super(BLIP , self).__init__()
        
        self.visual_encoder , vision_width = create_vit(vit , image_size , vit_grad_ckpt , vit_ckpt_layer)
        
        self.tokenizer = init_tokenizer()
        
        bert_config = BertConfig.from_json_file(bert_config)
        bert_config.encoder_width = vision_width
        self.text_decoder = BertLMHeadModel(config=bert_config)
        
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1
        
    def forward(self , image , caption):
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        text = self.tokenizer(caption, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(image.device) 
        
        text.input_ids[:,0] = self.tokenizer.bos_token_id
        
        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)         
        decoder_targets[:,:self.prompt_length] = -100
     
        decoder_output = self.text_decoder(text.input_ids, 
                                           attention_mask = text.attention_mask, 
                                           encoder_hidden_states = image_embeds,
                                           encoder_attention_mask = image_atts,                  
                                           labels = decoder_targets,
                                           return_dict = True,   
                                          )   
        loss_lm = decoder_output.loss
        
        return loss_lm
    
    def generate(self, image, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
        image_embeds = self.visual_encoder(image)

        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams,dim=0)
            
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask":image_atts}
        
        prompt = [self.prompt] * image.size(0)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device) 
        input_ids[:,0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1] 

        if sample:
            #nucleus sampling
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  do_sample=True,
                                                  top_p=top_p,
                                                  num_return_sequences=1,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id, 
                                                  repetition_penalty=1.1,                                            
                                                  **model_kwargs)
        else:
            #beam search
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  num_beams=num_beams,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id,     
                                                  repetition_penalty=repetition_penalty,
                                                  **model_kwargs)            
            
        captions = []    
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)    
            captions.append(caption[len(self.prompt):])
        return captions
    

        
        
        