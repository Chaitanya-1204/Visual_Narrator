import os
import logging

logging.basicConfig(
    filename="vit_opt350M.log",
    filemode="a",
    format="\n%(asctime)s | %(levelname)s\n%(message)s\n" + "-"*80,
    level=logging.INFO
)

os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), ".hf_cache")

import torch 
import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor ,  OPTForCausalLM 



# ViT Encoder class
class ViTEncoder(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224-in21k"):
        super(ViTEncoder , self).__init__()
        self.model = ViTModel.from_pretrained(model_name)
        self.image_processor = ViTImageProcessor.from_pretrained(model_name)

    def forward(self, images , device):
        
        inputs = self.image_processor(images=images, return_tensors="pt", do_rescale=False)
        inputs = inputs.to(device)  

        outputs = self.model(**inputs) # (B, num_patches+1, hidden_dim)
        context = outputs.last_hidden_state[: , 1: , :]  # (B, num_patches , hidden_dim) ignoring CLS token
        return context 
 
class OptDecoder(nn.Module):
    def __init__(self, model="facebook/opt-350m", num_cross_attn_layers = 4, num_heads=8, dropout=0.1, ffn_dim=2048, device='cuda'):
        super(OptDecoder, self).__init__()
        self.model = OPTForCausalLM.from_pretrained(model)
        hidden_dim = self.model.model.decoder.embed_tokens.embedding_dim  # usually 512 for opt-350m

        # Project context from ViT (usually 768 dim) → decoder embedding dim (e.g., 512)
        self.project_context = nn.Linear(768, hidden_dim).to(device)

        # Cross-attention blocks
        self.cross_attn_blocks = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True).to(device),
                "attn_ln": nn.LayerNorm(hidden_dim).to(device),
                "ffn": nn.Sequential(
                    nn.Linear(hidden_dim, ffn_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_dim, hidden_dim)
                ).to(device),
                "ffn_ln": nn.LayerNorm(hidden_dim).to(device)
            }) for _ in range(num_cross_attn_layers)
        ])

    def forward(self, context, input_ids=None, attention_mask=None, device='cuda'):
        B, T = input_ids.shape
        text_embeds = self.model.model.decoder.embed_tokens(input_ids.to(device))  # (B, T, D)
        prefix_length = context.shape[1]

     
        context = self.project_context(context)  # (B, 196, D)
        output = text_embeds  # initialize with decoder input
        

        for block in self.cross_attn_blocks:
            attn_out, _ = block["attn"](query=output, key=context, value=context)  # ✔️ correct direction
            output = block["attn_ln"](output + attn_out)

            ffn_out = block["ffn"](output)
            output = block["ffn_ln"](output + ffn_out)

        # # Concatenate projected + attended image tokens with text
        # full_inputs = torch.cat([context, text_embeds], dim=1)  # (B, prefix+T, D)
        full_inputs = output
        # # Masks
        # prefix_mask = torch.ones((B, prefix_length), dtype=attention_mask.dtype, device=device)
        # full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        full_mask = attention_mask

        # # Labels
        # prefix_labels = torch.full((B, prefix_length), -100, dtype=input_ids.dtype, device=device)
        # full_labels = torch.cat([prefix_labels, input_ids], dim=1)

        full_labels = input_ids
        outputs = self.model(inputs_embeds=full_inputs, attention_mask=full_mask, labels=full_labels)
        return outputs.loss, outputs.logits
    
    
class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        self.projection = nn.Linear(
            encoder.model.config.hidden_size,
            decoder.model.model.decoder.embed_tokens.embedding_dim  # Match OPT embedding size
        )
        

    def forward(self, images, input_ids, attention_mask, device):
        context = self.encoder(images, device)
    
        
        
        loss, logits = self.decoder(context, input_ids, attention_mask, device=device)
       
        return loss, logits
    
 
 
    
    
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total Parameters: {total_params:,}")
    logging.info(f"Trainable Parameters: {trainable_params:,}")
    





