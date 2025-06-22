import os
os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), ".hf_cache")
import torch 
import torch.nn as nn


# EVA-CLIP ViT Encoder class
from transformers import CLIPModel, CLIPProcessor , ViTModel, ViTFeatureExtractor , T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, OPTForCausalLM , AutoModelForCausalLM

class EVA_CLIP_ViTEncoder(nn.Module):
    def __init__(self, model_name="BAAI/EVA-CLIP-ViT-g-14"):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def forward(self, images):
        # images should be raw PIL images or already preprocessed tensors
        inputs = self.processor(images=images, return_tensors="pt")
        outputs = self.model.vision_model(**inputs)
        return outputs.last_hidden_state  # (B, num_patches+1, hidden_dim)





# ViT Encoder class
class ViTEncoder(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224-in21k"):
        super().__init__()
        self.model = ViTModel.from_pretrained(model_name)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

    def forward(self, images):
        # images should be raw PIL images or already preprocessed tensors
        inputs = self.feature_extractor(images=images, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.last_hidden_state  # (B, num_patches+1, hidden_dim)


# CLIP ViT Encoder class
class CLIPViTEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def forward(self, images):
        # images should be raw PIL images or already preprocessed tensors
        inputs = self.processor(images=images, return_tensors="pt")
        outputs = self.model.vision_model(**inputs)
        return outputs.last_hidden_state  # (B, num_patches+1, hidden_dim)
    

class QFormer(nn.Module):
    def __init__(self, vision_dim=768, hidden_dim=512, num_query_tokens=32, num_layers=6, num_heads=8):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_dim))
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, vision_feats):
        """
        vision_feats: Tensor of shape (B, N_patches, vision_dim)
        returns: Q-former outputs of shape (B, num_query_tokens, hidden_dim)
        """
        B = vision_feats.size(0)

        # Project vision features to decoder dimension
        vision_feats_proj = self.vision_proj(vision_feats)

        # Expand query tokens across the batch
        queries = self.query_tokens.expand(B, -1, -1)  # (B, num_query_tokens, hidden_dim)

        # Concatenate vision_feats_proj and queries for attention context
        # Each query attends to the entire vision context
        combined_input = torch.cat([queries, vision_feats_proj], dim=1)

        # Apply Transformer encoder
        attended = self.transformer_encoder(combined_input)

        # Return only the updated query embeddings
        return attended[:, :self.query_tokens.shape[1], :]  # (B, num_query_tokens, hidden_dim)


class T5Decoder(nn.Module):
    def __init__(self, model_name="google/flan-t5-base"):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def forward(self, inputs_embeds, decoder_input_ids=None, labels=None):
        return self.model(
            inputs_embeds=inputs_embeds,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )

    def generate(self, inputs_embeds, max_length=30, num_beams=3):
        return self.model.generate(inputs_embeds=inputs_embeds, max_length=max_length, num_beams=num_beams)

    def decode(self, generated_ids):
        return [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]

class GPT2Decoder(nn.Module):
    def __init__(self, model_name="gpt2"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def forward(self, inputs_embeds, attention_mask=None, labels=None):
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )

    def generate(self, inputs_embeds, attention_mask=None, max_length=30, num_beams=3):
        return self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            pad_token_id=self.tokenizer.pad_token_id
        )

    def decode(self, generated_ids):
        return [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]

class OPTDecoder(nn.Module):
    def __init__(self, model_name="facebook/opt-2.7b"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = OPTForCausalLM.from_pretrained(model_name)

    def forward(self, inputs_embeds, attention_mask=None, labels=None):
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )

    def generate(self, inputs_embeds, attention_mask=None, max_length=30, num_beams=3):
        return self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            pad_token_id=self.tokenizer.pad_token_id
        )

    def decode(self, generated_ids):
        return [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]

class ImageCaptioningModel(nn.Module):
    def __init__(self, vision_encoder, q_former, decoder, decoder_type="t5"):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.q_former = q_former
        self.decoder = decoder
        self.decoder_type = decoder_type.lower()

    def forward(self, images, labels=None, max_length=30, num_beams=3):
        # Step 1: Extract visual features
        vision_feats = self.vision_encoder(images)  # (B, N_patches, vision_dim)

        # Step 2: Pass through Q-Former
        query_outputs = self.q_former(vision_feats)  # (B, num_query_tokens, hidden_dim)

        # === TRAINING MODE ===
        if labels is not None:
            if self.decoder_type == "t5":
                output = self.decoder(
                    inputs_embeds=query_outputs,
                    labels=labels
                )
            elif self.decoder_type in {"gpt2", "opt"}:
                attention_mask = torch.ones(query_outputs.size()[:-1], dtype=torch.long).to(query_outputs.device)
                output = self.decoder(
                    inputs_embeds=query_outputs,
                    attention_mask=attention_mask,
                    labels=labels
                )
            else:
                raise ValueError(f"Unsupported decoder type: {self.decoder_type}")
            return output  # includes loss + logits

        # === INFERENCE MODE ===
        else:
            if self.decoder_type == "t5":
                generated_ids = self.decoder.generate(
                    inputs_embeds=query_outputs,
                    max_length=max_length,
                    num_beams=num_beams
                )
            elif self.decoder_type in {"gpt2", "opt"}:
                attention_mask = torch.ones(query_outputs.size()[:-1], dtype=torch.long).to(query_outputs.device)
                generated_ids = self.decoder.generate(
                    inputs_embeds=query_outputs,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    num_beams=num_beams
                )
            else:
                raise ValueError(f"Unsupported decoder type: {self.decoder_type}")
            
            captions = self.decoder.decode(generated_ids)
            return captions
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    return total_params, trainable_params


if __name__ == "__main__":
    
    print("Counting parameters for ViT Encoder + OPT Decoder")
    vision_encoder = ViTEncoder()

    q_former = QFormer(vision_dim=768, hidden_dim=512)
 
    decoder = OPTDecoder()
    
    model = ImageCaptioningModel(vision_encoder, q_former, decoder, decoder_type="opt")
    print("Image Captioning Model created")
    count_parameters(model)