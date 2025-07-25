import torch 
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import PatchEmbed
from timm.layers import DropPath , trunc_normal_
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map
    
    def forward(self, x, register_hook=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
                
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)        

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0. , 
                attn_drop=0.,drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                use_grad_checkpointing=False):
        
        super(Block , self).__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim , num_heads , qkv_bias , qk_scale , attn_drop , proj_drop = drop 
            
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.mlp = MLP(in_features = dim , hidden_features = mlp_hidden_dim , 
                       act_layer = act_layer , drop = drop)
        
        
        if use_grad_checkpointing:
            self.attn = checkpoint_wrapper(self.attn)
            self.mlp = checkpoint_wrapper(self.mlp)
            
        
    def forward(self , x , register_hook = False):
        
        x = x + self.drop_path(self.attn(self.norm1(x) , register_hook = register_hook))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x 
        

        
        

class VisionTransformer(nn.Module):
    
    
    def __init__(
                self , img_size = 224 , patch_size = 16 , in_chans = 3 , num_classes = 1000 , 
                embed_dim = 768 , depth = 12 , num_heads = 12 ,  mlp_ratio=4., qkv_bias=True, 
                qk_scale=None , representation_size=None,drop_rate=0., attn_drop_rate=0., 
                drop_path_rate=0., norm_layer=None, use_grad_checkpointing=False, ckpt_layer=0):
        

        """
        Args:
        
            img_size : input image size
            patch_size : patch size
            in_chans : number of input channels
            num_classes : number of classes for classification head
            embed_dim : embedding dimension
            depth : depth of transformer
            num_heads : number of attention heads
            mlp_ratio : ratio of mlp hidden dim to embedding dim
            qkv_bias : enable bias for qkv if True
            qk_scale : override default qk scale of head_dim ** -0.5 if set
            representation_size: enable and set representation layer (pre-logits) to this value if set
            drop_rate : dropout rate
            attn_drop_rate : attention dropout rate
            drop_path_rate : stochastic depth rate
            norm_layer : normalization layer
            
        """
        
        super(VisionTransformer , self).__init__()
        
        self.num_features = embed_dim
        self.embed_dim = embed_dim
        
        norm_layer = partial(nn.LayerNorm , eps = 1e-6)
        
        # Create pathes and returns a tensor of shape (B ,  number of patches , embedding dim)
        # This function is taken from timm library 
        
        self.patch_embed = PatchEmbed(
            img_size = img_size , 
            patch_size = patch_size , 
            in_chans = in_chans , 
            embed_dim = embed_dim
        )
        
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1 , 1,  embed_dim)) 
        self.pos_embed = nn.Parameter(torch.zeros(1 , num_patches + 1 , embed_dim))
        self.pos_drop = nn.Dropout(p = drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0 , drop_path_rate , depth)]
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_grad_checkpointing=(use_grad_checkpointing and i>=depth-ckpt_layer)
            )
            for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        
    
        
    
        
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0) 
            
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}
    
    def forward(self, x, register_blk=-1):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1)
  
        x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)

        for i,blk in enumerate(self.blocks):
            x = blk(x, register_blk==i)
        x = self.norm(x)
        
        return x
    
    
def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):        
    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.patch_embed.num_patches
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)

    if orig_size!=new_size:
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print('reshape position embedding from %d to %d'%(orig_size ** 2,new_size ** 2))
        
        return new_pos_embed    
    else:
        return pos_embed_checkpoint      
        