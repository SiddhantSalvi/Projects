import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
import math
from PIL import Image

class VAE(nn.Module):
    def __init__(self, device):
        super(VAE, self).__init__()
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
        self.vae.config.scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
        self.vae.config.shift_factor = 0
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.vae.requires_grad_(False)
        self.vae.eval()

    def encode_latents(self, x):
        latents = self.vae.encode(x).latent_dist.mode() - self.vae.config.shift_factor
        latents = latents * self.vae.config.scaling_factor
        return latents

    def decode_latents(self, latents):
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        decoded_latents = self.vae.decode(latents).sample
        return decoded_latents

class TextEncoder(nn.Module):
    def __init__(self, device, model_id="openai/clip-vit-large-patch14"):
        super().__init__()
        self.device = device
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.text_encoder = CLIPTextModel.from_pretrained(model_id).to(device)
        
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()

    def forward(self, prompts):
        # 1. Tokenize the strings (max_length 77 is standard for CLIP)
        inputs = self.tokenizer(
            prompts, 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)

        # 2. Get the embeddings
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            
            # Use pooler_output for a single (Batch, 768) vector
            # This is what your ContextEmbedding currently expects.
            embeddings = outputs.pooler_output
            
        return embeddings

class DiT(nn.Module):
    def __init__(self, 
                 input_size=32,
                 patch_size=2,
                 in_channels=4,
                 num_blocks=12,
                 embed_dim=1024,
                 num_heads=8,
                 mlp_ratio=4,
                 text_embedding_dim=768,
    ):
        super(DiT, self).__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_blocks = num_blocks
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.text_embedding_dim = text_embedding_dim
        self.num_patches = (input_size // patch_size) ** 2
        self.patch_embedding = PatchEmbedding(self.in_channels, self.input_size, self.embed_dim, self.patch_size, self.num_patches)
        self.context_embedding = ContextEmbedding(self.embed_dim, self.text_embedding_dim)
        self.dit_block = nn.ModuleList([
            DiTBlock(self.embed_dim, self.num_heads, self.mlp_ratio) for _ in range(self.num_blocks)
        ])
        self.norm = FinalNorm(self.embed_dim, self.patch_size, self.in_channels)
        self.reshape = Reshape(self.embed_dim, self.in_channels, self.patch_size, self.num_patches)
    
    def forward(self, x, timestep, text_logits):
        x = self.patch_embedding(x)
        condition = self.context_embedding(timestep, text_logits)
        for i in range(self.num_blocks):
            x = self.dit_block[i](x, condition)
        x = self.norm(x, condition)
        x = self.reshape(x)
        return x
    

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, input_size, embed_dim, patch_size, num_patches):
        super(PatchEmbedding, self).__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = num_patches
        pos_embed = self.get_position_embedding(self.num_patches, embed_dim)
        self.patch_embedding = nn.Conv2d(self.in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.register_buffer("pos_embed", pos_embed)
    
    @staticmethod
    def get_position_embedding(num_patches, embed_dim):
        # implementation for 1D token positional embedding
        def get_1d_position_embedding(num_patches, embed_dim):
            position_embedding = torch.zeros(num_patches, embed_dim)
            for pos in range(num_patches):
                for i in range(0, embed_dim, 2):
                    position_embedding[pos, i] = torch.sin(torch.tensor(pos) / (10000 ** (2*i / embed_dim)))
                    position_embedding[pos, i+1] = torch.cos(torch.tensor(pos) / (10000 ** (2*i / embed_dim)))
            return position_embedding
        
        # implementation for 2D token positional embedding
        grid_size = int(math.sqrt(num_patches))
        embed_dim_2d = embed_dim // 2
        grid_x = torch.arange(grid_size)
        grid_y = torch.arange(grid_size)
        grid = torch.stack(torch.meshgrid(grid_x, grid_y, indexing='ij'), dim=0) # (2, grid_size, grid_size)
        pos_x = get_1d_position_embedding(grid[0].flatten().shape[0], embed_dim_2d)
        pos_y = get_1d_position_embedding(grid[1].flatten().shape[0], embed_dim_2d)
        position_embedding = torch.cat([pos_y, pos_x], dim=1).unsqueeze(0)
        return position_embedding

    def forward(self, x):
        batch, channels, height, width = x.shape
        patches = self.patch_embedding(x)
        patches = patches.flatten(2).transpose(1, 2)
        patches = patches + self.pos_embed
        return patches

class ContextEmbedding(nn.Module):
    def __init__(self, embed_dim, text_embedding_dim):
        super(ContextEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.text_embedding_dim = text_embedding_dim
        
        self.time_embedding = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.text_embedding =  nn.Sequential(
            nn.Linear(text_embedding_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.final_projection = nn.Linear(embed_dim*2, embed_dim)
        
    @staticmethod
    def get_timesteps(timestep, embed_dim):
        timestep = timestep * 1000
        half = embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timestep.device)
        
        args = timestep[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding

    def forward(self, timestep, text_logits):
        timesteps_freqs = self.get_timesteps(timestep, self.embed_dim)
        time_embedding = self.time_embedding(timesteps_freqs)
        text_embedding = self.text_embedding(text_logits)
        condition = torch.cat([time_embedding, text_embedding], dim=-1)
        condition = self.final_projection(condition)
        return condition

class Reshape(nn.Module):
    def __init__(self, embed_dim, input_size, patch_size, num_patches):
        super(Reshape, self).__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        
    def forward(self, x):
        batch, seq_len, d_out = x.shape
        grid_size = int(self.num_patches ** 0.5)
        x = x.view(batch, grid_size, grid_size, self.input_size, self.patch_size, self.patch_size)
        x = x.permute(0,3,1,4,2,5).contiguous()
        x = x.view(batch, self.input_size, grid_size*self.patch_size, grid_size*self.patch_size)
        x = F.tanh(x)
        return x

class FinalNorm(nn.Module):
    def __init__(self, embed_dim, patch_size, in_channels):
        super(FinalNorm, self).__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 2*embed_dim),
        )
        self.linear = nn.Linear(embed_dim, patch_size * patch_size * in_channels, bias=True)
        nn.init.constant_(self.adaln[-1].weight, 0)
        nn.init.constant_(self.adaln[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)
        
    def forward(self, x, condition):
        shift, scale = self.adaln(condition).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        return x

class DiTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio):
        super(DiTBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 6*embed_dim),
        )
        self.mha = MultiHeadAttention(embed_dim, embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, embed_dim, mlp_ratio)
        # Initialize the adaln project layer to zero 
        # This makes the gates zero at the start of training
        nn.init.constant_(self.adaln[-1].weight, 0)
        nn.init.constant_(self.adaln[-1].bias, 0)
    
    def forward(self, x, condition):
        residual = x 
        res1 = self.norm1(x)
        scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp = self.adaln(condition).chunk(6, dim=-1)
        res1 = res1 * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        res1 = self.mha(res1)
        res1 = gate_msa.unsqueeze(1) * res1 
        y = res1 + residual
        residual = y
        y = self.norm2(y)
        res2 = y * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        res2 = self.mlp(res2)
        res2 = gate_mlp.unsqueeze(1) * res2 
        y = res2 + residual
        return y

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, dropout = 0.1, qkv_bias = False, proj_bias = False):
        super(MultiHeadAttention, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.value = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out, bias = proj_bias)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch, seq_len, d_in = x.shape
        queries = self.query(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.key(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = queries @ keys.transpose(2,3)
        d_keys = keys.shape[-1]
        normalized_scores = attn_scores / d_keys ** 0.5
        attn_weights = F.softmax(normalized_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vectors = attn_weights @ values
        context_vectors = context_vectors.transpose(1,2).contiguous().view(batch, seq_len, self.d_out)
        context_vectors = self.out_proj(context_vectors)
        context_vectors = self.dropout(context_vectors)
        return context_vectors


class MLP(nn.Module):
    def __init__(self, d_in, d_out, mlp_ratio, dropout = 0.1, bias = False):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(d_in, mlp_ratio * d_out, bias = bias)
        self.fc2 = nn.Linear(mlp_ratio * d_out, d_out, bias = bias)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(device)
    text_encoder = TextEncoder(device)
    dit = DiT().to(device)
    x = torch.randn(1, 3, 256, 256).to(device)
    encoded_x = vae.encode_latents(x)
    text_logits = text_encoder("a photo of a cat")
    timestep = torch.randint(0, 1000, (1,)).to(device)
    x = dit(encoded_x, timestep, text_logits)
    print(x.shape)
    decoded_x = vae.decode_latents(x)
    # save the decoded_x as an image
    decoded_x = decoded_x.clamp(0, 1)
    decoded_x = decoded_x.permute(0, 2, 3, 1).detach().cpu().numpy()
    decoded_x = (decoded_x * 255).astype(np.uint8)
    decoded_x = Image.fromarray(decoded_x[0])
    decoded_x.save("decoded_x.png")