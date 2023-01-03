import torch
from torch import nn


# Reference
'''
[1] GitHub Repo: lucidrains/vit-pytorch/vit_pytorch/vit.py
[2] Zhihu Blog: https://zhuanlan.zhihu.com/p/439554945
'''


# helper blocks
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(
        self, 
        dim, 
        hidden_dim, 
        dropout = 0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self, 
        dim, 
        heads = 8, 
        dim_head = 64, 
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, l, _ = x.shape
        # Q, K, V projection
        qkv = self.to_qkv(x)
        qkv = qkv.view(b, l, 3, self.heads, -1).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.chunk(3)
        q = q.squeeze(0)
        k = k.squeeze(0)
        v = v.squeeze(0)
        # Attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        # Output
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, l, -1)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self, 
        dim, 
        depth, 
        heads, 
        dim_head, 
        mlp_dim, 
        dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for att, ffn in self.layers:
            x = att(x) + x
            x = ffn(x) + x
        return x


# core: ViT module
class ViT(nn.Module):
    def __init__(
        self, 
        image_size: tuple, 
        patch_size: tuple, 
        output_dim, 
        dim=1024, 
        depth=6, 
        heads=8, 
        mlp_dim=2048, 
        channels = 3, 
        dim_head = 64, 
        dropout = 0., 
        emb_dropout = 0.,
        **kwargs
    ):
        super().__init__()
        image_height, image_width = image_size
        patch_height, patch_width = patch_size
        patch_dim = channels * patch_height * patch_width

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        self.dim = dim
        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_h = patch_height
        self.patch_w = patch_width
        self.patch_num_h = image_height // self.patch_h
        self.patch_num_w = image_width // self.patch_w
        self.patch_embedding = nn.Linear(patch_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, output_dim)
        )

    def forward(self, x):
        b, c, *_ = x.shape
        # Patch partition
        patches = x.view(b, c, self.patch_num_h, self.patch_h, self.patch_num_w, self.patch_w).permute(0, 2, 4, 3, 5, 1).reshape(b, self.num_patches, -1)
        # Patch embedding
        tokens = self.patch_embedding(patches) #+ self.pos_embedding
        print(tokens.shape)
        tokens = self.dropout(tokens)
        # transformer
        tokens = self.transformer(tokens)
        # pooling 
        tokens = tokens.mean(dim=1)
        # MLP
        logits = self.mlp_head(tokens)
        return logits