import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from masked_autoencoder.ViT import ViT, Transformer


class MAE(nn.Module):
    def __init__(
        self,
        mask_ratio=0.75,
        random_mask=True,
        decoder_dim=512,
        decoder_depth=6,
        decoder_heads=8,
        decoder_dim_head=64,
        encoder=None,
        **kwargs
    ):
        super().__init__()
        # encoder
        self.encoder = encoder if encoder is not None else ViT(**kwargs)
        # encoder dimension
        self.patch_h = self.encoder.patch_h
        self.patch_w = self.encoder.patch_w
        self.patch_num_h = self.encoder.patch_num_h
        self.patch_num_w = self.encoder.patch_num_w
        self.num_patches = self.encoder.num_patches
        num_pixel_per_patch = self.encoder.patch_embedding.weight.shape[-1]
        # encoder to decoder
        encoder_dim = self.encoder.dim
        self.enc2dec = nn.Linear(encoder_dim, decoder_dim)
        # mask 
        self.mask_ratio = mask_ratio
        self.mask_embedding = nn.Parameter(torch.randn(decoder_dim))
        self.random_mask = random_mask
        # decoder definition
        self.decoder = Transformer(
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            dim_head=decoder_dim_head,
            mlp_dim=decoder_dim * 4
        )
        self.decoder_pos_embedding = nn.Embedding(self.num_patches, decoder_dim)
        self.decoder_project_head = nn.Linear(decoder_dim, num_pixel_per_patch)

    def forward(self, x):
        b, c, *_ = x.shape
        batch_range = torch.arange(b).unsqueeze(-1)
        # patch partition
        patches = x.view(b, c, self.patch_num_h, self.patch_h, self.patch_num_w, self.patch_w).permute(0, 2, 4, 3, 5, 1).reshape(b, self.num_patches, -1)
        # shuffle
        shuffle_indices = torch.rand(b, self.num_patches).argsort().to(x.device)
        # masking 
        num_masked = int(self.mask_ratio * self.num_patches * random.randint(0, 1))
        mask_index = shuffle_indices[:, :num_masked]
        unmask_index = shuffle_indices[:, num_masked:]
        mask_patches = patches[batch_range, mask_index]
        unmask_patches = patches[batch_range, unmask_index]
        # encoder
        encoder_tokens = self.encoder.patch_embedding(unmask_patches)
        encoder_tokens += self.encoder.pos_embedding.repeat(b, 1, 1)[batch_range, unmask_index]
        encoder_tokens = self.encoder.transformer(encoder_tokens)
        # encoder to decoder 
        mid_tokens = self.enc2dec(encoder_tokens)
        # mask tokens 
        mask_tokens = self.mask_embedding[None, None, :].repeat(b, num_masked, 1)
        mask_tokens += self.decoder_pos_embedding(mask_index)
        # concatenate masked and unmasked tokens
        concat_tokens = torch.cat([mask_tokens, mid_tokens], dim=1)
        # un-shuffle
        decoder_tokens = torch.empty_like(concat_tokens)
        decoder_tokens[batch_range, shuffle_indices] = concat_tokens
        # decoder 
        decoder_tokens = self.decoder(decoder_tokens)
        # mlp prediction
        pred_pixel_values = self.decoder_project_head(decoder_tokens)
        # loss 
        return F.mse_loss(pred_pixel_values, patches)
        
    @torch.no_grad()
    def inference(
        self, 
        x, 
        mask_ratio=-1,
        **kwargs
    ):
        if mask_ratio < 0:
            mask_ratio = self.mask_ratio
        self.eval()
        # shape
        b, c, h, w = x.shape
        batch_range = torch.arange(b).unsqueeze(-1)
        # patch partition
        patches = x.view(b, c, self.patch_num_h, self.patch_h, self.patch_num_w, self.patch_w).permute(0, 2, 4, 3, 5, 1).reshape(b, self.num_patches, -1)
        # shuffle indices
        num_masked = int(mask_ratio * self.num_patches)
        if num_masked > 0:
            shuffle_indices = torch.rand(b, self.num_patches).argsort().to(x.device)
        else: 
            shuffle_indices = torch.arange(self.num_patches).repeat(b, 1).to(x.device)
        # masking
        mask_index = shuffle_indices[:, :num_masked]
        unmask_index = shuffle_indices[:, num_masked:]
        mask_patches = patches[batch_range, mask_index]
        unmask_patches = patches[batch_range, unmask_index]
        # encoder
        encoder_tokens = self.encoder.patch_embedding(unmask_patches)
        encoder_tokens += self.encoder.pos_embedding.repeat(b, 1, 1)[batch_range, unmask_index]
        encoder_tokens = self.encoder.transformer(encoder_tokens)
        # encoder to decoder
        mid_tokens = self.enc2dec(encoder_tokens)
        # mask tokens 
        mask_tokens = self.mask_embedding[None, None, :].repeat(b, num_masked, 1)
        mask_tokens += self.decoder_pos_embedding(mask_index)
        # concatenate masked and unmasked tokens
        concat_tokens = torch.cat([mask_tokens, mid_tokens], dim=1)
        # un-shuffle
        decoder_tokens = torch.empty_like(concat_tokens)
        decoder_tokens[batch_range, shuffle_indices] = concat_tokens
        # decoder 
        decoder_tokens = self.decoder(decoder_tokens)
        # mlp prediction
        pred_pixel_values = self.decoder_project_head(decoder_tokens)
        # reconstruct image 
        reconstruct_image = pred_pixel_values.view(b, self.patch_num_h, self.patch_num_w, self.patch_h, self.patch_w, c).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)
        return reconstruct_image