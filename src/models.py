import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ViTEncoder(nn.Module):
    def __init__(self, *, image_size, patch_size, num_frames, dim, depth, heads, mlp_dim, latent_map_size=(8,7), channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = image_size
        patch_height, patch_width = patch_size
        self.latent_h, self.latent_w = latent_map_size
        self.latent_dim = 128

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        
        self.to_patch_embedding = nn.Linear(patch_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        transformer_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=depth)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.latent_h * self.latent_w * self.latent_dim)
        )

    def forward(self, video):
        p1 = p2 = 16
        x = rearrange(video, 'b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=p1, p2=p2)
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        x += self.pos_embedding[:, :, :n]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.transformer(x)
        x = rearrange(x, '(b t) n d -> b t n d', b=b)

        x = x.mean(dim=2)
        x = x.mean(dim=1)
        
        x = self.mlp_head(x)
        latent_map = rearrange(x, 'b (c h w) -> b c h w', c=self.latent_dim, h=self.latent_h, w=self.latent_w)
        return latent_map

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=128, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, z):
        z_permuted = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z_permuted.view(-1, self.embedding_dim)
        d = torch.sum(z_flat**2, dim=1, keepdim=True) + \
            torch.sum(self.embeddings.weight**2, dim=1) - \
            2 * torch.matmul(z_flat, self.embeddings.weight.t())
        
        encoding_indices = torch.argmin(d, dim=1)
        quantized = self.embeddings(encoding_indices).view(z_permuted.shape)
        
        e_latent_loss = F.mse_loss(quantized.detach(), z_permuted)
        q_latent_loss = F.mse_loss(quantized, z_permuted.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = z_permuted + (quantized - z_permuted).detach()
        return quantized.permute(0, 3, 1, 2).contiguous(), loss, encoding_indices.view(z.shape[0], z.shape[2], z.shape[3])

class Decoder(nn.Module):
    def __init__(self, in_channels=128, out_channels=3, frame_size=(128, 128)):
        super().__init__()
        self.frame_size = frame_size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        return F.interpolate(x, size=self.frame_size, mode='bilinear', align_corners=False)

class WorldModel_ViT(nn.Module):
    def __init__(self, frame_size=(128, 128), patch_size=(16, 16), num_frames=16,
                 dim=512, depth=6, heads=8, mlp_dim=1024,
                 codebook_size=512, embedding_dim=128, commitment_cost=0.25):
        super().__init__()
        self.encoder = ViTEncoder(
            image_size=frame_size, patch_size=patch_size, num_frames=num_frames,
            dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim
        )
        self.vq = VectorQuantizer(num_embeddings=codebook_size, embedding_dim=embedding_dim, commitment_cost=commitment_cost)
        self.decoder = Decoder(in_channels=embedding_dim, out_channels=3, frame_size=frame_size)

    def forward(self, frame_sequence):
        z = self.encoder(frame_sequence)
        quantized, vq_loss, indices = self.vq(z)
        recon_next_frame = self.decoder(quantized)
        return recon_next_frame, vq_loss, indices

class ActionToLatentMLP(nn.Module):
    def __init__(self, input_dim=2, hidden1=512, hidden2=512, latent_dim=56, codebook_size=512, dropout=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, latent_dim * codebook_size)
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(-1, self.latent_dim, self.codebook_size)
        
    def sample(self, x, temperature=0.1):
        logits = self.forward(x)
        probs = F.softmax(logits / temperature, dim=-1)
        batch_size = probs.shape[0]
        return torch.multinomial(probs.view(-1, self.codebook_size), 1).view(batch_size, self.latent_dim)