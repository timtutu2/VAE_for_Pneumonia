"""
Variational Autoencoder (VAE) for Chest X-ray Image Generation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """
    Convolutional Variational Autoencoder for 224x224 grayscale images
    """
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder: 224x224 -> latent_dim
        self.encoder = nn.Sequential(
            # 224x224x1 -> 112x112x32
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 112x112x32 -> 56x56x64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 56x56x64 -> 28x28x128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 28x28x128 -> 14x14x256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 14x14x256 -> 7x7x512
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        # Latent space: mean and log variance
        self.fc_mu = nn.Linear(512 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(512 * 7 * 7, latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(latent_dim, 512 * 7 * 7)
        
        # Decoder: latent_dim -> 224x224
        self.decoder = nn.Sequential(
            # 7x7x512 -> 14x14x256
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 14x14x256 -> 28x28x128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 28x28x128 -> 56x56x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 56x56x64 -> 112x112x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 112x112x32 -> 224x224x1
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Encode input to latent space parameters"""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + std * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to image"""
        h = self.decoder_input(z)
        h = h.view(h.size(0), 512, 7, 7)
        return self.decoder(h)
    
    def forward(self, x):
        """Forward pass through VAE"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def sample(self, num_samples, device):
        """Generate new samples from the latent space"""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples


def vae_loss(recon_x, x, mu, logvar, kl_weight=1.0, reduction='sum'):
    """
    VAE loss = Reconstruction loss + KL divergence
    """
    if reduction == 'sum':
        # Reconstruction loss (Binary Cross Entropy)
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss
    elif reduction == 'mean':
        # Reconstruction loss (Binary Cross Entropy)
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='mean')
        # KL divergence (per latent dimension, averaged over batch)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")