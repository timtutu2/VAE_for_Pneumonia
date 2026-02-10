"""
Training script for VAE on Chest X-ray dataset
"""
import os
import argparse
import json
import yaml
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from vae_model import VAE, vae_loss
from dataset import get_data_loaders


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_config_with_args(config, args):
    """Merge config file with command-line arguments (args take precedence)"""
    # Create a flat dictionary from nested config
    merged = {}
    
    # Extract from config
    if config:
        merged['data_dir'] = config.get('data', {}).get('data_dir', './chest_xray')
        merged['image_size'] = config.get('data', {}).get('image_size', 224)
        merged['num_workers'] = config.get('data', {}).get('num_workers', 4)
        merged['latent_dim'] = config.get('model', {}).get('latent_dim', 128)
        merged['batch_size'] = config.get('training', {}).get('batch_size', 32)
        merged['epochs'] = config.get('training', {}).get('epochs', 50)
        merged['lr'] = config.get('training', {}).get('lr', 1e-3)
        merged['kl_weight'] = config.get('training', {}).get('kl_weight', 1.0)
        merged['output_dir'] = config.get('output', {}).get('output_dir', './outputs')
        merged['save_interval'] = config.get('output', {}).get('save_interval', 10)
        merged['sample_interval'] = config.get('output', {}).get('sample_interval', 5)
    
    # Override with command-line arguments if provided
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            merged[key] = value
    
    return argparse.Namespace(**merged)


def train_epoch(model, train_loader, optimizer, device, kl_weight=1.0):
    """Train for one epoch"""
    model.train()
    train_loss = 0
    train_recon_loss = 0
    train_kl_loss = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, data in enumerate(pbar):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        recon_batch, mu, logvar = model(data)
        loss, recon_loss, kl_loss = vae_loss(recon_batch, data, mu, logvar, kl_weight)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        train_loss += loss.item()
        train_recon_loss += recon_loss.item()
        train_kl_loss += kl_loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item() / len(data),
            'recon': recon_loss.item() / len(data),
            'kl': kl_loss.item() / len(data)
        })
    
    # Average losses
    avg_loss = train_loss / len(train_loader.dataset)
    avg_recon = train_recon_loss / len(train_loader.dataset)
    avg_kl = train_kl_loss / len(train_loader.dataset)
    
    return avg_loss, avg_recon, avg_kl


def test_epoch(model, test_loader, device, kl_weight=1.0):
    """Test for one epoch"""
    model.eval()
    test_loss = 0
    test_recon_loss = 0
    test_kl_loss = 0
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss, recon_loss, kl_loss = vae_loss(recon_batch, data, mu, logvar, kl_weight)
            
            test_loss += loss.item()
            test_recon_loss += recon_loss.item()
            test_kl_loss += kl_loss.item()
    
    # Average losses
    avg_loss = test_loss / len(test_loader.dataset)
    avg_recon = test_recon_loss / len(test_loader.dataset)
    avg_kl = test_kl_loss / len(test_loader.dataset)
    
    return avg_loss, avg_recon, avg_kl


def save_samples(model, epoch, device, save_dir, num_samples=16):
    """Generate and save sample images"""
    model.eval()
    with torch.no_grad():
        samples = model.sample(num_samples, device)
        samples = samples.cpu()
        
        # Create grid
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            if i < num_samples:
                ax.imshow(samples[i, 0], cmap='gray')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'samples_epoch_{epoch}.png'))
        plt.close()


def plot_losses(train_losses, test_losses, save_dir):
    """Plot and save training curves"""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, test_losses, 'r-', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train VAE on Chest X-ray dataset')
    
    # Config file argument
    parser.add_argument('--config', type=str, default='./config/train_config.yaml',
                        help='Path to config YAML file')
    
    # Data arguments (optional, override config)
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to chest_xray dataset directory')
    parser.add_argument('--image_size', type=int, default=None,
                        help='Image size')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of data loading workers')
    
    # Model arguments (optional, override config)
    parser.add_argument('--latent_dim', type=int, default=None,
                        help='Latent dimension size')
    
    # Training arguments (optional, override config)
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--kl_weight', type=float, default=None,
                        help='Weight for KL divergence loss')
    
    # Output arguments (optional, override config)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save outputs')
    parser.add_argument('--save_interval', type=int, default=None,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--sample_interval', type=int, default=None,
                        help='Generate sample images every N epochs')
    
    args = parser.parse_args()
    
    # Load config file
    config = None
    if os.path.exists(args.config):
        print(f'Loading config from {args.config}')
        config = load_config(args.config)
    else:
        print(f'Config file {args.config} not found, using command-line arguments only')
    
    # Merge config with command-line arguments
    args = merge_config_with_args(config, args)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    print('\n' + '='*50)
    print('Training Configuration')
    print('='*50)
    for key, value in vars(args).items():
        if key != 'config':
            print(f'{key}: {value}')
    print('='*50 + '\n')
    
    # Save hyperparameters
    config_dict = {k: v for k, v in vars(args).items() if k != 'config'}
    with open(os.path.join(args.output_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    print('Loading data...')
    train_loader, test_loader = get_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )
    
    # Create model
    print('Creating model...')
    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print('Starting training...')
    train_losses = []
    test_losses = []
    
    for epoch in range(1, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}')
        
        # Train
        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, device, args.kl_weight
        )
        
        # Test
        test_loss, test_recon, test_kl = test_epoch(
            model, test_loader, device, args.kl_weight
        )
        
        # Log results
        print(f'Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f})')
        print(f'Test Loss: {test_loss:.4f} (Recon: {test_recon:.4f}, KL: {test_kl:.4f})')
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        # Save samples
        if epoch % args.sample_interval == 0 or epoch == args.epochs:
            save_samples(model, epoch, device, args.output_dir)
        
        # Save checkpoint
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'vae_final.pth'))
    
    # Plot loss curves
    plot_losses(train_losses, test_losses, args.output_dir)
    
    print('\nTraining completed!')
    print(f'Results saved to {args.output_dir}')


if __name__ == '__main__':
    main()

