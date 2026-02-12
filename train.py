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
import wandb

from vae_model import VAE, vae_loss
from dataset import get_data_loaders


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(model, train_loader, optimizer, device, kl_weight=1.0, reduction='sum', use_wandb=False):
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
        loss, recon_loss, kl_loss = vae_loss(recon_batch, data, mu, logvar, kl_weight, reduction)
        
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


def test_epoch(model, test_loader, device, kl_weight=1.0, reduction='sum', use_wandb=False):
    """Test for one epoch"""
    model.eval()
    test_loss = 0
    test_recon_loss = 0
    test_kl_loss = 0
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss, recon_loss, kl_loss = vae_loss(recon_batch, data, mu, logvar, kl_weight, reduction)
            
            test_loss += loss.item()
            test_recon_loss += recon_loss.item()
            test_kl_loss += kl_loss.item()
    
    # Average losses
    avg_loss = test_loss / len(test_loader.dataset)
    avg_recon = test_recon_loss / len(test_loader.dataset)
    avg_kl = test_kl_loss / len(test_loader.dataset)
    
    return avg_loss, avg_recon, avg_kl


def save_samples(model, epoch, device, save_dir, num_samples=16, use_wandb=False):
    """Generate and save sample images"""
    model.eval()
    with torch.no_grad():
        # Use the underlying model if wrapped with DataParallel
        model_to_use = model.module if isinstance(model, torch.nn.DataParallel) else model
        samples = model_to_use.sample(num_samples, device)
        samples = samples.cpu()
        
        # Create grid
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            if i < num_samples:
                ax.imshow(samples[i, 0], cmap='gray')
            ax.axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'samples_epoch_{epoch}.png')
        plt.savefig(save_path)
        
        # Log to wandb
        if use_wandb:
            wandb.log({
                "generated_samples": wandb.Image(save_path),
                "epoch": epoch
            })
        
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
    parser.add_argument('--reduction', type=str, default=None,
                        help='Reduction type for loss calculation')
    
    # Output arguments (optional, override config)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save outputs')
    parser.add_argument('--save_interval', type=int, default=None,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--sample_interval', type=int, default=None,
                        help='Generate sample images every N epochs')
    
    # Wandb arguments (optional, override config)
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Wandb entity/username')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='Wandb run name')
    
    args = parser.parse_args()
    
    # Load config file
    print(f'Loading configuration from {args.config}...')
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command-line arguments
    if args.data_dir is not None:
        config['data']['data_dir'] = args.data_dir
    if args.image_size is not None:
        config['data']['image_size'] = args.image_size
    if args.num_workers is not None:
        config['data']['num_workers'] = args.num_workers
    if args.latent_dim is not None:
        config['model']['latent_dim'] = args.latent_dim
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.lr is not None:
        config['training']['lr'] = args.lr
    if args.kl_weight is not None:
        config['training']['kl_weight'] = args.kl_weight
    if args.reduction is not None:
        config['training']['reduction'] = args.reduction
    if args.output_dir is not None:
        config['output']['output_dir'] = args.output_dir
    if args.save_interval is not None:
        config['output']['save_interval'] = args.save_interval
    if args.sample_interval is not None:
        config['output']['sample_interval'] = args.sample_interval
    if args.use_wandb:
        config['wandb']['use_wandb'] = True
    if args.wandb_project is not None:
        config['wandb']['project'] = args.wandb_project
    if args.wandb_entity is not None:
        config['wandb']['entity'] = args.wandb_entity
    if args.wandb_name is not None:
        config['wandb']['name'] = args.wandb_name
    
    # Set default reduction if not specified (for backward compatibility)
    if 'reduction' not in config['training']:
        config['training']['reduction'] = 'sum'
    
    # Print configuration
    print('\nConfiguration:')
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create output directory
    os.makedirs(config['output']['output_dir'], exist_ok=True)
    
    # Set device and detect GPUs early
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()
    
    # Add GPU info to config
    config['system'] = {
        'num_gpus': num_gpus,
        'gpu_names': [torch.cuda.get_device_name(i) for i in range(num_gpus)] if num_gpus > 0 else []
    }
    
    # Save configuration
    config_save_path = os.path.join(config['output']['output_dir'], 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Configuration saved to {config_save_path}\n")
    
    # Initialize wandb
    if config['wandb']['use_wandb']:
        wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            name=config['wandb']['name'],
            config=config
        )
        print(f'Wandb initialized: {wandb.run.name}\n')
    
    # Print GPU information
    print('='*60)
    print('GPU Configuration:')
    if num_gpus > 1:
        print(f'  Found {num_gpus} GPUs! Using DataParallel for multi-GPU training')
        for i in range(num_gpus):
            print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    elif num_gpus == 1:
        print(f'  Using single GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('  Using CPU (no GPU available)')
    print('='*60 + '\n')
    
    # Load data
    print('\nLoading data...')
    train_loader, test_loader = get_data_loaders(
        config['data']['data_dir'],
        batch_size=config['training']['batch_size'],
        image_size=config['data']['image_size'],
        num_workers=config['data']['num_workers']
    )
    
    # Create model
    print('Creating model...')
    model = VAE(latent_dim=config['model']['latent_dim']).to(device)
    
    # Wrap model with DataParallel if multiple GPUs are available
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
        print(f'Model wrapped with DataParallel across {num_gpus} GPUs')
        print(f'Effective batch size: {config["training"]["batch_size"]} per GPU × {num_gpus} GPUs = {config["training"]["batch_size"] * num_gpus}')
    
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
    
    # Training loop
    print(f"\nStarting training for {config['training']['epochs']} epochs...")
    print('='*60)
    train_losses = []
    test_losses = []
    
    for epoch in range(1, config['training']['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['training']['epochs']}")
        
        # Train
        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, device, 
            kl_weight=config['training']['kl_weight'], 
            reduction=config['training']['reduction'],
            use_wandb=config['wandb']['use_wandb']
        )
        
        # Test
        test_loss, test_recon, test_kl = test_epoch(
            model, test_loader, device, 
            kl_weight=config['training']['kl_weight'], 
            reduction=config['training']['reduction'],
            use_wandb=config['wandb']['use_wandb']
        )
        
        # Log results
        print(f'Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f})')
        print(f'Test Loss: {test_loss:.4f} (Recon: {test_recon:.4f}, KL: {test_kl:.4f})')
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        # Log to wandb
        if config['wandb']['use_wandb']:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'train/recon_loss': train_recon,
                'train/kl_loss': train_kl,
                'test/loss': test_loss,
                'test/recon_loss': test_recon,
                'test/kl_loss': test_kl,
            })
        
        # Save samples
        if epoch % config['output']['sample_interval'] == 0 or epoch == config['training']['epochs']:
            save_samples(model, epoch, device, config['output']['output_dir'], 
                        use_wandb=config['wandb']['use_wandb'])
        
        # Save checkpoint
        if epoch % config['output']['save_interval'] == 0 or epoch == config['training']['epochs']:
            checkpoint_path = os.path.join(config['output']['output_dir'], f'checkpoint_epoch_{epoch}.pth')
            # Save the underlying model if using DataParallel
            model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
            }, checkpoint_path)
            
            # Log checkpoint to wandb
            if config['wandb']['use_wandb']:
                wandb.save(checkpoint_path)
    
    # Save final model
    final_model_path = os.path.join(config['output']['output_dir'], 'vae_final.pth')
    # Save the underlying model if using DataParallel
    model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
    torch.save(model_to_save.state_dict(), final_model_path)
    
    # Plot loss curves (only if not using wandb, since wandb creates them automatically)
    if not config['wandb']['use_wandb']:
        plot_losses(train_losses, test_losses, config['output']['output_dir'])
    
    # Save artifacts to wandb
    if config['wandb']['use_wandb']:
        wandb.save(final_model_path)
        wandb.finish()
    
    print('\n' + '='*60)
    print('Training completed!')
    print(f"Results saved to {config['output']['output_dir']}")
    print('='*60)


if __name__ == '__main__':
    main()

