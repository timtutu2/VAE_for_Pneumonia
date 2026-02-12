"""
Evaluation script for computing Inception Score and FID
"""
import os
import argparse
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from vae_model import VAE


def generate_images(model, num_images, device, output_dir):
    """Generate images using the trained VAE"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    print(f'Generating {num_images} images...')
    
    batch_size = 32
    num_batches = (num_images + batch_size - 1) // batch_size
    
    img_idx = 0
    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            current_batch_size = min(batch_size, num_images - img_idx)
            samples = model.sample(current_batch_size, device)
            samples = samples.cpu()
            
            # Save each image
            for j in range(current_batch_size):
                img = samples[j, 0].numpy()
                img = (img * 255).astype(np.uint8)
                img_pil = Image.fromarray(img, mode='L')
                img_pil.save(os.path.join(output_dir, f'generated_{img_idx:05d}.png'))
                img_idx += 1
    
    print(f'Saved {img_idx} images to {output_dir}')


def collect_real_images(data_dir, output_dir, num_images=1000):
    """Collect real images for FID computation"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f'Collecting {num_images} real images...')
    
    # Collect from test set
    test_dir = os.path.join(data_dir, 'test')
    image_paths = []
    
    for category in ['NORMAL', 'PNEUMONIA']:
        category_dir = os.path.join(test_dir, category)
        if os.path.exists(category_dir):
            for img_name in os.listdir(category_dir):
                if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                    image_paths.append(os.path.join(category_dir, img_name))
    
    # Sample random images
    np.random.seed(42)
    selected_paths = np.random.choice(image_paths, min(num_images, len(image_paths)), replace=False)
    
    # Resize and save
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
    ])
    
    for i, img_path in enumerate(tqdm(selected_paths)):
        img = Image.open(img_path).convert('L')
        img = transform(img)
        img.save(os.path.join(output_dir, f'real_{i:05d}.png'))
    
    print(f'Saved {len(selected_paths)} real images to {output_dir}')


def compute_inception_score(image_dir, batch_size=32, splits=10):
    """
    Compute Inception Score for generated images
    Note: For grayscale medical images, IS may not be very meaningful
    """
    try:
        from pytorch_fid.inception import InceptionV3
        from scipy.stats import entropy
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Inception model
        inception_model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device)
        inception_model.eval()
        
        # Load images
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        
        # Transform for Inception (needs RGB and specific size)
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Get predictions
        preds = []
        for img_file in tqdm(image_files, desc='Computing IS'):
            img = Image.open(os.path.join(image_dir, img_file))
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                pred = inception_model(img_tensor)[0]
                pred = torch.nn.functional.softmax(pred, dim=1)
                preds.append(pred.cpu().numpy())
        
        preds = np.concatenate(preds, axis=0)
        
        # Compute IS
        split_scores = []
        for k in range(splits):
            part = preds[k * (len(preds) // splits): (k + 1) * (len(preds) // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))
        
        is_mean = np.mean(split_scores)
        is_std = np.std(split_scores)
        
        return is_mean, is_std
    
    except Exception as e:
        print(f"Error computing Inception Score: {e}")
        print("Note: IS computation requires pytorch-fid and scipy")
        return None, None


def compute_fid(real_dir, fake_dir, batch_size=50, device=None):
    """
    Compute FID score between real and generated images
    Uses pytorch-fid library
    """
    try:
        from pytorch_fid import fid_score
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print('Computing FID score...')
        fid_value = fid_score.calculate_fid_given_paths(
            [real_dir, fake_dir],
            batch_size=batch_size,
            device=device,
            dims=2048
        )
        
        return fid_value
    
    except Exception as e:
        print(f"Error computing FID: {e}")
        print("Note: FID computation requires pytorch-fid library")
        return None


def main():
    parser = argparse.ArgumentParser(description='Evaluate VAE with IS and FID')
    parser.add_argument('--model_path', type=str, default='/mnt/tim/VAE_for_Pneumonia/outputs/checkpoint_epoch_50.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='/mnt/tim/VAE_for_Pneumonia/chest_xray',
                        help='Path to chest_xray dataset directory')
    parser.add_argument('--output_dir', type=str, default='/mnt/tim/VAE_for_Pneumonia/evaluation',
                        help='Directory to save evaluation results')
    parser.add_argument('--num_generated', type=int, default=1000,
                        help='Number of images to generate')
    parser.add_argument('--num_real', type=int, default=1000,
                        help='Number of real images to use')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Latent dimension size')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    generated_dir = os.path.join(args.output_dir, 'generated')
    real_dir = os.path.join(args.output_dir, 'real')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    print('Loading model...')
    model = VAE(latent_dim=args.latent_dim).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Checkpoint saved with additional metadata (epoch, optimizer, etc.)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        # Checkpoint is just the state dict
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Generate images
    generate_images(model, args.num_generated, device, generated_dir)
    
    # Collect real images
    collect_real_images(args.data_dir, real_dir, args.num_real)
    
    # Compute metrics
    print('\n' + '='*50)
    print('Computing Evaluation Metrics')
    print('='*50)
    
    # Inception Score
    print('\n1. Computing Inception Score...')
    is_mean, is_std = compute_inception_score(generated_dir)
    if is_mean is not None:
        print(f'Inception Score: {is_mean:.4f} ± {is_std:.4f}')
    else:
        print('Inception Score: N/A (computation failed)')
    
    # FID Score
    print('\n2. Computing FID Score...')
    fid_value = compute_fid(real_dir, generated_dir, args.batch_size, device)
    if fid_value is not None:
        print(f'FID Score: {fid_value:.4f}')
    else:
        print('FID Score: N/A (computation failed)')
    
    # Save results
    results = {
        'inception_score_mean': float(is_mean) if is_mean is not None else None,
        'inception_score_std': float(is_std) if is_std is not None else None,
        'fid_score': float(fid_value) if fid_value is not None else None,
        'num_generated_images': args.num_generated,
        'num_real_images': args.num_real
    }
    
    import json
    with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print('\n' + '='*50)
    print('Evaluation completed!')
    print(f'Results saved to {args.output_dir}')
    print('='*50)


if __name__ == '__main__':
    main()

