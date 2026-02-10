# VAE for Chest X-ray Image Generation

Implementation of Variational Autoencoder (VAE) for generating chest X-ray images as part of ECE 285 Homework 2.

## Project Structure

```
.
├── vae_model.py          # VAE model architecture
├── dataset.py            # Dataset loader for chest X-rays
├── train.py              # Training script
├── evaluate.py           # Evaluation script (IS and FID)
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

Download the Chest X-ray Pneumonia dataset from Kaggle:
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

Extract it so the structure looks like:
```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

## Usage

### Training

Train the VAE model:

```bash
python train.py \
    --data_dir /path/to/chest_xray \
    --output_dir ./outputs \
    --latent_dim 128 \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.001
```

**Key hyperparameters:**
- `--latent_dim`: Dimension of latent space (default: 128)
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--kl_weight`: Weight for KL divergence term (default: 1.0)
- `--image_size`: Image resolution (default: 224)

### Evaluation

Compute Inception Score and FID:

```bash
python evaluate.py \
    --model_path ./outputs/vae_final.pth \
    --data_dir /path/to/chest_xray \
    --output_dir ./evaluation \
    --num_generated 1000 \
    --num_real 1000 \
    --latent_dim 128
```

This will:
1. Generate 1000 images from the trained VAE
2. Sample 1000 real images from the test set
3. Compute Inception Score (IS)
4. Compute Fréchet Inception Distance (FID)
5. Save results to `evaluation/evaluation_results.json`

## Model Architecture

The VAE consists of:

**Encoder:**
- 5 convolutional layers with batch normalization
- Downsamples 224×224 → 7×7
- Outputs mean (μ) and log-variance (log σ²) for latent distribution

**Latent Space:**
- Default dimension: 128
- Uses reparameterization trick: z = μ + σ × ε

**Decoder:**
- 5 transposed convolutional layers with batch normalization
- Upsamples 7×7 → 224×224
- Sigmoid activation for output

**Loss Function:**
- Reconstruction loss (Binary Cross-Entropy)
- KL divergence regularization
- Total loss = Reconstruction Loss + β × KL Loss

## Expected Results

After training for 50 epochs, you should see:
- Training loss converging to ~10,000-15,000
- Generated images resembling chest X-rays
- FID score: typically 50-150 (lower is better)
- IS score: varies for medical images

## Outputs

Training outputs (in `./outputs/`):
- `vae_final.pth`: Final model weights
- `checkpoint_epoch_*.pth`: Periodic checkpoints
- `samples_epoch_*.png`: Generated samples during training
- `loss_curves.png`: Training and test loss curves
- `hyperparameters.json`: Training configuration

Evaluation outputs (in `./evaluation/`):
- `generated/`: Generated images
- `real/`: Real images for comparison
- `evaluation_results.json`: IS and FID scores

## Notes

- The model is trained on grayscale images (1 channel)
- Images are resized to 224×224 pixels
- Training on GPU is recommended (takes ~2-3 hours on a single GPU)
- For better results, consider training for more epochs or tuning hyperparameters

## References

- Dataset: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
- Evaluation metrics: https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/
- VAE paper: Kingma & Welling, "Auto-Encoding Variational Bayes" (2013)

