#!/bin/bash
# Helper script to run training and evaluation in Docker

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
DATA_DIR="${DATA_DIR:-./chest_xray}"
OUTPUT_DIR="./outputs"
EVAL_DIR="./evaluation"
USE_GPU="${USE_GPU:-true}"

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
}

# Function to check if dataset exists
check_dataset() {
    if [ ! -d "$DATA_DIR" ]; then
        print_error "Dataset directory not found: $DATA_DIR"
        print_info "Please set DATA_DIR environment variable or download the dataset."
        exit 1
    fi
}

# Build Docker image
build_image() {
    print_info "Building Docker image..."
    # Build from parent directory with Dockerfile in docker/ folder
    docker build -f docker/Dockerfile -t vae-pneumonia:latest ..
    print_info "Docker image built successfully!"
}

# Start training
train() {
    check_dataset
    
    print_info "Starting VAE training..."
    
    if [ "$USE_GPU" = "true" ]; then
        RUNTIME_FLAG="--gpus all"
        print_info "Using GPU for training"
    else
        RUNTIME_FLAG=""
        print_warning "Using CPU for training (this will be slow)"
    fi
    
    docker run --rm $RUNTIME_FLAG \
        -v "$(pwd)/$DATA_DIR:/workspace/data:ro" \
        -v "$(pwd)/$OUTPUT_DIR:/workspace/outputs" \
        vae-pneumonia:latest \
        python train.py \
        --data_dir /workspace/data \
        --output_dir /workspace/outputs \
        --latent_dim "${LATENT_DIM:-128}" \
        --batch_size "${BATCH_SIZE:-32}" \
        --epochs "${EPOCHS:-50}" \
        --lr "${LEARNING_RATE:-0.001}" \
        --kl_weight "${KL_WEIGHT:-1.0}" \
        --image_size "${IMAGE_SIZE:-224}" \
        --num_workers "${NUM_WORKERS:-4}"
    
    print_info "Training completed! Results saved to $OUTPUT_DIR"
}

# Run evaluation
evaluate() {
    check_dataset
    
    if [ ! -f "$OUTPUT_DIR/vae_final.pth" ]; then
        print_error "Model not found: $OUTPUT_DIR/vae_final.pth"
        print_info "Please train the model first using: $0 train"
        exit 1
    fi
    
    print_info "Starting VAE evaluation..."
    
    if [ "$USE_GPU" = "true" ]; then
        RUNTIME_FLAG="--gpus all"
    else
        RUNTIME_FLAG=""
    fi
    
    docker run --rm $RUNTIME_FLAG \
        -v "$(pwd)/$DATA_DIR:/workspace/data:ro" \
        -v "$(pwd)/$OUTPUT_DIR:/workspace/outputs:ro" \
        -v "$(pwd)/$EVAL_DIR:/workspace/evaluation" \
        vae-pneumonia:latest \
        python evaluate.py \
        --model_path /workspace/outputs/vae_final.pth \
        --data_dir /workspace/data \
        --output_dir /workspace/evaluation \
        --num_generated "${NUM_GENERATED:-1000}" \
        --num_real "${NUM_REAL:-1000}" \
        --latent_dim "${LATENT_DIM:-128}" \
        --batch_size "${EVAL_BATCH_SIZE:-50}"
    
    print_info "Evaluation completed! Results saved to $EVAL_DIR"
}

# Interactive shell
shell() {
    check_dataset
    
    print_info "Starting interactive shell..."
    
    if [ "$USE_GPU" = "true" ]; then
        RUNTIME_FLAG="--gpus all"
    else
        RUNTIME_FLAG=""
    fi
    
    docker run --rm -it $RUNTIME_FLAG \
        -v "$(pwd)/$DATA_DIR:/workspace/data:ro" \
        -v "$(pwd)/$OUTPUT_DIR:/workspace/outputs" \
        -v "$(pwd)/$EVAL_DIR:/workspace/evaluation" \
        vae-pneumonia:latest \
        /bin/bash
}

# Show usage
usage() {
    cat << EOF
Usage: $0 [command] [options]

Commands:
    build       Build the Docker image
    train       Run VAE training
    evaluate    Run evaluation (IS and FID)
    shell       Start interactive shell in container
    help        Show this help message

Environment Variables:
    DATA_DIR         Path to dataset directory (default: ./chest_xray)
    USE_GPU          Use GPU for training (default: true)
    
    Training options:
    LATENT_DIM       Latent dimension (default: 128)
    BATCH_SIZE       Batch size (default: 32)
    EPOCHS           Number of epochs (default: 50)
    LEARNING_RATE    Learning rate (default: 0.001)
    KL_WEIGHT        KL divergence weight (default: 1.0)
    IMAGE_SIZE       Image size (default: 224)
    NUM_WORKERS      Data loading workers (default: 4)
    
    Evaluation options:
    NUM_GENERATED    Number of images to generate (default: 1000)
    NUM_REAL         Number of real images (default: 1000)
    EVAL_BATCH_SIZE  Evaluation batch size (default: 50)

Examples:
    # Build image
    $0 build
    
    # Train with default settings
    DATA_DIR=/path/to/chest_xray $0 train
    
    # Train with custom hyperparameters
    DATA_DIR=/path/to/chest_xray EPOCHS=100 BATCH_SIZE=64 $0 train
    
    # Evaluate trained model
    DATA_DIR=/path/to/chest_xray $0 evaluate
    
    # Train on CPU (not recommended)
    USE_GPU=false DATA_DIR=/path/to/chest_xray $0 train
    
    # Interactive shell
    DATA_DIR=/path/to/chest_xray $0 shell

EOF
}

# Main script
check_docker

case "$1" in
    build)
        build_image
        ;;
    train)
        train
        ;;
    evaluate)
        evaluate
        ;;
    shell)
        shell
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        usage
        exit 1
        ;;
esac

