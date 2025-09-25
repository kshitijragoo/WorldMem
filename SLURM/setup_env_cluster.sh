#!/bin/bash
# Environment setup script for HPC cluster - run this ONCE before submitting jobs

echo "Setting up WorldMem Python environment on HPC cluster..."

# Load modules (adjust for your cluster)
module purge
module load python/3.9
module load cuda/11.8
module load gcc/9.3.0

# Create directories
mkdir -p slurm_logs
mkdir -p data/minecraft

# Option 1: Virtual Environment Setup
echo "Creating virtual environment with all dependencies..."
if [ -d "venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf venv
fi

python -m venv venv
source venv/bin/activate

# Upgrade pip and install wheel for faster installs
pip install --upgrade pip setuptools wheel

# Install PyTorch first (often faster to install separately)
echo "Installing PyTorch..."
pip install torch~=2.4.0 torchvision~=0.19.1 --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
echo "Installing other dependencies..."
pip install lightning~=2.1.2
pip install wandb~=0.17.0
pip install hydra-core~=1.3.2
pip install omegaconf~=2.3.0
pip install "torchmetrics[image]==0.11.4"
pip install wandb-osh==1.2.1
pip install "gluonts[torch]==0.13.1"
pip install pytorchvideo~=0.1.5
pip install colorama tqdm opencv-python matplotlib click
pip install moviepy==1.0.3
pip install imageio einops pandas pyzmq h5py
pip install rotary_embedding_torch diffusers timm
pip install gradio spaces transformers

# Try to install pyrealsense2 (might not be available on all clusters)
pip install pyrealsense2 || echo "Warning: pyrealsense2 not available, skipping..."

# Install internetarchive
pip install internetarchive

echo "Virtual environment setup complete!"

# Test the installation
echo "Testing installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "Environment setup complete!"
echo "Virtual environment location: $(pwd)/venv"
echo ""
echo "To use this environment in your SLURM jobs, the script will automatically activate it."
