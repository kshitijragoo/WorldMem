#!/bin/bash
# Conda environment setup for HPC cluster

echo "Setting up WorldMem Conda environment on HPC cluster..."

# Load modules (adjust for your cluster)
module purge
module load miniconda3  # or anaconda3, depending on your cluster

# Initialize conda (if not already done)
source ~/miniconda3/etc/profile.d/conda.sh  # or ~/anaconda3/etc/profile.d/conda.sh

# Create directories
mkdir -p slurm_logs
mkdir -p data/minecraft

# Create conda environment
echo "Creating conda environment from environment.yml..."
if conda env list | grep -q "worldmem"; then
    echo "Removing existing worldmem environment..."
    conda env remove -n worldmem -y
fi

conda env create -f environment.yml

echo ""
echo "Conda environment setup complete!"
echo "Environment name: worldmem"
echo ""
echo "To test the environment:"
echo "conda activate worldmem"
echo "python -c \"import torch; print(torch.__version__)\""
