#!/bin/bash
# Setup script for HPC cluster environment

echo "Setting up WorldMem environment on HPC cluster..."

# Create necessary directories
mkdir -p slurm_logs
mkdir -p data/minecraft

# Set up Python virtual environment
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "Virtual environment created and packages installed."
else
    echo "Virtual environment already exists."
fi

# Make SLURM script executable
chmod +x run_infer_slurm.sh

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Set your WANDB token: export WANDB_API_KEY=\"your_token_here\""
echo "2. Edit run_infer_slurm.sh to adjust:"
echo "   - Partition name (--partition=gpu)"
echo "   - Resource requirements (memory, CPUs, time)"
echo "   - Module names for your specific cluster"
echo "   - Email address for notifications"
echo "3. Submit the job: sbatch run_infer_slurm.sh"
