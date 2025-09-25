#!/bin/bash
# Job monitoring script for SLURM

if [ $# -eq 0 ]; then
    echo "Usage: $0 <job_id>"
    echo "Example: $0 123456"
    exit 1
fi

JOB_ID=$1

echo "Monitoring job $JOB_ID"
echo "========================"

# Show job status
echo "Job Status:"
squeue -j $JOB_ID

echo ""
echo "Job Details:"
scontrol show job $JOB_ID

echo ""
echo "Recent output (last 20 lines):"
if [ -f "slurm_logs/worldmem_infer_${JOB_ID}.out" ]; then
    tail -20 "slurm_logs/worldmem_infer_${JOB_ID}.out"
else
    echo "Output file not found yet."
fi

echo ""
echo "Recent errors (last 10 lines):"
if [ -f "slurm_logs/worldmem_infer_${JOB_ID}.err" ]; then
    tail -10 "slurm_logs/worldmem_infer_${JOB_ID}.err"
else
    echo "Error file not found or empty."
fi
