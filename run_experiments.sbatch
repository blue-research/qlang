#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=quantum_experiment
#SBATCH --array=0-15  # Adjust based on total combinations (4 Ansatzes * 4 Rewriters)
#SBATCH --output=slurm/R-%x.%j.out
#SBATCH --error=slurm/R-%x.%j.err

# Set up environment and variables
export ANSATZ_INDEX=$((SLURM_ARRAY_TASK_ID / 4))
export REWRITER_INDEX=$((SLURM_ARRAY_TASK_ID % 4))
export TIME=$(date +"%m_%d_%H:%M")

#Adjust paths as necessary
OVERLAY_PATH="/scratch/jd5018/my_env/overlay-15GB-500K.ext3"
SINGULARITY_IMAGE="/scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif"
SCRIPT_PATH="/scratch/jd5018/qlang/run_experiments.py"

singularity exec --nv \
    --overlay $OVERLAY_PATH:ro \
    $SINGULARITY_IMAGE \
    /bin/bash -c "source /ext3/env.sh; cd /scratch/jd5018/qlang/; python -u $SCRIPT_PATH $ANSATZ_INDEX $REWRITER_INDEX $TIME"
