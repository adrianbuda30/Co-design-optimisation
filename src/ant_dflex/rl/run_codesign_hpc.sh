#!/bin/bash

#PBS -l select=1:ncpus=16:mem=128gb:ngpus=1:gpu_type=RTX6000
#PBS -l walltime=24:00:00
#PBS -o output_dflex_ant.txt
#PBS -e error_dlfex_ant.txt

# Cluster Environment Setup
cd $PBS_O_WORKDIR


# Activate your Python environment
module load anaconda3/personal
source activate base
conda init
conda activate codesign
conda activate diffrl

ln -s $CONDA_PREFIX/lib $CONDA_PREFIX/lib64

# Run the Python training script
python train.py


