#!/bin/bash

#SBATCH --account=cs678fl22
#SBATCH --job-name=mnist
#SBATCH --partition=gpuq
#SBATCH --qos=gpu

#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=50GB  
#SBATCH --export=ALL 
#SBATCH --time=0-2:00:00 

#SBATCH --output=mnist.%j.out
#SBATCH --error=mnist.%j.err

# to see ID and state of GPUs assigned
nvidia-smi 

## Load the necessary modules
module load gnu10
module load python

## Execute your script
## pip install --upgrade pip
## pip install wandb
## pip install transformers==3.1.0
## pip install pytorch-lightning==1.0.6
## pip install spacy==3.0
## pip install sentence-transformers==2.1.0
## python -m spacy download en_core_web_sm
./scripts/train_kairos.sh
# ./scripts/test_kairos.sh