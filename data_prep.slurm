#!/bin/bash

#SBATCH --job-name=data_prep_infersent
#SBATCH --open-mode=append
#SBATCH --output=/scratch/str8775/conda_env/compositional-generalisation/role-decomposition/%x_%j.out
#SBATCH --error=/scratch/str8775/conda_env/compositional-generalisation/role-decomposition/%x_%j.err
#SBATCH --export=ALL
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 8

singularity exec --nv --bind /scratch --overlay /scratch/str8775/conda_env/compositional-generalisation_env.ext3:rw /scratch/str8775/conda_env/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
python3 /scratch/str8775/conda_env/compositional-generalisation/role-decomposition/data_preparation.py
"
