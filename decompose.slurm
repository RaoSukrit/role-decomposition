#!/bin/bash

#SBATCH --job-name=role-learning-cogs
#SBATCH --open-mode=append
#SBATCH --output=/scratch/str8775/conda_env/compositional-generalisation/role-decomposition/%x_train_run1_%j.out
#SBATCH --error=/scratch/str8775/conda_env/compositional-generalisation/role-decomposition/%x_train_run1_%j.err
#SBATCH --export=ALL
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 8

singularity exec --nv --bind /scratch --overlay /scratch/str8775/conda_env/compositional-generalisation_env.ext3:rw /scratch/str8775/conda_env/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda activate
cd /scratch/str8775/conda_env/compositional-generalisation/role-decomposition/
echo 'Running in /scratch/str8775/conda_env/compositional-generalisation/role-decomposition/'
python3 decompose.py --data_path ./role_data --data_prefix cogs_1_example_lstm_bi_1_embd --num_roles 50 --filler_dim 512 --role_dim 50 --role_learning --vocab_size 747 --digits False --hidden_size 512
"
