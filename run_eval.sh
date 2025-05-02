#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=bert_output_%j.log
#SBATCH --error=bert_error_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=bigTiger


conda activate multimodal-popularity

cd /project/jlglliam/final_project

python eval_and_stat_graphs.py