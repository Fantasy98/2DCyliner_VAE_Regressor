#!/usr/bin/env bash  
#SBATCH -A NAISS2023-5-102 -p alvis # project name, cluster name
#SBATCH -N 1 --gpus-per-node=A40:1     #A40:4 #A100fat:4    #V100:2  A100fat:4  A100:4  # number of nodes, gpu name   
#SBATCH -t 0-11:00:00 # time


source ../../load_modules.sh


python train_deepONet.py