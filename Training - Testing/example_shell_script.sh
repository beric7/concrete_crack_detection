#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH -t 2:00:00
#SBATCH -p v100_dev_q
#SBATCH -A infraeval
module load gcc cmake
module load cuda/9.0.176 
module load cudnn/7.1
module load Anaconda
source activate TF2

cd $PBS_O_WORKDIR
cd ~/COCO-Bridge-2020/MODELS/deeplabv3plus_seg_crack/

python main_plus.py -data_directory './DATA/' \
-exp_directory './stored_weights_plus/var_1plus/' \
--epochs 20 --batch 4

exit
