#!/bin/bash
# One GPU job that runs the whole ImageNet control experiment and stops.
# Works on SLURM (sbatch) and LSF (bsub): each scheduler reads its own lines and ignores the other's.
#
# Submit FROM A SHELL WHERE (myenv) IS ACTIVE -- the job inherits that Python environment:
#   SLURM:  sbatch run_imagenet_control.sh
#   LSF:    bsub < run_imagenet_control.sh
#
# ---- SLURM ----
#SBATCH --job-name=imagenet_ctrl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=imagenet_ctrl_%j.log
#SBATCH --export=ALL
# ---- LSF ----
#BSUB -J imagenet_ctrl
#BSUB -gpu "num=1"
#BSUB -n 8
#BSUB -R "rusage[mem=64G]"
#BSUB -W 72:00
#BSUB -o imagenet_ctrl_%J.log

set -e
cd /path/to/data/PSH_project/IPMI2027/first_paper

echo "Python: $(which python)"
nvidia-smi || echo "WARNING: no GPU visible"
python -c "import torch; assert torch.cuda.is_available(), 'No GPU in this job'; print('GPU:', torch.cuda.get_device_name(0))"

echo "===== 1/3 prepare (checks data, saves contact sheet) ====="
python imagenet_control_experiment.py prepare

echo "===== 2/3 four binary runs + DeLong/Holm comparison + 3-class origin ====="
python imagenet_control_experiment.py all

echo "===== 3/3 done ====="
cat /path/to/data/PSH_project/IPMI2027/output_first_paper/imagenet_control/comparison/SUMMARY.txt
cat /path/to/data/PSH_project/IPMI2027/output_first_paper/imagenet_control/dataset_origin_3class/metrics.csv
