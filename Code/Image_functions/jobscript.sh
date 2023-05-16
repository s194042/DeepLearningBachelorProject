#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J CE_L1_5
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 23:30
# request 32GB of system-memory
#BSUB -R "rusage[mem=32GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o JobOutputFiles/gpu_%J.out
#BSUB -e JobOutputFiles/gpu_%J.err
#BSUB -R "select[gpu80gb]"
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load cuda/11.6

cd /work3/s194042/
source .env/bin/activate
cd DeepLearningBachelorProject/Code
python Image_functions/HPC_Loss_v3.py LossThirdRun
