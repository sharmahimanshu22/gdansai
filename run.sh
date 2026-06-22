#BSUB -J monarch                           # Job name
#BSUB -P acc_pejaverlab   # allocation account
#BSUB -n 1
#BSUB -R rusage[mem=80000]              # 80GB of memory                                                                                                                               
#BSUB -R span[hosts=1]
#BSUB -W 12:00                                   # walltime in HH:MM                                                                                                                 
#BSUB -o %J.stdout                              # output log (%J : JobID)                                                                                                            
#BSUB -eo %J.stderr                             # error log                                                                                                                          
#BSUB -q gpu
#BSUB -R a100
#BSUB -gpu num=1
#BSUB -L /bin/bash                               # Initialize the execution environment

ml anaconda3
ml java/11.0.2
ml cuda/13.0

conda activate torch-2.9.1+cu130
python main.py
