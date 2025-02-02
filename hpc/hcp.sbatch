#!/bin/bash
#SBACTH --job-name="run_neural_SBS"                                                                                                                                                               
#SBACTH --time=0-00:50                                                                                                                                                                            
#SBACTH --gres=cpu:1                                                                                                                                                                              
#SBACTH --output="stdout-%j.txt"                                                                                                                                                                  
#SBACTH --error="stderr-%j.txt"                                                                                                                                                                   
module load Python/Anaconda_v03.2023                                                                                                                                                              
module load CUDA/11.4                                                                                                                                                                             
#conda create -n myenv_sbs python                                                                                                                                                                 
source activate myenv_sbs                                                                                                                                                                         
pip install torch pandas matplotlib numpy scikit-learn transformers                                                                                                                               
python testing.py