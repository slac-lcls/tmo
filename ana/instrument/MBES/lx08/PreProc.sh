#!/bin/bash
#SBATCH --output=/reg/d/psdm/tmo/tmolx0819/scratch/xiangli/preprocess.out
#SBATCH --error=/reg/d/psdm/tmo/tmolx0819/scratch/xiangli/preprocess.err
#SBATCH --partition=psfehq
#SBATCH --ntasks=10
#SBATCH --ntasks-per-node=5
#SBATCH --mail-user=xiangli@slac.stanford.edu
#SBATCH --mail-type=FAIL,END 

mpirun python PreProc.py
