#!/bin/bash
#SBATCH --output=/reg/d/psdm/tmo/tmolw5618/results/xiangli/preprocess_arp.out
#SBATCH --error=/reg/d/psdm/tmo/tmolw5618/results/xiangli/preprocess_arp.err
#SBATCH --partition=psfehq
#SBATCH --ntasks=10
#SBATCH --ntasks-per-node=10
#SBATCH --mail-user=xiangli@slac.stanford.edu
#SBATCH --mail-type=FAIL,END 

mpirun python PreProc.py
