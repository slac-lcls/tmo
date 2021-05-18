#!/bin/bash
#SBATCH --output=/reg/d/psdm/tmo/tmolw5618/results/xiangli/preprocess_99.out
#SBATCH --error=/reg/d/psdm/tmo/tmolw5618/results/xiangli/preprocess_99.err
#SBATCH --partition=psfehq
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=xiangli@slac.stanford.edu
#SBATCH --mail-type=FAIL,END 

python PreProc0.py
