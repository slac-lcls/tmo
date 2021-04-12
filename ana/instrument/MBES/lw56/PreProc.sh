#!/bin/bash
#SBATCH --output=/reg/d/psdm/tmo/tmolw5618/results/xiangli/preprocess_99tst.out
#SBATCH --error=/reg/d/psdm/tmo/tmolw5618/results/xiangli/preprocess_99tst.err
#SBATCH --partition=psfehq
#SBATCH --ntasks=10
#SBATCH --ntasks-per-node=5
#SBATCH --mail-user=xiangli@slac.stanford.edu
#SBATCH --mail-type=FAIL,END 

mpirun -n 10 python PreProc.py
