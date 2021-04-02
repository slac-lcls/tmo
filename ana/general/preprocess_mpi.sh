#!/bin/bash
#SBATCH --output=[folder path]/preprocess_mpi.out
#SBATCH --error=[folder path]/preprocess_mpi.err
#SBATCH --partition=psfehq
#SBATCH --ntasks=10
#SBATCH --ntasks-per-node=5
#SBATCH --mail-user=[email address]
#SBATCH --mail-type=FAIL,END 

mpirun python preprocess_mpi.py -i preprocess_mpi.ini
