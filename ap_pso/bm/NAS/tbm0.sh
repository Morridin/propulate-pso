#!/bin/bash
#SBATCH --partition=gpu_4_a100
#SBATCH --gres=gpu:4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=16:00:00
#SBATCH --mem=510000
#SBATCH --output="/pfs/work7/workspace/scratch/pa1164-propulate_bm_1/async-parallel-pso/ap_pso/bm/NAS/NAS.out"
#SBATCH --job-name="NAS"
#SBATCH --mail-type=all
#SBATCH --mail-user=pa1164@partner.kit.edu

cd \$(ws_find propulate_bm_1)
ml purge
ml restore propulate
mkdir /pfs/work7/workspace/scratch/pa1164-propulate_bm_1/async-parallel-pso/ap_pso/bm/NAS/tbm/
source /pfs/work7/workspace/scratch/pa1164-propulate_bm_1/.venvs/async-parallel-pso/bin/activate
mpirun --bind-to core --map-by core --mca btl ^ofi python -u /pfs/work7/workspace/scratch/pa1164-propulate_bm_1/async-parallel-pso/ap_pso/bm/NAS/torch_benchmark.py 0 64 4 400 /pfs/work7/workspace/scratch/pa1164-propulate_bm_1/async-parallel-pso/ap_pso/bm/NAS/tbm/
