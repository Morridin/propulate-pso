#!/bin/bash
# BASE_DIR="/pfs/work7/workspace/scratch/pa1164-propulate_bm_1/async-parallel-pso"
BASE_DIR="."
for RACE in {0..4}
do
  NODES=$(( 2 ** RACE ))
  ITERATIONS=$(( 2000 / NODES ))
  QUEUE="single"
  if [[ $NODES -gt 1 ]]
  then
    QUEUE="multiple"
  fi
#  case "$RACE" in
#    5)
#      ITERATIONS=$((ITERATIONS * 10))
#      ;;
#    6)
#      ITERATIONS=-1
#      ;;
#    *)
#      echo "Error: Race $RACE was called."
#      exit
#      ;;
#  esac
  SCRIPT="#!/bin/bash
#SBATCH --nodes=${NODES}
#SBATCH --partition=${QUEUE}
#SBATCH --job-name=\"hyppopy_${RACE}\"
#SBATCH --time=15:00
#SBATCH --mem=10000
#SBATCH --cpus-per-task=40
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pa1164@partner.kit.edu

cd \$(ws_find propulate_bm_1)
ml purge
ml restore propulate
source ${BASE_DIR}/../.venvs/async-parallel-pso/bin/activate
"
  for FUNCTION in "sphere" "rosenbrock" "step" "quartic" "rastrigin" "griewank" "schwefel" "bisphere" "birastrigin"
  do
    DIRNAME="bm_H_${FUNCTION}_${RACE}"
    EXECUTION_DIR="${BASE_DIR}/ap_pso/bm/${DIRNAME}"
    mkdir "$EXECUTION_DIR"

    SCRIPT+="mpirun python -u ${BASE_DIR}/ap_pso/bm/hyppopy_benchmark.py ${FUNCTION} ${ITERATIONS} ${EXECUTION_DIR}
"
  done
  SCRIPT+="deactivate
"
  FILE="${BASE_DIR}/ap_pso/bm/start_bm_H_${RACE}.sh"
  echo "${SCRIPT}" > "${FILE}"
  sbatch -p "${QUEUE}" -N "${NODES}" "${FILE}"
done
