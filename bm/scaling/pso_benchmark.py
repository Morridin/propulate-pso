#!/usr/bin/env python3
"""
This is the script with which I conducted the main part of the scaling benchmarks.
I cleaned up the script a little bit by removing lines of code that were commented already when running the
benchmarks and also by rewriting the imports so that the script would still work today.

Vanilla Propulate was benchmarked on a slightly modified islands_example.py, in which I only added the time printing
prior optimization start and past optimization end.
"""
import random
import sys
import time
from pathlib import Path

from mpi4py import MPI

from propulate import Islands
from propulate.propagators import Conditional, pso, Propagator
from tutorials.function_benchmark import get_function_search_space

############
# SETTINGS #
############

function_name = sys.argv[1]  # Get function to optimize from command-line.
NUM_GENERATIONS: int = int(sys.argv[2])  # Set number of generations.
POP_SIZE = 2 * MPI.COMM_WORLD.size  # Set size of breeding population.
PSO_TYPE = int(sys.argv[3])  # selects the propagator below
CHECKPOINT_PLACE = sys.argv[4]
num_migrants = 1

function, limits = get_function_search_space(function_name)

if __name__ == "__main__":
    rng = random.Random(MPI.COMM_WORLD.rank)

    propagator: Propagator = [
        pso.Basic(0.729, 1.49445, 1.49445, MPI.COMM_WORLD.rank, limits, rng),
        pso.VelocityClamping(
            0.729, 1.49445, 1.49445, MPI.COMM_WORLD.rank, limits, rng, 0.6
        ),
        pso.Constriction(2.05, 2.05, MPI.COMM_WORLD.rank, limits, rng),
        pso.Canonical(2.05, 2.05, MPI.COMM_WORLD.rank, limits, rng),
    ][PSO_TYPE]

    init = pso.InitUniform(limits, rng=rng, rank=MPI.COMM_WORLD.rank)
    propagator = Conditional(POP_SIZE, propagator, init)
    if MPI.COMM_WORLD.rank == 0:
        print("#-----------------------------------#")
        print(f"| Current time: {time.time_ns()} |")
        print("#-----------------------------------#")
        print(f"\nSaving files to: {Path(CHECKPOINT_PLACE).name}")

    islands = Islands(
        function,
        propagator,
        rng,
        generations=NUM_GENERATIONS,
        checkpoint_path=CHECKPOINT_PLACE,
        migration_probability=0,
        pollination=False,
    )
    islands.evolve(top_n=1, debug=0)

    if MPI.COMM_WORLD.rank == 0:
        print("#-----------------------------------#")
        print(f"| Current time: {time.time_ns()} |")
        print("#-----------------------------------#")
