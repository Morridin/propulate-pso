#!/usr/bin/env python3
import random
import sys

from mpi4py import MPI

from ap_pso.propagators import PSOInitUniform, VelocityClampingPropagator, ConstrictionPropagator, PSOCompose, \
    BasicPSOPropagator, StatelessPSOPropagator
from propulate import Islands
from propulate.propagators import Conditional, InitUniform

############
# SETTINGS #
############

fname = sys.argv[1]  # Get function to optimize from command-line.
NUM_GENERATIONS: int = int(sys.argv[2])  # Set number of generations.
POP_SIZE = 2 * MPI.COMM_WORLD.size  # Set size of breeding population.
num_migrants = 1


# SPHERE
# continuous, convex, separable, non-differentiable, non-multimodal
# input domain: -5.12 <= x, y <= 5.12
# global minimum 0 at (x, y) = (0, 0)
def sphere(params):
    x = params["x"]
    y = params["y"]
    return x ** 2 + y ** 2


if fname == "sphere":
    function = sphere
    limits = {"x": (-5.12, 5.12), "y": (-5.12, 5.12), }
else:
    sys.exit("ERROR: Function undefined...exiting")

if __name__ == "__main__":
    # migration_topology = num_migrants*np.ones((4, 4), dtype=int)
    # np.fill_diagonal(migration_topology, 0)

    rng = random.Random(MPI.COMM_WORLD.rank)

    propagator = PSOCompose(
        [
            # VelocityClampingPropagator(0.7298, 1.49618, 1.49618, MPI.COMM_WORLD.rank, limits, rng, 0.6)
            ConstrictionPropagator(2.49618, 2.49618, MPI.COMM_WORLD.rank, limits, rng)
            # BasicPSOPropagator(0.7298,1.49618,1.49618, MPI.COMM_WORLD.rank, limits, rng)
            # StatelessPSOPropagator(0, 1.49618, 1.49618, MPI.COMM_WORLD.rank, limits, rng) # Attention! Does not work with current chart drawing script!
        ]
    )

    init = PSOInitUniform(limits, rng=rng)
    propagator = Conditional(POP_SIZE, propagator, init)

    islands = Islands(function, propagator, rng, generations=NUM_GENERATIONS, checkpoint_path='./checkpoints/',
                      migration_probability=0)
    islands.evolve(top_n=1, logging_interval=1, DEBUG=2)
