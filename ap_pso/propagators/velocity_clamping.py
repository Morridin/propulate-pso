"""
This file contains a PSO propagator relying on the standard one but additionally performing velocity clamping.
"""
from random import Random

import numpy as np

from ap_pso import Particle
from ap_pso.propagators import BasicPSOPropagator


class VelocityClampingPropagator(BasicPSOPropagator):
    def __init__(self,
                 w_k: float,
                 c_cognitive: float,
                 c_social: float,
                 rank: int,
                 limits: dict[str, tuple[float, float]],
                 rng: Random,
                 v_limits: float | np.ndarray):
        """
        Class constructor.
        :param w_k: The particle's inertia factor
        :param c_cognitive: constant cognitive factor to scale p_best with
        :param c_social: constant social factor to scale g_best with
        :param rank: the rank of the worker the propagator is living on in MPI.COMM_WORLD
        :param limits: a dict with str keys and 2-tuples of floats associated to each of them
        :param rng: random number generator
        :param v_limits: a numpy array containing values that work as relative caps for their corresponding search space dimensions. If this is a float instead, it does its job for all axes.
        """
        super().__init__(w_k, c_cognitive, c_social, rank, limits, rng)
        self.v_cap = v_limits

    def __call__(self, particles: list[Particle]) -> Particle:
        p: Particle = super().__call__(particles)
        p.position -= p.velocity
        p.velocity = p.velocity.clip(*(self.v_cap * self.laa))
        p.position += p.velocity
        return p
