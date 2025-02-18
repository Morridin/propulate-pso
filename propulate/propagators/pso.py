import logging
from random import Random
from typing import Dict, Tuple, Union, List

import numpy as np

from ..propagators import Propagator, Stochastic
from ..population import Particle, Individual
from ..utils import make_particle


class BasicPSO(Propagator):
    """
    This propagator implements the most basic PSO variant one possibly could think of.

    It features an inertia factor applied to the old velocity in the velocity update, a social and a cognitive factor.

    With the help of the random number generator required as creation parameter, non-linearity is added to the particle
    update in order to not collapse to linear regression.

    This basic PSO propagator can only explore real-valued search spaces, i.e., continuous parameters.
    It works on ``Particle`` objects and serves as the foundation of all other PSO propagators.
    Further PSO propagators should be derived from this propagator or from one that is derived from this.

    This variant was first proposed in 1998 by Y. Shi and R. Eberhart, "A modified particle swarm optimizer"
    https://doi.org/10.1109/ICEC.1998.699146.
    """

    def __init__(
        self,
        inertia: float,
        c_cognitive: float,
        c_social: float,
        rank: int,
        limits: Dict[str, Tuple[float, float]],
        rng: Random,
    ):
        """
        Instantiate a basic PSO propagator.

        In theory, it should be no problem to hand over numpy arrays instead of the float-type hyperparameters inertia,
        cognitive factor, and social factor. In this case, please ensure that the dimension of the passed arrays fits
        the search domain.

        Parameters
        ----------
        inertia : float
                  inertia weight.
        c_cognitive : float
                      constant cognitive factor for scaling the distance to the particle's personal best value
        c_social : float
                   constant social factor for scaling the distance to the swarm's global best value
        rank : int
               global rank of the worker the propagator is living on
        limits : Dict[str, Tuple[float, float]]
                 borders of the continuous search domain
        rng : random.Random
              random number generator for introducing non-linearity
        """
        super().__init__(parents=-1, offspring=1)
        self.c_social = c_social
        self.c_cognitive = c_cognitive
        self.inertia = inertia
        self.rank = rank
        self.limits = limits
        self.rng = rng
        self.limits_as_array: np.ndarray = np.array(list(limits.values())).T

    def __call__(self, individuals: List[Individual]) -> Particle:
        """
        Apply the standard PSO update rule with inertia.

        Return a ``Particle`` object containing the updated values of the youngest passed ``Particle`` or ``Individual``
        that belongs to the worker the propagator is living on.

        Parameters
        ----------
        individuals: List[Individual]
                     list of individuals that must at least contain one individual that belongs to the propagator
                     This list is used to calculate personal and global best of the particle and the swarm,
                     respectively, and then to update the particle based on the retrieved results. Individuals that
                     cannot be used as ``Particle`` objects are converted to particles first.

        Returns
        -------
        propulate.population.Particle
            updated particle
        """
        old_p, p_best, g_best = self._prepare_data(individuals)

        new_velocity: np.ndarray = (
            self.inertia * old_p.velocity
            + self.rng.uniform(0, self.c_cognitive) * (p_best.position - old_p.position)
            + self.rng.uniform(0, self.c_social) * (g_best.position - old_p.position)
        )
        new_position: np.ndarray = old_p.position + new_velocity

        return self._make_new_particle(new_position, new_velocity, old_p.generation + 1)

    def _prepare_data(
        self, individuals: List[Individual]
    ) -> Tuple[Particle, Particle, Particle]:
        """
        Given a list of ``Individual`` or ``Particle`` objects, determine the particle to be updated on this rank, its
        current personal best, and the currently known global best of the swarm to perform a particle update step.

        Parameters
        ----------
        individuals : List[Individual]
                      ``Individual`` objects that shall be used as data basis for a PSO update step

        Returns
        -------
        Tuple[propulate.population.Particle, propulate.population.Particle, propulate.population.Particle]
            The following particles in this very order:
            1.  old_p: the current particle to be updated now
            2.  p_best: the personal best value of this particle
            3.  g_best: the global best value currently known
        """
        if len(individuals) < self.offspring:
            raise ValueError("Not enough Particles")

        particles = []
        for individual in individuals:
            if isinstance(individual, Particle):
                particles.append(individual)
            else:
                particles.append(make_particle(individual))
                logging.warning(
                    f"Got Individual instead of Particle. If this is on purpose, you can ignore this warning. "
                    f"Converted the Individual to Particle. Continuing."
                )

        own_p = [
            x
            for x in particles
            if (isinstance(x, Particle) and x.global_rank == self.rank)
            or x.rank == self.rank
        ]
        if len(own_p) > 0:
            old_p: Individual = max(own_p, key=lambda p: p.generation)
            if not isinstance(old_p, Particle):
                old_p = make_particle(old_p)

        else:
            victim = max(particles, key=lambda p: p.generation)
            old_p = self._make_new_particle(
                victim.position, victim.velocity, victim.generation
            )

        g_best = min(particles, key=lambda p: p.loss)
        p_best = min(own_p, key=lambda p: p.loss)

        return old_p, p_best, g_best

    def _make_new_particle(
        self, position: np.ndarray, velocity: np.ndarray, generation: int
    ) -> Particle:
        """
        Create a new ``Particle`` with the position dictionary set to the values provided by the numpy array.

        Parameters
        ----------
        position : np.ndarray
                   position of the particle to be created
        velocity : np.ndarray
                   velocity of the particle to be created
        generation : int
                     generation of the new particle

        Returns
        -------
        propulate.population.Particle
            new ``Particle`` object resulting from the PSO update step
        """
        new_p = Particle(position, velocity, generation, self.rank)
        for i, k in enumerate(self.limits):
            new_p[k] = new_p.position[i]
        return new_p


class VelocityClampingPSO(BasicPSO):
    """
    This propagator implements velocity clamping PSO.

    In addition to the parameters known from the basic PSO propagator, it features a clamping factor within [0, 1] used
    to determine each parameter's maximum velocity value relative to its search-space limits.

    Based on these values, the velocities of the particles are cut down to a reasonable value.
    """

    def __init__(
        self,
        inertia: float,
        c_cognitive: float,
        c_social: float,
        rank: int,
        limits: Dict[str, Tuple[float, float]],
        rng: Random,
        v_limits: Union[float, np.ndarray],
    ):
        """
        Instantiate a velocity clamping PSO propagator.

        Parameters
        ----------
        inertia : float
                  inertia factor
        c_cognitive : float
                      constant cognitive factor for scaling the distance to the particle's personal best value
        c_social : float
                   Constant social factor for scaling the distance to the swarm's global best value
        rank : int
               global rank of the worker the propagator is living on
        limits : Dict[str, Tuple[float, float]]
                 borders of the continuous search domain
        rng : random.Random
              random number generator for introducing non-linearity
        v_limits : Union[float, np.ndarray]
                   clamping factor to be multiplied with the clamping limit in order to reduce it further
                   Should be in (0, 1). If this parameter has float type, it is applied to all dimensions of the search
                   domain; else, each of its elements is applied to the corresponding dimension of the search domain.
        """
        super().__init__(inertia, c_cognitive, c_social, rank, limits, rng)
        x_min, x_max = self.limits_as_array
        x_range = abs(x_max - x_min)
        v_limits = abs(v_limits)
        self.v_cap: np.ndarray = np.array([-v_limits * x_range, v_limits * x_range])

    def __call__(self, individuals: List[Individual]) -> Particle:
        """
        Apply the standard PSO update rule with inertia, extended by cutting off too high velocities.

        Return a ``Particle`` object containing the updated values of the youngest passed ``Particle`` or ``Individual``
        that belongs to the worker the propagator is living on.

        Parameters
        ----------
        individuals: List[Individual]
                     list of individuals that must at least contain one individual that belongs to the propagator
                     This list is used to calculate personal and global best of the particle and the swarm,
                     respectively, and then to update the particle based on the retrieved results. Individuals that
                     cannot be used as ``Particle`` objects are converted to particles first.

        Returns
        -------
        propulate.population.Particle
            updated particle
        """
        old_p, p_best, g_best = self._prepare_data(individuals)

        new_velocity: np.ndarray = (
            self.inertia * old_p.velocity
            + self.rng.uniform(0, self.c_cognitive) * (p_best.position - old_p.position)
            + self.rng.uniform(0, self.c_social) * (g_best.position - old_p.position)
        ).clip(*self.v_cap)
        new_position: np.ndarray = old_p.position + new_velocity

        return self._make_new_particle(new_position, new_velocity, old_p.generation + 1)


class ConstrictionPSO(BasicPSO):
    """
    This propagator subclass features constriction PSO as proposed by Clerc and Kennedy in 2002.

    Reference publication: R. Poli, J. Kennedy, and T. Blackwell. Particle swarm optimization. Swarm Intell 1, 33–57
    (2007). https://doi.org/10.1007/s11721-007-0002-0

    Instead of an inertia factor that affects the old velocity value within the velocity update, a constriction factor
    is applied to the new velocity *after* the update.

    The constriction factor is calculated from cognitive and social factors and thus no separate hyperparameter.

    This propagator runs on ``Particle`` objects.
    """

    def __init__(
        self,
        c_cognitive: float,
        c_social: float,
        rank: int,
        limits: Dict[str, Tuple[float, float]],
        rng: Random,
    ):
        """
        Instantiate a constriction PSO propagator.

        *Important note:* ``c_cognitive`` and ``c_social`` have to sum up to a number greater than 4!

        Parameters
        ----------
        c_cognitive : float
                      constant cognitive factor for scaling the distance to the particle's personal best value
                      *Has to sum up with ``c_social`` to a number greater than 4!*
        c_social : float
                   Constant social factor for scaling the distance to the swarm's global best value
                   *Has to sum up with ``c_cognitive`` to a number greater than 4!*
        rank : int
               The global rank of the worker the propagator is living on
        limits : Dict[str, Tuple[float, float]]
                 borders of the continuous search domain
        rng : random.Random
              random number generator for introducing non-linearity

        Raises
        ------
        ValueError
            If ``c_social`` and ``c_cognitive`` do not sum up to a number greater than 4.
        """
        if c_cognitive + c_social <= 4:
            raise ValueError(
                "c_cognitive + c_social < 4 but should sum up to a number > 4!"
            )
        phi: float = c_cognitive + c_social
        chi: float = 2.0 / (phi - 2.0 + np.sqrt(phi * (phi - 4.0)))
        super().__init__(chi, c_cognitive, c_social, rank, limits, rng)

    def __call__(self, individuals: List[Individual]) -> Particle:
        """
        Apply the constriction PSO update rule.

        Return a ``Particle`` object containing the updated values of the youngest passed ``Particle`` or ``Individual``
        that belongs to the worker the propagator is living on.

        Parameters
        ----------
        individuals: List[Individual]
                     list of individuals that must at least contain one individual that belongs to the propagator
                     This list is used to calculate personal and global best of the particle and the swarm,
                     respectively, and then to update the particle based on the retrieved results. Individuals that
                     cannot be used as ``Particle`` objects are converted to particles first.

        Returns
        -------
        propulate.population.Particle
            updated particle
        """
        old_p, p_best, g_best = self._prepare_data(individuals)

        new_velocity = self.inertia * (
            old_p.velocity
            + self.rng.uniform(0, self.c_cognitive) * (p_best.position - old_p.position)
            + self.rng.uniform(0, self.c_social) * (g_best.position - old_p.position)
        )
        new_position = old_p.position + new_velocity

        return self._make_new_particle(new_position, new_velocity, old_p.generation + 1)


class CanonicalPSO(ConstrictionPSO):
    """
    This propagator subclass features a combination of constriction PSO and velocity clamping.

    The velocity clamping uses a clamping factor of 1, the constriction is done as in the parental ``Constriction``
    propagator.

    For information on the method parameters, please refer to the ``Constriction`` propagator.

    Original publications:
    R. Poli, J. Kennedy, and T. Blackwell. Particle swarm optimization. Swarm Intell 1, 33–57 (2007).
    https://doi.org/10.1007/s11721-007-0002-0
    R. C. Eberhart and Y. Shi. Comparing inertia weights and constriction factors in particle swarm optimization.
    Proceedings of the 2000 Congress on Evolutionary Computation. CEC00 (Cat. No.00TH8512), La Jolla, CA, USA, 2000,
    pp. 84-88 vol.1, https://10.1109/CEC.2000.870279.

    See Also
    --------
    :class:`ParentClassName` : Parent class
    """

    def __init__(self, c_cognitive, c_social, rank, limits, rng):
        """
        Initialize a canonical PSO propagator.

        In theory, it should be no problem to hand over numpy arrays instead of the float-type hyperparameters inertia,
        cognitive factor, and social factor. In this case, please ensure that the dimension of the passed arrays fits
        the search domain.

        Parameters
        ----------
        c_cognitive : float
                      constant cognitive factor for scaling the distance to the particle's personal best value
        c_social : float
                   constant social factor to scaling the distance to the swarm's global best value
        rank : int
               global rank of the worker the propagator is living on
        limits : Dict[str, Tuple[float, float]]
                 Borders of the continuous search domain
        rng : random.Random
              random number generator for introducing non-linearity
        """
        super().__init__(c_cognitive, c_social, rank, limits, rng)
        x_min, x_max = self.limits_as_array
        x_range = np.abs(x_max - x_min)
        self.v_cap: np.ndarray = np.array([-x_range, x_range])

    def __call__(self, individuals: List[Individual]) -> Particle:
        """
        Apply the canonical PSO variant update rule.

        Return a ``Particle`` object containing the updated values of the youngest passed ``Particle`` or ``Individual``
        that belongs to the worker the propagator is living on.

        Parameters
        ----------
        individuals: List[Individual]
                     list of individuals that must at least contain one individual that belongs to the propagator
                     This list is used to calculate personal and global best of the particle and the swarm,
                     respectively, and then to update the particle based on the retrieved results. Individuals that
                     cannot be used as ``Particle`` objects are converted to particles first.

        Returns
        -------
        propulate.population.Particle
            update particle
        """
        # Abuse Constriction's update rule, so I don't have to rewrite it.
        victim = super().__call__(individuals)

        # Set new position and speed.
        v = victim.velocity.clip(*self.v_cap)
        p = victim.position - victim.velocity + v

        # Create and return new particle.
        return self._make_new_particle(p, v, victim.generation)


class InitUniformPSO(Stochastic):
    """
    Initialize ``Particle`` by uniformly sampling specified limits for each trait.
    """

    def __init__(
        self,
        limits: Dict[str, Tuple[float, float]],
        rank: int,
        parents=0,
        probability=1.0,
        rng: Random = None,
        v_init_limit: Union[float, np.ndarray] = 0.1,
    ):
        """
        Instantiate a uniform-initialization PSO propagator.

        In case of parents > 0 and probability < 1., call returns input individual without change.

        Parameters
        ----------
        limits : dict[str, tuple[float, float]]
                 limits of the search space, i.e., limits of (hyper-)parameters to be optimized
        rank : int
               rank of the worker in MPI.COMM_WORLD
        parents : int
                  number of input individuals (-1 for any)
        probability : float
                      probability of creating a completely new individual
        rng : random.Random
              random number generator
        v_init_limit: float | np.ndarray
                      multiplicative constant to reduce initial random velocity values
        """
        super().__init__(parents, 1, probability, rng)
        self.limits = limits
        self.laa = np.array(list(limits.values())).T
        if isinstance(v_init_limit, np.ndarray):
            assert v_init_limit.shape[-1] == self.laa.shape[-1]
        self.v_limits = v_init_limit
        self.rank = rank

    def __call__(self, individuals: List[Individual]) -> Particle:
        """
        Apply uniform-initialization propagator.

        Parameters
        ----------
        individuals : List[Individual]
            individuals the propagator is applied to

        Returns
        -------
        propulate.population.Particle
            one particle object
        """
        if (
            len(individuals) == 0 or self.rng.random() < self.probability
        ):  # Apply only with specified `probability`.
            position = np.array(
                [self.rng.uniform(*self.laa[..., i]) for i in range(self.laa.shape[-1])]
            )
            velocity = np.array(
                [
                    self.rng.uniform(*(self.v_limits * self.laa)[..., i])
                    for i in range(self.laa.shape[-1])
                ]
            )

            particle = Particle(
                position, velocity, rank=self.rank
            )  # Instantiate new particle.

            for index, limit in enumerate(self.limits):
                # Since Py 3.7, iterating over dicts is stable, so we can do the following.

                if not isinstance(
                    self.limits[limit][0], float
                ):  # Check search space for validity
                    raise TypeError("PSO only works on continuous search spaces!")

                # Randomly sample from specified limits for each trait.
                particle[limit] = particle.position[index]
            return particle
        else:
            particle = individuals[0]
            if isinstance(particle, Particle):
                return particle  # Return 1st input individual w/o changes.
            else:
                return make_particle(particle)


class StatelessPSO(Propagator):
    """
    This propagator performs PSO without the need of Particles, but as a consequence, also without velocity.
    Thus, it is called stateless.

    As this propagator works without velocity, there is also no inertia weight used.

    It uses only classes provided by vanilla Propulate.
    """

    def __init__(
        self,
        c_cognitive: float,
        c_social: float,
        rank: int,
        limits: Dict[str, Tuple[float, float]],
        rng: Random,
    ):
        """
        Instantiate a stateless PSO propagator.

        Parameters
        ----------
        c_cognitive : float
                      constant cognitive factor for scaling individual's personal best value
        c_social : float
                   constant social factor for scaling swarm's global best value
        rank : int
               global rank of the worker the propagator is living on
        limits : Dict[str, Tuple[float, float]
                 Borders of the continuous search domain
        rng : random.Random
              random number generator required for non-linearity of update
        """
        super().__init__(parents=-1, offspring=1)
        self.c_social = c_social
        self.c_cognitive = c_cognitive
        self.rank = rank
        self.limits = limits
        self.rng = rng

    def __call__(self, individuals: List[Individual]) -> Individual:
        """
        Apply standard PSO update without inertia and old velocity.

        Parameters
        ----------
        individuals : List[Individual]
                      individuals are used as data basis for the PSO update

        Returns
        -------
        propulate.population.Individual
            An updated Individual

        Raises
        ------
        ValueError
            If the individuals list passed is empty and the propagator thus has no data to work on.
        """
        if len(individuals) < self.offspring:
            raise ValueError("Not enough particles.")
        own_p = [x for x in individuals if x.rank == self.rank]
        if len(own_p) > 0:
            old_p = max(own_p, key=lambda p: p.generation)
        else:  # No own particle found in given parameters, thus creating new one.
            old_p = Individual(0, self.rank)
            for k in self.limits:
                old_p[k] = self.rng.uniform(*self.limits[k])
            return old_p
        g_best = min(individuals, key=lambda p: p.loss)
        p_best = min(own_p, key=lambda p: p.loss)
        new_p = Individual(generation=old_p.generation + 1)
        for k in self.limits:
            new_p[k] = (
                old_p[k]
                + self.rng.uniform(0, self.c_cognitive) * (p_best[k] - old_p[k])
                + self.rng.uniform(0, self.c_social) * (g_best[k] - old_p[k])
            )
        return new_p
