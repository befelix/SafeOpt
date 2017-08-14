"""
General class for constrained swarm optimization.

Authors: - Felix Berkenkamp (befelix at inf dot ethz dot ch)
         - Nicolas Carion (nicolas dot carion at gmail dot com)
"""

from __future__ import print_function, absolute_import, division

import numpy as np
from builtins import range


__all__ = ['SwarmOptimization']


class SwarmOptimization(object):
    """Constrained swarm optimization.

    Parameters
    ----------
    swarm_size: int
        The number of particles
    velocity: ndarray
        The base velocities of particles for each dimension.
    fitness: callable
        A function that takes particles positions and returns two values. The
        first one corresponds to the fitness of the particle, while the second
        one is an array of booleans indicating whether the particle fulfills
        the constraints.
    bounds: list, optional
        A list of constraints to which particle exploration is limited. Of the
        form [(x1_min, x1_max), (x2_min, x2_max)...].
    """

    def __init__(self, swarm_size, velocity, fitness, bounds=None):
        """Initialization, see `SwarmOptimization`."""
        super(SwarmOptimization, self).__init__()

        self.c1 = self.c2 = 1
        self.fitness = fitness

        self.bounds = bounds
        if self.bounds is not None:
            self.bounds = np.asarray(self.bounds)

        self.initial_inertia = 1.0
        self.final_inertia = 0.1
        self.velocity_scale = velocity

        self.ndim = len(velocity)
        self.swarm_size = swarm_size

        self.positions = np.empty((swarm_size, len(velocity)), dtype=np.float)
        self.velocities = np.empty_like(self.positions)

        self.best_positions = np.empty_like(self.positions)
        self.best_values = np.empty(len(self.best_positions), dtype=np.float)
        self.global_best = None

    @property
    def max_velocity(self):
        """Return the maximum allowed velocity of particles."""
        return 10 * self.velocity_scale

    def init_swarm(self, positions):
        """Initialize the swarm.

        Parameters
        ----------
        positions: ndarray
            The initial positions of the particles.
        """
        self.positions = positions
        self.velocities = (np.random.rand(*self.velocities.shape) *
                           self.velocity_scale)

        values, safe = self.fitness(self.positions)

        # Initialize best estimates
        self.best_positions[:] = self.positions
        self.best_values = values

        self.global_best = self.best_positions[np.argmax(values), :]

    def run_swarm(self, max_iter):
        """Let the swarm explore the parameter space.

        Parameters
        ----------
        max_iter : int
            The number of iterations for which to run the swarm.
        """
        # run the core swarm optimization
        inertia = self.initial_inertia
        inertia_step = (self.final_inertia - self.initial_inertia) / max_iter

        for _ in range(max_iter):
            # update velocities
            delta_global_best = self.global_best - self.positions
            delta_self_best = self.best_positions - self.positions

            # Random update vectors
            r = np.random.rand(2 * self.swarm_size, self.ndim)
            r1 = r[:self.swarm_size]
            r2 = r[self.swarm_size:]

            # Update the velocities
            self.velocities *= inertia
            self.velocities += ((self.c1 * r1 * delta_self_best +
                                 self.c2 * r2 * delta_global_best) /
                                self.velocity_scale)

            inertia += inertia_step

            # clip
            # np.clip(velocities, -4, 4, out=velocities)
            np.clip(self.velocities,
                    -self.max_velocity,
                    self.max_velocity,
                    out=self.velocities)

            # update position
            self.positions += self.velocities

            # Clip particles to domain
            if self.bounds is not None:
                np.clip(self.positions,
                        self.bounds[:, 0],
                        self.bounds[:, 1],
                        out=self.positions)

            # compute fitness
            values, safe = self.fitness(self.positions)

            # find out which particles are improving
            update_set = values > self.best_values

            # update whenever safety and improvement are guaranteed
            update_set &= safe

            self.best_values[update_set] = values[update_set]
            self.best_positions[update_set] = self.positions[update_set]

            best_value_id = np.argmax(self.best_values)
            self.global_best = self.best_positions[best_value_id, :]
