"""
Provides an abstract class so that users can build custom samplers upon
"""

from abc import ABC, abstractmethod


class CustomSampler(ABC):
    """Abstract subclass for user defined samplers"""

    @property
    def seed_group(self):
        """The generator state this sampler shares, if it shares one.

        Samplers are independent by default and each is seeded on its own. Two
        wrappers over one generator, as the correlated wind pair in the
        documentation are, should both return that generator here, so the pair
        is seeded once as a unit rather than one of them silently overwriting
        the other's seed.

        Returns
        -------
        object
            Identity is what counts, not equality. ``self`` by default, which
            makes every sampler its own group.
        """
        return self

    @abstractmethod
    def sample(self, n_samples=1):
        """Generates samples from the custom distribution

        Parameters
        ----------
        n_samples : int, optional
            Numbers of samples to be generated

        Returns
        -------
        samples_list : list
            A list with n_samples elements, each of which is a valid sample
        """

    @abstractmethod
    def reset_seed(self, seed=None):
        """Resets the seeds of all associated stochastic generators

        Parameters
        ----------
        seed : int, optional
            Seed for the random number generator. The default is None

        Returns
        -------
        None
        """
