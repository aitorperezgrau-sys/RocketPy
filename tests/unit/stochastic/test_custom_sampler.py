import numpy as np
import pytest

from rocketpy.stochastic import StochasticRocket
from rocketpy.stochastic.custom_sampler import CustomSampler
from rocketpy.environment.environment import Environment


def test_create_object(stochastic_environment_custom_sampler):
    """Test create object method of StochasticEnvironment class.

    This test checks if the create_object method of the StochasticEnvironment
    class creates a StochasticEnvironment object from the randomly generated
    input arguments.

    Parameters
    ----------
    stochastic_environment : StochasticEnvironment
        StochasticEnvironment object to be tested.

    Returns
    -------
    None
    """
    obj = stochastic_environment_custom_sampler.create_object()
    assert isinstance(obj, Environment)


class _Gaussian(CustomSampler):
    """A sampler of the shape the documentation teaches."""

    def __init__(self, mean, sd):
        self.mean, self.sd = mean, sd
        self.rng = np.random.default_rng()

    def sample(self, n_samples=1):
        return list(self.rng.normal(self.mean, self.sd, n_samples))

    def reset_seed(self, seed=None):
        self.rng = np.random.default_rng(seed)


def _deviates(drawn):
    """The standard normal behind each draw, so samplers with different means
    and spreads can still be compared."""
    return (
        (drawn["mass"] - 14.426) / 0.5,
        (drawn["radius"] - 0.0635) / 0.001,
    )


def _two_sampler_model(calisto_robust):
    return StochasticRocket(
        rocket=calisto_robust,
        mass=_Gaussian(14.426, 0.5),
        radius=_Gaussian(0.0635, 0.001),
    )


def test_two_samplers_do_not_draw_the_same_deviate(calisto_robust):
    """Every sampler on a model used to be reset with the model's own seed, so
    two backed by ``default_rng`` started from the same state and drew the same
    underlying value. Not nearly identical: the same, to every digit."""
    model = _two_sampler_model(calisto_robust)
    model._set_stochastic(4242)

    mass, radius = _deviates(next(model.dict_generator()))

    assert mass != pytest.approx(radius, abs=1e-12)


def test_a_seed_still_reproduces_the_same_samples(calisto_robust):
    """The control. Independence must not have been bought with fresh entropy
    per reseed, which would decorrelate the samplers and lose the seed."""
    model = _two_sampler_model(calisto_robust)

    model._set_stochastic(4242)
    first = _deviates(next(model.dict_generator()))
    model._set_stochastic(4242)
    again = _deviates(next(model.dict_generator()))
    model._set_stochastic(99)
    other = _deviates(next(model.dict_generator()))

    assert first == again
    assert first != other


def test_adding_a_parameter_leaves_the_others_where_they_were(calisto_robust):
    """Seeds are keyed by the input's name, not its position, so declaring one
    more sampler does not move the streams of the ones already there."""
    model = _two_sampler_model(calisto_robust)
    model._set_stochastic(4242)
    before = _deviates(next(model.dict_generator()))

    wider = StochasticRocket(
        rocket=calisto_robust,
        mass=_Gaussian(14.426, 0.5),
        radius=_Gaussian(0.0635, 0.001),
        inertia_11=_Gaussian(6.321, 0.1),
    )
    wider._set_stochastic(4242)
    after = _deviates(next(wider.dict_generator()))

    assert after == before
