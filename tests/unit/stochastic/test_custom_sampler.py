from types import SimpleNamespace

import numpy as np
import pytest

from rocketpy.environment.environment import Environment
from rocketpy.stochastic import StochasticRocket
from rocketpy.stochastic.custom_sampler import CustomSampler
from rocketpy.stochastic.stochastic_model import StochasticModel, _sampler_seed


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


class _SharedPair:
    """Two wrappers over one generator, as the wind example in the docs does."""

    def __init__(self):
        self.rng = np.random.default_rng()
        self.last_seed = None
        self.reset_count = 0

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        self.last_seed = seed
        self.reset_count += 1

    def draw(self):
        return float(self.rng.normal())


class _SharedWrapper(CustomSampler):
    def __init__(self, shared):
        self.shared = shared

    def sample(self, n_samples=1):
        return [self.shared.draw() for _ in range(n_samples)]

    def reset_seed(self, seed=None):
        self.shared.reset(seed)


def _seed_the_shared_generator_received(declare_second_first):
    shared = _SharedPair()
    first, second = _SharedWrapper(shared), _SharedWrapper(shared)
    inputs = (
        {"wind_y": second, "wind_x": first}
        if declare_second_first
        else {"wind_x": first, "wind_y": second}
    )
    model = StochasticModel(SimpleNamespace(wind_x=0.0, wind_y=0.0), **inputs)
    shared.reset_count = 0  # the constructor has already seeded once
    model._set_stochastic(4242)
    return shared.last_seed, shared.reset_count


def test_a_shared_generator_lands_on_the_same_seed_whatever_the_order():
    """Samplers may share one generator on purpose, and each reset overwrites
    the last, so whichever is reset last decides the stream. Seeding runs in
    sorted order for that reason: the same seed has to mean the same stream
    whichever order the model was written in.

    On the values drawn, not the stream: two wrappers reading one generator
    take successive values, so swapping the declaration swaps which wrapper
    gets which. That is inherent to sharing a generator and is not seeding.
    """
    ordered, reversed_ = (
        _seed_the_shared_generator_received(False),
        _seed_the_shared_generator_received(True),
    )

    assert ordered == reversed_
    assert ordered[1] == 2, "each wrapper still resets the generator it wraps"


def test_two_names_that_a_hash_would_collide_get_different_streams():
    """Keying by a 32-bit hash put these two back on one stream, which is the
    bug this keying exists to prevent. Both are valid identifiers and their
    CRC32 is 1560575156."""
    assert _sampler_seed(4242, "wd4s4xka50") != _sampler_seed(4242, "p56cjcee10")


def test_the_sampler_seed_keeps_the_full_width():
    """128 bits, matching the width the Monte Carlo seeding uses, so a study
    spawning many streams does not run into birthday collisions."""
    assert _sampler_seed(4242, "mass").bit_length() > 64


def test_declaring_a_sampler_does_not_reorder_the_other_inputs(calisto_robust):
    """Seeding is sorted; the validation loop below it is not. Sorting that one
    too would set __dict__ alphabetically and move every tuple's draw."""
    plain = StochasticRocket(rocket=calisto_robust, mass=(14.426, 0.5))
    plain._set_stochastic(42)
    before = next(plain.dict_generator())

    with_sampler = StochasticRocket(
        rocket=calisto_robust, mass=(14.426, 0.5), radius=_Gaussian(0.0635, 0.001)
    )
    with_sampler._set_stochastic(42)
    after = next(with_sampler.dict_generator())

    # `radius` is declared either way, so it is in both. Only its kind changed.
    assert list(after) == list(before)
    assert after["mass"] == before["mass"]
