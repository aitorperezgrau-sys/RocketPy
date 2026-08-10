import pytest

from rocketpy.rocket.plate import Plate


@pytest.fixture()
def test_circular_plate():
    return Plate(
        shape="circular",
        dimensions=0.03,
        material="carbon_steel",
        thickness=0.0002,
        z_points=40,
        angular_points=40,
        name="test_circular_plate",
    )


@pytest.fixture()
def test_squared_plate():
    return Plate(
        shape="circular",
        dimensions=0.03,
        material="carbon_steel",
        thickness=0.0002,
        z_points=40,
        angular_points=40,
        name="test_squared_plate",
    )


@pytest.fixture()
def test_personalized_plate():
    return Plate(
        shape="personalized",
        material="carbon_steel",
        grid_spacing=0.002,
        name="test_personalized_plate",
    )
