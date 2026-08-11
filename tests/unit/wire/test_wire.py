import pytest

from rocketpy.mathutils import Vector
from rocketpy.plots.wire_plots import _WirePlots
from rocketpy.rocket import Wire


@pytest.mark.parametrize(
    "wire_type, ignition_wire_funciton, current, extra_ignition_time",
    [
        (0, "motor_ignition", 1, 0.4),
        (  # wrong wire type: number
            "ignition",
            None,
            1,
            0.4,
        ),
        (  # wrong ignition_wire_type: None
            "ignition",
            0,
            1,
            0.4,
        ),
        (  # wrong ignition_wire_type: number
            "ignition",
            "motor",
            "1",
            0.4,
        ),
        (  # wrong current: str
            "ignition",
            "motor",
            1,
            "0.4",
        ),
        (  # wrong extra_igntion_time: str
            "ignition",
            "motor",
            1,
            -0.4,
        ),  # wrong extra_igntion_time: -
    ],
)
def validate_wire_type(wire_type, ignition_wire_function, current, extra_ignition_time):
    with pytest.raises(ValueError):
        Wire(current, wire_type, ignition_wire_function, extra_ignition_time)


def test_define_magnetic_field(test_communications_wire):
    position_vector = [0.003, 0.002, 1]
    magnetic_field = [10, 25, 10]
    test_communications_wire.define_magnetic_field(position_vector, magnetic_field)
    list_key = list(test_communications_wire._magnetic_field)
    assert list_key[0] == tuple(position_vector)
    assert (
        test_communications_wire._magnetic_field[tuple(position_vector)]
        == magnetic_field
    )
    assert test_communications_wire._magnetic_field[tuple(position_vector)] == Vector(
        magnetic_field
    )


def test_current_directon(calisto_robust):
    horizontal_wire_1 = Wire(10, wire_type="communications", name="horizontal_wire")
    horizontal_wire_2 = Wire(10, wire_type="communications", name="horizontal_wire")
    calisto_robust.add_wire(horizontal_wire_1, [[0.001, 0.001, 0], [0.001, -0.001, 0]])
    calisto_robust.add_wire(horizontal_wire_2, [[0.001, -0.001, 0], [0.001, 0.001, 0]])

    horizontal_wire_1.measure_magnetic_field([0, 0, 0])
    horizontal_wire_2.measure_magnetic_field([0, 0, 0])

    # calisto_robust has the same user defined coordinate  as the body axis coordinate system (center at cdm and orientation tail to nose)
    b_field_1 = horizontal_wire_1._magnetic_field[(0, 0, 0)]
    b_field_2 = horizontal_wire_2._magnetic_field[(0, 0, 0)]
    assert b_field_1[0] == 0
    assert b_field_1[1] == 0
    assert b_field_2[0] == 0
    assert b_field_2[1] == 0
    assert b_field_2[2] == -b_field_1[2]


def test_from_dict():
    wire_dict = {
        "current": 3,
        "wire_type": "ignition",
        "ignition_wire_function": "motor_ignition",
        "extra_ignition_time": 2,
    }
    wire_from_dict = Wire.from_dict(wire_dict)

    assert isinstance(wire_from_dict, Wire)
    assert wire_from_dict.current == 3
    assert wire_from_dict.wire_type == "ignition"
    assert wire_from_dict.ignition_wire_function == "motor_ignition"
    assert wire_from_dict.extra_ignition_time == 2


def test_rocket_belonging(test_communications_wire, calisto_robust):
    calisto_robust.add_wire(
        test_communications_wire,
        position_edges=[[0.001, 0.002, -0.3], [0.001, 0.002, 0.3]],
    )
    assert isinstance(test_communications_wire.plots, _WirePlots)


def test_wire_prints_and_plots(test_communications_wire, calisto_robust):
    """Test the print methods of the Wire class. Checks if all attributes are
    printed and plotted correctly.
    """
    calisto_robust.add_wire(
        test_communications_wire, [[0.001, 0.002, -0.3], [0.001, 0.002, 0.3]]
    )
    test_communications_wire.prints.all()
    test_communications_wire.plots.all()
    assert True
