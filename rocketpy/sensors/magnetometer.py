import math
from datetime import datetime

import numpy as np
from pywmm import WMMv2
from pywmm.calculator import calculate_geomagnetic
from pywmm.date_utils import decimal_year

from rocketpy.mathutils.vector_matrix import Matrix, Vector
from rocketpy.rocket import Rocket
from rocketpy.sensors.sensor import InertialSensor
from rocketpy.tools import inverted_haversine


class Magnetometer(InertialSensor):
    """
    Class for simulating a magnetometer sensor during rocket flight.

    Inherits from the InertialSensor subclass. Models Earth's geomagnetic
    field using the World Magnetic Model (WMM) and simulates physical sensor
    distortions, including hard iron, soft iron (via plates), power wire
    interferences (via communication & ignition wires), thermal drifts, noise, and
    quantization.


    Attributes
    ----------
    sampling_rate : float
        Sample rate of the sensor in Hz.
    orientation : tuple, list
        Orientation of the sensor in the rocket.
    magnetic_interference : list
        Magnetic interference on the magnetometer, it is the sum of the
        hard iron distortion and power interference and soft_iron_distortion_difference.
    hard_iron_distortion : list
        Hard iron distortion in T.
    _soft_iron_distortion_matrix : Matrix
        Sum of the soft iron distortion matrixes applied to the magnetic
        reading.
    soft_iron_distortion_difference : list
        Difference between the vector after aplication of soft iron distortion
        and before in T.
    power_interference : list
        Holds the total magnetic distortion due to the system interference in T
        regardless of the initialization mode, for a given measurement.
    communications_interference : list
        Holds the magnetic distortion due to the communication wires in T.
    activation_signal_interference: list
        Holds the magnetic distortion due to the ignition signals in T.
    measurement_range : float or tuple
        The measurement range of the sensor in T.
    resolution : float
        The resolution of the sensor in T/LSB.
    noise_density : float list
        The noise density of the sensor in T/√Hz.
    noise_variance : float, list
        The variance of the noise of the sensor in T^2.
    random_walk_density : float, list
        The random walk density of the sensor in T/√Hz.
    random_walk_variance : float, list
        The variance of the random walk of the sensor in T^2.
    constant_bias : float, list
        The constant bias of the sensor in T.
    operating_temperature : float
        The operating temperature of the sensor in Kelvin.
    temperature_bias : float, list
        The temperature bias of the sensor in T/K.
    temperature_scale_factor : float, list
        The temperature scale factor of the sensor in %/K.
    cross_axis_sensitivity : float
        The cross axis sensitivity of the sensor in percentage.
    name : str
        The name of the sensor.
    _random_walk_drift : Vector
        The random walk drift of the sensor in T.
    measurement : float
        The measurement of the sensor after quantization, noise, temperature
        drift and magnetic interference in T.
    measured_data : list
        The stored measured data of the sensor after quantization, noise and
        temperature drift.
    wmm : WMM object from pywmm library
        More details on the pywmm librayon its github repository:
          https://github.com/dougc95/pywmm/tree/main
    year : float
        Current decimal year.
    rotation_sensor_to_body : Matrix
        The rotation matrix of the sensor from the sensor frame to the rocket
        frame of reference.
    normal_vector : Vector
        The normal vector of the sensor in the rocket frame of reference.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        sampling_rate,
        orientation=(0, 0, 0),
        measurement_range=np.inf,
        resolution=0,
        hard_iron_distortion=0,
        soft_iron_distortion=Matrix.identity(),
        power_interference=0,
        activation_signal_interference=None,
        communications_interference=None,
        noise_density=0,
        noise_variance=1,
        random_walk_density=0,
        random_walk_variance=1,
        constant_bias=0,
        operating_temperature=298,
        temperature_bias=0,
        temperature_scale_factor=0,
        cross_axis_sensitivity=0,
        name="Magnetometer",
    ):
        """
        Initializes the Magnetometer sensor:

        Parameters
        -----------
        sampling_rate : float
            Sample rate of the sensor in Hz.
        orientation : tuple, list, optional
            Orientation of the sensor in the rocket. The orientation can be
            given as either:

            - A list of length 3, where the elements are the Euler angles for
              the rotation yaw (ψ), pitch (θ) and roll (φ) in radians. The
              standard rotation sequence is z-y-x (3-2-1) is used, meaning the
              sensor is first rotated by ψ around the x axis, then by θ around
              the new y axis and finally by φ around the new z axis.
            - A list of lists (matrix) of shape 3x3, representing the rotation
              matrix from the sensor frame to the rocket frame. The sensor frame
              of reference is defined as to have z axis along the sensor's normal
              vector pointing upwards, x and y axes perpendicular to the z axis
              and each other.

            The rocket frame of reference is defined as to have z axis
            along the rocket's axis of symmetry pointing upwards, x and y axes
            perpendicular to the z axis and each other. A rotation around the x
            axis configures a pitch, around the y axis a yaw and around z axis a
            roll. Default is (0, 0, 0), meaning the sensor is aligned with all
            of the rocket's axis.
        measurement_range : float, tuple, optional
            The measurement range of the sensor in T. If a float, the
            same range is applied both for positive and negative values. If a
            tuple, the first value is the positive range and the second value is
            the negative range. Default is np.inf.
        resolution : float, optional
            The resolution of the sensor in T/LSB. Default is 0, meaning no
            quantization is applied.
        hard_iron_distortion : float, list, optional
            The hard iron distortion desired for the sensor in T.
            If a float, the same value is applied to each axis
            If a list or float, the distortion will be taken considering these
            values.
        soft_iron_distortion : Matrix, string
            soft iron distortion is caused by materials with high permeability on
            the rocket, (circuit board copper traces, nearby metal casing), that do
            not generate the field, but distort the existing external field lines
            passing through them. This is because the magnetic permeability measures
            how easily a material allows magnetic field lines to pass through it.
            Therefore, because it has lower resistance than air  magnetic field lines
            will bent, to go through the material.

            If a Matrix,   a direct 3x3
            transformation matrix to the magnetic field. If string 'plates',
            computes distortion dynamically based on plates attached to the Rocket object.
            Default is Matrix.identity().
        power_interference : int, float, list, str, optional
            The power interference is the magnetic distortion due to the
            magnetic field generated by the current flowign through wires in the
            avionics bay in T. It is formed by the activation signal interference
            and the communication wires interference

            - If an int or float, the same magnetic field will be applied for
            each axis for the total power interference
            - If a list, the given magnetic field will be considered as
            the power interference

            - If a string, the accepted option are:
                - If 'wires': models total interference dynamically using rocket wires.
                  activation_signal_interference and communications_interference.
                - If 'personalized': requires passing activation_signal_interference and
                    communications_interference explicitly.
            Default is 0.
        activation_signal_interference : int, float, list, str, optional
            It is the magnetic field in T generated by the ignition wires,
            this depend on the conditions.

            - If an int or float, the same magnetic field will be applied for
            each axis for the total activation signal interference, regardless
            of the ignition wires defined.

            - If a list, the given magnetic field will be considered as
            the total activation signal interference, regardless of the
            ignition wires defined

            - If a string, the accepted option is 'wires'. then the
            activation_signal_interference will be considered using the wires
            attached to the rocket

            Default is None.
        communications_interference : int, float, list, str, optional
            It is the magnetic field in T generated by the communication wires,
            this is a constant value that it is added to the read magnetic field.

            - If an int or float, the same magnetic field will be applied for
            each axis for the communications interference, regardless
            of the ignition wires defined.

            - If a list, the given magnetic field will be considered as
            the communications interference, regardless of the
            ignition wires defined.

            - If a string, the accepted option is 'wires'. then the
            communications interference will be considered using the wires
            attached to the rocket

            Default is None.
        noise_density : float, list, optional
            The noise density of the sensor for a Gaussian white noise in T/√Hz.
            Sometimes called "white noise drift", "angular random walk" for
            gyroscopes, "velocity random walk" for accelerometers or
            "(rate) noise density". Default is 0, meaning no noise is applied.
            If a float or int is given, the same noise density is applied to all
            axes. The values of each axis can be set individually by passing a
            list of length 3.
        noise_variance : float, list, optional
            The noise variance of the sensor for a Gaussian white noise in T^2.
            Default is 1, meaning the noise is normally distributed with a
            standard deviation of 1 T. If a float or int is given, the same
            variance is applied to all axes. The values of each axis can be set
            individually by passing a list of length 3.
        random_walk_density : float, list, optional
            The random walk of the sensor for a Gaussian random walk in T/√Hz.
            Sometimes called "bias (in)stability" or "bias drift"". Default is 0,
            meaning no random walk is applied. If a float or int is given, the
            same random walk is applied to all axes. The values of each axis can
            be set individually by passing a list of length 3.
        random_walk_variance : float, list, optional
            The random walk variance of the sensor for a Gaussian random walk in
            T^2. Default is 1, meaning the noise is normally distributed
            with a standard deviation of 1 T. If a float or int is given,
            the same variance is applied to all axes. The values of each axis
            can be set individually by passing a list of length 3.
        constant_bias : float, list, optional
            The constant bias of the sensor in T. Default is 0, meaning no
            constant bias is applied. If a float or int is given, the same bias
            is applied to all axes. The values of each axis can be set
            individually by passing a list of length 3.
        operating_temperature : float, optional
            The operating temperature of the sensor in Kelvin.
            At 298.15 K (25 °C), the sensor is assumed to operate ideally, no
            temperature related noise is applied. Default is 298.15.
        temperature_bias : float, list, optional
            The temperature bias of the sensor in T/K. Default is 0,
            meaning no temperature bias is applied. If a float or int is given,
            the same temperature bias is applied to all axes. The values of each
            axis can be set individually by passing a list of length 3.
        temperature_scale_factor : float, list, optional
            The temperature scale factor of the sensor in %/K. Default is 0,
            meaning no temperature scale factor is applied. If a float or int is
            given, the same temperature scale factor is applied to all axes. The
            values of each axis can be set individually by passing a list of
            length 3.
        cross_axis_sensitivity : float, optional
            Skewness of the sensor's axes in percentage. Default is 0, meaning
            no cross-axis sensitivity is applied.
        name : str, optional
            The name of the sensor. Default is 'Magnetometer'.

        """
        self.magnetic_interference = [0, 0, 0]

        if isinstance(power_interference, str):
            if power_interference.lower() == "wires":
                self.initial_power_interference = "wires"
                self.validate_power_wires_parameters()
            elif power_interference.lower() == "personalized":
                self.initial_power_interference = "personalized"
                self.validate_power_personalized_parameters(
                    activation_signal_interference, communications_interference
                )
            else:
                raise ValueError("The accepted strings are 'wires' or 'personalized'")
        else:
            if isinstance(power_interference, (int, float)):
                power_interference = [power_interference] * 3
            self.power_interference = list(power_interference)
            self._power_interference = Vector(self.power_interference)

        self.validate_soft_iron(soft_iron_distortion)
        self.validate_hard_iron(hard_iron_distortion)

        # Get current decimal year
        current_date = datetime.now().strftime("%Y-%m-%d")
        self.year = decimal_year(current_date)

        # initialize the magnetic model
        self.wmm = WMMv2()

        # initialize InertialSensor class
        super().__init__(
            sampling_rate=sampling_rate,
            orientation=orientation,
            measurement_range=measurement_range,
            resolution=resolution,
            noise_density=noise_density,
            noise_variance=noise_variance,
            random_walk_density=random_walk_density,
            random_walk_variance=random_walk_variance,
            constant_bias=constant_bias,
            operating_temperature=operating_temperature,
            temperature_bias=temperature_bias,
            temperature_scale_factor=temperature_scale_factor,
            cross_axis_sensitivity=cross_axis_sensitivity,
            name=name,
        )

    def validate_soft_iron(self, soft_iron_distortion):
        """
        Checks and defines the soft_iron_distortion parameter.
        """
        # initialize soft_iron_distortion attribute
        if isinstance(soft_iron_distortion, Matrix):
            self._soft_iron_distortion_matrix = soft_iron_distortion
            self.soft_iron_distortion_difference = []
        elif isinstance(soft_iron_distortion, str):
            if soft_iron_distortion == "plates":
                self._soft_iron_distortion_matrix = Matrix.identity()
                self.initial_soft_iron_distortion_matrix = "plates"
                self.total_soft_iron_distortion_matrix_computed = False
                self.soft_iron_distortion_difference = []
            else:
                raise ValueError("The accepted string must be plates")

    def validate_hard_iron(self, hard_iron_distortion):
        """
        Checks and defines the soft_iron_distortion parameter.
        """
        if isinstance(hard_iron_distortion, (float, int)):
            hard_iron_distortion = [hard_iron_distortion] * 3

        self.hard_iron_distortion = list(hard_iron_distortion)
        self._hard_iron_distortion = Vector(self.hard_iron_distortion)

    def validate_power_personalized_parameters(
        self, activation_signal_interference, communications_interference
    ) -> None:

        if (
            activation_signal_interference is None
            or communications_interference is None
        ):
            raise ValueError(
                "For 'personalized' interference, you must provide values for both "
                "activation_signal_interference and communications_interference."
            )

        self.validate_activation_signal_interference(activation_signal_interference)
        self.validate_communications_interferemce(communications_interference)

    def validate_activation_signal_interference(self, activation_signal_interference):
        """
        Checks activation signal interference and defines the related attributes
        """

        if isinstance(activation_signal_interference, str):
            if activation_signal_interference.lower() == "wires":
                self.activation_signal_interference = [0, 0, 0]
                self.initial_activation_signal_interference = "wires"
            else:
                raise ValueError("The accepted string is wires")
        else:
            if isinstance(activation_signal_interference, (int, float)):
                activation_signal_interference = [activation_signal_interference] * 3
            self.initial_activation_signal_interference = "number"
            self.activation_signal_interference = list(activation_signal_interference)
        self._activation_signal_interference = Vector(
            self.activation_signal_interference
        )

    def validate_communications_interferemce(self, communications_interference):
        """
        Checks communications interference and defines the related attributes.
        """

        if isinstance(communications_interference, str):
            if communications_interference.lower() == "wires":
                self.communications_interference = [0, 0, 0]
                self.initial_communications_interference = "wires"
                self.communications_computed = False
            else:
                raise ValueError("The accepted string is wires")
        else:
            if isinstance(communications_interference, (int, float)):
                communications_interference = [communications_interference] * 3
            self.initial_communications_interference = "number"
            self.communications_interference = list(communications_interference)

        self._communications_interference = Vector(self.communications_interference)

    def validate_power_wires_parameters(self):
        self.power_interference = [0, 0, 0]
        self.communications_interference = [0, 0, 0]
        self.communications_computed = False
        self.activation_signal_interference = [0, 0, 0]

        self.initial_communications_interference = "wires"
        self.initial_activation_signal_interference = "wires"

    def measure(self, time: float, **kwargs) -> None:
        """
        obtain the simulated reading of the magnetometer for a given time step

        Parameters
        ----------
        time : float
            Current time in seconds.
        kwargs : dict
            Keyword arguments dictionary containing the following keys:

             - u : np.array
                State vector of the rocket.
                u = [x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]

            rocket: Rocket
                Rocketpy Rocket class

            - u_dot : np.array
                Derivative of the state vector of the rocket.

            - relative_position: Vector
                Position of the sensor relative to the rocket center of dry mass in m.

            - parachute_events : list only required if the ignition_wire_function
                is 'parachute_ignition'
                List that stores parachute events triggered during flight.
                it is a list formed by lists which contain the trigger time
                as the first element and the parachute object as the second.

            - environment : Environment
                Environment object containing the atmospheric conditions.

        Returns
        -------
        None
        """
        # initialization of parameters
        u = kwargs["u"]  # state vector
        parachute_events = kwargs.get("parachute_events", None)
        sensor_from_bacs = kwargs[
            "relative_position"
        ]  # sensor position from body axis coordinate system
        current_time = time
        lat0, lon0, launch_site_elevation = (
            kwargs["environment"].latitude,
            kwargs["environment"].longitude,
            kwargs["environment"].elevation,
        )
        earth_radius = kwargs["environment"].earth_radius
        rocket = kwargs["rocket"]

        # u[6:10]: Quaternion represents the com orientation with respect to the inertial frame.
        rotation_bacs_to_inertial = Matrix.transformation(
            u[6:10]
        )  # rotation matrix from com to inertial frame

        # --- obtain the current longitude, latitude and elevation ---
        # obtain the sensor coordinates in the inertial frame, by adding the offset to the positon vector
        x_inertial, y_inertial, z_inertial = (
            rotation_bacs_to_inertial @ sensor_from_bacs
            + Vector(
                u[0:3]
            )  # Vector(u[0:3]) is the coordinates center of dry mass in the inertial frame
        )
        b_north, b_east, b_down = self.obtain_magnetic_field(
            x_inertial,
            y_inertial,
            z_inertial,
            launch_site_elevation,
            earth_radius,
            lat0,
            lon0,
        )

        # --- Transform from NED to Rocketpy's inertial frame ---
        b_field_inertial = Vector([b_east, b_north, -b_down])  # T

        # --- from Rocketpy's inertial frame to bacs frame ---
        b_field_bacs = rotation_bacs_to_inertial.transpose @ b_field_inertial  # T

        # --- Apply magnetic interference ---
        b_field_bacs = self.apply_magnetic_interference(
            b_field_bacs, rocket, current_time, parachute_events
        )  # T

        # Transform body frame (bacs) to sensor frame, includes the cross-axis sensitivity adjustment
        b_field_sensor = (
            self._total_rotation_sensor_to_body.transpose @ b_field_bacs
        )  # T

        # --- apply noise and quantize ---
        b_field_sensor = self.apply_temperature_drift(b_field_sensor)  # T
        b_field_sensor = self.apply_noise(b_field_sensor)  # T
        b_field_sensor = self.quantize(b_field_sensor)  # T

        self.measurement = (b_field_sensor.x, b_field_sensor.y, b_field_sensor.z)  # T
        self._save_data((time, *b_field_sensor))

    def obtain_magnetic_field(
        self,
        x_inertial,
        y_inertial,
        z_inertial,
        launch_site_elevation,
        earth_radius,
        lat0,
        lon0,
    ):
        """
        Returns from the magnetic model the magnetic field components in the NED
        frame based on the coordinates in the intertial frame, the launch site elevation
        and the earth radius and initial longitude and latitude of the launch site.
        """
        # z is calculated in meters above the sea level, we must change to WGS84 in km
        altitude_wgs84_km = (z_inertial + launch_site_elevation) / 1000.0

        # Convert x and y to current latitude and longitude
        drift = math.hypot(x_inertial, y_inertial)
        bearing = math.atan2(x_inertial, y_inertial) * (180 / math.pi) % 360
        latitude, longitude = inverted_haversine(
            lat0, lon0, drift, bearing, earth_radius
        )

        # --- obtain the magnetic field in the NED (North-East-Down axis) --
        # Calculate all field components at once
        calculate_geomagnetic(
            self.wmm, latitude, longitude, self.year, altitude_wgs84_km
        )

        # components of the magnetic field
        b_north = self.wmm.bx / 1e9  # T
        b_east = self.wmm.by / 1e9  # T
        b_down = self.wmm.bz / 1e9  # T

        return b_north, b_east, b_down

    def apply_magnetic_interference(
        self,
        b_field: Vector,
        rocket: Rocket,
        current_time: float | int,
        parachute_events: list | None = None,
    ) -> Vector:
        """
        Applies the magnetic distortion due to the power interference,
        hard iron and soft iron.

        Parameters
        ----------
        b_field : Vector
            Magnetic field Vector.
        rocket : Rocket
            Rocketpy Rocket class.
        current_time : float, only required if the ignition_wire_function
            is 'motor_ignition'
            Current time of the simulation.
        parachute_events : list only required if the ignition_wire_function
            is 'parachute_ignition'
            List that stores parachute events triggered during flight.
            it is a list formed by lists which contain the trigger time
            as the first element and the parachute object as the second.

        Returns
        -------
        b_field : Vector
            Magnetic field after adjustment of the hard iron,
            and power interference.

        """
        self.magnetic_interference = [0, 0, 0]

        b_field = self.apply_soft_iron(b_field, rocket)  # T
        b_field = self.apply_hard_iron(b_field)  # T
        b_field = self.apply_power_interference(
            b_field, rocket, current_time, parachute_events
        )  # T

        self.magnetic_interference = [
            self.power_interference[0]
            + self.hard_iron_distortion[0]
            + self.soft_iron_distortion_difference[0],
            self.power_interference[1]
            + self.hard_iron_distortion[1]
            + self.soft_iron_distortion_difference[1],
            self.power_interference[2]
            + self.hard_iron_distortion[2]
            + self.soft_iron_distortion_difference[2],
        ]
        return b_field

    def apply_soft_iron(self, b_field: Vector, rocket: Rocket) -> Vector:
        """
        Applies the soft iron distortion which is the distoriton
        of the magnetic field due to the higher magnetic permeability of
        some materials relative to the permeability of vacuum. This entails,
        that they have smaller magnetic resistance resulting in a bending of
        the magnetic field lines, that are forced to pass through them.

        Parameters
        ----------
        b_field : Vector
            Vector reading of the magnetic field of the earth.
        rocket : Rocket
            Rocketpy Rocket class.

        Returns
        -------
        b_field_distorted : Vector
            Magnetic field vector after the soft iron distortion.
        """
        if self.initial_soft_iron_distortion_matrix == "plates":
            if not self.total_soft_iron_distortion_matrix_computed:
                for plate in rocket.plates:
                    if (
                        not self.sensor_from_cso_t
                        in plate._magnetic_distortion_matrixes
                    ):
                        plate.calculate_soft_iron_distortion_matrix(
                            self._sensor_from_cso
                        )
                        self._soft_iron_distortion_matrix = (
                            self._soft_iron_distortion_matrix
                            + plate._magnetic_distortion_matrixes[
                                self.sensor_from_cso_t
                            ]
                        )

                self.total_soft_iron_distortion_matrix_computed = True

            b_field_distorted = self._soft_iron_distortion_matrix @ b_field
        else:
            b_field_distorted = self._soft_iron_distortion_matrix @ b_field

        self.soft_iron_distortion_difference = list(b_field_distorted - b_field)

        return b_field_distorted

    def apply_hard_iron(self, b_field: Vector) -> Vector:
        """
        Applies the hard iron distortion. This magnetic distortion is
        caused by permanent magnets or magnetized materials on the
        rocket itself that move along with the sensor (from steel screws,
        battery casing, feerromagnetic components), thus it is a constant
        value. It shifts the center of the magnetic data.

        Parameters
        ----------
        b_field : Vector
            Magnetic field Vector.

        Returns
        -------
        b_field : Vector
            Magnetic field after hard_iron_distortion.
        """
        b_field = b_field + self._hard_iron_distortion
        return b_field

    def apply_power_interference(
        self,
        b_field: Vector,
        rocket: Rocket,
        current_time: float,
        parachute_events: list | None = None,
    ) -> Vector:
        """
        Applies the electromagnetic interference to the magnetic field vector,
        when a signal is triggered. It considers that there is electron flow,
        thus, a generation of magnetic field, when the conditions for the trigger
        are fulfilled if it is a ignition wire, or always when it is a
        communication wire.

        Parameters
        ----------
        b_field : Vector
            Magnetic field Vector.
        rocket : Rocket
            Rocketpy Rocket class.
        current_time : float, only required if the ignition_wire_function
            is 'motor_ignition'
            current time of the simulation.
        parachute_events : list only required if the ignition_wire_function
            is 'parachute_ignition'
            List that stores parachute events triggered during flight.
            it is a list formed by lists which contain the trigger time
            as the first element and the parachute object as the second.

        Returns
        -------
        b_field : Vector
            Magnetic field after adjustment of both the activation signal
            interference and communications interference.

        """
        if self.initial_power_interference in ("wires" or "personalized"):
            self.power_interference = [0, 0, 0]
            b_field = self.apply_communications_interference(b_field, rocket)
            b_field = self.apply_activation_signal_interference(
                b_field, rocket, current_time, parachute_events
            )
            self.power_interference = [
                self.activation_signal_interference[0]
                + self.communications_interference[0],
                self.activation_signal_interference[1]
                + self.communications_interference[1],
                self.activation_signal_interference[2]
                + self.communications_interference[2],
            ]
            self._power_interference = Vector(self.power_interference)
        else:
            b_field = b_field + self._power_interference

        return b_field

    def apply_communications_interference(
        self, b_field: Vector, rocket: Rocket
    ) -> Vector:
        """
        Applies the interference caused due to the current flowing
        through the communication wires.

        Parameters
        ----------
        b_field : Vector
            Magnetic field Vector.
        rocket : Rocket
            Rocketpy Rocket class.

        Returns
        -------
        b_field : Vector
            Magnetic field after adjustment of communications magnetic
            interference.
        """
        if self.initial_communications_interference == "wires":
            if rocket.communication_wires:
                if not self.communications_computed:
                    for communication_wire in rocket.communication_wires:
                        communication_wire.measure_magnetic_field(self._sensor_from_cso)
                        self.communications_interference = [
                            self.communications_interference[0]
                            + communication_wire.magnetic_field[self.sensor_from_cso_t][
                                0
                            ],
                            self.communications_interference[1]
                            + communication_wire.magnetic_field[self.sensor_from_cso_t][
                                1
                            ],
                            self.communications_interference[2]
                            + communication_wire.magnetic_field[self.sensor_from_cso_t][
                                2
                            ],
                        ]

                    self._communications_interference = Vector(
                        self.communications_interference
                    )
                    b_field = b_field + self._communications_interference
                    self.communications_computed = True
                else:
                    b_field = b_field + self._communications_interference
            else:
                raise ValueError(
                    "You must define first some communication wires, to be able to consider the magnetic distrubance created by them"
                )
        else:
            b_field = b_field + self._communications_interference

        return b_field

    def apply_activation_signal_interference(
        self,
        b_field: Vector,
        rocket: Rocket,
        current_time: float,
        parachute_events: list | None = None,
    ) -> Vector:
        """
        Applies the magnetic interference caused due to the current
        flowing through the ignition wires, during an activation signal.

        Parameters
        ----------
        b_field : Vector
            Magnetic field Vector.
        rocket : Rocket
            Rocketpy Rocket class.
        current_time : float, only required if the ignition_wire_function
            is 'motor_ignition'
            Current time of the simulation.
        parachute_events : list only required if the ignition_wire_function
            is 'parachute_ignition'
            List that stores parachute events triggered during flight.
            it is a list formed by lists which contain the trigger time
            as the first element and the parachute object as the second.

        Returns
        -------
        b_field : Vector
            Magnetic field after adjustment activation signal interference
            interference.
        """

        if self.initial_activation_signal_interference == "wires":
            if rocket.ignition_wires:
                self.activation_signal_interference = [0, 0, 0]
                for ignition_wire in rocket.ignition_wires:
                    if ignition_wire.ignition_wire_function == "parachute_ignition":
                        b_field = (
                            self.calculate_activation_signal_interference_parachute(
                                b_field, current_time, parachute_events, ignition_wire
                            )
                        )

                    elif ignition_wire.ignition_wire_function == "motor_ignition":
                        b_field = self.calculate_activation_signal_interference_motor(
                            b_field, rocket, current_time, ignition_wire
                        )
                    else:
                        raise ValueError(
                            "The accepted strings for the ignition_wire_function are motor_ignition and parachute_ignition"
                        )
            else:
                raise ValueError(
                    "You must define some ignition wire to be able to consider its magnetic disturbance."
                )
        else:
            b_field = b_field + self._activation_signal_interference

        return b_field

    def calculate_activation_signal_interference_parachute(
        self, b_field, current_time, parachute_events, ignition_wire
    ):
        """
        Applies the activation signal interference due to the parachute.

        Parameters
        ----------
        b_field : Vector
            Magnetic field Vector.
        current_time : float, only required if the ignition_wire_function
            is 'motor_ignition'
            Current time of the simulation.
        parachute_events : list only required if the ignition_wire_function
            is 'parachute_ignition'
            List that stores parachute events triggered during flight.
            it is a list formed by lists which contain the trigger time
            as the first element and the parachute object as the second.
        igntion_wire : wire
            Wire with wire type ignition and parachute_ignition as a function.

        Returns
        -------
        b_field : Vector
            Vector containing the magnetic field after the
            checking and calculation of the parachute ignition
            conditions.
        """
        if not parachute_events is None:
            for parachute_event in parachute_events:
                ejection_time = parachute_event[0]
                parachute = parachute_event[1]
                if (
                    parachute.name == ignition_wire.parachute_name
                    and ejection_time != 0
                    and ejection_time - ignition_wire.lead_ignition_time
                    <= current_time
                    <= ejection_time + ignition_wire.extra_ignition_time
                ):
                    if not self.sensor_from_cso_t in ignition_wire._magnetic_field:
                        ignition_wire.measure_magnetic_field(self._sensor_from_cso)
                    self.activation_signal_interference = [
                        self.activation_signal_interference[0]
                        + ignition_wire.magnetic_field[self.sensor_from_cso_t][0],
                        self.activation_signal_interference[1]
                        + ignition_wire.magnetic_field[self.sensor_from_cso_t][1],
                        self.activation_signal_interference[2]
                        + ignition_wire.magnetic_field[self.sensor_from_cso_t][2],
                    ]

                    b_field = (
                        b_field + ignition_wire._magnetic_field[self.sensor_from_cso_t]
                    )
        else:
            raise ValueError(
                "The parachute events should be passed if a wire has ignition_wire_function == parachute_ignition"
            )

        return b_field

    def calculate_activation_signal_interference_motor(
        self, b_field, rocket, current_time, ignition_wire
    ):
        """
        Parameters
        ----------
        b_field : Vector
            Magnetic field Vector.
        rocket : Rocket
            Rocketpy Rocket class.
        current_time : float, only required if the ignition_wire_function
            is 'motor_ignition'
            Current time of the simulation.
        igntion_wire : wire
            Wire with wire type ignition and motor_ignition as a function.

        Returns
        -------
        b_field : Vector
            Magnetic field after adjustment activation signal interference
            interference.
        """
        if (
            rocket.motor.burn_start_time - ignition_wire.lead_ignition_time
            <= current_time
            <= rocket.motor.burn_start_time + ignition_wire.extra_ignition_time
        ):
            if not self.sensor_from_cso_t in ignition_wire._magnetic_field:
                ignition_wire.measure_magnetic_field(self._sensor_from_cso)
            self.activation_signal_interference = [
                self.activation_signal_interference[0]
                + ignition_wire.magnetic_field[self.sensor_from_cso_t][0],
                self.activation_signal_interference[1]
                + ignition_wire.magnetic_field[self.sensor_from_cso_t][1],
                self.activation_signal_interference[2]
                + ignition_wire.magnetic_field[self.sensor_from_cso_t][2],
            ]
            b_field = b_field + ignition_wire._magnetic_field[self.sensor_from_cso_t]
        return b_field

    def export_measured_data(self, filename, file_format="csv"):
        """
        Exports the measured values to a file

        Parameters
        ----------
        filename : str
            Name of the file to export the values to.
        file_format : str
            Format of the file to export the values to. Options are "csv" and
            "json". Default is "csv".

        Returns
        -------
        None

        """
        self._generic_export_measured_data(
            filename=filename,
            file_format=file_format,
            data_labels=("t", "Bx", "By", "Bz"),
        )

    @classmethod
    def from_dict(cls, data: dict) -> "Magnetometer":
        """
        Creates an instance of Magnetometer from a dictionary object, data.
        Data is a dictionary that must contain the same keys as the initialization
        parameter of the Magnetometer class. In the case some parameter is not
        defined, the default value matches the default intializaiton of the constructor

        Returns
        -------
            Magnetometer object
        """
        return cls(
            # Mandatory Parameter
            sampling_rate=data["sampling_rate"],
            # Optional Parameters
            orientation=data.get("orientation", (0, 0, 0)),
            measurement_range=data.get("measurement_range", np.inf),
            resolution=data.get("resolution", 0),
            hard_iron_distortion=data.get("hard_iron_distortion", 0.0),
            soft_iron_distortion=data.get("soft_iron_distortion", Matrix.identity()),
            power_interference=data.get("power_interference", 0),
            activation_signal_interference=data.get(
                "activation_signal_interference", None
            ),
            communications_interference=data.get("communications_interference", None),
            # Noise Profiles
            noise_density=data.get("noise_density", 0),
            noise_variance=data.get("noise_variance", 1),
            random_walk_density=data.get("random_walk_density", 0),
            random_walk_variance=data.get("random_walk_variance", 1),
            constant_bias=data.get("constant_bias", 0),
            # Environmental & Structural Shifts
            operating_temperature=data.get("operating_temperature", 298),
            temperature_bias=data.get("temperature_bias", 0),
            temperature_scale_factor=data.get("temperature_scale_factor", 0),
            cross_axis_sensitivity=data.get("cross_axis_sensitivity", 0),
        )
