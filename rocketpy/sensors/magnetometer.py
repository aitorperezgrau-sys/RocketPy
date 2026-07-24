import math
import csv
import numpy as np
from rocketpy.sensors.sensor import Sensor, InertialSensor, ScalarSensor
from rocketpy.mathutils.vector_matrix import Matrix, Vector
from rocketpy.tools import inverted_haversine
from pywmm import WMMv2
from datetime import datetime
from pywmm.date_utils import decimal_year
from pywmm.calculator import calculate_geomagnetic
from rocketpy.rocket import Rocket


class Magnetometer(InertialSensor):
    ''' 
    Class for the magnetometer sensor, this class inherits from 
    InertialSensor rocketpy subclass which in turn inhertis from Sensor class.
    This class replicates the simulated measuremente of the magnetometer sensor
    during a flight and allows to define the uncertainties associated with it. 


    Attributes: 
    -----------------
    sampling_rate : float
        Sample rate of the sensor in Hz.

    orientation : tuple, list
        Orientation of the sensor in the rocket.
    
    magnetic_interference: list
        Magnetic interference on the magnetometer, it is the sum of the
        hard iron distortion and power interference and soft_iron_distortion_difference
        
    hard_iron_distortion: list
        Hard iron distortion in T
    
    _soft_iron_distortion_matrix: Matrix
        Sum of the soft iron distortion matrixes applied to the magnetic 
        reading.

    soft_iron_distortion_difference: list
        Difference between the vector after aplication of soft iron distortion 
        and before in T
    
    power_interference: list 
        Holds the total magnetic distortion due to the system interference in T
        regardless of the initialization mode, for a given measurement 

    communications_interference: list
        Holds the magnetic distortion due to the communication wires in T
    
    activation_signal_interference: list
        Holds the magnetic distortion due to the ignition signals in T

    measurement_range : float, tuple
        The measurement range of the sensor in T.

    resolution : float
        The resolution of the sensor in T/LSB.

    noise_density : float, list
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
        The measurement of the sensor after quantization, noise and temperature
        drift.

    measured_data : list
        The stored measured data of the sensor after quantization, noise and
        temperature drift.

    wmm: WMM object from pywmm library
        More details on the pywmm librayon its github repository:
          https://github.com/dougc95/pywmm/tree/main

    year: float
        Current decimal year

    rotation_sensor_to_body : Matrix
        The rotation matrix of the sensor from the sensor frame to the rocket
        frame of reference.

    normal_vector : Vector
        The normal vector of the sensor in the rocket frame of reference.

    rotation_matrix : Matrix
        The rotation matrix of the sensor from the rocket frame to the sensor
        frame of reference.
    '''
    def __init__(
            self,
            sampling_rate,
            orientation = (0,0,0),
            measurement_range = np.inf,
            resolution = 0,
            hard_iron_distortion = 0,
            soft_iron_distortion = Matrix.identity(),
            power_interference = 0, 
            activation_signal_interference = None,
            communications_interference = None,
            noise_density = 0,
            noise_variance = 1,
            random_walk_density = 0,
            random_walk_variance = 1,
            constant_bias = 0,
            operating_temperature = 298,
            temperature_bias = 0,
            temperature_scale_factor = 0,
            cross_axis_sensitivity = 0,
            name = 'Magnetometer',
        ):
        '''
        initialize the magnetometer sensor

        Parameters
        ----------
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

        hard_iron_distortion: float, list, optional
            The hard iron distortion desired for the sensor in T. 
            If a float, the same value is applied to each axis 
            If a list or float, the distortion will be taken considering these 
            values. 

        soft_iron_distortion: Matrix, string
            soft iron distortion is caused by materials with high permeability on 
            the rocket, (circuit board copper traces, nearby metal casing), that do
            not generate the field, but distort the existing external field lines
            passing through them. This is because the magnetic permeability measures
            how easily a material allows magnetic field lines to pass through it.
            Therefore, because it has lower resistance than air  magnetic field lines
            will bent, to go through the material. 

            If a Matrix the matrix will be taken as a scaling factor 
            applied to the magnetic field reading.
            If a string, the accepted input is 'plates', which will
            consider the soft iron distortion based on the plates added to the 
            rocket. 

            Default is the identity matrix, meaning no soft iron distortion is applied

        power_interference: int, float, list, str, optional
            The power interference is the magnetic distortion due to the 
            magnetic field generated by the current flowign through wires in the 
            avionics bay in T. It is formed by the activation signal interference
            and the communication wires interference

            If an int or float, the same magnetic field will be applied for
            each axis for the total power interference
            If a list, the given magnetic field will be considered as
            the power interference

            If a string, the accepted option are wires. In this case, the power 
            interference can be modeled using the wires included in the rocket
            regardless of the inputs of the parameters: activation_signal_interference 
            and communications_interference. If personalized, 
            the activation_signal_interference and communications_interference
            can be defined independently. Thus this parameters are compulsory
            Default is 0.

        activation_signal_interference: int, float, list, str, optional
            It is the magnetic field in T generated by the ignition wires, 
            this depend on the conditions. 

            If an int or float, the same magnetic field will be applied for
            each axis for the total activation signal interference, regardless
            of the ignition wires defined.

            If a list, the given magnetic field will be considered as
            the total activation signal interference, regardless of the 
            ignition wires defined

            If a string, the accepted option is wires. then the 
            activation_signal_interference will be considered using the wires 
            attached to the rocket
            Default is None

        communications_interference: int, float, list, str, optional
            It is the magnetic field in T generated by the communication wires, 
            this is a constant value that it is added to the read magnetic field.

            If an int or float, the same magnetic field will be applied for
            each axis for the communications interference, regardless
            of the ignition wires defined.

            If a list, the given magnetic field will be considered as
            the communications interference, regardless of the 
            ignition wires defined. 

            If a string, the accepted option is wires. then the 
            communications interference will be considered using the wires 
            attached to the rocket          
            Default is None

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

        '''
        self.magnetic_interference = [0, 0, 0]

        if isinstance(power_interference, (int,float)):
            self.power_interference = [power_interference, power_interference, power_interference]
            self._power_interference = Vector(self.power_interference)
            self.initial_power_interference = 'number'
        elif isinstance(power_interference, (list, tuple)):
            if len(power_interference) == 3:
                self.power_interference = list(power_interference)
                self._power_interference = Vector(self.power_interference)
                self.initial_power_interference = 'number'
            else:
                raise ValueError('The length of the list must be 3: x,y,z')
        elif isinstance(power_interference, str):
            if power_interference.lower() == 'wires':
                self.power_interference = [0, 0, 0]
                self.communications_interference = [0, 0, 0]
                self.communications_computed = False
                self.activation_signal_interference = [0, 0, 0]

                self.initial_power_interference = 'wires'
                self.initial_communications_interference = 'wires'
                self.initial_activation_signal_interference = 'wires'
            elif power_interference.lower() == 'personalized':
                self.initial_power_interference = 'personalized'

                if activation_signal_interference is None or communications_interference is None:
                    raise ValueError(
                        "For 'personalized' interference, you must provide values for both "
                        "activation_signal_interference and communications_interference."
                    )
                
                if isinstance(activation_signal_interference, (int, float)):
                        self.activation_signal_interference = [activation_signal_interference, activation_signal_interference, activation_signal_interference]
                        self._activation_signal_interference = Vector(self.activation_signal_interference)
                        self.initial_activation_signal_interference = 'number'
                elif isinstance(activation_signal_interference, (list, tuple)):
                    if len(activation_signal_interference) == 3:
                        self.activation_signal_interference = list(activation_signal_interference)
                        self._activation_signal_interference = Vector(self.activation_signal_interference)
                        self.initial_activation_signal_interference = 'number'
                    else:
                        raise ValueError('The length of the list must be 3: x,y,z')
                elif isinstance(activation_signal_interference, str):
                    if activation_signal_interference.lower() == 'wires':
                        self.activation_signal_interference = [0, 0, 0]
                        self.initial_activation_signal_interference = 'wires'
                    else:
                        raise ValueError('The accepted string is wires')  
                else:
                    raise ValueError('The accepted values are list, tuple, str, int or float')
                    
                if isinstance(communications_interference, (int, float)):
                        self.communications_interference = [communications_interference, communications_interference, communications_interference]
                        self._communications_interference = Vector(self.communications_interference)
                        self.initial_communications_interference = 'number'
                elif isinstance(communications_interference, (list, tuple)):
                    if len(communications_interference) == 3:
                        self.communications_interference = list(communications_interference)
                        self._communications_interference = Vector(self.communications_interference)
                        self.initial_communications_interference = 'number'
                    else:
                        raise ValueError('The length of the list must be 3: x,y,z')
                elif isinstance(communications_interference, str):
                    if communications_interference.lower() == 'wires':
                        self.communications_interference = [0, 0, 0]
                        self.initial_communications_interference = 'wires'
                        self.communications_computed = False
                    else:
                        raise ValueError('The accepted string is wires')
                else: 
                    raise ValueError('The accepted values are list, tuple, str, int or float')
            else:
                raise ValueError('The accepted strings are wires or personalized')
        else:
            raise ValueError('The accepted values are list, tuple, str, int or float')

        # initialize hard_iron_distortion attribute
        if isinstance(hard_iron_distortion, (float, int)):
            self.hard_iron_distortion = [hard_iron_distortion, hard_iron_distortion, hard_iron_distortion]
        elif isinstance(hard_iron_distortion, (list, tuple)):
            if len(hard_iron_distortion) == 3: 
                self.hard_iron_distortion = list(hard_iron_distortion)
            else:
                raise Exception('If a list is passed, it must have a value for each axis. Therefore, it must have length 3')
        else:
            raise ValueError('Hard iron must be a float, int or a list with 3 elements')
        
        self._hard_iron_distortion = Vector(self.hard_iron_distortion)
        
        # initialize soft_iron_distortion attribute
        if isinstance(soft_iron_distortion, Matrix):
            for element in soft_iron_distortion:
                if not isinstance(element, (float, int)):
                    raise ValueError('The elements inside the matrix must be float or int')
                
            self._soft_iron_distortion_matrix = soft_iron_distortion
            self.initial_soft_iron_distortion_matrix = 'number'
            self.soft_iron_distortion_difference = []
        elif isinstance(soft_iron_distortion, str):
            if soft_iron_distortion == 'plates':
                self._soft_iron_distortion_matrix = Matrix.identity()
                self.initial_soft_iron_distortion_matrix = 'plates'
                self.total_soft_iron_distortion_matrix_computed = False
                self.soft_iron_distortion_difference = []
            else:
                raise ValueError('The accepted string must be plates')
        else:
            raise ValueError('The soft iron distortion can only be a Matrix or a string')

        # Get current decimal year
        current_date = datetime.now().strftime('%Y-%m-%d')
        self.year = decimal_year(current_date)

        # initialize the magnetic model
        self.wmm = WMMv2()

        # initialize InertialSensor class
        super().__init__(
            sampling_rate = sampling_rate,
            orientation = orientation,
            measurement_range = measurement_range,
            resolution = resolution,
            noise_density = noise_density,
            noise_variance = noise_variance,
            random_walk_density = random_walk_density,
            random_walk_variance = random_walk_variance,
            constant_bias = constant_bias,
            operating_temperature = operating_temperature,
            temperature_bias = temperature_bias,
            temperature_scale_factor = temperature_scale_factor,
            cross_axis_sensitivity = cross_axis_sensitivity,
            name = name,
        )


    def measure(self, time, **kwargs):
        '''
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

            - parachute_events: list only required if the ignition_wire_function
                is 'parachute_ignition'
                List that stores parachute events triggered during flight.
                it is a list formed by lists which contain the trigger time 
                as the first element and the parachute object as the second. 

            - extra_time_signal_motor: float, only required if the ignition_wire_function
                is 'motor_ignition'. 
                Time after the ignition of the motor, in which the igniton signal is 
                wanted to be sent. 

            - extra_time_signal_parachute: float, only required if the ignition_wire_function
                is 'parachute_ignition'. 
                Time after the ignition of the motor, in which the ignition signal is 
                wanted to be sent. It is considered to have the same value for all parachutes.
                
            - lead_time_signal_motor: float, optional
                Time before the ignition of the motor in which the ignition signal 
                is considered to be sent. 

            - lead_time_signal_parachute: float, optional
                Time before the ignition of the parachutes in which the ignition signal 
                is considered to be sent. it is considered to be the same for each parachutes. 
            
            - environment : Environment
                Environment object containing the atmospheric conditions.

        '''
        # initialization of input parameters
        u = kwargs["u"] #state vector
        parachute_events = kwargs.get("parachute_events", None)
        sensor_from_com = kwargs["sensor_from_com"]
        current_time = time
        lat0, lon0, launch_site_elevation = kwargs["environment"].latitude, kwargs["environment"].longitude, kwargs["environment"].elevation
        earth_radius = kwargs["environment"].earth_radius
        rocket = kwargs["rocket"]

        extra_time_signal_parachute = kwargs.get('extra_time_signal_parachute', 1)
        extra_time_signal_motor = kwargs.get('extra_time_signal_motor', 1.5)
        lead_time_signal_motor = kwargs.get('lead_time_signal_motor', 0.1)
        lead_time_signal_parachute = kwargs.get('lead_time_signal_parachute', 0.1)

        quaternion = u[6:10]  # Quaternion represents the com orientation with respect to the inertial frame.
        rotation_com_to_inertial = Matrix.transformation(quaternion) # rotation matrix from com to inertial frame

        #--- obtain the current longitude, latitude and elevation ---
        sensor_from_inertial = rotation_com_to_inertial @ sensor_from_com

        # obtain the sensor coordinates in the inertial frame, by adding the offset to the positon vector 
        com_from_inertial = Vector(u[0:3])
        x_inertial, y_inertial, z_inertial =  sensor_from_inertial + com_from_inertial
            
        # z is calculated in meters above the sea level, we must change to WGS84 in km
        altitude_wgs84_m = z_inertial + launch_site_elevation
        altitude_wgs84_km = altitude_wgs84_m / 1000.0

        # Convert x and y to current latitude and longitude
        drift = math.hypot(x_inertial, y_inertial)
        bearing = math.atan2(x_inertial, y_inertial) * (180 / math.pi) % 360
        latitude, longitude = inverted_haversine(lat0, lon0, drift, bearing, earth_radius)

        #--- obtain the magnetic field in the NED (North-East-Down axis) --
        # Calculate all field components at once 
        calculate_geomagnetic(self.wmm, latitude, longitude, self.year, altitude_wgs84_km)

        # components of the magnetic field
        b_north = self.wmm.bx / 1e9   # T
        b_east = self.wmm.by / 1e9   # T
        b_down = self.wmm.bz / 1e9   # T

        #--- Transform to Rocketpy's inertial frame ---
        b_inertial_x = b_east       # T
        b_inertial_y = b_north      # T
        b_inertial_z = - b_down     # T
        B_inertial = Vector([b_inertial_x, b_inertial_y, b_inertial_z])  # T
     
        # --- from Rocketpy's inertial frame to com frame ---
        rotation_inertial_to_com = rotation_com_to_inertial.transpose 
        B_com = rotation_inertial_to_com @ B_inertial  # T

        # B_com is already aligned with the body axes (CSO) because the axes are parallel
        B_cso = B_com 

        #--- Transform body frame to sensor frame ---
        rotation_cso_to_sensor = self._total_rotation_sensor_to_body.transpose # total includes the cross-axis sensitivity

        B_sensor = rotation_cso_to_sensor @ B_cso  # T

        #--- Apply noise + bias and quantize ---
        B_sensor = self.apply_magnetic_interference(B_sensor, 
                                                    rocket, 
                                                    current_time, 
                                                    parachute_events, 
                                                    extra_time_signal_parachute, 
                                                    extra_time_signal_motor,
                                                    lead_time_signal_parachute,
                                                    lead_time_signal_motor)  # T
        B_sensor = self.apply_temperature_drift(B_sensor)                                                                             # T
        B_sensor = self.apply_noise(B_sensor)                                                                                         # T
        B_sensor = self.quantize(B_sensor)                                                                                            # T

        self.measurement = (B_sensor.x, B_sensor.y, B_sensor.z)   # T                                  
        self._save_data((time, *B_sensor))        
    

    def apply_magnetic_interference(self, 
                                 B: Vector, 
                                 rocket: Rocket, 
                                 current_time, 
                                 parachute_events = None, 
                                 extra_time_signal_parachute = 1, 
                                 extra_time_signal_motor = 1.5,
                                 lead_time_signal_parachute: float = 0.1,
                                 lead_time_signal_motor: float = 0.1,
                                ):
        '''
        This funciton applies the magnetic distortion due to 
        the power interference, hard iron and soft iron.

        Input:
        --------
        B: Vector
            magnetic field Vector

        rocket: Rocket
            Rocketpy Rocket class

        current_time: float, only required if the ignition_wire_function
            is 'motor_ignition'
            current time of the simulation, from the initial time

        parachute_events: list only required if the ignition_wire_function
            is 'parachute_ignition'
            List that stores parachute events triggered during flight.
            it is a list formed by lists which contain the trigger time 
            as the first element and the parachute object as the second. 

        extra_time_signal_motor: float, only required if the ignition_wire_function
            is 'motor_ignition'. 
            Time after the ignition of the motor, in which the igniton signal is 
            wanted to be sent. 

        extra_time_signal_parachute: float, only required if the ignition_wire_function
            is 'parachute_ignition'. 
            Time after the ignition of the motor, in which the ignition signal is 
            wanted to be sent. It is considered to have the same value for all parachutes.
            
        lead_time_signal_motor: float, optional
            Time before the ignition of the motor in which the ignition signal 
            is considered to be sent

        lead_time_signal_parachute: float, optional
            Time before the ignition of the parachutes in which the ignition signal 
            is considered to be sent. it is considered to be the same for each parachutes. 

        Returns:
        -------
        B: Vector
            Magnetic field after adjustment of the hard iron, 
            and power interference

        '''
        self.magnetic_interference = [0, 0, 0]

        B = self.apply_soft_iron(B, rocket)                                                                                                                                 # T
        B = self.apply_hard_iron(B)                                                                                                                                         # T
        B = self.apply_power_interference(B, 
                                          rocket, 
                                          current_time, 
                                          parachute_events, 
                                          extra_time_signal_parachute, 
                                          extra_time_signal_motor,
                                          lead_time_signal_parachute,
                                          lead_time_signal_motor)   # T

        self.magnetic_interference = [
            self.power_interference[0] + self.hard_iron_distortion[0] + self.soft_iron_distortion_difference[0],
            self.power_interference[1] + self.hard_iron_distortion[1] + self.soft_iron_distortion_difference[1], 
            self.power_interference[2] + self.hard_iron_distortion[2] + self.soft_iron_distortion_difference[2]
        ]
        return B 
    

    def apply_soft_iron(self, B, rocket: Rocket):
        '''
        This function applies the soft iron distortion which is the distoriton 
        of the magnetic field due to the higher magnetic permeability of 
        some materials relative to the permeability of vacuum. This entails, 
        that they have smaller magnetic resistance resulting in a bending of 
        the magnetic field lines, that are forced to pass through them. 


        input
        ----------
        B: Vector
        Vector reading of the magnetic field of the earth

        rocket: Rocket
            Rocketpy Rocket class


        Returns
        ---------
        B_distorted: Vector
            Magnetic field vector after the soft iron distortion
        '''
        if self.initial_soft_iron_distortion_matrix == 'plates':
            if not self.total_soft_iron_distortion_matrix_computed:

                for plate in rocket.plates:
                    if not self.sensor_from_cso_t in plate._magnetic_distortion_matrixes: 
                        plate.calculate_soft_iron_distortion_matrix(self._sensor_from_cso)
                        self._soft_iron_distortion_matrix  = self._soft_iron_distortion_matrix + plate._magnetic_distortion_matrixes[self.sensor_from_cso_t]

                self.total_soft_iron_distortion_matrix_computed = True

            B_distorted = self._soft_iron_distortion_matrix @ B
        elif self.initial_soft_iron_distortion_matrix == 'number':
            B_distorted = self._soft_iron_distortion_matrix @ B

        self.soft_iron_distortion_difference = list(B_distorted - B)

        return B_distorted
    

    def apply_hard_iron(self, B):
        '''
        This funtion applies the hard iron distortion. 
        This magnetic distortion is caused by permanent magnets or 
        magnetized materials on the rocket itself that move along with 
        the sensor (from steel screws, battery casing, feerromagnetic components), 
        thus it is a constant value. It shifts the center of the magnetic data

        Input:
        --------
        B: Vector
            magnetic field Vector
        
        Returns: 
        -------
        B: Vector
            Magnetic field after hard_iron_distortion. 

        '''
        B = B + self._hard_iron_distortion
        return B
    


    def apply_power_interference(self, 
                                 B: Vector, 
                                 rocket: Rocket, 
                                 current_time, 
                                 parachute_events = None, 
                                 extra_time_signal_parachute: float = 1, 
                                 extra_time_signal_motor: float = 1.5,
                                 lead_time_signal_motor: float = 0.1,
                                 lead_time_signal_parachute: float = 0.1
                                ):
        
        '''
        This funtion applies the electromagnetic interference to the 
        magnetic field vector, when a signal is triggered. 
        It considers that there is electron flow, thus, 
        a generation of magnetic field, when the conditions
        for the trigger are fulfilled, when it is a 
        ignition wire, or always when it is a communication wire. 

        Input:
        --------
        B: Vector
            magnetic field Vector

        rocket: Rocket
            Rocketpy Rocket class

        current_time: float, only required if the ignition_wire_function
            is 'motor_ignition'
            current time of the simulation, from the initial time

        parachute_events: list only required if the ignition_wire_function
            is 'parachute_ignition'
            List that stores parachute events triggered during flight.
            it is a list formed by lists which contain the trigger time 
            as the first element and the parachute object as the second. 

        extra_time_signal_motor: float, only required if the ignition_wire_function
            is 'motor_ignition'. 
            Time after the ignition of the motor, in which the igniton signal is 
            wanted to be sent. 

        extra_time_signal_parachute: float, only required if the ignition_wire_function
            is 'parachute_ignition'. 
            Time after the ignition of the motor, in which the ignition signal is 
            wanted to be sent. It is considered to have the same value for all parachutes.
            
        lead_time_signal_motor: float, optional
            Time before the ignition of the motor in which the ignition signal 
            is considered to be sent

        lead_time_signal_parachute: float, optional
            Time before the ignition of the parachutes in which the ignition signal 
            is considered to be sent. it is considered to be the same for each parachutes. 
            
        Returns:
        -------
        B: Vector
            Magnetic field after adjustment of both the 
            activation signal interference and communications
            interference

        '''
        if self.initial_power_interference == 'wires' or self.initial_power_interference == 'personalized':
            self.power_interference = [0,0,0]
            B = self.apply_communications_interference(B, rocket)
            B = self.apply_activation_signal_interference(B, 
                                                          rocket, 
                                                          current_time, 
                                                          parachute_events, 
                                                          extra_time_signal_parachute, 
                                                          extra_time_signal_motor,
                                                          lead_time_signal_parachute,
                                                          lead_time_signal_motor)
            self.power_interference = [
                self.activation_signal_interference[0] + self.communications_interference[0],
                self.activation_signal_interference[1] + self.communications_interference[1], 
                self.activation_signal_interference[2] + self.communications_interference[2]
            ]
            self._power_interference = Vector(self.power_interference)
        elif self.initial_power_interference == 'number':
            B = B + self._power_interference

        return B
    

    def apply_communications_interference(self, B: Vector, rocket: Rocket):
        '''
        This function applies the interference caused due to the current
        flowing through the communication wires. 

         Input:
        --------
        B: Vector
            magnetic field Vector

        rocket: Rocket
            Rocketpy Rocket class

        Returns
        -------
        B: Vector
            Magnetic field after adjustment of communications
            magnetic interference. 
        
        '''
        if self.initial_communications_interference == 'wires':
            if rocket.communication_wires: 
                if not self.communications_computed:
                    for communication_wire in rocket.communication_wires:
                        communication_wire.measure_magnetic_field(self._sensor_from_cso)
                        self.communications_interference = [
                                                        self.communications_interference[0] + communication_wire.magnetic_field[self.sensor_from_cso_t][0],
                                                        self.communications_interference[1] + communication_wire.magnetic_field[self.sensor_from_cso_t][1], 
                                                        self.communications_interference[2] + communication_wire.magnetic_field[self.sensor_from_cso_t][2]
                                                        ]
                    
                    self._communications_interference = Vector(self.communications_interference)
                    B = B + self._communications_interference
                    self.communications_computed = True
                else:
                    B = B + self._communications_interference
            else:
                raise ValueError('You must define first some communication wires, to be able to consider the magnetic distrubance created by them')
        elif  self.initial_communications_interference == 'number':
            B = B + self._communications_interference

        return B


    def apply_activation_signal_interference(self, 
                                 B: Vector, 
                                 rocket: Rocket, 
                                 current_time, 
                                 parachute_events = None, 
                                 extra_time_signal_parachute: float = 1, 
                                 extra_time_signal_motor: float = 1.5,
                                 lead_time_signal_motor: float = 0.1,
                                 lead_time_signal_parachute: float = 0.1
                                ):
        '''
        This function applies the interference caused due to the current
        flowing through the ignition wires, during an activation signal

         Input:
        --------
        B: Vector
            magnetic field Vector

        rocket: Rocket
            Rocketpy Rocket class

        current_time: float, only required if the ignition_wire_function
            is 'motor_ignition'
            current time of the simulation, from the initial time

        parachute_events: list only required if the ignition_wire_function
            is 'parachute_ignition'
            List that stores parachute events triggered during flight.
            it is a list formed by lists which contain the trigger time 
            as the first element and the parachute object as the second. 

        extra_time_signal_motor: float, only required if the ignition_wire_function
            is 'motor_ignition'. 
            Time after the ignition of the motor, in which the igniton signal is 
            wanted to be sent. 

        extra_time_signal_parachute: float, only required if the ignition_wire_function
            is 'parachute_ignition'. 
            Time after the ignition of the motor, in which the ignition signal is 
            wanted to be sent. It is considered to have the same value for all parachutes.

        lead_time_signal_motor: float, optional
            Time before the ignition of the motor in which the ignition signal 
            is considered to be sent

        lead_time_signal_parachute: float, optional
            Time before the ignition of the parachutes in which the ignition signal 
            is considered to be sent. it is considered to be the same for each parachutes. 

        Returns: 
        -------
        B: Vector
            Magnetic field after adjustment activation signal interference
            interference
        '''
        
        if self.initial_activation_signal_interference == 'wires':
            if rocket.ignition_wires: 
                self.activation_signal_interference = [0, 0, 0]
                for ignition_wire in rocket.ignition_wires:
                    if ignition_wire.ignition_wire_function == 'parachute_ignition':
                        if not parachute_events == None:
                            for parachute_event in parachute_events:
                                ejection_time = parachute_event[0]
                                parachute = parachute_event[1]
                                initial_time = ejection_time - lead_time_signal_parachute
                                final_time = ejection_time + extra_time_signal_parachute
                                if parachute.name == ignition_wire.parachute_name and ejection_time != 0 and  initial_time <= current_time <= final_time:
                                    if not self.sensor_from_cso_t in ignition_wire._magnetic_field:
                                        ignition_wire.measure_magnetic_field(self._sensor_from_cso)
                                    self.activation_signal_interference = [
                                                                            self.activation_signal_interference[0] + ignition_wire.magnetic_field[self.sensor_from_cso_t][0],
                                                                            self.activation_signal_interference[1] + ignition_wire.magnetic_field[self.sensor_from_cso_t][1],
                                                                            self.activation_signal_interference[2] + ignition_wire.magnetic_field[self.sensor_from_cso_t][2],
                                                                            ]
                                    
                                    B = B + ignition_wire._magnetic_field[self.sensor_from_cso_t]
                        else:
                            raise ValueError('The parachute events should be passed if a wire has ignition_wire_function == parachute_ignition')
                    elif ignition_wire.ignition_wire_function == 'motor_ignition':
                        initial_time = rocket.motor.burn_start_time  - lead_time_signal_motor
                        final_time = rocket.motor.burn_start_time +  extra_time_signal_motor
                        if initial_time <= current_time <= final_time:
                            if not self.sensor_from_cso_t in ignition_wire._magnetic_field:
                                ignition_wire.measure_magnetic_field(self._sensor_from_cso)
                            self.activation_signal_interference = [
                                                                    self.activation_signal_interference[0] + ignition_wire.magnetic_field[self.sensor_from_cso_t][0],
                                                                    self.activation_signal_interference[1] + ignition_wire.magnetic_field[self.sensor_from_cso_t][1],
                                                                    self.activation_signal_interference[2] + ignition_wire.magnetic_field[self.sensor_from_cso_t][2],
                                                                    ]
                            B = B + ignition_wire._magnetic_field[self.sensor_from_cso_t]
                    else:
                        raise ValueError('The accepted strings for the ignition_wire_function are motor_ignition and parachute_ignition')
            else:
                raise ValueError('You must define some ignition wire to be able to consider its magnetic disturbance.')
        elif self.initial_activation_signal_interference == 'number':
            B = B + self._activation_signal_interference

        return B
    

    def export_measured_data(self, filename, file_format):
        '''
        Export the measured values to a file

        Parameters
        ----------
        filename : str
            Name of the file to export the values to
        file_format : str
            Format of the file to export the values to. Options are "csv" and
            "json". Default is "csv".

        Returns:
        -------
        None

        '''
        self._generic_export_measured_data(
            filename    = filename,
            file_format = file_format,
            data_labels = ("t", "Bx", "By", "Bz"),
        )
        

    @classmethod
    def from_dict(cls, data: dict):
        '''
        Creates an instance of Magnetometer from a dictionary object, data. 
        Data is a dictionary that must contain the same keys as the initialization
        parameter of the Magnetometer class. In the case some parameter is not 
        defined, the default value matches the default intializaiton of the constructor

        Returns: 
            Magnetometer object
        '''
        return cls(
            # Mandatory Parameter 
            sampling_rate = data['sampling_rate'],
            
            # Optional Parameters 
            orientation = data.get('orientation', (0, 0, 0)),
            measurement_range = data.get('measurement_range', np.inf),
            resolution = data.get('resolution', 0),
            hard_iron_distortion = data.get('hard_iron_distortion', 0.0),
            soft_iron_distortion = data.get('soft_iron_distortion', Matrix.identity()),
            power_interference = data.get('power_interference', 0),
            activation_signal_interference = data.get('activation_signal_interference', None),
            communications_interference = data.get('communications_interference', None),

            # Noise Profiles
            noise_density = data.get('noise_density', 0),
            noise_variance = data.get('noise_variance', 1),
            random_walk_density = data.get('random_walk_density', 0),
            random_walk_variance = data.get('random_walk_variance', 1),
            constant_bias = data.get('constant_bias', 0),
            
            # Environmental & Structural Shifts
            operating_temperature = data.get('operating_temperature', 298),
            temperature_bias = data.get('temperature_bias', 0),
            temperature_scale_factor = data.get('temperature_scale_factor', 0),
            cross_axis_sensitivity = data.get('cross_axis_sensitivity', 0),
        )
            