import math
import csv
import numpy as np
from rocketpy.sensors import Sensor, InertialSensor, ScalarSensor
from rocketpy.mathutils.vector_matrix import Matrix, Vector
from rocketpy.tools import inverted_haversine
from pywmm import WMMv2
from datetime import datetime
from pywmm.date_utils import decimal_year
from pywmm.calculator import calculate_geomagnetic
from rocketpy.rocket import Rocket



class Magnetometer(InertialSensor):

    ''' 
    class for the magnetometer sensor, this class inherits from 
    InertialSensor rocketpy subclass which, in turn inhertis from Sensor class.
    This class replicates the simulated measured value of the magnetometer sensor. 


    Attributes: 
    -----------------
    sampling_rate : float
        Sample rate of the sensor in Hz.

    orientation : tuple, list
        Orientation of the sensor in the rocket.

    hard_iron_distortion: list
        Hard iron distortion in T
    
    power_interference: list 
        Holds the total magnetic distortion due to the system interference in T
        regardless of the initalization mode.  

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

    '''

    def __init__(
            self,
            sampling_rate,
            orientation                   = (0,0,0),
            measurement_range             = np.inf,
            resolution                    = 0,
            hard_iron_distortion          = 0,
            noise_density                 = 0,
            noise_variance                = 1,
            random_walk_density           = 0,
            random_walk_variance          = 1,
            constant_bias                 = 0,
            operating_temperature         = 298,
            temperature_bias              = 0,
            temperature_scale_factor      = 0,
            cross_axis_sensitivity        = 0,
            name                          = 'Magnetometer',
        ):


        '''
        Initialize the magnetometer sensor

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
        
        # define hard_iron_distortion attribute
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
        

        # Get current decimal year
        current_date = datetime.now().strftime('%Y-%m-%d')
        self.year = decimal_year(current_date)

        # Initialize the magnetic model
        self.wmm = WMMv2()


        # Initialize InertialSensor class
        super().__init__(
            sampling_rate             = sampling_rate,
            orientation               = orientation,
            measurement_range         = measurement_range,
            resolution                = resolution,
            noise_density             = noise_density,
            noise_variance            = noise_variance,
            random_walk_density       = random_walk_density,
            random_walk_variance      = random_walk_variance,
            constant_bias             = constant_bias,
            operating_temperature     = operating_temperature,
            temperature_bias          = temperature_bias,
            temperature_scale_factor  = temperature_scale_factor,
            cross_axis_sensitivity    = cross_axis_sensitivity,
            name                      = name,
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
            - u_dot : np.array
                Derivative of the state vector of the rocket.
            - rocket: Rocket
                rocket with which the simulation is being executed.  
            - relative_position : np.array
                Position of the sensor relative to the rocket center of mass.
            - environment : Environment
                Environment object containing the atmospheric conditions.
        '''
            
        u = kwargs["u"] #state vector
        relative_position = kwargs["relative_position"] # it is already a Vector instance
        lat0, lon0, launch_site_elevation = kwargs["environment"].latitude, kwargs["environment"].longitude, kwargs["environment"].elevation
        earth_radius = kwargs["environment"].earth_radius
        avionics_signal = kwargs["avionics_signal"]
        rocket = kwargs["rocket"]
        
        quaternion = u[6:10]  # Quaternion represents the body orientation with respect to the inertial frame.
        rotation_body_to_inertial = Matrix.transformation(quaternion) # rotation matrix from rocket frame to inertial frame


        #--- obtain the current longitude, latitude and elevation ---

        # offset of the sensor in the inertial frame, from center of mass
        offset_sensor_inertial = rotation_body_to_inertial @ relative_position

        # obtain the sensor coordinates in the inertial frame, by adding the offset to the positon vector 
        x_inertial, y_inertial, z_inertial =  offset_sensor_inertial + Vector(u[0:3])
            
        # z is calculated in meters above the sea level, we must change to WGS84 in km
        altitude_wgs84_m = z_inertial + launch_site_elevation
        altitude_wgs84_km = altitude_wgs84_m / 1000.0

        # Convert x and y to current latitude and longitude
        drift = math.hypot(x_inertial, y_inertial)
        bearing = (2 * math.pi - math.atan2(-x_inertial, y_inertial)) * (180 / math.pi)
        latitude, longitude = inverted_haversine(lat0, lon0, drift, bearing, earth_radius)



        #--- obtain the magnetic field in the NED (North-East-Down axis) --

        # Calculate all field components at once (this the most efficient way)
        calculate_geomagnetic(self.wmm, latitude, longitude, self.year, altitude_wgs84_km)

        # components of the magnetic field
        b_north    = self.wmm.bx / 1e9   # T
        b_east     = self.wmm.by / 1e9   # T
        b_down     = self.wmm.bz / 1e9   # T

        #--- Transform to Rocketpy's inertial frame ---
        b_inertial_x = b_east       # T
        b_inertial_y = b_north      # T
        b_inertial_z = - b_down     # T
        B_inertial   = Vector([b_inertial_x, b_inertial_y, b_inertial_z])  # T
     

        # --- from Rocketpy's inertial frame to body frame ---
        rotation_inertial_to_body = rotation_body_to_inertial.transpose 
        
        B_body = rotation_inertial_to_body @ B_inertial # T

        #--- Transform body frame to sensor frame ---
        rotation_body_to_sensor = self._total_rotation_sensor_to_body.transpose

        B_sensor = rotation_body_to_sensor @ B_body  # T


        #--- Apply noise + bias and quantize ---
        B_sensor = self.apply_temperature_drift(B_sensor)              # T
        B_sensor = self.apply_hard_iron(B_sensor)                      # T
        B_sensor = self.power_interference(B_sensor, rocket, u)        # T
        B_sensor = self.apply_noise(B_sensor)                          # T
        B_sensor = self.quantize(B_sensor)                             # T


        self.measurement = (B_sensor.x, B_sensor.y, B_sensor.z)   # T                                  
        self._save_data((time, *B_sensor))        
    


    

    def power_interference(self, B: Vector, rocket: Rocket, u: list):

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
        u: list
            state vector of the rocket
            u = [x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]

        
        Returns:
        -------
        B: Vector
            Magnetic field after adjustment of both the 
            activation signal interference and communications
            interference
    

        '''

        B = B + self.standard_communications_interference(B, rocket)
        B = B + self.activation_signal_interference(B, rocket, u)

        return B
    

    def standard_communications_interference(self, B: Vector, rocket: Rocket):
        '''
        This function applies the interference caused due to the current
        flowing through the communication wires. 

         Input:
        --------
        B: Vector
            magnetic field Vector
        rocket: Rocket
            Rocketpy Rocket class
        '''

        for communication_wire in rocket.communication_wires:
            B = B + communication_wire._mangetic_interference

        return B



    def activation_signal_interference(self, B: Vector, rocket: Rocket, u: list):

        '''
        This function applies the interference caused due to the current
        flowing through the ignition wires, during an activation signal

         Input:
        --------
        B: Vector
            magnetic field Vector
        rocket: Rocket
            Rocketpy Rocket class
        u: list
            state vector of the rocket
            u = [x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]

        Returns: 
        -------
        B: Vector
            Magnetic field after adjustment activation signal interference
            interference
        '''

        for ingition_wire in rocket.ignition_wires:

            wire = ingition_wire[0]
            parachute_trigger = ingition_wire[1]

            if parachute_trigger == 'apogee' and u[5] < 0:
                    B = B + wire._magnetic_interference
            elif isinstance(parachute_trigger, (float, int)) and parachute_trigger >= u[2]:
                    B = B + wire._magnetic_interference
        return B
    

    def apply_hard_iron(self, B):

        '''
        This funtion applies the hard iron distortion. 
        This magnetic distortion is caused by permanent magnets or 
        magnetized materials on the rocket itself that move along with 
        the sensor (from steel screws, battery casing, feerromagnetic components), 
        thus it is a constant value. It shifts the center of the magnetic data
        '''

        B = B + self._hard_iron_distortion
        
        return B
    


    def export_measured_data(self, filename, file_format):
        '''Export the measured values to a file

        Parameters
        ----------
        filename : str
            Name of the file to export the values to
        file_format : str
            Format of the file to export the values to. Options are "csv" and
            "json". Default is "csv".

        Returns
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
        '''

        return cls(
            # Mandatory Parameter 
            sampling_rate                 = data['sampling_rate'],
            
            # Optional Parameters 
            orientation                   = data.get('orientation', (0, 0, 0)),
            measurement_range             = data.get('measurement_range', np.inf),
            resolution                    = data.get('resolution', 0),
            hard_iron_distortion          = data.get('hard_iron_distortion', 0.0),
          
            # Noise Profiles
            noise_density                 = data.get('noise_density', 0),
            noise_variance                = data.get('noise_variance', 1),
            random_walk_density           = data.get('random_walk_density', 0),
            random_walk_variance          = data.get('random_walk_variance', 1),
            constant_bias                 = data.get('constant_bias', 0),
            
            # Environmental & Structural Shifts
            operating_temperature         = data.get('operating_temperature', 298),
            temperature_bias              = data.get('temperature_bias', 0),
            temperature_scale_factor      = data.get('temperature_scale_factor', 0),
            cross_axis_sensitivity        = data.get('cross_axis_sensitivity', 0),
        )
            