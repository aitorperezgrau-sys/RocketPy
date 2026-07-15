import math
import numpy as np
from rocketpy.mathutils.vector_matrix import Matrix, Vector




class Wire():

    '''
    Wire class, that the physics of a wire, and the magnetic field it can create.
    It is used to model the magnetic interference in the magnetometer. 

    
    Attributes: 
    -----------------
    wire_current: float
        Intensity of the current through the wire in A

    wire_current_direction: string
        Direction of the current through the wire, it can either be 
        clockwise or anticlockwise

    magnetic_interference:
        distortion of the magnetic field due to the charge flow 
        through the wire.

    wire_length: float, optional
        length of the wire in m. 

    wire_distance_to_magnetometer: float, optional
        Distance of the activation wires to the magnetic sensor in m.

    wire_angles_magnetometer: list, optional
        angle between the edge of the wires and the magnetometer in degrees.
        The angles are defined between the wire and the line joining the edge
        of the wire and the magnetometer

    '''

    def __init__(
            self,
            wire_current,
            wire_current_direction,
            magnetic_interference = 'physical',
            wire_angles_magnetometer = 'physical',
            wire_distance_to_magnetometer = 1e-2,
            wire_length = 8e-2,
    ):
        
        '''
        wire_current: float, int, list, optional
            Intensity of the current through the communication wires to calculate 
            the magnetic distortion experienced by the sensor due to activation signals
            in Amperes (A). Default is 1 A. 

        wire_current_direction: string
            Direction of the current passing the communication wires to calculate the
            magnetic distortion experienced by the sensor due to activation signals 
            in Amperes (A). Default is anticlockwise

        magnetic_interference: float, list, str, optional
            Magnetic influence on the magnetometer due to activation signal in T: 

            - If a float, in T the same value is applied to each axis 

            - If a list or float, in T the distortion will be taken considering 
              these values. 

            - If str: 'physical' the activation_signal_distortion can be modelled 
              using the followign arguments, wire_current, wire_current_direction 
              wire_distance_to_magnetometer, wire_length, assuming that the magnetometer is placed 
              in the middle of the wire, and that magnetometer and wire are in the 
              same plane. 

            - If str: 'angles', the activation_signal_distortion can be modelled 
              using the following arguments, wire_current, wire_current_direction
              wire_distance, wire_length, assuming that the magnetometer and wire
              are in the same plane

            - If 0, there is no magnetic distortion due to activation signal
            
            Default is 'physical', meaning, it can be defined with the physical
            parameters. 

        wire_angles_magnetometer: tuple, list, float, int, str, optional
            Angles between the edge of the wires and the magnetometer in degrees. 
            The angles are defined between the wire and the line joining the edge
            of the wire and the magnetometer. This parameter is necesary if the 
            activation_signal_distortion is 'angles'. The angles must be between 0 and 90,
            without including them

            If a float, the same angle applies for the left and rigth edges
            If a list or tuple, the first angle is considered to be the one to the
            left from the sensor perspective, and the second to the left from the 
            sensor perspective
            If a str: 'physical' indicates that the activation_signal_distortion 
            is in the physical mode. 

            Default is 'physical'. 

        wire_distance_to_magnetometer: float, int, optional  
            Distance from the wires to the magnetic sensor to to calculate the
            magnetic distortion experienced by the sensor due to activation signals. 
            If a float, the wires are assumed to be at the same distance of each
            magnetometer axis. 
            If a list, each value is the distance to the x,y,z axis of the magnetometer
            with a list of length 3.
            
            Default is 1e-2 m, 1 cm. 

        wire_length: float, optional, str
            Length of the wire to calculate the magnetic distortion experienced by
            the sensor due to activation signals.
            str: If the activation_signal_distortion is 'angles' this parameters must be 
            initiated with 'angles'. 

            Default is 8 * 1e-2 m, 8cm. 

        '''



        # define mangetic interference:

        if isinstance(magnetic_interference, (float, int)) and magnetic_interference != 0:

            self.magnetic_interference = [magnetic_interference, magnetic_interference, magnetic_interference]

        elif  isinstance(magnetic_interference, (list, tuple)):

            if len(magnetic_interference) == 3: 
                self.magnetic_interference = list(magnetic_interference)
            else:
                raise ValueError('If a list is passed, it must have a value for each axis. Therefore, it must have length 3')

        elif isinstance(magnetic_interference, str) and (magnetic_interference == 'physical' or magnetic_interference == 'angles'): 

            # define wire_current
            if not isinstance(wire_current, (float, int)):
                raise ValueError('The current through the wire must be a float or int') 
            else: 
                self.wire_current = wire_current

            # define wire_distance
            if not isinstance(wire_distance_to_magnetometer, (float, int)): 
                raise ValueError('The distance from the wire to the magnetometer must be a float or int') 
            elif wire_distance_to_magnetometer > 0:
                self.wire_distance_to_magnetometer = wire_distance_to_magnetometer 
            else:
                raise ValueError('The distance from the wire to the magnetometer must be greater than 0')

            if wire_length <= 0:
                raise ValueError('The length of the wire must be greater than 0')

            # definition of the angles
            if isinstance(wire_angles_magnetometer, (float, int)):
                if not 0 < wire_angles_magnetometer < 90: 
                    raise ValueError('The angle must be between 0 and 90 degrees excluded')
                
                self.wire_angles_magnetometer = [wire_angles_magnetometer, wire_angles_magnetometer] 
                self._wire_angles_magnetometer = [self.wire_angles[0] * (np.pi / 180), self.wire_angles[1] * (np.pi / 180) ]

            elif isinstance(wire_angles_magnetometer, (list, tuple)) and len(wire_angles_magnetometer) == 2:

                for i in wire_angles_magnetometer: 
                    if not 0 < i < 90: 
                        raise ValueError('The angle must be between 0 and 90 degrees excluded')
                    
                self.wire_angles_magnetometer = list(wire_angles_magnetometer) 
                self._wire_angles_magnetometer = [self.wire_angles[0] * (np.pi / 180), self.wire_angles[1] * (np.pi / 180) ]

            elif isinstance(wire_angles_magnetometer, str) and wire_angles_magnetometer == 'physical':
                
                angle = math.atan(self.wire_distance_to_magnetometer / (wire_length / 2))

                if angle <= 0:
                    raise ValueError('The angle must be between 0 and 90 excluded')
                
                self._wire_angles_magnetometer = [angle, angle]  
                self._wire_angles_magnetometer = [angle * (180 / np.pi), angle * (180 / np.pi)]

            else: 
                raise ValueError('Introduce a valid combination')


            # creation of wire_length attribute
            if isinstance(wire_length, str) and wire_length == 'angles':
                self.wire_length = (self.wire_distance_to_magnetometer / math.tan(self._wire_angles_magnetometer[0])) + (self.wire_distance / math.tan(self._wire_angles[1]))
            elif isinstance(wire_length, (float, int)):
                self.wire_length = wire_length
            else:
                raise ValueError('Introduce a valid wire length')

            

            # handling of direction of the current
            if isinstance(wire_current_direction, str) and not (wire_current_direction == 'anticlockwise' or wire_current_direction == 'clockwise'): 
                raise ValueError('The accepted strings are anticlockwise and clockwise')
            
            elif not isinstance(wire_current_direction, str):
                raise ValueError('The current direction parameter must be a string')
            
            elif isinstance(wire_current_direction, str) and wire_current_direction == 'anticlockwise':
                self._wire_angles_magnetometer = [self._wire_angles_magnetometer[1], self._wire_angles_magnetometer[0]]


            b_field_wire = ((1e-7 * wire_current) / wire_distance_to_magnetometer) * (math.cos(self._wire_angles_magnetometer[0]) - math.cos(self._wire_angles_magnetometer[1]))

            self.magnetic_interference = [0, 0, b_field_wire]
            
            
        elif magnetic_interference == 0:
            self.magnetic_interference = [0,0,0]

            self.wire_current = None
            self.wire_distance = None
            self.wire_length = None
            self.wire_angles = None
        else:
            raise ValueError('Introduce a correct value for the activation signal')

        self._magnetic_interference = Vector(self.magnetic_interference) 
