import math
import numpy as np
from rocketpy.mathutils.vector_matrix import Matrix, Vector




class Wire():

    '''
    Wire class, that the physics of a wire, and the magnetic field it can create.
    It is used to model the magnetic field in the magnetometer. 

    
    Attributes: 
    -----------------
    wire_current: float, int
        Intensity of the current through the wire in A

    wire_current_direction: string
        Direction of the current through the wire, it can either be 
        clockwise or anticlockwise

    magnetic_field:
        magnetic field due to the charge flow through the wire.

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
            current,
            current_direction,
            type,
            ignition_wire_function = 'parachute',
            parachute_name = 'main', 
    ):
        
        '''
        current: float, int, list, optional
            Intensity of the current through the communication wires to calculate 
            the magnetic distortion experienced by the sensor due to activation signals
            in Amperes (A). Default is 1 A. 

        current_direction: string
            Direction of the current passing the communication wires to calculate the
            magnetic distortion experienced by the sensor due to activation signals 
            in Amperes (A). Default is anticlockwise
        
        type: str
            type of wire.
            If 'communications', the wire will be consider to
            have only information communicated from one component to 
            another, thus they will have a constatn effect on 
            the magnetic field. If 'ignition', the wire is considered
            to have flow of current only when there is a ignition.

        ignition_wire_function: str
            type of ingnition wire. 


        parachute_name: str, mandatory when it is a ignition wire whose function is parachute
            ,otherwise None. 

            Name of the parachtue in whose deployment we want the wire to have
            charge flow, it must be the same as the name assigned for the parachute
            The magnetic disturbance will ocurr during the selected parachute 
            ejection, simulating the signal sent by the avionics. The ejection 
            conditions will be taken from the parachute definition.

        '''
            

        # handling of direction of the current
        if not isinstance(self.current_direction, str):
            raise ValueError('The current direction parameter must be a string')
        elif  not (current_direction == 'anticlockwise' or current_direction == 'clockwise'): 
            raise ValueError('The accepted strings are anticlockwise and clockwise')
        else: 
            self.current_direction = current_direction
       

        # define current of the wire
        if not isinstance(current, (float, int)):
            raise ValueError('The current through the wire must be a float or int') 
        else: 
            self.current = current


        # define the type of wire
        if isinstance(type, str):
            if type == 'communications':
                self.type = 'communications'
            elif type == 'ignition':
                self.type == 'ignition'
                if isinstance(ignition_wire_function,str):
                    if ignition_wire_function.lower() == 'parachute':
                        if not isinstance(self.parachute_name, str):
                            raise ValueError('The name of the parachute must be a string')
                        else: 
                            self.parachute_name = parachute_name
                    else: 
                        raise ValueError(f'There is not ignition type {ignition_wire_function}', ignition_wire_function)
                else: 
                    raise ValueError('The type of ignition wire must be a str')
            else:
                raise ValueError('The type must be a ignition or communication')
            
        else: 
            raise ValueError('The type must be a string')


        self._magnetic_field = {}

    def measure_magnetic_field(
        self,
        position_vector, 
        magnetic_field = 'physical',
        ):


        '''
        magnetic_field: float, list, str, optional
            Magnetic influence on a certain position due to activation signal in T: 

            - If a float, it assumes that the wire genertes the same magnetic field
              on each axis, with the given value

            - If a tuple or float, it assumes that the wire genertes the given magnetic
              field

            - If str: 'physical' the magnetic field on the point position_vector is 
            calculated using physical parameters, assuming that the magnetometer and wire
            are in the same plane
            
            Default is 'physical', meaning, it is calculated using the physical parameters of the rocket

        '''


        # define mangetic field:

        if isinstance(magnetic_field, (float, int)) and magnetic_field != 0:

            self.magnetic_field = [magnetic_field, magnetic_field, magnetic_field]

        elif  isinstance(magnetic_field, (list, tuple)):

            if len(magnetic_field) == 3: 
                self.magnetic_field = list(magnetic_field)
            else:
                raise ValueError('If a list is passed, it must have a value for each axis. Therefore, it must have length 3')



        elif isinstance(magnetic_field, str) and (magnetic_field == 'physical' or magnetic_field == 'angles'): 

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

            
            
            if isinstance(self.wire_current_direction, str) and self.wire_current_direction == 'anticlockwise':
                self._wire_angles_magnetometer = [self._wire_angles_magnetometer[1], self._wire_angles_magnetometer[0]]


            b_field_wire = ((1e-7 * self.current) / wire_distance_to_magnetometer) * (math.cos(self._wire_angles_magnetometer[0]) - math.cos(self._wire_angles_magnetometer[1]))

            self.magnetic_field = [0, 0, b_field_wire]
            
            
        elif magnetic_field == 0:
            self.magnetic_field = [0,0,0]

            self.wire_current = None
            self.wire_distance = None
            self.wire_length = None
            self.wire_angles = None
        else:
            raise ValueError('Introduce a correct value for the activation signal')



        self._magnetic_field[position_vector] = Vector(self.magnetic_field) 
