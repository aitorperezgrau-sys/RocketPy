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

    magnetic_field: dict
        Dictionary of all the magnetic fields calculated, 
        the key is the position vector of the point relative
        to the cso, and the value is the magnetic field due to
        the wire at that point as a list in T. 

    wire_length: float
        length of the wire in m. 

    wire_type: str
        type of wire, it can either be ignition or communications

    ignition_wire_function: str
        sub-type of the wire, when it is an ignition wire
        because it is a HIL wire it can be:
        'parachute' or 'solid_motor_ignition
    
    parachute_name: str
        Name of the parachute to which it is attached the wire
        in the case ignition_wire_function is parachute
    
    '''
    def __init__(
            self,
            current,
            wire_type,
            ignition_wire_function = None,
            parachute_name = None, 
    ):
        '''
        current: float, int, list
            Intensity of the current through the communication wires to calculate 
            the magnetic distortion experienced by the sensor due to activation signals
            in Amperes (A). Default is 1 A. The current goes from the first to the second
            edge defined in add_wire
        
        wire_type: str
            type of wire.
            If 'communications', the wire will be consider to
            have only information communicated from one component to 
            another, thus they will have a constatn effect on 
            the magnetic field. If 'ignition', the wire is considered
            to have flow of current only when there is a ignition.

        ignition_wire_function: str, mandatory if type is ingition
            type of ingnition wire. This parameter must be a string, for a solid rocket
            the only ignitions are the parachutes and the motor at the beggining 
            of the flight. In this case, the valid arguments are 'solid_motor_ignition'
            or 'parachute_ignitions'. Default is None. 

        parachute_name: str, mandatory when it is a ignition wire whose function is parachute 

            In the case ignition_wire_function is parachute: 
                Name of the parachtue in whose deployment we want the wire to have
                charge flow, it must be the same as the name assigned for the parachute
                The magnetic disturbance will ocurr during the selected parachute 
                ejection, simulating the signal sent by the avionics. The ejection 
                conditions will be taken from the parachute definition.

        '''
       
        # define current of the wire
        if not isinstance(current, (float, int)):
            raise ValueError('The current through the wire must be a float or int') 
        else: 
            self.current = current

        # definition of the type of wire
        if isinstance(wire_type, str):
            if wire_type == 'communications':
                self.wire_type = 'communications'
                self.ignition_wire_function = None                
            elif wire_type == 'ignition':
                self.wire_type = 'ignition'
                if ignition_wire_function == None:
                    raise ValueError('The ignition type is compulsory when it is a ignition wire')
                elif isinstance(ignition_wire_function, str):
                    if ignition_wire_function.lower() == 'parachute_ignition':
                        self.ignition_wire_function = 'parachute_ignition'
                        if parachute_name == None:
                            raise ValueError('The name of the parachute is compulsory if the ignition_wire_function is parachute')
                        elif not isinstance(parachute_name, str):
                            raise ValueError('The name of the parachute must be a string')
                        else: 
                            self.parachute_name = parachute_name
                    else: 
                        self.ignition_wire_function = ignition_wire_function
                else: 
                    raise ValueError('The type of ignition wire must be a str')
            else:
                raise ValueError('The type must be a ignition or communication')
        else: 
            raise ValueError('The type must be a string')

        self._magnetic_field = {}
        self.magnetic_field = {}
        self._wire_edges_from_cso = None
        self.wire_length = 0
    

    def measure_magnetic_field(
        self,
        position_vector, 
        ):
        '''
        This function measures the magnetic field on a given position_vector
        based on the position of the edges of the wire. the magnetic field is
        calculated assuming that the wire is straight. 

        input:
        --------

        position_vector: list, tuple or Vector
            position vector of the point in which the magnetic field
            is going to be measured relative to the coordiante system origin choosen
            by the user. 

        Returns:
        ---------
            None
        '''
        # definition of the required values
        r1 = self._wire_edges_from_cso[0]  # m
        r2 = self._wire_edges_from_cso[1]  # m

        if len(position_vector) == 3:
            if isinstance(position_vector, (list, tuple)):
                r_V = Vector(position_vector)  # m
                r_t   = tuple(position_vector)    # m 
            elif isinstance(position_vector, Vector):
                r_V = position_vector          # m
                r_t   = tuple(position_vector) # m
            else: 
                raise ValueError('The only accepted parameters are list, tuple or Vector')
        else:
            raise ValueError('The length of the position vector must be 3, x,y,z')    
        
        l = r2 - r1 #m
        self.wire_length = abs(l) # m

        r1_V =  r_V - r1 # m
        r2_V =  r_V - r2 # m

        cross_r1_r2 = (r1_V ^ r2_V)
        cross_norm_r1_r2 = abs(cross_r1_r2)

        dot_term = l @ (r1_V.unit_vector - r2_V.unit_vector)

        if cross_norm_r1_r2 < 1e-12: #along the same line, cross product is zero -> magnetic field is 0
            b_V = Vector([0,0,0]) 
        else:
            b_V = (1e-7 * self.current) * (cross_r1_r2 / cross_norm_r1_r2) * dot_term # T
       
        self.magnetic_field[r_t]  = list(b_V)
        self._magnetic_field[r_t] = b_V

    
    def define_magnetic_field(self, position_vector, magnetic_field):
        '''
        This function allows to defined the magnetic field at a certain point
        and this will the value used for the calculations. 

        inputs:
        -----------
        position_vector: list, tuple
            position vector of the point where the magnetic field is
            defined

        magnetic_field: int, float, list, tuple
            Magnetic influence on the position given by position_vector in T: 

            - If a float, it assumes that the wire genertes the same magnetic field
              on each axis, with the given value. 

            - If a tuple or float, it assumes that the wire genertes the given magnetic
              field. 

        Returns:
        ---------
            None
        '''
        if isinstance(position_vector, (tuple, list, Vector)):
            if len(position_vector) == 3:
                position_vector_t = tuple(position_vector)
                if isinstance(magnetic_field, (float, int)):
                    self._magnetic_field[position_vector_t] = [magnetic_field, magnetic_field, magnetic_field]
                elif isinstance(magnetic_field, (list, tuple)):
                    if len(magnetic_field) == 3: 
                        self._magnetic_field[position_vector_t] = list(magnetic_field)
                    else:
                        raise Exception('If a list is passed, it must have a value for each axis. Therefore, it must have length 3')
            else:
                raise ValueError('The position_vector must be a list, tuple or Vector with 3 elements')
        else:
            raise ValueError('The position_vector must be a list, tuple or Vector')
        

    def _set_wire_edges_from_cso (self, _wire_edges_from_cso):
        '''
        save as an attribute the position of the wire edges from the coordiante system
        origin, chosen by the user.
        
        input:
        ------------------
        wire_edges_from_cso: list, tuple, Vector formed by Vectors 
            containing the edges position relative to the the coordiante system
            origin. 

        Returns:
        -------
            None

        '''

        self._wire_edges_from_cso = _wire_edges_from_cso

    @classmethod
    def from_dict(cls, data: dict):
        '''
        Creates an instance of Wire from a dictionary object, data. 
        Data is a dictionary that must contain the same keys as the initialization
        parameter of the Wire class. In the case some parameter is not 
        defined, the default value matches the default intializaiton of the constructor

        Returns:
        ---------
            Wire object
        '''
        return cls(
            # Mandatory Parameters 
            current                      = data['current'],
            current_direction            = data['current_direction'],
            wire_type                    = data['wire_type'],
            
            # Optional Parameters 
            ignition_wire_function       = data.get('ignition_wire_function', None),
            parachute_name               = data.get('parachute_name', None)
        )