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

    '''

    def __init__(
            self,
            current,
            current_direction,
            type,
            ignition_wire_function = None,
            parachute_name = None, 
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

        ignition_wire_function: str, mandatory if type is ingition
            type of ingnition wire. This parameter must be a string, for a solid rocket
            the only ignitions are the parachutes and the motor at the beggining 
            of the flight. In this case, the valid arguments are 'solid_motor' or 'parachutes'.
            Default is None. 

        parachute_name: str, mandatory when it is a ignition wire whose function is parachute 

            In the case ignition_wire_function is parachute: 
                Name of the parachtue in whose deployment we want the wire to have
                charge flow, it must be the same as the name assigned for the parachute
                The magnetic disturbance will ocurr during the selected parachute 
                ejection, simulating the signal sent by the avionics. The ejection 
                conditions will be taken from the parachute definition.

        '''
        

        # handling of direction of the current
        if not isinstance(current_direction, str):
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






        # definition of the type of wire
        if isinstance(type, str):

            if type == 'communications':
                self.type = 'communications'
                self.ignition_wire_function = None                

            elif type == 'ignition':
                self.type == 'ignition'

                if ignition_wire_function == None:
                    raise ValueError('The ignition type is compulsory when it is a ignition wire')

                elif isinstance(ignition_wire_function, str):

                    if ignition_wire_function.lower() == 'parachute':

                        self.ignition_wire_function = 'parachute'

                        if parachute_name == None:
                            raise ValueError('The name of the parachute is compulsory if the ignition_wire_function is parachute')
                        
                        elif not isinstance(self.parachute_name, str):
                            raise ValueError('The name of the parachute must be a string')
                        
                        else: 
                            self.parachute_name = parachute_name

                    elif ignition_wire_function.lower() == 'solid_motor':
                        self.ignition_wire_function = 'solid_motor'

                    else: 
                        raise ValueError(f'There is not ignition type {ignition_wire_function}', ignition_wire_function)
                    
                else: 
                    raise ValueError('The type of ignition wire must be a str')
                
            else:
                raise ValueError('The type must be a ignition or communication')
            
        else: 
            raise ValueError('The type must be a string')



        self._magnetic_field = {}
        self._position_edges = Vector([])




    def measure_magnetic_field(
        self,
        position_vector, 
        ):

        '''
        This function measures the magnetic field on a given position_vector
        based on the position of the edges of the wire. the magnetic field is calculated assuming 
        that the wire is straight. 
        '''

        







    
    def define_magnetic_field(self, position_vector, magnetic_field):
        '''
        This function allows to defined the magnetic field at a certain point
        and this will the value used for the calculations. 

        inputs:
        -----------
        position_vector:
            position vector of the point where the magnetic field is
            defined

        magnetic_field: int, float, list, tuple
            Magnetic influence on the position given by position_vector in T: 

            - If a float, it assumes that the wire genertes the same magnetic field
              on each axis, with the given value. 

            - If a tuple or float, it assumes that the wire genertes the given magnetic
              field. 
        '''
        

        if isinstance(magnetic_field, (float, int)):

            self._magnetic_field[position_vector] = [magnetic_field, magnetic_field, magnetic_field]

        elif isinstance(magnetic_field, (list, tuple)):

            if len(magnetic_field) == 3: 
                self._magnetic_field[position_vector] = list(magnetic_field)
            else:
                raise Exception('If a list is passed, it must have a value for each axis. Therefore, it must have length 3')
        else:
            raise ValueError('Hard iron must be a float, int or a list with 3 elements')
        


