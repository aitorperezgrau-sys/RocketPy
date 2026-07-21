from rocketpy.rocket import Rocket
from rocketpy.rocket.aero_surface import NoseCone
from rocketpy.mathutils import Vector, Matrix
import math as m
import numpy as np

class Plate():
    '''
    This class allows to define surfaces on the rocket.
    It is used to account for the soft iron distortion
    on the rocket that affect the magnetometer reading.

    Attributes:
    --------------
    material: str
        Material from which the plate is composed
        Allowed strings are 'iron', 'carbon steel', or 
        'personalized' if we want to define the material 
        based on the magnetic permeability. 

    magnetic_permeability: 
        Magnetic permeability of the material, 

    magnetic_distortion: 
        Dictionary formed by the magnetic distortion
        matrix caused by the plate. The keys are the position
        vector of the point relative to the cso, and the value 
        is the magnetic distortion Matrix. 

    '''
    

    def __init__(
            self,
            material,
            magnetic_permeability = None
    ):
    
        '''

        Parameters:
        --------------

        material: str
            Material from which the plate is composed
            Allowed strings are 'iron', 'carbon steel', or 
            'personalized' if we want to define the material 
            based on the magnetic permeability. 

        magnetic_permeability: float, int, optional
            Magnetic permeability of the material, which is 
            the measure of a material abilty to allow magnetic
            field lines to pass through it. 
        '''

        self._magnetic_distortion_matrixes = {}
        self._vertixes = []



    

    def define_plate_position(self, shape, dimensions, position, height, rocket):
        '''
        This function defines the geometry of the plate
        with respect to the cso from the shape, position,
        dimensions and height defined in the add_plate()
        rocket class method. 

        Input:
        ------------
        shape: str
            The shape of the plate, allowed parameters are:

            'circular': then the plate is assumed to be 
            a circle, and the input 'dimension' refers to
            the radius
            'squared': then the plate is assumed to be a 
            square and the input 'dimensions' refers to the 
            side 
            'personalized': then the plate will have the shape 
            specified by the vertexes defined in 'dimensions'

        dimensions: float, int, list
            Dimensions of the plate, which depend on 'shape' 
            definition:

            When it is 'circular', the dimension is a float or int,
            which represents the radius, when the shape is flat. 

            when it is 'squared', the dimension is a float or int,
            which represents the side lenght, when the shape is flat.

            when it is 'personalized', dimensions must be a list
            with the vertixes that form the shape. 

        position: str, optional
            position of the plate, when the shape is not 'personalized'
            Allowed entries are:
            'left', 'right', 'back', 'front'
            The plate will be located with the geometric center
            along the chosen lateral position

        height: float, int, optional
            Position of the geometric center of plate when the shape is not 
            'personalized' along the z axis. 

        rocket: Rocket
            RocketPy class.
        '''

        # definition of the position of the NoseCone relative to the cso
        for aerodynamic_surface, _position_relative_to_cso in rocket.aerodynamic_surfaces:
            if  isinstance(aerodynamic_surface, NoseCone):
                self.nose_cone = aerodynamic_surface
                limiting_z_nose_cone = _position_relative_to_cso[2] - self.nose_cone.length
            else: 
                raise ValueError('To determine the plates position, first it must be added a NoseCone to the rocket')
            
        if shape == 'squared': 
            
            if (height + dimensions / 2) < limiting_z_nose_cone: 
                upper_z = height - dimensions / 2
                lower_z = height - dimensions / 2

                angle = dimensions / rocket.radius   # rad
                lateral_abs = m.sin(angle / 2) * rocket.radius
                central_abs = m.cos(angle / 2) * rocket.radius

                if position == 'rigth':
                    self._vertixes = [
                        Vector(central_abs, lateral_abs, upper_z),
                        Vector(central_abs, - lateral_abs, upper_z),
                        Vector(central_abs, lateral_abs, lower_z),
                        Vector(central_abs, - lateral_abs, lower_z)
                        ]
                    
                elif position == 'front':
                    self._vertixes = [
                        Vector(lateral_abs, central_abs, upper_z),
                        Vector( - lateral_abs, central_abs, upper_z),
                        Vector(lateral_abs, central_abs, lower_z),
                        Vector( - lateral_abs, central_abs, lower_z)
                        ]
                    
                elif position == 'left':
                    self._vertixes = [
                        Vector( - central_abs, lateral_abs, upper_z),
                        Vector( - central_abs, - lateral_abs, upper_z),
                        Vector( - central_abs, lateral_abs, lower_z),
                        Vector( - central_abs, - lateral_abs, lower_z)
                        ]
                    
                else: 
                    self._vertixes = [
                        Vector(lateral_abs, - central_abs, upper_z),
                        Vector( - lateral_abs, - central_abs, upper_z),
                        Vector(lateral_abs, - central_abs, lower_z),
                        Vector( - lateral_abs, - central_abs, lower_z)
                        ]
            else: 
         

        elif shape == 'circular': 
            if height + dimensions  < limiting_z_nose_cone: 
                total_angle = 2 * dimensions / rocket.radius
                z_offset_abs = m.cos(np.pi / 3) * (dimensions / 2)
                upper_z = height + z_offset_abs
                midd_z = height
                lower_z = height - z_offset_abs
                total_lateral_abs = m.sin(total_angle/ 2) * rocket.radius
                total_central_abs = m.cos(total_angle / 2) * rocket.radius
                inner_angle = dimensions / 2 / rocket.radius
                inner_lateral_abs = m.sin(inner_angle/ 2) * rocket.radius
                inner_central_abs = m.cos(inner_angle / 2) * rocket.radius

                if position == 'rigth':
                    self._vertixes = [
                        Vector(total_central_abs, total_lateral_abs, midd_z),
                        Vector(total_central_abs, - total_lateral_abs, midd_z),
                        Vector(inner_central_abs, inner_lateral_abs, upper_z),
                        Vector(inner_central_abs, - inner_lateral_abs, upper_z),
                        Vector(inner_central_abs, inner_lateral_abs, lower_z),
                        Vector(inner_central_abs, - inner_lateral_abs, lower_z)
                        ]
                    
                elif position == 'front':
                    self._vertixes = [
                        Vector(total_lateral_abs, total_central_abs, midd_z),
                        Vector(- total_lateral_abs, total_central_abs, midd_z),
                        Vector(inner_lateral_abs, inner_central_abs, upper_z),
                        Vector( - inner_lateral_abs, inner_central_abs, upper_z),
                        Vector(inner_lateral_abs, inner_central_abs, lower_z),
                        Vector( - inner_lateral_abs, inner_central_abs, lower_z)
                        ]
                    
                elif position == 'left':
                    self._vertixes = [
                        Vector( - total_central_abs, total_lateral_abs, midd_z),
                        Vector( - total_central_abs, - total_lateral_abs, midd_z),
                        Vector( - inner_central_abs, inner_lateral_abs, upper_z),
                        Vector( - inner_central_abs, - inner_lateral_abs, upper_z),
                        Vector( - inner_central_abs, inner_lateral_abs, lower_z),
                        Vector( - inner_central_abs, - inner_lateral_abs, lower_z)
                        ]
                    
                else: 
                    self._vertixes = [
                        Vector(total_lateral_abs, - total_central_abs, midd_z),
                        Vector(- total_lateral_abs, - total_central_abs, midd_z),
                        Vector(inner_lateral_abs, - inner_central_abs, upper_z),
                        Vector( - inner_lateral_abs, - inner_central_abs, upper_z),
                        Vector(inner_lateral_abs, - inner_central_abs, lower_z),
                        Vector( - inner_lateral_abs, - inner_central_abs, lower_z)
                        ]
            else:
                


        else:
            self._vertixes = dimensions




        
    



    def calculate_soft_iron_distortion_matrix(self, position_vector):

        '''
        This function allows to calculate the soft iron
        distortion matrix from the position of the points
        and the parameters defined for the surface
        '''

