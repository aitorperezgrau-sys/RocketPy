
from rocketpy.rocket.aero_surface import NoseCone
from rocketpy.mathutils import Vector, Matrix
import math as m
import numpy as np
from rocketpy.tools import calculate_area_3D



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

    absolute_magnetic_permeability: float
        Magnetic permeability of the material,

    relative_magnetic_permeabiltiy: float
        ratio of the magnetic permeability to 
        the magnetic permeability of vacuum

    thickness: float, int
        Thickness of the plate

    area: float, int
        Area of the plate. When the parameter is 
        circular or squared, then the common expression
        is used, while with personalized, the area is
        the minimal area of the region contained within the points. 

    volume: float, int
        Volume of the plate. 

    _magnetic_distortion_matrixes: dict
        Dictionary formed by the magnetic distortion
        matrix caused by the plate. The keys are the position
        vector of the point relative to the cso given as a tuple,
        and the value is the magnetic distortion Matrix. 
    
    _vertices: list
        list formed by the Vectors representing the 
        components of each vertex of the surface relative
        to the cso. 

    '''
    

    def __init__(
            self,
            material,
            thickness, 
            absolute_magnetic_permeability = None
    ):
    
        '''

        Parameters:
        --------------

        material: str
            Material from which the plate is composed
            Allowed strings are 'iron', 'carbon_steel', or 
            'personalized' if we want to define the material 
            based on the magnetic permeability. 

        thickness: float, int
            Thickness of the plate in m. 

        magnetic_permeability: float, int, optional
            Magnetic permeability of the material, which is 
            the measure of a material abilty to allow magnetic
            field lines to pass through it. 
        '''

        self._magnetic_distortion_matrixes = {}
        self._vertices = []

        if isinstance(material, str):

            if material == 'iron':

                self.material = 'iron'
                self.absolute_magnetic_permeability = 1.25e-3
               
            elif material == 'carbon_steel':

                self.material = 'carbon_steel'
                self.absolute_magnetic_permeability = 1.2e-4

            elif material == 'personalized':

                self.material = 'personalized'

                if absolute_magnetic_permeability == None:

                    raise ValueError('The magnetic permeability is compulsory when personalized is chosen')
                
                elif not isinstance(absolute_magnetic_permeability, (int, float)):

                    raise ValueError('The magnetic permeability must be an int or float')
                
                else:
                    self.absolute_magnetic_permeability = absolute_magnetic_permeability

            else:
                raise ValueError('Material argument can only be iron, carbon_steel or personalized')

        else: 
            raise ValueError('material argument can only be a string')


        self.relative_magnetic_permeability = self.absolute_magnetic_permeability / (4 * np.pi * 1e-7)

        if not isinstance(thickness, (float, int)):
            raise ValueError('Thickness must be a float or int')
        else:
            self.thickness = thickness
    

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
            specified by the vertices defined in 'dimensions'

        dimensions: float, int, list
            Dimensions of the plate, which depend on 'shape' 
            definition:

            When it is 'circular', the dimension is a float or int,
            which represents the radius, when the shape is flat. 

            when it is 'squared', the dimension is a float or int,
            which represents the side lenght, when the shape is flat.

            when it is 'personalized', dimensions must be a list
            with the vertices that form the shape. The vertices
            components are given as a Vector. 

        position: str, optional
            position of the plate, when the shape is not 'personalized'
            Allowed entries are:
            'left', 'right', 'back', 'front'
            The plate will be located with the geometric center
            along the chosen lateral position, which is defined based
            on the coordinate system origin. 

        height: float, int, optional
            Position of the geometric center of plate when the shape is not 
            'personalized' along the z axis relative to the cso. 

        rocket: Rocket
            RocketPy class.
        '''
 
        # definition of the position of the NoseCone relative to the cso
        nose_cone = None

        for aerodynamic_surface, _position_relative_to_cso in rocket.aerodynamic_surfaces:

            if  isinstance(aerodynamic_surface, NoseCone):

                nose_cone = aerodynamic_surface
                nose_tip_from_cso = _position_relative_to_cso[2]
                limiting_z_nose_cone = nose_tip_from_cso - nose_cone.length
 
        if nose_cone == None:
            raise ValueError('To define the plate, first the nose cone must be added to the rocket.')


        if shape == 'squared': 
            
            self.area = dimensions * dimensions
            self.volume = self.area * self.thickness

            upper_z = height + dimensions / 2.0
            lower_z = height - dimensions / 2.0


            # upper radius
            if upper_z > limiting_z_nose_cone:
                z_local = abs(nose_tip_from_cso - (height + upper_z))
                r_upper = nose_cone.radius(z_local)
            else: 
                r_upper = rocket.radius

            # lower radius
            if lower_z > limiting_z_nose_cone:
                z_local = abs(nose_tip_from_cso - (height + lower_z))
                r_lower = nose_cone.radius(z_local)
            else: 
                r_lower = rocket.radius
            

            # upper part 
            angle_upper = dimensions / r_upper        # rad
            lateral_upper = m.sin(angle_upper / 2) * r_upper
            central_upper = m.cos(angle_upper / 2) * r_upper

            # lower part 
            angle_l = dimensions / r_lower        # rad
            lateral_l = m.sin(angle_l / 2) * r_lower
            central_l = m.cos(angle_l / 2) * r_lower

            match position:

                case 'right':
                    self._vertices = [
                        Vector([central_upper, lateral_upper, upper_z]),
                        Vector([central_upper, -lateral_upper, upper_z]),
                        Vector([central_l, lateral_l, lower_z]),
                        Vector([central_l, -lateral_l, lower_z]),
                    ]

                case 'front':
                    self._vertices = [
                        Vector([lateral_upper, central_upper, upper_z]),
                        Vector([-lateral_upper, central_upper, upper_z]),
                        Vector([lateral_l, central_l, lower_z]),
                        Vector([-lateral_l, central_l, lower_z]),
                    ]

                case 'left':
                    self._vertices = [
                        Vector([-central_upper, lateral_upper, upper_z]),
                        Vector([-central_upper, -lateral_upper, upper_z]),
                        Vector([-central_l, lateral_l, lower_z]),
                        Vector([-central_l, -lateral_l, lower_z]),
                    ]

                case 'back':
                    self._vertices = [
                        Vector([lateral_upper, -central_upper, upper_z]),
                        Vector([-lateral_upper, -central_upper, upper_z]),
                        Vector([lateral_l, -central_l, lower_z]),
                        Vector([-lateral_l, -central_l, lower_z]),
                    ]
            
         

        elif shape == 'circular': 

            self.area = np.pi * (dimensions ** 2)
            self.volume = self.area * self.thickness

            z_offset_abs = m.cos(np.pi / 3) * (dimensions / 2)
            upper_z = height + z_offset_abs
            mid_z = height
            lower_z = height - z_offset_abs

            # upper radius
            if upper_z > limiting_z_nose_cone:
                z_local = abs(nose_tip_from_cso - (height + upper_z))
                r_upper = nose_cone.radius(z_local)
            else: 
                r_upper = rocket.radius

            # mid radius
            if mid_z > limiting_z_nose_cone:
                z_local = abs(nose_tip_from_cso - (height + mid_z))
                r_mid = nose_cone.radius(z_local)
            else: 
                r_mid = rocket.radius

            # lower radius
            if lower_z > limiting_z_nose_cone:
                z_local = abs(nose_tip_from_cso - (height + lower_z))
                r_lower = nose_cone.radius(z_local)
            else: 
                r_lower = rocket.radius

            # Mid-level calculations
            total_angle = 2 * dimensions / r_mid
            total_lateral = m.sin(total_angle / 2) * r_mid
            total_central = m.cos(total_angle / 2) * r_mid

            # Upper and Lower inner angle calculations
            inner_angle_upper = (dimensions / 2) / r_upper
            inner_lateral_upper = m.sin(inner_angle_upper / 2) * r_upper
            inner_central_upper = m.cos(inner_angle_upper / 2) * r_upper

            inner_angle_lower = (dimensions / 2) / r_lower
            inner_lateral_lower = m.sin(inner_angle_lower / 2) * r_lower
            inner_central_lower = m.cos(inner_angle_lower / 2) * r_lower      
            
            match position:

                case 'right':
                    self._vertices = [
                        Vector([total_central, total_lateral, mid_z]),
                        Vector([total_central, -total_lateral, mid_z]),
                        Vector([inner_central_upper, inner_lateral_upper, upper_z]),
                        Vector([inner_central_upper, -inner_lateral_upper, upper_z]),
                        Vector([inner_central_lower, inner_lateral_lower, lower_z]),
                        Vector([inner_central_lower, -inner_lateral_lower, lower_z]),
                    ]

                case 'front':
                    self._vertices = [
                        Vector([total_lateral, total_central, mid_z]),
                        Vector([-total_lateral, total_central, mid_z]),
                        Vector([inner_lateral_upper, inner_central_upper, upper_z]),
                        Vector([-inner_lateral_upper, inner_central_upper, upper_z]),
                        Vector([inner_lateral_lower, inner_central_lower, lower_z]),
                        Vector([-inner_lateral_lower, inner_central_lower, lower_z]),
                    ]

                case 'left':
                    self._vertices = [
                        Vector([- total_central, total_lateral, mid_z]),
                        Vector([-total_central, -total_lateral, mid_z]),
                        Vector([-inner_central_upper, inner_lateral_upper, upper_z]),
                        Vector([-inner_central_upper, -inner_lateral_upper, upper_z]),
                        Vector([-inner_central_lower, inner_lateral_lower, lower_z]),
                        Vector([-inner_central_lower, -inner_lateral_lower, lower_z]),
                    ]

                case 'back':
                    self._vertices = [
                        Vector([total_lateral, -total_central, mid_z]),
                        Vector([-total_lateral, -total_central, mid_z]),
                        Vector([inner_lateral_upper, -inner_central_upper, upper_z]),
                        Vector([-inner_lateral_upper, -inner_central_upper, upper_z]),
                        Vector([inner_lateral_lower, -inner_central_lower, lower_z]),
                        Vector([-inner_lateral_lower, -inner_central_lower, lower_z]),
                    ]

        else:
            self._vertices = dimensions
            self.area = calculate_area_3D(self._vertices)
            self.volume = self.area * self.thickness



        
    

    

    def calculate_soft_iron_distortion_matrix(self, position_vector: Vector):

        '''
        This function allows to calculate the soft iron
        distortion matrix from the position of the vertices 
        of the plate and the parameters defined for the surface. 


        input: 
        ------------

        position_vector: Vector, list, tuple
            Vector containing the position relative 
            to the coordinate system origin 
            of the point in m for which we want to calculate the 
            soft iron distortion matrix. 
        '''

        induced_matrix = Matrix.zeros()
        diff_magnetic = self.relative_magnetic_permeability - 1.0
        num_vertices = len(self._vertices)
        dV = self.volume / num_vertices
        dipole_scalar = (diff_magnetic * dV) / (4.0 * np.pi) 


        if isinstance(position_vector, (list, tuple)):
            position_vector = Vector(position_vector)
        elif isinstance(position_vector, Vector):
            position_vector = position_vector
        else:
            raise ValueError('Position Vector can only be a tuple, list or Vector')


        for _vertex in self._vertices: 

            r_V = position_vector - _vertex 
            r = abs(r_V)
            r_unit = r_V / r 


            
            rx, ry, rz = r_unit[0], r_unit[1], r_unit[2]


            projection_tensor = Matrix([
            [rx * rx, rx * ry, rx * rz],
            [ry * rx, ry * ry, ry * rz],
            [rz * rx, rz * ry, rz * rz]
            ])
            
            adjustment_dimensions = 1e-4
            dipole_kernel = (
            3.0 * projection_tensor - Matrix.identity()
             ) / ((r ** 3) + adjustment_dimensions) 

            induced_matrix = induced_matrix + (dipole_scalar * dipole_kernel)
        
        distortion_matrix = Matrix.identity() + induced_matrix 

        self._magnetic_distortion_matrixes[tuple(position_vector)] = distortion_matrix




    @classmethod
    def from_dict(cls, data: dict):

        '''
        Creates an instance of Plate class from a dictionary object, data. 
        Data is a dictionary that must contain the same keys as the initialization
        parameter of the Plate class. In the case some parameter is not 
        defined, the default value matches the default intializaiton of the constructor
        '''

        return cls(
            # Mandatory Parameter 
            material                       = data['material'],

            # Optional Parameter 
            absolute_magnetic_permeability = data.get('absolute_magnetic_permeability', None)
        )