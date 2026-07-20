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



    

    def define_plate_position(self, shape, dimensions, position, height):
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
            Position of the plate when the shape is not 
            'personalized' along the z axis. 
              
        
        

        '''
    



    def calculate_soft_iron_distortion_matrix(self, position_vector):

        '''
        This function allows to calculate the soft iron
        distortion matrix from the position of the point
        and the parameters defined for the surface
        '''

