
from rocketpy.rocket.aero_surface import NoseCone
from rocketpy.mathutils import Vector, Matrix
import math as m
import numpy as np
from matplotlib.path import Path  
from rocketpy.mathutils.function import Function
from rocketpy.plots.plate_plots import _PlatePlots
from rocketpy.prints.plate_prints import _PlatePrints




class Plate():
    '''
    This class allows to define surfaces on the rocket.
    It is used to account for the soft iron distortion
    on the rocket that affects the magnetometer reading.

    Attributes:
    --------------

    shape: str
        Shape of the plate. It can be 'circular', 
        'squared' or 'personalized'.

    dimensions: float, int or list[list]
        Dimensions of the plate. When the shape is 
        squared or circular it has a float or int, whereas
        it is a list of lists when shape is 'personalized'.

    material: str
        Material from which the plate is composed
        Allowed strings are 'iron', 'carbon steel', or 
        'personalized' if we want to define the material 
        based on the magnetic permeability. 

    absolute_magnetic_permeability: float
        Magnetic permeability of the material,

    relative_magnetic_permeability: float
        ratio of the magnetic permeability to 
        the magnetic permeability of vacuum

    thickness: float or int
        Thickness of the plate

    area: float or int
        Area of the plate. When the parameter is 
        circular or squared, then the common expression
        is used, while with personalized, the area is
        the minimal area of the region contained within the points.

    volume: float or int
        Volume of the plate. 

    _magnetic_distortion_matrixes: dict
        Dictionary formed by the magnetic distortion
        matrix caused by the plate. The keys are the position
        vector of the point relative to the cso given as a tuple,
        and the value is the magnetic distortion Matrix. 
    
    points: list[list]
        list formed by the vectors representing the 
        components of each points of the surface relative
        to the cso. 

    name: str
        Name of the plate.
    '''
    def __init__(
        self,
        shape: str,
        dimensions: float | int | list,
        material,
        thickness: int | float = 0.001,
        absolute_magnetic_permeability: float | int | None = None,             
        grid_spacing: int | float = 0.001, 
        name: str = 'Plate', 
    ):
        '''
        Parameters:
        --------------
        shape: str
            The shape of the plate, allowed parameters are:

            'circular': then the plate is assumed to be 
            a circle, and the input 'dimension' refers to
            the radius. The plate will be located in the 
            rocket body or nose cone. 

            'squared': then the plate is assumed to be a 
            square and the input 'dimensions' refers to the 
            side. The plate will be located in the 
            rocket body or nose cone. 

            'personalized': then the plate will have the shape 
            specified by the vertexes defined in 'dimensions'

        dimensions: float, int, list
            Dimensions of the plate, which depend on 'shape' 
            definition:

            When it is 'circular', the dimension is a float or int,
            which represents the radius, when the shape is flat. 

            when it is 'squared', the dimension is a float or int,
            which represents the side length, when the shape is flat.

            when it is 'personalized', dimensions must be a list
            with lists as the vertixes that form the shape. They must be
            in order and at least 3 vertices must be defined. 

        material: str
            Material from which the plate is composed
            Allowed strings are 'iron', 'carbon_steel', or 
            'personalized' if we want to define the material 
            based on the magnetic permeability. 

        thickness: float or int
            Thickness of the plate in m. 

        absolute_magnetic_permeability: float, int, optional
            Magnetic permeability of the material, which is 
            the measure of a material ability to allow magnetic
            field lines to pass through it. 

        grid_spacing: float, optional, 
            it is used only when the shape is personalized and determines 
            the space between the points of the approximated shape defined
            by the vertices. Default is 0.001. 

        name: str, optional
            Name of the plate. Default value is 'Plate'.

        '''
        self._magnetic_distortion_matrixes = {}
        self.points = []
        self.plots = None
        self.prints = _PlatePrints(self)

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

        if isinstance(shape, str):
            if shape == 'circular' or shape == 'squared':
                self.shape = shape
                if not isinstance(dimensions, (float, int)):
                    raise ValueError('The dimensions must be a float or int, when the shape is circular or squared')
                else:
                    self.dimensions = dimensions
            elif shape == 'personalized':
                self.shape = shape
                if not isinstance(dimensions, (tuple, list)):
                    raise ValueError('The dimensions must be a list or tuple, when the shape is personalized')
                elif len(dimensions) < 3:
                    raise ValueError('At least 3 points must be defined to create a surface')
                else:
                    for num_vertex in range(len(dimensions)):
                        if len(dimensions[num_vertex]) == 3:
                            if not isinstance(dimensions[num_vertex], (list, tuple)):
                                raise ValueError('When the shape is personalized the dimensions must be a list of vertex whose components must be defined in a list or tupe')
                        else:
                            raise ValueError('The vertex must be defiened with 3 components')      
                        self.dimensions = dimensions      
            else:
                raise ValueError('The accepted shapes are circular, squared and personalized')
        else:
            raise ValueError('The shape must be defined as a string')
        

        self.relative_magnetic_permeability = self.absolute_magnetic_permeability / (4 * np.pi * 1e-7)
        
        if not isinstance(thickness, (float, int)):
            raise ValueError('Thickness must be a float or int')
        else:
            self.thickness = thickness

        if isinstance(grid_spacing, (float, int)):
            if grid_spacing <= 0:
                raise ValueError('Grid spacing must be greater than 0')
            else:
                self.grid_spacing = grid_spacing
        else:
            raise ValueError('Grid spacing must be a float or int')  
                
        if isinstance(name, str):
            self.name = name
        else: 
            raise ValueError('The name must be a str')
        
    

    def define_plate_position(
        self,    
        rocket,
        position: str | None = None,
        height: float | int | None = None,
    ) -> None:
        '''
        This function defines the geometry of the plate
        with respect to the cso from the shape, position,
        dimensions and height defined in the add_plate()
        rocket class method. 

        Parameters:
        ------------
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
        if self.shape == 'squared':
            self.generate_points(rocket, position, height)
            self.area = self.dimensions * self.dimensions
            self.volume = self.area * self.thickness
        elif self.shape == 'circular': 
            self.generate_points(rocket, position, height)
            self.area = np.pi * (self.dimensions ** 2)
            self.volume = self.area * self.thickness
        elif self.shape == 'personalized':
            self.generate_points(rocket, position, height)
            self.area = len(self.points) * (self.grid_spacing ** 2) 
            self.volume = self.area * self.thickness

    def generate_points(
        self,
        rocket,
        position: str | None = None,
        height: float | int | None = None,
    ) -> None:
        '''
        This function generates the points required to calculate the
        soft iron distortion matrix: 

        Parameters:
        ------------
        rocket: Rocket
            RocketPy class.

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

        '''
        self.points = []
        if self.shape == 'circular':
            upper_z = height + self.dimensions 
            lower_z = height - self.dimensions 
            center_z = (lower_z + upper_z) / 2
            self.points = []

            for z in np.linspace(lower_z, upper_z, 40):
                z_points = []
                r = rocket.general_radius(z)
                dz = z - center_z

                inside_sqrt = self.dimensions ** 2 - dz ** 2
                if inside_sqrt < 0:
                    inside_sqrt = 0

                arc_length = 2 * m.sqrt(inside_sqrt)
                
                if arc_length > np.pi * r:
                    raise ValueError(f'The side length, {arc_length} cannot be bigger than half of the perimeter for a given radius {r}')      

                angle = arc_length / r
                lateral = m.sin(angle / 2) * r

                match position:
                    case 'right':
                        for y in np.linspace(-lateral, lateral, 70):
                            x = m.sqrt(r**2 - y**2)
                            z_points.append([x, y, z])
                    case 'front':
                        for x in np.linspace(lateral, -lateral, 70):
                            y = m.sqrt(r**2 - x**2)
                            z_points.append([x, y, z])
                    case 'left':
                        for y in np.linspace(lateral, -lateral, 70):
                            x = -m.sqrt(r**2 - y**2)
                            z_points.append([x, y, z])
                    case 'back':
                        for x in np.linspace(-lateral, lateral, 70):
                            y = -m.sqrt(r**2 - x**2)
                            z_points.append([x, y, z])

                self.points.extend(z_points)

        elif self.shape == 'squared':
            upper_z = height + self.dimensions / 2.0
            lower_z = height - self.dimensions / 2.0
            for z in np.linspace(lower_z, upper_z, 30):
                z_points = []
                r = rocket.general_radius(z)
                
                if self.dimensions > np.pi * r:
                    raise ValueError('The side length cannot be bigger than the radius')

                angle = self.dimensions / r
                lateral = m.sin(angle / 2) * r

                match position:
                    case 'right':
                        for y in np.linspace(-lateral, lateral, 60):
                            x = m.sqrt(r**2 - y**2)
                            z_points.append([x, y, z])
                    case 'front':
                        for x in np.linspace(lateral, -lateral, 60):
                            y = m.sqrt(r**2 - x**2)
                            z_points.append([x, y, z])
                    case 'left':
                        for y in np.linspace(lateral, -lateral, 60):
                            x = -m.sqrt(r**2 - y**2)  # Fixed: added negative sign
                            z_points.append([x, y, z])
                    case 'back':
                        for x in np.linspace(-lateral, lateral, 60):
                            y = -m.sqrt(r**2 - x**2)
                            z_points.append([x, y, z])

                self.points.extend(z_points)   

        elif self.shape == 'personalized':
            self.generate_personalized_internal_plate(rocket.general_radius, rocket.z_bounds_check)


    def generate_personalized_internal_plate(
        self, 
        radius_func: Function, 
        z_checking_function: Function, 
    ) -> None:
        '''
        Generates a 3D grid of points bounded by an arbitrary set of vertices,
        forced flat, and filtered to remain inside the rocket hull.
        
        Parameters:
        ---------
        radius_func:
            A callable function that takes Z and returns the rocket radius

        z_checking_function:
            A callable function that takes z and returns True if it is inside the rocket, False, otherwise

        '''
        for point in self.dimensions:
            x, y, z = point[0], point[1], point[2]
            
            # height boundds
            flag, range_z = z_checking_function(z)
            if not flag:
                raise ValueError(f'The z component: {z} of {point} is outside the rocket range {range_z}')

            # radial bounds
            r_point = m.sqrt(x ** 2 + y ** 2)
            r = radius_func(z)
            if r_point > r:
                raise ValueError(f'The point with coordinates {point} is outside the rocket since the radius {r_point} is bigger than the radius of the rocket at that z: {z}, which is: {r}')
 
        total_x, total_y, total_z = 0.0, 0.0, 0.0
        num_pts = len(self.dimensions)
        
        for pt in self.dimensions:
            total_x += pt[0]
            total_y += pt[1]
            total_z += pt[2]
            
        centroid_x = total_x / num_pts
        centroid_y = total_y / num_pts
        centroid_z = total_z / num_pts

        v0 = self.dimensions[0]
        v1 = self.dimensions[1]
        v2 = self.dimensions[2]

        a_x, a_y, a_z = v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]
        b_x, b_y, b_z = v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]

        nx = a_y * b_z - a_z * b_y
        ny = a_z * b_x - a_x * b_z
        nz = a_x * b_y - a_y * b_x

        norm_val = m.sqrt(nx**2 + ny**2 + nz**2)
        nx, ny, nz = nx / norm_val, ny / norm_val, nz / norm_val

        # Define 2D Local Coordinate System (u_vec, v_vec)
        if abs(nx) > 0.5 or abs(ny) > 0.5:
            arb_x, arb_y, arb_z = 0.0, 0.0, 1.0
        else:
            arb_x, arb_y, arb_z = 1.0, 0.0, 0.0

        ux = ny * arb_z - nz * arb_y
        uy = nz * arb_x - nx * arb_z
        uz = nx * arb_y - ny * arb_x
        u_norm = m.sqrt(ux**2 + uy**2 + uz**2)
        ux, uy, uz = ux / u_norm, uy / u_norm, uz / u_norm

        vx = ny * uz - nz * uy
        vy = nz * ux - nx * uz
        vz = nx * uy - ny * ux

        uv_vertices = []
        u_coords = []
        v_coords = []

        for pt in self.dimensions:
            cx = pt[0] - centroid_x
            cy = pt[1] - centroid_y
            cz = pt[2] - centroid_z

            u_val = cx * ux + cy * uy + cz * uz
            v_val = cx * vx + cy * vy + cz * vz

            u_coords.append(u_val)
            v_coords.append(v_val)
            uv_vertices.append((u_val, v_val))

        # Generate 2D bounding grid 
        u_min, u_max = min(u_coords), max(u_coords)
        v_min, v_max = min(v_coords), max(v_coords)

        plate_path = Path(uv_vertices)
        final_3d_points = []


        curr_u = u_min
        while curr_u <= u_max:
            curr_v = v_min
            while curr_v <= v_max:
                if plate_path.contains_point((curr_u, curr_v)):
                    p3d_x = centroid_x + (curr_u * ux) + (curr_v * vx)
                    p3d_y = centroid_y + (curr_u * uy) + (curr_v * vy)
                    p3d_z = centroid_z + (curr_u * uz) + (curr_v * vz)

                    # Final geometry check: point must lie within internal radius at the corresponding height 
                    r_point = m.sqrt(p3d_x**2 + p3d_y**2)
                    r_allowed = radius_func(p3d_z)

                    if r_point < r_allowed:
                        final_3d_points.append([p3d_x, p3d_y, p3d_z])

                curr_v += self.grid_spacing
            curr_u += self.grid_spacing

        self.points = final_3d_points


    def calculate_soft_iron_distortion_matrix(self, position_vector: Vector) -> None:
        '''
        This function allows to calculate the soft iron
        distortion matrix from the position of the points 
        of the plate and the parameters defined for the surface. 

        Parameters: 
        ------------
        position_vector: Vector, list, tuple
            Vector containing the position relative 
            to the coordinate system origin 
            of the point in m for which we want to calculate the 
            soft iron distortion matrix. 
        
        Returns: 
        --------
        None
        '''
        if isinstance(self.points, list):
            if self.points != []:
                induced_matrix = Matrix.zeros()
                diff_magnetic = self.relative_magnetic_permeability - 1.0
                num_points = len(self.points)
                dV = self.volume / num_points
                dipole_scalar = (diff_magnetic * dV) / (4.0 * np.pi) 

                if isinstance(position_vector, (list, tuple)):
                    position_vector = Vector(position_vector)
                elif isinstance(position_vector, Vector):
                    position_vector = position_vector
                else:
                    raise ValueError('Position Vector can only be a tuple, list or Vector')

                for point in self.points: 
                    r_V = position_vector - Vector(point) 
                    r = abs(r_V)
                    r_unit = r_V / r 
                    
                    rx, ry, rz = r_unit[0], r_unit[1], r_unit[2]

                    projection_tensor = Matrix([
                    [rx * rx, rx * ry, rx * rz],
                    [ry * rx, ry * ry, ry * rz],
                    [rz * rx, rz * ry, rz * rz]
                    ])
                    
                    dipole_kernel = (
                    3.0 * projection_tensor - Matrix.identity()
                    ) / ((r ** 3)) 

                    induced_matrix = induced_matrix + (dipole_scalar * dipole_kernel)
                
                self._magnetic_distortion_matrixes[tuple(position_vector)] = induced_matrix

            else:
                raise ValueError('To calculate the soft iron distortion matrix, first the plate must be added to the rocket, points list cannot be empty')
            
        else: raise ValueError('The points defining the plate must be a list')


    def draw_3D(
        self, 
        color: str = 'blue', 
        marker: str = 'o',
        filename = None
    ) -> None:
        '''
        Draws the plate in a matplotlib figure

        Parameters
        ----------
        color : str
            Color of the points. 
            A full list of color names can be found at:
            https://matplotlib.org//gallery/color/named_colors
            
        marker: str
            shape of the points from which the plate is formed. 
            A full list of markers can be found at: 
            https://matplotlib.org/stable/api/markers_api.html

        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        '''
        self.plots.draw_3D(color, marker, filename)

    def _rocket_belonging(self, rocket) -> None:
        '''
        This funciton is used to initialize _WirePlot class by
        passing the rocket instance to which it belongs

        Parameter:
        ----------
        rocket: Rocket
            rocket instance to which it belongs
        
        Returns: 
        -------
        None
        '''
        self.plots = _PlatePlots(self, rocket)


    def info(self) -> None:
        '''
        print a summary of the information stored in the plate object

        Returns
        -------
        None
        '''
        self.prints.all()


    def all_info(self) -> None:
        '''
        Prints out all data and graphs available about the Plate.

        Returns
        -------
        None
        '''
        self.plots.all()
        self.prints.all()


    @classmethod
    def from_dict(cls, data: dict) -> "Plate":
        '''
        Creates an instance of Plate class from a dictionary object, data. 
        Data is a dictionary that must contain the same keys as the initialization
        parameter of the Plate class. In the case some parameter is not 
        defined, the default value matches the default intializaiton of the constructor

        '''
        return cls(
            # Mandatory Parameter 
            shape = data['shape'],
            dimensions = data['dimensions'],
            material = data['material'],


            # Optional Parameter 

            thickness = data.get('thickness', 0.001),
            absolute_magnetic_permeability = data.get('absolute_magnetic_permeability', None),
            grid_spacing = data.get('grid_spacing', 0.001),
            name = data.get('name', 'Plate')
        )
    


            
        