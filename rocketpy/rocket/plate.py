import math as m

import numpy as np
from matplotlib.path import Path

from rocketpy.mathutils import Matrix, Vector
from rocketpy.mathutils.function import Function
from rocketpy.plots.plate_plots import _PlatePlots
from rocketpy.prints.plate_prints import _PlatePrints
from rocketpy.rocket.aero_surface import NoseCone


class Plate:
    """
    This class allows to define surfaces on the rocket. It is used
    to account for the soft iron distortion on the rocket that
    affects the magnetometer reading.

    Attributes
    ----------
    Plate.shape: str
        Shape of the plate. It can be 'circular', 'squared'
        or 'personalized'.
    Plate.dimensions: float, int or list[list]
        Dimensions of the plate. When the shape is squared or
        circular it has a float or int, whereas it is a list of
        lists when shape is 'personalized'.
    Plate.material: str
        Material from which the plate is composed. Allowed strings
        are 'iron', 'carbon steel', or 'personalized' if we want to
        define the material based on the magnetic permeability.
    Plate.absolute_magnetic_permeability: float
        Magnetic permeability of the material,
    Plate.relative_magnetic_permeability: float
        Ratio of the magnetic permeability to the magnetic permeability
        of vacuum.
    Plate.thickness: float or int
        Thickness of the plate.
    Plate.area: float or int
        Area of the plate. When the parameter is circular or squared,
        then the common expression is used, while with personalized,
        the area is the minimal area of the region contained within the
        points.
    Plate.volume: float or int
        Volume of the plate.
    Plate._magnetic_distortion_matrixes: dict
        Dictionary formed by the magnetic distortion matrix caused by
        the plate. The keys are the position vector of the point relative
        to the cso given as a tuple, and the value is the magnetic
        distortion Matrix.
    Plate.points: list[list]
        List formed by the vectors representing the
        components of each points of the surface in the body axis coordinate
        system.
    Plate.grid_spacing: float or int
        Space between points when it is 'personalized'
    Plate.z_points: int 
        Number of points when it is 'circular' or 'squared'
        along the z axis. 
    Plate.angular_points: int
        Number of angles when it is 'circular' or 'squared'
    Plate.name: str
        Name of the plate.

    """

    def __init__(
        self,
        shape: str,
        dimensions: float | int | list,
        material,
        thickness: int | float = 0.001,
        absolute_magnetic_permeability: float | int | None = None,
        grid_spacing: int | float = 0.001,
        z_points: float | int = 40,
        angular_points: float | int = 70,
        name: str = "Plate",
    ):
        """
        Initializes the Plate.

        Parameters
        ----------
        shape: str
            The shape of the plate, allowed parameters are:

            If 'circular': then the plate is assumed to be
            a circle, and the input 'dimension' refers to
            the radius. The plate will be located in the
            rocket body or nose cone.

            If 'squared': then the plate is assumed to be a
            square and the input 'dimensions' refers to the
            side. The plate will be located in the
            rocket body or nose cone.

            If 'personalized': then the plate will have the shape
            specified by the vertexes defined in 'dimensions'
        dimensions: float, int, list
            Dimensions of the plate, which depend on 'shape'
            definition:

            - If shape is 'circular', the dimension is a float or int,
            which represents the radius when the shape is flat.

            - If shape is 'squared', the dimension is a float or int,
            which represents the side length when the shape is flat.

            - If shape is 'personalized', dimensions must be a list
            with lists as the vertixes that form the shape. They must be
            in sequential order (clockwise or counter-clockwise) and at
            least 3 non-collinear vertices must be defined in the user 
            defined coordinate system.
        material: str
            Material from which the plate is composed Allowed strings
            are 'iron', 'carbon_steel', or 'personalized' if we want
            to define the material based on the magnetic permeability.
        thickness: float or int
            Thickness of the plate in m.
        absolute_magnetic_permeability: float, int, optional
            Magnetic permeability of the material, which is the measure
            of a material ability to allow magnetic field lines to pass
            through it.
        grid_spacing: float, optional,
            Only used when the shape is personalized and it determines
            the space between the points of the approximated shape defined
            by the vertices. Default is 0.001.
        z_points: int, optional
            Number of points that will be taken in the z axis to create the
            plate  when it is 'circular' or 'squared'. Default is 40. 
        angular_points: int, optional
            Number of angles that will be taken to create the plate when it 
            is 'circular' or 'squared'. Default is 70. 
        name: str, optional
            Name of the plate. Default value is 'Plate'

        """
        self._magnetic_distortion_matrixes = {}
        self.points = []
        self.plots = None
        self.prints = _PlatePrints(self)

        if isinstance(material, str):
            if material == "iron":
                self.material = "iron"
                self.absolute_magnetic_permeability = 1.25e-3
            elif material == "carbon_steel":
                self.material = "carbon_steel"
                self.absolute_magnetic_permeability = 1.2e-4
            elif material == "personalized":
                self.material = "personalized"

                if absolute_magnetic_permeability == None:
                    raise ValueError(
                        "The magnetic permeability is compulsory when personalized is chosen"
                    )
                elif not isinstance(absolute_magnetic_permeability, (int, float)):
                    raise ValueError(
                        "The magnetic permeability must be an int or float"
                    )
                else:
                    self.absolute_magnetic_permeability = absolute_magnetic_permeability
            else:
                raise ValueError(
                    "Material argument can only be iron, carbon_steel or personalized"
                )
        else:
            raise ValueError("material argument can only be a string")

        if isinstance(shape, str):
            if shape == "circular" or shape == "squared":
                self.shape = shape
                if not isinstance(dimensions, (float, int)):
                    raise ValueError(
                        "The dimensions must be a float or int, when the shape is circular or squared"
                    )
                else:
                    self.dimensions = dimensions
                if isinstance(z_points, int):
                    self.z_points = z_points
                else:
                    raise ValueError('The number points along the z axis must be an int')
                
                if isinstance(angular_points, int):
                    self.angular_points = angular_points
                else:
                    raise ValueError('The number angles that will be evaluated to get the plate must be an int')
                
                self.grid_spacing = None
                
            elif shape == "personalized":
                self.shape = shape
                if not isinstance(dimensions, (tuple, list)):
                    raise ValueError(
                        "The dimensions must be a list or tuple, when the shape is personalized"
                    )
                elif len(dimensions) < 3:
                    raise ValueError(
                        "At least 3 points must be defined to create a surface"
                    )
                else:
                    for num_vertex in range(len(dimensions)):
                        if len(dimensions[num_vertex]) == 3:
                            if not isinstance(dimensions[num_vertex], (list, tuple)):
                                raise ValueError(
                                    "When the shape is personalized the dimensions must be a list of vertex whose components must be defined in a list or tupe"
                                )
                        else:
                            raise ValueError(
                                "The vertex must be defiened with 3 components"
                            )
                self.dimensions = dimensions

                if isinstance(grid_spacing, (float, int)):
                    if grid_spacing <= 0:
                        raise ValueError("Grid spacing must be greater than 0")
                    else:
                        self.grid_spacing = grid_spacing
                else:
                    raise ValueError("Grid spacing must be a float or int")
                self.z_points = None
                self.angular_points = None
            else:   
                raise ValueError(
                    "The accepted shapes are circular, squared and personalized"
                )
        else:
            raise ValueError("The shape must be defined as a string")

        self.relative_magnetic_permeability = self.absolute_magnetic_permeability / (
            4 * np.pi * 1e-7
        )

        if not isinstance(thickness, (float, int)):
            raise ValueError("Thickness must be a float or int")
        else:
            self.thickness = thickness

        if isinstance(name, str):
            self.name = name
        else:
            raise ValueError("The name must be a str")

    def define_plate_position(
        self,
        rocket,
        position: str | None = None,
        height: float | int | None = None,
    ) -> None:
        """
        Defines the geometry of the plate from the
        shape, position, dimensions and height defined in the add_plate()
        rocket class method.

        Parameters
        ----------
        position: float, int, optional
            Position of the plate, when the shape is 'squared' or 'circular'
            It is the angle between the y axis of the user defined coordinate system
            and the geometric center of the plate in degrees. The positive direction is defined 
            as the direciton in which the right hand rule coincides with the z direction
            based on the coordinate system orientation. 
        height: float, int, optional
            Position of the geometric center of plate when the shape is not
            'personalized' along the z axis relative to the user defined coordiante
            system.
        rocket: Rocket
            RocketPy class.

        Returns
        -------
        None
        """
        if self.shape == "squared":
            self.generate_points(rocket, position, height)
            self.area = self.dimensions * self.dimensions
            self.volume = self.area * self.thickness
        elif self.shape == "circular":
            self.generate_points(rocket, position, height)
            self.area = np.pi * (self.dimensions**2)
            self.volume = self.area * self.thickness
        elif self.shape == "personalized":
            self.generate_points(rocket, position, height)
            self.area = len(self.points) * (self.grid_spacing**2)
            self.volume = self.area * self.thickness

    def generate_points(
        self,
        rocket,
        position: str | None = None,
        height: float | int | None = None,
    ) -> None:
        """
        Generates the points required to calculate the soft iron distortion
        matrix relative to the body axis coordinate system:

        Parameters
        ------------
        rocket: Rocket
            RocketPy class.
        position: float, int, optional
            Position of the plate, when the shape is 'squared' or 'circular'
            It is the angle between the y axis of the user defined coordinate system
            and the geometric center of the plate in degrees. The positive direction is defined 
            as the direciton in which the right hand rule coincides with the z direction
            based on the coordinate system orientation. 
        height: float, int, optional
            Position of the geometric center of plate when the shape is 'circular'
            or 'squared' along the z axis relative to the user defined 
            coordiante system.

        Returns
        -------
        None

        """
        self.points = []

        if self.shape == "circular":
            upper_z = height + self.dimensions
            lower_z = height - self.dimensions
            center_z = height 
            geometric_center_angle = position * (np.pi / 180)

            for z in np.linspace(lower_z, upper_z, self.z_points):
                flag, _ = rocket.z_bounds_check(z, frame = 'ucs')
                if not flag:
                    continue

                r = rocket.general_radius(z, frame = 'ucs')
                if r <= 1e-6:
                    continue

                dz = z - center_z
                inside_sqrt = self.dimensions**2 - dz**2
                if inside_sqrt < 0:
                    inside_sqrt = 0

                arc_length = m.sqrt(inside_sqrt)
                extension_angle = arc_length / r
                alpha = geometric_center_angle - extension_angle / 2
                beta = geometric_center_angle + extension_angle / 2

                for theta in np.linspace(alpha, beta, self.angular_points):
                    x = r * np.sin(theta)
                    y = r * np.cos(theta)

                    # change to body axis coordiante system 
                    z_bacs = (z - rocket.center_of_dry_mass_position) * rocket._csys
                    if rocket._csys == -1: 
                        self.points.append([-x, y, z_bacs])
                    else:
                        self.points.append([x, y, z_bacs])
                

        elif self.shape == "squared":
            upper_z = height + self.dimensions / 2
            lower_z = height - self.dimensions / 2
            geometric_center_angle = position * (np.pi / 180)

            for z in np.linspace(lower_z, upper_z, self.z_points):
                flag, _ = rocket.z_bounds_check(z, frame = 'ucs')
                if not flag:
                    continue

                r = rocket.general_radius(z, frame = 'ucs')
                if r <= 1e-6:
                    continue

                extension_angle = self.dimensions / r
                alpha = geometric_center_angle - extension_angle / 2
                beta = geometric_center_angle + extension_angle / 2

                for theta in np.linspace(alpha, beta, self.angular_points):
                    x = r * np.sin(theta)
                    y = r * np.cos(theta)
                    
                    # change to body axis coordiante system 
                    z_bacs = (z - rocket.center_of_dry_mass_position) * rocket._csys
                    if rocket._csys == -1: 
                        self.points.append([-x, y, z_bacs])
                    else:
                        self.points.append([x, y, z_bacs])

        elif self.shape == "personalized":
            self.generate_personalized_internal_plate(
                rocket.general_radius, rocket.z_bounds_check, rocket
            )

    def generate_personalized_internal_plate(
        self,
        radius_func: Function,
        z_checking_function: Function,
        rocket, 
    ) -> None:
        """
        Generates a 3D grid of points bounded by an arbitrary set of vertices,
        forced flat, and filtered to remain inside the rocket.

        Parameters
        ----------
        radius_func:
            A callable function that takes Z relative to the and returns the rocket radius.
        z_checking_function:
            A callable function that takes z and returns True if it is inside
            the rocket, False, otherwise.
        rocket: Rocket
            Rocket to which the plate belongs. 

        Returns
        -------
        None
        """
        # processing of points
        vertices = []
        for point in self.dimensions:            
            
            # check height bounds and radial bounds in ucs
            # height boundds
            flag, range_z = z_checking_function(point[2], frame = 'ucs')
            if not flag:
                raise ValueError(
                    f"The z component: {point[2]} of {point} is outside the rocket range {range_z}"
                )

            # radial bounds
            r_point = m.sqrt(point[0]**2 + point[1]**2)
            r = radius_func(point[2], frame = 'ucs')
            if r_point > r:
                raise ValueError(
                    f"The point with coordinates {point} is outside the rocket since the radius {r_point} is bigger than the radius of the rocket at that z: {point[2]}, which is: {r}"
                )
            
            # transforming from user defined coordinate system to body axis coordinate system
            cdm_user_frame = Vector([0, 0, rocket.center_of_dry_mass_position])
            sensor_from_cdm_user_frame = Vector(point) - cdm_user_frame

            if rocket._csys == -1: # nose to tail
                point_bacs_frame = Vector([-sensor_from_cdm_user_frame[0], sensor_from_cdm_user_frame[1], -sensor_from_cdm_user_frame[2]])
            elif rocket._csys == 1: #tail to nose
                point_bacs_frame = sensor_from_cdm_user_frame


            x = point_bacs_frame[0]
            y = point_bacs_frame[1]
            z = point_bacs_frame[2]
            vertices.append([x,y,z])

        total_x, total_y, total_z = 0.0, 0.0, 0.0
        num_pts = len(self.dimensions)

        for pt in vertices:
            total_x += pt[0]
            total_y += pt[1]
            total_z += pt[2]

        centroid_x = total_x / num_pts
        centroid_y = total_y / num_pts
        centroid_z = total_z / num_pts

        v0 = vertices[0]
        v1 = vertices[1]
        v2 = vertices[2]

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

        for pt in vertices:
            cx = pt[0] - centroid_x
            cy = pt[1] - centroid_y
            cz = pt[2] - centroid_z

            u_val = cx * ux + cy * uy + cz * uz
            v_val = cx * vx + cy * vy + cz * vz

            u_coords.append(u_val)
            v_coords.append(v_val)
            uv_vertices.append((u_val, v_val))

        # Generate 2D bounding grid
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
                    r_allowed = radius_func(p3d_z, frame = 'bacs')

                    if r_point < r_allowed:
                        final_3d_points.append([p3d_x, p3d_y, p3d_z])

                curr_v += self.grid_spacing
            curr_u += self.grid_spacing

        self.points = final_3d_points 

    def calculate_soft_iron_distortion_matrix(self, position_vector: Vector) -> None:
        """
        Calculates the soft iron distortion matrix from the position of the points 
        relative to the body axis coordinate system of the plate and the parameters 
        defined for the surface.

        Parameters
        ----------
        position_vector: Vector, list, tuple
            Vector containing the position in the body axis coordinate system
            of the point in m for which we want to calculate the soft iron
            distortion matrix.

        Returns
        -------
        None
        """
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
                    raise ValueError(
                        "Position Vector can only be a tuple, list or Vector"
                    )

                for point in self.points:
                    r_V = position_vector - Vector(point)
                    r = abs(r_V)
                    r_unit = r_V / r

                    rx, ry, rz = r_unit[0], r_unit[1], r_unit[2]

                    projection_tensor = Matrix(
                        [
                            [rx * rx, rx * ry, rx * rz],
                            [ry * rx, ry * ry, ry * rz],
                            [rz * rx, rz * ry, rz * rz],
                        ]
                    )

                    dipole_kernel = (3.0 * projection_tensor - Matrix.identity()) / (
                        r**3
                    )

                    induced_matrix = induced_matrix + (dipole_scalar * dipole_kernel)

                self._magnetic_distortion_matrixes[tuple(position_vector)] = (
                    induced_matrix
                )
            else:
                raise ValueError(
                    "To calculate the soft iron distortion matrix, first the plate must be added to the rocket, points list cannot be empty"
                )

        else:
            raise ValueError("The points defining the plate must be a list")

    def draw_3D(self, color: str = "teal", marker: str = "h", filename=None) -> None:
        """
        Draws the plate in a matplotlib figure

        Parameters
        ----------
        color : str
            Color of the points.
            A full list of color names can be found at:
            https://matplotlib.org//gallery/color/named_colors
            Default is 
        marker: str
            shape of the points from which the plate is formed.
            A full list of markers can be found at:
            https://matplotlib.org/stable/api/markers_api.html
            Default is 'h', hexagon. 
        filename : str, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """
        self.plots.draw_3D(color, marker, filename)

    def _rocket_belonging(self, rocket) -> None:
        """
        Initializate _PlatePlot class with the rocket instance to which it belogns.


        Parameters
        ----------
        rocket: Rocket
            rocket instance to which it belongs

        Returns
        -------
        None
        """
        self.plots = _PlatePlots(self, rocket)

    def info(self) -> None:
        """
        Prints a summary of the information stored in the plate object.

        Returns
        -------
        None
        """
        self.prints.all()

    def all_info(self) -> None:
        """
        Prints out all data and graphs available about the Plate.

        Returns
        -------
        None
        """
        self.plots.all()
        self.prints.all()

    @classmethod
    def from_dict(cls, data: dict) -> "Plate":
        """
        Creates an instance of Plate class from a dictionary object, data.
        Data is a dictionary that must contain the same keys as the initialization
        parameter of the Plate class. In the case some parameter is not
        defined, the default value matches the default intializaiton of the constructor

        """
        return cls(
            # Mandatory Parameter
            shape=data["shape"],
            dimensions=data["dimensions"],
            material=data["material"],
            # Optional Parameter
            thickness=data.get("thickness", 0.001),
            absolute_magnetic_permeability=data.get(
                "absolute_magnetic_permeability", None
            ),
            grid_spacing=data.get("grid_spacing", 0.001),
            name=data.get("name", "Plate"),
        )
