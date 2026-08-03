import math as m

import numpy as np
from matplotlib.path import Path

from rocketpy.mathutils import Matrix, Vector
from rocketpy.mathutils.function import Function
from rocketpy.plots.plate_plots import _PlatePlots
from rocketpy.prints.plate_prints import _PlatePrints


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
        self._validate_material(material, absolute_magnetic_permeability)
        self._validate_shape(shape, dimensions, z_points, angular_points, grid_spacing)

        self.relative_magnetic_permeability = self.absolute_magnetic_permeability / (
            4 * np.pi * 1e-7
        )

        self.thickness = thickness
        self.name = name

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
        else:  # personalized
            self.generate_points(rocket, position, height)
            self.area = len(self.points) * (self.grid_spacing**2)
            self.volume = self.area * self.thickness

    def _validate_material(self, material, absolute_magnetic_permeability):
        """
        Validates and defines the input parameters related to the material
        and magnetic permeability.
        """
        if isinstance(material, str):
            if material == "iron":
                self.material = "iron"
                self.absolute_magnetic_permeability = 1.25e-3
            elif material == "carbon_steel":
                self.material = "carbon_steel"
                self.absolute_magnetic_permeability = 1.2e-4
            elif material == "personalized":
                self.material = "personalized"

                if absolute_magnetic_permeability is None:
                    raise ValueError(
                        "The magnetic permeability is compulsory when personalized is chosen"
                    )
                self.absolute_magnetic_permeability = absolute_magnetic_permeability
            else:
                raise ValueError(
                    "Material argument can only be iron, carbon_steel or personalized"
                )
        else:
            raise ValueError("material argument can only be a string")

    def _validate_shape(
        self, shape, dimensions, z_points, angular_points, grid_spacing
    ):
        """
        Validates and defines the input parameters related to the shape.
        """
        if isinstance(shape, str):
            if shape in ("circular", "squared"):
                self.shape = shape
                self.dimensions = dimensions
                self.z_points = z_points
                self.angular_points = angular_points
                self.grid_spacing = None

            elif shape == "personalized":
                self.shape = shape
                self.dimensions = dimensions
                self.grid_spacing = grid_spacing
                self.z_points = None
                self.angular_points = None
            else:
                raise ValueError(
                    "The accepted strings are 'circular', 'squared' or 'personalized'"
                )
        else:
            raise ValueError("The shape must be defined as a string")

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

        if self.shape == "personalized":
            self.generate_personalized_internal_plate(
                rocket.general_radius, rocket.z_bounds_check, rocket
            )
        else:
            upper_z = height + self.dimensions / 2
            lower_z = height - self.dimensions / 2
            geometric_center_angle = position * (np.pi / 180)
            for z in np.linspace(lower_z, upper_z, self.z_points):
                if not rocket.z_bounds_check(z, frame="ucs")[0]:
                    continue

                r = rocket.general_radius(z, frame="ucs")
                if r <= 1e-6:
                    continue

                if self.shape == "circular":
                    center_z = height
                    alpha, beta = self._circular_angle_calculation(
                        z, center_z, geometric_center_angle, r
                    )
                else:  # squared
                    alpha, beta = self._squared_angle_calculation(
                        geometric_center_angle, r
                    )

                for theta in np.linspace(alpha, beta, self.angular_points):
                    x = r * np.sin(theta)
                    y = r * np.cos(theta)

                    # change to body axis coordiante system
                    z_bacs = (z - rocket.center_of_dry_mass_position) * rocket._csys
                    if rocket._csys == -1:
                        self.points.append([-x, y, z_bacs])
                    else:
                        self.points.append([x, y, z_bacs])

    def _circular_angle_calculation(self, z, center_z, geometric_center_angle, r):
        dz = z - center_z
        inside_sqrt = max(self.dimensions**2 - dz**2, 0)
        extension_angle = m.sqrt(inside_sqrt) / r
        alpha = geometric_center_angle - extension_angle / 2
        beta = geometric_center_angle + extension_angle / 2
        return alpha, beta

    def _squared_angle_calculation(self, geometric_center_angle, r):
        extension_angle = self.dimensions / r
        alpha = geometric_center_angle - extension_angle / 2
        beta = geometric_center_angle + extension_angle / 2
        return alpha, beta

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
        # Processing of points
        vertices = self.vertices_definition(z_checking_function, radius_func, rocket)

        centroid = [sum(col) / len(vertices) for col in zip(*vertices)]
        cx, cy, cz = centroid[0], centroid[1], centroid[2]

        ux, uy, uz, vx, vy, vz = self._calculate_uv_frame(vertices)

        uv_vertices = [
            (
                (pt[0] - cx) * ux + (pt[1] - cy) * uy + (pt[2] - cz) * uz,
                (pt[0] - cx) * vx + (pt[1] - cy) * vy + (pt[2] - cz) * vz,
            )
            for pt in vertices
        ]

        u_coords, v_coords = zip(*uv_vertices)
        plate_path = Path(uv_vertices)
        final_3d_points = []

        curr_u = min(u_coords)
        while curr_u <= max(u_coords):
            curr_v = min(v_coords)
            while curr_v <= max(v_coords):
                if plate_path.contains_point((curr_u, curr_v)):
                    p3d_x = cx + (curr_u * ux) + (curr_v * vx)
                    p3d_y = cy + (curr_u * uy) + (curr_v * vy)
                    p3d_z = cz + (curr_u * uz) + (curr_v * vz)

                    if m.hypot(p3d_x, p3d_y) < radius_func(p3d_z, frame="bacs"):
                        final_3d_points.append([p3d_x, p3d_y, p3d_z])

                curr_v += self.grid_spacing
            curr_u += self.grid_spacing

        self.points = final_3d_points

    def vertices_definition(self, z_checking_function, radius_func, rocket):
        vertices = []
        cdm_user_frame = Vector([0, 0, rocket.center_of_dry_mass_position])
        for pt in self.dimensions:
            self.check_entry_dimensions(pt, z_checking_function, radius_func)

            # Transform to BACS
            sensor_vec = Vector(pt) - cdm_user_frame
            if rocket._csys == -1:
                vertices.append([-sensor_vec[0], sensor_vec[1], -sensor_vec[2]])
            else:
                vertices.append([sensor_vec[0], sensor_vec[1], sensor_vec[2]])
        return vertices

    def check_entry_dimensions(self, pt, z_checking_function, radius_func):
        """
        Check whether the points passed when shape is personalized are inside the rocket
        and in the case it is wrong prints why.
        """
        # Bounds checking
        if not z_checking_function(pt[2], frame="ucs")[0]:
            raise ValueError(
                f"The z component: {pt[2]} of {pt} is outside the rocket range."
            )

        if m.hypot(pt[0], pt[1]) > radius_func(pt[2], frame="ucs"):
            raise ValueError(f"Point {pt} is outside the rocket radius at z={pt[2]}.")

    def _calculate_uv_frame(self, vertices):
        """Helper method to calculate the 2D local coordinate system vectors."""
        v0, v1, v2 = vertices[:3]
        nx = (v1[1] - v0[1]) * (v2[2] - v0[2]) - (v1[2] - v0[2]) * (v2[1] - v0[1])
        ny = (v1[2] - v0[2]) * (v2[0] - v0[0]) - (v1[0] - v0[0]) * (v2[2] - v0[2])
        nz = (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0])

        n_norm = m.hypot(nx, ny, nz)
        nx, ny, nz = nx / n_norm, ny / n_norm, nz / n_norm

        arb_x, arb_y, arb_z = (
            (0.0, 0.0, 1.0) if (abs(nx) > 0.5 or abs(ny) > 0.5) else (1.0, 0.0, 0.0)
        )

        ux = ny * arb_z - nz * arb_y
        uy = nz * arb_x - nx * arb_z
        uz = nx * arb_y - ny * arb_x
        u_norm = m.hypot(ux, uy, uz)
        ux, uy, uz = ux / u_norm, uy / u_norm, uz / u_norm

        vx, vy, vz = ny * uz - nz * uy, nz * ux - nx * uz, nx * uy - ny * ux

        return ux, uy, uz, vx, vy, vz

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
            if self.points:
                induced_matrix = Matrix.zeros()
                diff_magnetic = self.relative_magnetic_permeability - 1.0
                num_points = len(self.points)
                dv = self.volume / num_points
                dipole_scalar = (diff_magnetic * dv) / (4.0 * np.pi)

                if isinstance(position_vector, (list, tuple)):
                    position_vector = Vector(position_vector)
                elif not isinstance(position_vector, Vector):
                    raise ValueError(
                        "Position Vector can only be a tuple, list or Vector"
                    )

                for point in self.points:
                    r_v = position_vector - Vector(point)
                    r = abs(r_v)
                    r_unit = r_v / r

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

    def draw_3d(self, color: str = "teal", marker: str = "h", filename=None) -> None:
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
        self.plots.draw_3d(color, marker, filename)

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
