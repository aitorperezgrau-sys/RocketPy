import matplotlib.pyplot as plt

from rocketpy.plots.plot_helpers import show_or_save_plot

import numpy as np

 
class _PlatePlots:
    """
    Class that holds plot methods for the Plate class.

    Attributes
    ----------
    plate: Plate
        Plate object that will be used for the plots.
    """

    def __init__(self, plate, rocket):
        """
        Parameters
        ----------
        plate: Plate
            Plate instance.
        rocket: Rocket
            Rocket instance to which the plate
            is attached.

        """
        self.plate = plate
        self.rocket = rocket

    def draw_3D(
        self,
        color: str = "teal",
        marker: str = "h",
        elev: float | int | None = None,
        azim: float | int | None = None,
        filename: str | None = None,
    ) -> None:
        """
        Plots the scatter plot of the plate formed by the points
        used to model the magnetic distortion in 3D.

        Parameters
        ----------
        color : str, optional
            Color of the points.
            A full list of color names can be found at:
            https://matplotlib.org//gallery/color/named_colors
            Default: 'teal'.
        marker : str, optional
            shape of the points from which the plate is formed.
            A full list of markers can be found at:
            https://matplotlib.org/stable/api/markers_api.html
            Default is 'h'.
        elev : float, optional
            The elevation angle in degrees rotates the camera above the plane
            pierced by the vertical axis, with a positive angle corresponding
            to a location above that plane. For example, with the default
            vertical axis of 'z', the elevation defines the angle of the camera
            location above the x-y plane.
            If None, then the initial value as specified in the `Axes3D`
            constructor is used. Default is None.
        azim : float, optional
            The azimuthal angle in degrees rotates the camera about the
            vertical axis, with a positive angle corresponding to a
            right-handed rotation. For example, with the default vertical axis
            of 'z', a positive azimuth rotates the camera about the origin from
            its location along the +x axis towards the +y axis.
            If None, then the initial value as specified in the `Axes3D`
            constructor is used. Default is None.
        filename : str, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """
        # Unpack columns into separate x, y, and z components
        x, y, z = zip(*self.plate.points)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        # plot individual points
        ax.scatter(x, y, z, color=color, marker=marker, label=self.plate.name)
        ax.view_init(elev=elev, azim=azim)

        # Labels & formatting
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend()


        print(f"\n{self.plate.name} representation: ")
        show_or_save_plot(filename)

    def draw(
        self,
        vis_args: dict | None = None,
        plane: str = "xz",
        color: str = "darkgree",
        marker="o",
        filename=None,
    ) -> None:
        """
        Plots the plate together with the outline of the rocket
        in 2D. 

        Parameters
        ----------
        vis_args : dict, optional
            Determines the visual aspects when drawing the rocket. If ``None``,
            default values are used. Default values are:

            .. code-block:: python

                {
                    "background": "#EEEEEE",
                    "tail": "black",
                    "nose": "black",
                    "body": "black",
                    "fins": "black",
                    "motor": "black",
                    "line_width": 2.0,
                }

            A full list of color names can be found at: \
            https://matplotlib.org/stable/gallery/color/named_colors
        plane : str, optional
            Plane that it is wanted to be represented:
            Accepted options are 'xz' and 'yz'.
            Default value is 'xz'. 
        color : str, optional
            Color of the points. 
            A full list of color names can be found at:
            https://matplotlib.org//gallery/color/named_colors
            Default is 'darkgreen'. 
        filename : str, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None

        """
        if vis_args is None:
            vis_args = {
                "background": "#EEEEEE",
                "tail": "black",
                "nose": "black",
                "body": "black",
                "fins": "black",
                "motor": "black",
                "buttons": "black",
                "line_width": 1.0,
            }

        ax, _, _ = self.rocket.plots._rocket_shape_plot(vis_args, plane)
        self._plot_plate_rocket(ax, plane, marker, color)

        plt.title(f"Plate representation")
        plt.xlim()
        plt.ylim([-self.rocket.radius * 4, self.rocket.radius * 6])
        plt.xlabel("Position (m)")
        plt.ylabel("Radius (m)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        show_or_save_plot(filename)

    def _plot_plate_rocket(
        self, ax, plane: str = "xz", color: str = "darkgreen"
    ) -> None:
        """'
        Plots the plate on the rocket:

        Parameters
        ----------
        plane : str, optional
            Plane that it is wanted to be represented:
            Accepted options are 'xz' and 'yz'
            Default value is 'xz'.
        color : str, optional
            Color of the points.
            A full list of color names can be found at:
            https://matplotlib.org//gallery/color/named_colors
            Default is 'darkgreen'.

        Returns
        -------
        None
        """
        x, y, z = zip(*self.plate.points) # in the bacs frame z tail to nose

        # change nose to tail with nose origin
        nose_to_cdm_dist = self.rocket.center_of_dry_mass_position - self.rocket._nose_tip_from_ucs
        z = nose_to_cdm_dist - np.array(z)
        x = -np.array(x)

        if plane == "xz":
            r = x
        elif plane == "yz":
            r = y
        else:
            raise ValueError("Plane value can only be xz or yz")

        # Connect points with a solid line
        if isinstance(self.plate.dimensions, float) and self.plate.dimensions > 0.1:
            ax.plot(z, r, color=color, linewidth=3, linestyle="-", label=self.plate.name, zorder=1)
        else:
            ax.plot(z, r, color=color, linewidth=2, linestyle="-", label=self.plate.name, zorder=1)

    def all(self) -> None:
        """
        Prints out all graphs available about the Plate. It simply calls
        all the other plotter methods in this class with all the
        default parameters.

        Returns
        -------
        None
        """
        self.draw_3D()
        self.draw()
