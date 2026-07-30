import math as m

import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes

from rocketpy.plots.plot_helpers import show_or_save_plot


class _WirePlots:
    """
    Class that holds plot methods for the Wire class.

    Attributes
    ----------
    _WirePlots.wire: Wire
        Wire object that will be used for the plots.
    """

    def __init__(self, wire, rocket):
        """
        Parameters
        ----------
        wire: Wire
            Wire instance.
        rocket: Rocket
            Rocket instance to which the wire
            is attached.
        """
        self.wire = wire
        self.rocket = rocket

    def draw(
        self,
        vis_args: dict | None = None,
        plane: str = "xz",
        color: str = "salmon",
        marker: str = "o",
        linestyle: str = "-",
        edges_names: bool = True,
        filename: str | None = None,
    ) -> None:
        """
        Plots the wire and the rocket together. 

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
        plane: str, optional
            Plane that it is wanted to be represented:
            Accepted options are 'xz' and 'yz'
            Default value is 'xz'. 
        color : str, optional
            Color of the points. 
            A full list of color names can be found at:
            https://matplotlib.org//gallery/color/named_colors
            Default is 'salmon'. 
        marker : str, optional
            shape of the points from which the plate is formed. 
            A full list of markers can be found at: 
            https://matplotlib.org/stable/api/markers_api.html
            Default is 'o'. 
        linestyle : str, optional
            type of the line that will represent the wire. 
            A full list of linestyles can be found at: 
            https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
            Default is '-'. 
        edges_names : bool, optional
            boolean defining whether the names of the edges are displayed.
            If False, they will not be displayed
            If True, the name Edge A, and Edge B will be show.
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
        self._draw_wires(ax, plane, color, marker, linestyle, edges_names)

        plt.title(f"{self.wire.name} representation")
        plt.xlim()
        plt.ylim([-self.rocket.radius * 4, self.rocket.radius * 6])
        plt.xlabel("Position (m)")
        plt.ylabel("Radius (m)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        show_or_save_plot(filename)

    def _draw_wires(
        self,
        ax: Axes,
        plane: str = "xz",
        color: str = "salmon",
        marker: str = "o",
        linestyle: str = "-",
        edges_names: bool = True,
    ) -> None:
        """
        Plot the edges and the wire in the rocket on the axes 'ax'.

        Parameters
        ----------
        ax: Axes
            matplotlib instance in which the wire will be plotted.
        color : str, optional
            Color of the points.
            A full list of color names can be found at:
            https://matplotlib.org//gallery/color/named_colors
            Default is 'salmon'.
        marker : str, optional
            shape of the points from which the plate is formed.
            A full list of markers can be found at:
            https://matplotlib.org/stable/api/markers_api.html
            Default is 'o'.
        linestyle : str, optional
            type of the line that will represent the wire.
            A full list of linestyles can be found at:
            https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
            Default is '-'.
        edges_names : bool, optional
            boolean defining whether the names of the edges are displayed.
            If False, they will not be displayed
            If True, the name Edge A, and Edge B will be show.

        Returns
        -------
        None
        """
        # change nose to tail with nose origin
        edge_a_x = self.wire._wire_edges_from_cdm[0][0] * self.rocket._csys
        edge_b_x = self.wire._wire_edges_from_cdm[1][0] * self.rocket._csys

        edge_a_y = self.wire._wire_edges_from_cdm[0][1]
        edge_b_y = self.wire._wire_edges_from_cdm[1][1]

        edge_a_z = self.rocket.center_of_dry_mass_position + (
            self.wire._wire_edges_from_cdm[0][2] * self.rocket._csys
        )
        edge_b_z = self.rocket.center_of_dry_mass_position + (
            self.wire._wire_edges_from_cdm[1][2] * self.rocket._csys
        )
        if plane == "xz":
            r_a = edge_a_x
            r_b = edge_b_x

        elif plane == "yz":
            r_a = edge_a_y
            r_b = edge_b_y

        z = [edge_a_z, edge_b_z]
        r = [r_a, r_b]

        # plot lines connecting edges
        ax.plot(z, r, color=color, linestyle=linestyle, label=self.wire.name)

        if edges_names == True:
            ax.scatter(z, r, marker=marker, color=color, zorder=5, label="Wire edges")

            # Add text labels
            ax.annotate(
                "Edge A",
                (edge_a_z, r_a),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=9,
            )
            ax.annotate(
                "Edge B",
                (edge_b_z, r_b),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=9,
            )
        elif edges_names == False:
            ax.scatter(z, r, marker=marker, color=color, zorder=5)

    def all(self) -> None:
        """
        Prints out all graphs available about the Wire. It simply calls
        all the other plotter methods in this class with all the
        default parameters.

        Returns
        -------
        None
        """
        print(f"\n{self.wire.name} representation: ")
        self.draw()
