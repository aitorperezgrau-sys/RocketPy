import matplotlib.pyplot as plt
from rocketpy.plots.plot_helpers import show_or_save_plot


class _PlatePlots():
    '''
    Class that holds plot methods for the Plate class

    Attributes: 
    -------------
    plate: Plate
        Plate object that will be used for the plots
    
    '''

    def __init__(self, plate, rocket):
        '''
        Parameters: 
        -----------
        plate: Plate
            Plate instance

        rocket: Rocket
            Rocket instance to which the plate
            is attached

        '''
        self.plate = plate
        self.rocket = rocket

    def draw_3D(self, color = 'blue', marker = 'o', filename = None):
        '''
        plots the scatter plot of the plate formed by the points
        used to model the magnetic distortion. 

        color : str, optional
            Color of the points. 
            A full list of color names can be found at:
            https://matplotlib.org//gallery/color/named_colors
            Default is 'blue'. 

            
        marker: str, optional
            shape of the points from which the plate is formed. 
            A full list of markers can be found at: 
            https://matplotlib.org/stable/api/markers_api.html
            Default is 'o'. 


        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).
        '''
        # Unpack columns into separate x, y, and z components
        x, y, z = zip(*self.plate.points)

        fig = plt.figure(figsize=(8, 6))
        ax= fig.add_subplot(111, projection="3d")

        # plot individual points
        ax.scatter(x, y, z, color = color, marker = marker, label = self.plate.name)

        # Labels & formatting
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend()

        print(f'\n{self.plate.name} representation: ')
        show_or_save_plot(filename)


    def draw(self,
             vis_args = None,
             plane = 'xz',
             color = 'green',
             marker = 'o',
             filename = None):
        '''
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
            Accepted options are 'xz' and 'yz'
            Default value is 'xz'

        color : str, optional
            Color of the points. 
            A full list of color names can be found at:
            https://matplotlib.org//gallery/color/named_colors
            Default is 'green'. 
        
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).
        '''


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

        plt.title(f'Plate representation')
        plt.xlim()
        plt.ylim([-self.rocket.radius * 4, self.rocket.radius * 6])
        plt.xlabel("Position (m)")
        plt.ylabel("Radius (m)")    
        ax.legend()

        plt.tight_layout()
        show_or_save_plot(filename)
    

    def _plot_plate_rocket(self,
                            ax,
                            plane = 'xz',
                            color = 'green'):
        ''''
        This function plots the plate on the rocket:
        '''
        x, y, z = zip(*self.plate.points)

        if plane == 'xz':
            r = y
            z = z
        elif plane == 'yz':
            r = x
            z = z
        else: 
            raise ValueError('Plane value can only be xz or yz')

        # Connect points with a solid line
        ax.plot(z, r, color=color, linewidth=2, linestyle="-", label=self.plate.name)




    def all(self):
        '''
        Prints out all graphs available about the Plate. It simply calls
        all the other plotter methods in this class.

        Returns:
        --------
        None
        '''
        self.draw_3D()