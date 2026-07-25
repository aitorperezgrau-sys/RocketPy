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

    def __init__(self, plate):
        '''
        Parameters: 
        -----------
        plate: Plate
            Plate instance

        '''
        self.plate = plate

    def points(self, color = 'blue', marker = 'o', filename = None):
        '''
        plots the scatter plot of the plate formed by the points
        used to model the magnetic distortion. 
        '''
        # Unpack columns into separate x, y, and z components
        x, y, z = zip(*self.plate.points)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        # plot individual points
        ax.scatter(x, y, z, color = color, marker = marker, label = self.plate.name)

        # Labels & formatting
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend()

        print(f'\n{self.plate.name} representation: ')
        show_or_save_plot(filename)


    def all(self):
        '''
        Prints out all graphs available about the Plate. It simply calls
        all the other plotter methods in this class.

        Returns:
        --------
        None
        '''
        self.points()