class _PlatePrints():
    ''''
    Class that holds prints methods for Plate class.

    Attributes
    ----------
    _PlatePrints.plate : Plate
        Plate object that will be used for the prints.
    '''

    def __init__(self, plate):
        '''
        Parameters: 
        -----------
        plate: Plate
            Plate instance

        '''
        self.plate = plate


    def len_points_print(self):
        len_points = len(self.plate.points)
        print(f'Number of points that form the plate: {len_points}')
    

    def relative_magnetic_print(self):
        print(f'Relative magnetic permeability: {self.plate.relative_magnetic_permeability}')


    def magnetic_distortion_matrix(self):
        for key in self.plate._magnetic_distortion_matrixes:
            print(f'Magnetic distortion matrix at position: {key} is {self.plate._magnetic_distortion_matrixes[key]}')


    def all(self):
        print(f'\n{self.plate.name} information: ')
        self.len_points_print()
        self.relative_magnetic_print()
        self.magnetic_distortion_matrix()