class _WirePrints:
    """Class that holds print methods for the Wire class.

    Attributes
    ----------
    _WirePrints.wire : Wire
        Wire object that will be used for the prints.
    """

    def __init__(self, wire) -> None:
        """Initializes _WirePrints class.

        Parameters
        ----------
        wire: Wire
            Instance of the Wire class.

        Returns
        -------
        None
        """
        self.wire = wire

    def current(self) -> None:
        """Prints the current flowing through the wire.

        Returns
        -------
        None
        """
        print(f"Current: {self.wire.current} A")

    def edges(self) -> None:
        """Prints the wire edges coordinates in the body axis coordinate system.

        Returns
        -------
        None
        """
        print(
            f"Edge A is {self.wire._wire_edges_bacs[0]}, Edge B is {self.wire._wire_edges_bacs[1]} in the body axis coordinate system"
        )

    def wire_type(self) -> None:
        """Prints the wire type and, if applicable, the ignition wire function.

        Returns
        -------
        None
        """
        if self.wire.wire_type == "communications":
            print("Wire type: communications type")
        elif self.wire.wire_type == "ignition":
            print("Wire type: ignition type")
            print(f"Ignition wire function: {self.wire.ignition_wire_function}")

    def magnetic_field_vectors(self) -> None:
        """Prints the magnetic field vectors stored in the wire for all recorded positions.

        Returns
        -------
        None
        """
        for position, b_field in self.wire.magnetic_field.items():
            if b_field is None:
                print(
                    f"Magnetic field vector at position {position} hasn't been measured yet."
                )
            else:
                print(
                    f"Magnetic field vector at position {position} due to the wire is {b_field}"
                )

    def all(self):
        """Prints all print methods about the Wire.

        Returns
        -------
        None
        """
        print(f"\n{self.wire.name} information: ")
        self.wire_type()
        self.current()
        self.magnetic_field_vectors()
        self.edges()
