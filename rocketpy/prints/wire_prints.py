class _WirePrints:
    """Class that holds print methods for the Wire class.

    Attributes
    ----------
    _WirePrints.wire : Wire
        Wire object that will be used for the prints.
    """

    def __init__(self, wire):
        """
        Parameters
        ----------
        wire: Wire
            Wire instance.
        """
        self.wire = wire

    def current(self):
        print(f"Current: {self.wire.current} A")

    def edges(self):
        print(
            f"Edge A is {self.wire._wire_edges_bacs[0]}, Edge B is {self.wire._wire_edges_bacs[1]} in the body axis coordinate system"
        )

    def wire_type(self):
        if self.wire.wire_type == "communications":
            print("Wire type: communications type")
        elif self.wire.wire_type == "ignition":
            print("Wire type: ignition type")
            print(f"Ignition wire function: {self.wire.ignition_wire_function}")

    def magnetic_field_vectors(self) -> None:
        """Prints the magnetic field vectors stored in the wire for all recorded positions."""
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
