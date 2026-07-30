class _WirePrints:
    """
    Class that holds prints methods for Wire class.

    Attributes
    ----------
    _WirePrints.plate : Wire
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
            f"Edge A is {self.wire._wire_edges_from_cso[0]}, Edge B is {self.wire._wire_edges_from_cso[1]}"
        )

    def wire_type(self):
        if self.wire.wire_type == "communications":
            print("Wire type: communications type")
        elif self.wire.wire_type == "ignition":
            print("Wire type: ignition type")
            print(f"Ignition wire function: {self.wire.ignition_wire_function}")

    def magnetic_field_vector(self):
        for key in self.wire.magnetic_field:
            print(
                f"Magnetic field vector at position: {key} due to the wire is {self.wire.magnetic_field[key]}"
            )

    def all(self):
        print(f"\n{self.wire.name} information: ")
        self.wire_type()
        self.current()
        self.magnetic_field_vector()
        self.edges()
