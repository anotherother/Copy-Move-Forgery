class ObjectDescription:
    def __init__(self,description,key_coordinates,reference_coordinates=None):
        self.description=description
        self.key_coordinates=key_coordinates
        self.reference_coordinates=reference_coordinates
    def getDescription(self):
        return self.description