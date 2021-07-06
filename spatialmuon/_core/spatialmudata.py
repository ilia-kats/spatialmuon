class SpatialMuData:
    def __init__(self, modalities):
        self.mod = modalities

    def __getitem__(self, key):
        return self.mod[key]
