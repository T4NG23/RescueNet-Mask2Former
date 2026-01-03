DATASET_REGISTRY = {}

def register_dataset(name):
    def deco(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return deco
