import runpy, inspect, importlib
from nh_datasets.registry import DATASET_REGISTRY

# ensure modules are imported so registration runs
importlib.import_module("nh_datasets.floodnet")
importlib.import_module("nh_datasets.crarsar")
importlib.import_module("nh_datasets.rescuenet")
importlib.import_module("nh_datasets.rescuenet_optimized")

def filter_kwargs_for_init(cls, kwargs):
    sig = inspect.signature(cls.__init__)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}

def build_dataset_from_py(cfg_path: str, **overrides):
    cfg = runpy.run_path(cfg_path)
    name = cfg.get("DATASET_NAME")
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}")

    cls = DATASET_REGISTRY[name]
    kwargs = dict(cfg.get("DATASET_KWARGS", {}))
    kwargs.update(overrides)
    kwargs = filter_kwargs_for_init(cls, kwargs)
    return cls(**kwargs)

# if __name__ == "__main__":
#     ds = build_dataset_from_py("config.py", split="val")
#     print(type(ds))
