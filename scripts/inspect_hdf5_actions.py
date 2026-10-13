import h5py
import numpy as np

with h5py.File("/Users/skywalker/Downloads/dataset_annotated_gr1_gr00t.hdf5", "r") as f:
    for demo_name, demo in f["data"].items():
        for key in ("actions", "processed_actions"):
            if key not in demo:
                continue
            x = np.asarray(demo[key][:], dtype=np.float64)
            print(demo_name, key, "shape:", x.shape)
            print("  min :", np.round(x.min(axis=0), 6))
            print("  max :", np.round(x.max(axis=0), 6))
            print("  std :", np.round(x.std(axis=0), 6))
            print("  changed:", np.flatnonzero(np.ptp(x, axis=0) > 1e-6))
