import os, json
import numpy as np

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_json(data, filename):
    def convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.float32, np.float64, np.int32, np.int64)):
            return o.item()
        raise TypeError(f"Object of type {type(o)} not JSON serializable")
    ensure_dir(os.path.dirname(filename))
    with open(filename, "w") as f:
        json.dump(data, f, indent=4, default=convert)