import os
import numpy as np


def make_filename(boundary_type, b, sigma, dt, n_parts, extension=".npy"):

    filename = (f"{boundary_type}" f"_b-{b}" f"_s-{sigma}" f"_dt-{dt}" f"_N-{n_parts}" 
                f"{extension}")

    return filename.replace(" ", "")


def save_solution(X, filename, data_path="./data"):
    os.makedirs(data_path, exist_ok=True)

    filepath = os.path.join(data_path, filename)
    np.save(filepath, X)

    return filepath


def load_solution(filename, data_path="./data"):
    filepath = os.path.join(data_path, filename)

    return np.load(filepath)