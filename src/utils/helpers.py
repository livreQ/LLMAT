import dill
import pandas as pd
import numpy as np
import torch
import time


def dill_save(obj, filepath):
    with open(filepath, "wb") as f:
        dill.dump(obj, f)


def dill_load(filepath):
    with open(filepath, "rb") as f:
        obj = dill.load(f)
    return obj


def pop_df(df: pd.DataFrame, idx: int):
    row = df.loc[idx]
    df.drop(idx, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return row


def y_unscale(y, scaler):
    return scaler.inverse_transform(np.array(y).reshape(-1, 1)).item()


def y_transform(y, maximize: bool):
    """
    Negating y if maximize=False.
    Rationale: we define the BayesOpt as a maximization problem.
    """
    return y if maximize else -y


def check_device():
    """
    Check the available device and return it.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def trace_times(start=None, end=None, device="cuda"):
    """
    Trace the time taken for each step in the optimization process.
    """
    if device == torch.device("cuda"):
        if start is None and end is None:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            return start, end
        elif start is not None and end is not None:
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / 1000
        else:
            raise ValueError("Invalid arguments for trace_times. Use start=None and end=None to start timing, or provide both start and end to get elapsed time.")
    elif device == torch.device("mps"):
        if start is None and end is None:
            start = torch.mps.Event(enable_timing=True)
            end = torch.mps.Event(enable_timing=True)
            torch.mps.synchronize()
            start.record()
            return start, end
        elif start is not None and end is not None:
            #
            end.record()
            torch.mps.synchronize()
            return start.elapsed_time(end) / 1000
        else:
            raise ValueError("Invalid arguments for trace_times. Use start=None and end=None to start timing, or provide both start and end to get elapsed time.")
    else:
        if start is None and end is None:
            start = time.time()
            return start, None
        elif start is not None and end is None:
            end = time.time()
            return end - start
        else:
            raise ValueError("Invalid arguments for trace_times. Use start=None and end=None to start timing, or provide both start and end to get elapsed time.")
