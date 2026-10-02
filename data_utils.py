"""Selected data-preparation utilities."""
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def chronological_split(array, train_ratio=0.60, val_ratio=0.20):
    n = len(array)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return array[:train_end], array[train_end:val_end], array[val_end:]

def build_continuity_mask(datetimes, input_window=48, horizon=24):
    dt = np.asarray(datetimes, dtype="datetime64[m]")
    delta = (dt[1:] - dt[:-1]).astype("timedelta64[m]").astype(int)
    continuous = delta == 60
    valid = np.zeros(len(dt), dtype=bool)
    for i in range(input_window, len(dt) - horizon + 1):
        start, end = i - input_window + 1, i + horizon - 1
        if continuous[start-1:end].all():
            valid[i] = True
    return valid

def make_windows(features, target, input_window=48, horizon=24, valid_mask=None, offset=0):
    X, y = [], []
    for i in range(input_window, len(features) - horizon + 1):
        if valid_mask is not None and not valid_mask[offset + i]:
            continue
        X.append(features[i-input_window:i])
        y.append(target[i:i+horizon].reshape(-1))
    return np.asarray(X, np.float32), np.asarray(y, np.float32)

def fit_train_scalers(x_train, y_train):
    return MinMaxScaler((0,1)).fit(x_train), MinMaxScaler((0,1)).fit(y_train)
