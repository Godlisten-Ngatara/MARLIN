import pandas as pd
import numpy as np
import torch


def create_causal_sequences(data, window_size, features=['prev_true', 'EIR_true']):
    xs, ys = [], []
    feature_data = data[features].to_numpy()

    has_target = 'incall' in data.columns  # check if incidence is in the data

    for i in range(len(data)):
        if i < window_size:
            # Pad with the first row
            pad_size = window_size - i
            first_row = feature_data[0].reshape(1, -1)
            padding = np.tile(first_row, (pad_size, 1))
            actual = feature_data[0:i+1]  # from 0 to i (inclusive)
            x_values = np.concatenate((padding, actual), axis=0)
        else:
            x_values = feature_data[i - window_size:i + 1]

        xs.append(x_values)

        # Only append target if it's available
        if has_target:
            y = data.iloc[i]['incall']
            ys.append([y])

    xs = torch.tensor(np.array(xs), dtype=torch.float32)  

    if has_target:
        ys = torch.tensor(np.array(ys), dtype=torch.float32)
        return xs, ys
    else:
        return xs, None
    

def create_sequences_assymetric(data, window_size):
    xs, ys = [], []
    has_targets = all(col in data.columns for col in ['EIR_true'])  #, 'incall'# Check if target columns exist
    half_window_size = int(np.ceil(window_size / 2))

    for i in range(len(data)-half_window_size):
        # if i + half_window_size >= len(data):
        #     break  # Not enough future steps
        if i < window_size:
            # Pad beginning of sequence
            pad_size = window_size - i
            first_values = data.iloc[0][['prev_true']].values
            replicated_values = np.tile(first_values, (pad_size, 1))
            x_values = np.concatenate((replicated_values, data.iloc[0:i + half_window_size + 1][['prev_true']].values), axis=0)
        else:
            x_values = data.iloc[i - window_size:i + half_window_size + 1][['prev_true']].values

        xs.append(x_values.flatten())

        if has_targets:
            y = data.iloc[i][['EIR_true']].values#, 'incall'
            ys.append(y)

    xs = np.array(xs, dtype=np.float32)

    if has_targets:
        ys = np.array(ys, dtype=np.float32)
        return torch.tensor(xs), torch.tensor(ys)
    else:
        return torch.tensor(xs), None  # Return None for ys if targets are missing