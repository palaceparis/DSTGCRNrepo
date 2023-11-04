import numpy as np
import pandas as pd


def load_st_dataset(dataset):
    # output B, N, D
    if dataset == "CARBON":
        data = pd.read_csv(
            "data/interim/emissionsWithoutHeader.csv", header=None
        ).values
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print(
        "Load %s Dataset shaped: " % dataset,
        data.shape,
        data.max(),
        data.min(),
        data.mean(),
        np.median(data),
    )
    return data
