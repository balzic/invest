import pickle

import typing
import numpy as np
import stock_data as sd
import os
from dataclasses import dataclass
DATAFILE = 'data.pkl'

@dataclass
class Returns:
    yearly: list[float]
    mean: float

def save_data(data: dict[str, sd.StockData]):
    pickle.dump(data, open(DATAFILE, 'wb'))

def load_data() -> dict[str, sd.StockData]:
    return pickle.load(open(DATAFILE, 'rb'))

def data_exists() -> bool:
    return os.path.exists(DATAFILE)


@dataclass
class PortfolioResult:
    balanced_return: np.ndarray[np.float64, typing.Any]
    optimum_return: np.ndarray[np.float64, typing.Any]



def compute_covariance_matrix(returns_matrix: list[list[float]]) -> np.ndarray[np.float64, typing.Any]:
    returns_np = np.asarray(returns_matrix, dtype=float)
    # Each row is one asset over time; compute asset-by-asset covariance.
    covariance_matrix = np.cov(returns_np, rowvar=True)
    return covariance_matrix


def show_weights(weights: np.ndarray[np.float64, typing.Any], non_SPY_funds: list[str], normalized: bool = False):
    for i, fund in enumerate(non_SPY_funds):
        if weights[i] > 0 or not normalized:
            print(f"Weight for {fund}: {weights[i]:.4f}")