import pickle
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