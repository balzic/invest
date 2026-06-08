import stock_data as sd
import math
import numpy as np

import utility


def log_return(x_vals: list[float], y_vals: list[float]) -> list[float]:
    return [math.log(x / y) for x, y in zip(x_vals, y_vals)]


def yearly_returns(chunks: list[list[float]]) -> list[float]:
    return [sum(chunk) for chunk in chunks]

def mean(x: list[float]) -> float:
    return sum(x) / len(x) if x else 0.0

def efficient_frontier(
    erate: float,
    covariance: list[list[float]],
    r1: float,
    r2: float,
    expected_returns: list[float],
    pins: int = 10000,
) -> tuple[list[float], list[float]]:
    covariance_np = np.asarray(covariance, dtype=float)
    returns_np = np.asarray(expected_returns, dtype=float)

    excess_return1 = returns_np - r1
    excess_return2 = returns_np - r2

    try:
        z1 = np.linalg.solve(covariance_np, np.column_stack((excess_return1, excess_return2)))
    except np.linalg.LinAlgError as exc:
        raise ValueError("Covariance matrix is singular or ill-conditioned.") from exc

    a = np.array([[1.0, r1], [1.0, r2]], dtype=float)
    c1 = np.linalg.solve(a, z1.T)

    if pins < 2:
        pins = 2
    rates = np.linspace(-20.0, erate, pins)
    new_a = np.ones((pins, 2), dtype=float)
    new_a[:, 1] = rates

    z = new_a @ c1
    row_sums = np.sum(z, axis=1)
    row_sums_list = row_sums.tolist()
    if any(abs(v) < 1e-12 for v in row_sums_list):
        raise ValueError("Encountered zero row sum while normalizing portfolio weights.")

    x = z / row_sums[:, np.newaxis]
    exp_return_p = x @ returns_np
    variance_p = np.array([float(row @ covariance_np @ row) for row in x], dtype=float)

    return exp_return_p.tolist(), variance_p.tolist()


def efficient_frontier_sbl(
    covariance: list[list[float]],
    r: float,
    expected_returns: list[float],
    points: int = 100,
) -> tuple[float, list[float], list[float], float, float, list[float]]:
    covariance_np = np.asarray(covariance, dtype=float)
    returns_np = np.asarray(expected_returns, dtype=float)
    excess_return = returns_np - r

    try:
        z = np.linalg.solve(covariance_np, excess_return)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Covariance matrix is singular or ill-conditioned.") from exc

    z_sum = float(np.sum(z))
    if abs(z_sum) < 1e-12:
        raise ValueError("Sum of solved weights is zero; cannot normalize portfolio weights.")
    weights = z / z_sum

    exp_return_p = float(weights.T @ returns_np)
    variance_p = float(weights.T @ covariance_np @ weights)

    slope = (exp_return_p - r) / math.sqrt(variance_p)

    if points < 2:
        points = 2
    x = np.linspace(0.0, 0.9, points)
    y = slope * x + r

    return slope, x.tolist(), y.tolist(), exp_return_p, variance_p, weights.tolist()

def compute_returns(data: sd.StockData) -> tuple[list[float], list[float]]:
    years = [d.year for d in data.date]
    counts: list[int] = []
    if years:
        current_year = years[0]
        current_count = 0
        for year in years:
            if year == current_year:
                current_count += 1
            else:
                counts.append(current_count)
                current_year = year
                current_count = 1
        counts.append(current_count)

    if counts:
        counts[0] -= 1

    x_x = data.adjclose[1:]
    x_y = data.adjclose[:-1]
    r = log_return(x_x, x_y)

    split: list[list[float]] = []
    start = 0
    for count in counts:
        end = start + count
        split.append(r[start:end])
        start = end

    yearly_r = yearly_returns(split)
    return r, yearly_r


def load_data(funds: list[str], last_year: int, ten_years_ago: int) -> dict[str, sd.StockData]:
    data: dict[str, sd.StockData] = dict()
    if not utility.data_exists():
        for fund in funds:
            data[fund] = sd.hist_stock_data_period(f'0101{ten_years_ago}', f'0101{last_year}', '1d', 'history', fund)
        utility.save_data(data)
    else:
        data.update(utility.load_data())
    return data


def validate_data(data: dict[str, sd.StockData], funds: list[str], current_year: int, ten_years_ago: int):
    for fund in funds:
        if data[fund].from_date.year != ten_years_ago:
            print(f"Expected from_date year {ten_years_ago}, got {data[fund].from_date.year} for fund {fund}")
            exit(1)
        if data[fund].to_date.year != current_year - 1:
            print(f"Expected to_date year {current_year - 1}, got {data[fund].to_date.year} for fund {fund}")
            exit(1)


def main():
    print("Loading data for the following funds:")
    funds = ['SPY', 'FOCPX', 'IWM', 'DIA', 'XLY', 'XLF', 'XLK', 'XLE']
    print(funds)
    current_year = 2022
    ten_years_ago = current_year - 10
    print("Year range:", ten_years_ago, "-", current_year)
    N = len(funds)
    W = 10  # number of windows - 1
    data = load_data(funds, current_year, ten_years_ago)
    validate_data(data, funds, current_year, ten_years_ago)
    print("Data loaded and validated successfully.")

    _, year_market = compute_returns(data['SPY'])
    SPY_return = mean(year_market)
    print(f"SPY average yearly return: {SPY_return:.4f}")

    non_SPY_funds = [fund for fund in funds if fund != 'SPY']
    returns: dict[str, utility.Returns] = dict()
    for fund in non_SPY_funds:
        _, year_fund = compute_returns(data[fund])
        fund_return = mean(year_fund)
        returns[fund] = utility.Returns(yearly=year_fund, mean=fund_return)
        print(year_fund)
        print(f"{fund} average yearly return: {fund_return:.4f}")

if __name__ == "__main__":
    main()