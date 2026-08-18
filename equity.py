from dataclasses import dataclass
import sys
import typing

from long_only import backtest_only_long
import stock_data as sd
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats # type: ignore
from utility import PortfolioResult, compute_covariance_matrix, show_weights
import utility

def log_return(x_vals: list[float], y_vals: list[float]) -> list[float]:
    return [math.log(x / y) for x, y in zip(x_vals, y_vals)]


def yearly_returns(chunks: list[list[float]]) -> list[float]:
    return [sum(chunk) for chunk in chunks]

def mean(x: list[float]) -> float:
    return sum(x) / len(x) if x else 0.0

def efficient_frontier(
    erate: float,
    covariance: np.ndarray[np.float64, typing.Any],
    r1: float,
    r2: float,
    expected_returns: list[float],
    pins: int = 10000,
) -> tuple[list[float], list[float]]:
    returns_np = np.asarray(expected_returns, dtype=float)
    if covariance.shape[0] != covariance.shape[1] or covariance.shape[0] != returns_np.size:
        raise ValueError(
            f"Covariance matrix shape {covariance.shape} must be square and match expected returns length {returns_np.size}."
        )
    excess_return1 = returns_np - r1
    excess_return2 = returns_np - r2
    excess_returns = np.array([excess_return1, excess_return2], dtype=float).T
    try:
        z1 = np.linalg.solve(covariance, excess_returns)
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
    variance_p = np.zeros(pins, dtype=float)
    for i in range(pins):
        variance_p[i] = float(x[i] @ covariance @ x[i])
    return exp_return_p.tolist(), variance_p.tolist()


def efficient_frontier_sbl(
    covariance: np.ndarray[np.float64, typing.Any],
    r: float,
    expected_returns: list[float],
    points: int = 100,
) -> tuple[float, list[float], list[float], float, float, list[float]]:
    returns_np = np.asarray(expected_returns, dtype=float)
    excess_return = returns_np - r

    try:
        z = np.linalg.solve(covariance, excess_return.T)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Covariance matrix is singular or ill-conditioned.") from exc

    z_sum = float(np.sum(z))
    if abs(z_sum) < 1e-12:
        raise ValueError("Sum of solved weights is zero; cannot normalize portfolio weights.")
    weights = z / z_sum

    exp_return_p = float(weights.T @ returns_np.T)
    variance_p = float(weights.T @ covariance @ weights)

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


def load_data(funds: list[str], last_year: int, start_year: int) -> dict[str, sd.StockData]:
    data: dict[str, sd.StockData] = dict()
    if not utility.data_exists():
        for fund in funds:
            data[fund] = sd.hist_stock_data_period(f'0101{start_year}', f'0101{last_year}', '1mo', 'history', fund)
        utility.save_data(data)
    else:
        data.update(utility.load_data())
    return data


def validate_data(data: dict[str, sd.StockData], funds: list[str], current_year: int, start_year: int):
    for fund in funds:
        if data[fund].from_date.year != start_year:
            print(f"Expected from_date year {start_year}, got {data[fund].from_date.year} for fund {fund}")
            exit(1)
        if data[fund].to_date.year != current_year - 1:
            print(f"Expected to_date year {current_year - 1}, got {data[fund].to_date.year} for fund {fund}")
            exit(1)

def compute_correlation_matrix(returns_matrix: list[list[float]]) -> np.ndarray[np.float64, typing.Any]:
    returns_np = np.asarray(returns_matrix, dtype=float)
    correlation_matrix = np.corrcoef(returns_np, rowvar=True)
    return correlation_matrix

def plot_efficient_frontier(W: int, returns: dict[str, utility.Returns], non_SPY_funds: list[str]):
    plt.figure() # type: ignore
    for i in range(W+1):
        slice_start = i
        slice_end = i + W
        yearly_returns = [mean(returns[fund].yearly[slice_start:slice_end]) for fund in non_SPY_funds]
        returns_matrix: list[list[float]] = [returns[fund].yearly[slice_start:slice_end] for fund in non_SPY_funds]
        covariance_matrix = compute_covariance_matrix(returns_matrix)
        #correlation_matrix = compute_correlation_matrix(returns_matrix)
        exp_return_p,variance_p = efficient_frontier(10,covariance_matrix,0.05,0.02,yearly_returns)
        plt.plot(np.sqrt(variance_p), exp_return_p, label=f'Window {i + 1}') # type: ignore

        I = np.argmin(np.sqrt(variance_p))
        plt.text(np.sqrt(variance_p[I]), exp_return_p[I],f"{i+1}") # type: ignore
        ax = plt.gca()
        ax.set_xlim(left=0.0, right=0.15)
        ax.set_ylim(bottom=-0.2, top=0.6)
        plt.xlabel("Standard Deviation") # type: ignore
        plt.ylabel("Expected Return") # type: ignore
    plt.show() # type: ignore

@dataclass
class TobinsSeparation:
    slope: list[float]
    exp_return: list[float]
    variance: list[float]
    weights: np.ndarray[np.float64, typing.Any]

def plot_tobins_separation(W: int, returns: dict[str, utility.Returns], non_SPY_funds: list[str], rate: float = 0.01) -> TobinsSeparation:
    plt.figure() # type: ignore
    ts = TobinsSeparation(slope=[], exp_return=[], variance=[], weights=np.zeros((W+1, len(non_SPY_funds)), dtype=float))
    for i in range(W+1):
        slice_start = i
        slice_end = i + W
        yearly_returns = [mean(returns[fund].yearly[slice_start:slice_end]) for fund in non_SPY_funds]
        returns_matrix: list[list[float]] = [returns[fund].yearly[slice_start:slice_end] for fund in non_SPY_funds]
        covariance_matrix = utility.compute_covariance_matrix(returns_matrix)
        slope, x, y, exp_return_p, variance_p, weights = efficient_frontier_sbl(covariance_matrix, rate, yearly_returns)
        ts.slope.append(slope)
        ts.exp_return.append(exp_return_p)
        ts.variance.append(variance_p)
        ts.weights[i, :] = np.array(weights, dtype=float)
        plt.plot(x, y, label=f'Window {i + 1}') # type: ignore
        plt.text(x=0.5+i/100,y=(0.5+i/100)*slope+rate,s=""+str(i+1)) # type: ignore
        plt.scatter(np.sqrt(variance_p), exp_return_p, color='red') # type: ignore
        print(weights)
        plt.xlabel("Standard Deviation") # type: ignore
        plt.ylabel("Expected Return") # type: ignore
    plt.show() # type: ignore
    return ts

def compute_portfolio_return(W: int, ts: TobinsSeparation, rate: float= 0.01, required_return: float = 0.15) -> np.ndarray[np.float64, typing.Any]:
    R_pEnd = 1000
    turnover = np.zeros(W, dtype=float)
    allocation = np.zeros((W+1), dtype=float)
    constant_return = required_return
    plt.figure() # type: ignore
    for i in range(W+1):
        R_pStart = R_pEnd
        A = min(1.0, (constant_return - rate) / (ts.exp_return[i] - rate))
        allocation[i] = A
        risk = (constant_return - rate) / ts.slope[i]
        x = np.linspace(0.0, 0.9, 100)
        plt.plot(x, ts.slope[i] * x + rate, label=f'Window {i + 1}') # type: ignore
        plt.scatter(risk, constant_return, color='red') # type: ignore
        print("Allocation for window", i+1, ":", A)
        R_pEnd = R_pStart * (1 + ((1-A)*rate+A*ts.exp_return[i]))
        if i > 0:
            turnover[i-1] = abs(allocation[i] - allocation[i-1])
        plt.text(x=0.2+i/100,y=(0.2+i/100)*ts.slope[i]+rate,s="" + str(i+1)) # type: ignore
    plt.show() # type: ignore
    print("Final portfolio value after", W+1, "windows:", R_pEnd)
    print("Total turnover over all windows:", np.mean(turnover))
    return allocation

def normalize_weights(weights: np.ndarray[np.float64, typing.Any]) -> np.ndarray[np.float64, typing.Any]:
    weights_non_zero = weights.copy()
    weights_non_zero[weights_non_zero < 0] = 0
    row_sums = np.sum(weights_non_zero)
    normalized_weights = weights_non_zero / row_sums
    return normalized_weights

# this function performs a backtest, where lending and borrowing is not allowed
# to find the return, when borrowing and lending is allowed, simpy remove
# usage of normalize_weights
def backtest(
    W: int,
    returns: dict[str, utility.Returns],
    non_SPY_funds: list[str],
    ts: TobinsSeparation,
    allocations: np.ndarray[np.float64, typing.Any],
    start_year: int,
    rate: float = 0.01,
    required_return: float = 0.15,
    normalized: bool = False,
):
    plt.figure() # type: ignore
    plt.title("Backtest") # type: ignore
    plt.xlabel("Year") # type: ignore
    plt.ylabel("Actual yearly return") # type: ignore

    balanced_return = np.zeros(W, dtype=float)
    optimum_return = np.zeros(W, dtype=float)

    year_returns = np.asarray([returns[fund].yearly for fund in non_SPY_funds], dtype=float)
    value_optimum = 1000.0
    value_balanced = 1000.0
    for i in range(W):
        window_returns = year_returns[:, W + i]
        normalized_weights = ts.weights[i, :]
        if normalized:
            normalized_weights = normalize_weights(ts.weights[i, :])
        optimum_return[i] = float(normalized_weights @ window_returns)
        value_optimum = (1 + optimum_return[i]) * value_optimum
        balanced_return[i] = (1 - allocations[i]) * rate + allocations[i] * optimum_return[i]
        value_balanced = (1 + balanced_return[i]) * value_balanced

        plt.scatter(i+1, optimum_return[i], color='red') # type: ignore
        plt.scatter(i+1, balanced_return[i], color='black') # type: ignore
    output_optimum = "normalized " if normalized else ""
    print(f"{output_optimum}optimum portfolio after", W, "windows:", value_optimum)
    print(f"{output_optimum}balanced portfolio after", W, "windows:", value_balanced)
    if normalized:
        show_weights(normalize_weights(ts.weights[W, :]), non_SPY_funds, normalized=True)
    else:
        show_weights(ts.weights[W, :], non_SPY_funds)

    plt.plot([0, W + 1], [0.0, 0.0], color='gray', linestyle='--', label='zero return') # type: ignore
    plt.plot([0, W + 1], [required_return, required_return], color='blue', linestyle='--', label='required return') # type: ignore

    labels = [str(start_year + W + i) for i in range(W)]
    plt.xticks(np.arange(1, W + 1), labels) # type: ignore
    plt.legend() # type: ignore
    plt.show() # type: ignore
    return PortfolioResult(balanced_return=balanced_return, optimum_return=optimum_return)


@dataclass
class SingleIndexModelResult:
    beta: float
    alpha: float
    sigma2_e: float
    alpha_p: float
    beta_p: float
    r2: float


# Compute single index model for a single stock
# stock_returns: [N] stock returns
# market_returns: [N] market returns
def single_index_model(
    stock_returns: typing.Union[np.ndarray[np.float64, typing.Any], list[float]],
    market_returns: typing.Union[np.ndarray[np.float64, typing.Any], list[float]],
) -> SingleIndexModelResult:
    s = np.asarray(stock_returns, dtype=float)
    m = np.asarray(market_returns, dtype=float)
    if s.size != m.size:
        raise ValueError(f"Stock returns length {s.size} must match market returns length {m.size}.")
    n = s.size
    market_return = float(np.mean(m))
    stock_return = float(np.mean(s))
    diff_m = m - market_return
    diff_s = s - stock_return
    sigma_sm = float(diff_s @ diff_m)
    sigma_m = float(diff_m @ diff_m)
    beta = sigma_sm / sigma_m
    alpha = stock_return - beta * market_return
    diff_prediction = s - (alpha + beta * m)
    sigma2_e = float(diff_prediction @ diff_prediction) / n
    error = float(diff_prediction @ diff_prediction) / (n - 2)
    devterm = 1.0 / n + market_return * market_return / sigma_m
    std_err_alpha = math.sqrt(error) * math.sqrt(devterm)
    std_err_beta = math.sqrt(error) / math.sqrt(sigma_m)
    alpha_p = 2.0 * float(stats.t.pdf(alpha / std_err_alpha, n - 2)) # type: ignore
    beta_p = 2.0 * float(stats.t.pdf(beta / std_err_beta, n - 2)) # type: ignore
    r2 = beta * math.sqrt(sigma_m) / math.sqrt(float(diff_s @ diff_s))
    return SingleIndexModelResult(beta=beta, alpha=alpha, sigma2_e=sigma2_e, alpha_p=alpha_p, beta_p=beta_p, r2=r2)


def security_market_line(
    W: int,
    portfolio: utility.PortfolioResult,
    year_market: list[float],
    returns: dict[str, utility.Returns],
    non_SPY_funds: list[str],
    rate: float = 0.01,
):
    plt.figure() # type: ignore
    plt.title("Comparison to Security Market Line of CAPM") # type: ignore
    plt.xlabel("Beta") # type: ignore
    plt.ylabel("Expected return") # type: ignore

    market_returns = year_market[W:2 * W]
    market_slope = mean([r - rate for r in market_returns])

    beta_plot = np.linspace(0.0, 2.0, 100)
    plt.plot(beta_plot, rate + market_slope * beta_plot) # type: ignore

    for name, portfolio_returns in (("Balanced portfolio", portfolio.balanced_return),
                                    ("Optimum portfolio", portfolio.optimum_return)):
        sim = single_index_model(portfolio_returns, market_returns)
        print(f"({name})")
        print("Jensens alpha:", sim.alpha - rate)
        print("The P values for Alpha:", sim.alpha_p)
        print("The P values for Beta:", sim.beta_p)
        print("The goodness of fit R2:", sim.r2)
        plt.scatter(sim.beta, float(np.mean(portfolio_returns)), s=100, marker='d', label=name) # type: ignore

    for fund in non_SPY_funds:
        fund_returns = returns[fund].yearly[W:2 * W]
        sim = single_index_model(fund_returns, market_returns)
        plt.scatter(sim.beta, mean(fund_returns)) # type: ignore
        plt.text(sim.beta, mean(fund_returns), fund) # type: ignore

    plt.legend() # type: ignore
    plt.show() # type: ignore

def main():
    arg = ""
    if len(sys.argv) > 1:
        arg = sys.argv[1]
    print("Loading data for the following funds:")
    funds = ['SPY', 'FOCPX', 'DIA', 'XLY', 'XLK']
    #N = len(funds)
    W = 10  # number of windows - 1
    print(funds)
    current_year = 2026
    required_return = 0.15
    start_year = current_year - W*2
    print("Year range:", start_year, "-", current_year)
    data = load_data(funds, current_year, start_year)
    validate_data(data, funds, current_year, start_year)
    print("Data loaded and validated successfully.")

    _, year_market = compute_returns(data['SPY'])
    SPY_return = mean(year_market[W:])
    print(f"SPY average yearly return: {SPY_return:.4f}")

    non_SPY_funds = [fund for fund in funds if fund != 'SPY']
    returns: dict[str, utility.Returns] = dict()
    for fund in non_SPY_funds:
        _, year_fund = compute_returns(data[fund])
        fund_return = mean(year_fund[W:])
        returns[fund] = utility.Returns(yearly=year_fund, mean=fund_return)
        print(f"{fund} average yearly return: {fund_return:.4f}")
    plot_efficient_frontier(W, returns, non_SPY_funds)
    ts = plot_tobins_separation(W, returns, non_SPY_funds)
    allocations = compute_portfolio_return(W, ts, rate=0.01, required_return=required_return)
    if arg == "long-only":
        print("Running backtest with long-only portfolios...")
        portfolioResult = backtest_only_long(W, returns, non_SPY_funds, allocations,
             start_year=start_year, rate=0.01, required_return=required_return)
        security_market_line(W, portfolioResult, year_market, returns, non_SPY_funds, rate=0.01)
    elif arg == "normalize":
        print("Running backtest with normalized portfolios...")
        portfolioResult = backtest(W, returns, non_SPY_funds, ts, allocations,
             start_year=start_year, rate=0.01, required_return=required_return, normalized=True)
        security_market_line(W, portfolioResult, year_market, returns, non_SPY_funds, rate=0.01)
    else:
        portfolioResult = backtest(W, returns, non_SPY_funds, ts, allocations,
             start_year=start_year, rate=0.01, required_return=required_return)
        security_market_line(W, portfolioResult, year_market, returns, non_SPY_funds, rate=0.01)

if __name__ == "__main__":
    main()