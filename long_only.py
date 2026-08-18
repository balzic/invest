


import math
import typing

import numpy as np
from scipy import optimize # pyright: ignore[reportMissingTypeStubs]
import utility
from utility import PortfolioResult, compute_covariance_matrix, show_weights
from matplotlib import pyplot as plt
from utility import show_weights
# max-Sharpe tangency portfolio with no shorting and no lending/borrowing (w >= 0, sum(w) = 1)
def long_only_tangency(
    covariance: np.ndarray[np.float64, typing.Any],
    r: float,
    expected_returns: list[np.float64],
) -> np.ndarray[np.float64, typing.Any]:
    returns_np = np.asarray(expected_returns, dtype=float)
    n = returns_np.size

    def neg_sharpe(w: np.ndarray[np.float64, typing.Any]) -> float:
        variance = float(w @ covariance @ w)
        if variance <= 1e-16:
            return 0.0
        return -(float(w @ returns_np) - r) / math.sqrt(variance)

    result = optimize.minimize( # type: ignore
        neg_sharpe,
        x0=np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0}], # type: ignore
    )
    if not result.success: # type: ignore
        raise ValueError(f"Long-only optimization failed: {result.message}") # type: ignore
    weights = np.asarray(result.x, dtype=float) # type: ignore
    weights[weights < 0] = 0.0
    return weights / np.sum(weights)

def backtest_only_long(
    W: int,
    returns: dict[str, utility.Returns],
    non_SPY_funds: list[str],
    allocations: np.ndarray[np.float64, typing.Any],
    start_year: int,
    rate: float = 0.01,
    required_return: float = 0.15,
)-> PortfolioResult:
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
        expected = [np.mean(returns[fund].yearly[i:i + W]) for fund in non_SPY_funds]
        returns_matrix: list[list[float]] = [returns[fund].yearly[i:i + W] for fund in non_SPY_funds]
        covariance = compute_covariance_matrix(returns_matrix)
        weights = long_only_tangency(covariance, rate, expected)
        optimum_return[i] = float(weights @ window_returns)
        value_optimum = (1 + optimum_return[i]) * value_optimum
        balanced_return[i] = (1 - allocations[i]) * rate + allocations[i] * optimum_return[i]
        value_balanced = (1 + balanced_return[i]) * value_balanced

        plt.scatter(i+1, optimum_return[i], color='red') # type: ignore
        plt.scatter(i+1, balanced_return[i], color='black') # type: ignore

    print("long-only optimum portfolio after", W, "windows:", value_optimum)
    print("long-only balanced portfolio after", W, "windows:", value_balanced)
    plt.plot([0, W + 1], [0.0, 0.0], color='gray', linestyle='--', label='zero return') # type: ignore
    plt.plot([0, W + 1], [required_return, required_return], color='blue', linestyle='--', label='required return') # type: ignore


    expected = [np.mean(returns[fund].yearly[W:W*2]) for fund in non_SPY_funds]
    returns_matrix: list[list[float]] = [returns[fund].yearly[W:W*2] for fund in non_SPY_funds]
    covariance = compute_covariance_matrix(returns_matrix)
    weights = long_only_tangency(covariance, rate, expected)
    show_weights(weights, non_SPY_funds, normalized=True)

    labels = [str(start_year + W + i) for i in range(W)]
    plt.xticks(np.arange(1, W + 1), labels) # type: ignore
    plt.legend() # type: ignore
    plt.show() # type: ignore
    return PortfolioResult(balanced_return=balanced_return, optimum_return=optimum_return)
