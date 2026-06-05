from datetime import datetime, timezone
from urllib.parse import urlencode
import urllib.request as r
import json
from dataclasses import dataclass

@dataclass
class StockData:
    symbol: str
    from_date: datetime
    to_date: datetime
    open: list[float]
    high: list[float]
    low: list[float]
    close: list[float]
    adjclose: list[float]
    volume: list[int]

def hist_stock_data_period(start_date, end_date, interval_string, events_string, ticker) -> StockData:
    """
    Python equivalent of the MATLAB hist_stock_data_period function.

    Args:
        start_date (str): Date in 'ddmmyyyy' format.
        end_date (str): Date in 'ddmmyyyy' format.
        interval_string (str): Yahoo interval (e.g. '1d', '1wk', '1mo').
        events_string (str): Yahoo events type (e.g. 'history', 'div').
        ticker (str): Stock symbol (e.g. 'GOOGL').

    Returns:
        StockData: StockData object containing the historical stock data.
    """
    start_dt = datetime.strptime(start_date, "%d%m%Y").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%d%m%Y").replace(tzinfo=timezone.utc)

    # MATLAB datenum conversion in this case maps to Unix epoch seconds.
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    base_url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
    query = urlencode(
        {
            "period1": start_ts,
            "period2": end_ts,
            "interval": interval_string,
            "events": events_string,
        }
    )
    url = f"{base_url}?{query}"
    print(f"Requesting URL: {url}")
    req = r.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with r.urlopen(req, timeout=30) as f:
        data = f.read()
        json_data = json.loads(data.decode("utf-8"))
        result = json_data["chart"]["result"][0]
        indicators = result["indicators"]["quote"][0]
        adjclose = result["indicators"]["adjclose"][0]["adjclose"]
        return StockData(
            symbol=ticker,
            from_date=datetime.fromtimestamp(result["timestamp"][0], tz=timezone.utc),
            to_date=datetime.fromtimestamp(result["timestamp"][-1], tz=timezone.utc),
            open=indicators["open"],
            high=indicators["high"],
            low=indicators["low"],
            close=indicators["close"],
            adjclose=adjclose,
            volume=indicators["volume"],
        )


if __name__ == "__main__":
    stocks = hist_stock_data_period("01012024", "01012025", "1wk", "history", "GOOGL")
    print(stocks)