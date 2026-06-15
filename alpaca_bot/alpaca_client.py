import json
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from logger import setup_logger

logger = setup_logger("alpaca_client")


class AlpacaClient:
    def __init__(self, config):
        self.config = config
        self.api = None
        self._connect()

    def _connect(self):
        api_key = self.config.get("alpaca_api_key", "")
        secret_key = self.config.get("alpaca_secret_key", "")
        if not api_key or not secret_key:
            logger.warning("API keys not configured. Running in simulation mode.")
            self.api = None
            return
        try:
            import alpaca_trade_api as tradeapi
        except ImportError:
            logger.warning("alpaca-trade-api not installed. Running in simulation mode.")
            self.api = None
            return
        paper = self.config.get("paper_trading", True)
        base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        self.api = tradeapi.REST(api_key, secret_key, base_url, api_version="v2")
        account = self.api.get_account()
        logger.info(f"Connected to Alpaca ({'paper' if paper else 'live'})")
        logger.info(f"  Account: {account.status}, Equity: ${float(account.equity):.2f}")

    def get_account_info(self):
        if self.api is None:
            return {"equity": self.config.get("initial_capital", 10000), "cash": self.config.get("initial_capital", 10000)}
        acct = self.api.get_account()
        return {"equity": float(acct.equity), "cash": float(acct.cash)}

    def get_positions(self):
        if self.api is None:
            return []
        positions = self.api.list_positions()
        return [{
            "ticker": p.symbol, "qty": int(p.qty),
            "avg_entry_price": float(p.avg_entry_price),
            "current_price": float(p.current_price),
            "market_value": float(p.market_value),
            "unrealized_pl": float(p.unrealized_pl),
            "unrealized_plpc": float(p.unrealized_plpc),
        } for p in positions]

    def get_bars(self, tickers, lookback_days=60, timeframe="15Min"):
        if self.api is None:
            logger.warning("No API connection. Cannot fetch bars.")
            return {}
        end = datetime.now()
        start = end - timedelta(days=lookback_days)
        try:
            bars = self.api.get_bars(
                tickers, timeframe,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                adjustment="raw",
            ).df
        except Exception as e:
            logger.error(f"Failed to fetch bars: {e}")
            return {}
        result = {}
        for ticker in tickers:
            try:
                ticker_bars = bars.loc[ticker].copy()
                ticker_bars = ticker_bars.rename(columns={
                    "open": "open", "high": "high", "low": "low",
                    "close": "close", "volume": "volume",
                })
                ticker_bars.index = pd.to_datetime(ticker_bars.index)
                result[ticker] = ticker_bars
            except Exception:
                continue
        logger.info(f"Fetched bars for {len(result)}/{len(tickers)} tickers")
        return result

    def submit_order(self, ticker, qty, side, order_type="market"):
        if self.api is None:
            logger.info(f"[SIM] Order: {side} {qty} {ticker}")
            return {"id": "sim", "status": "simulated", "filled_qty": qty}
        try:
            order = self.api.submit_order(
                symbol=ticker, qty=qty, side=side,
                type=order_type, time_in_force="day",
            )
            logger.info(f"Order placed: {side} {qty} {ticker} (id={order.id})")
            return {"id": order.id, "status": order.status, "filled_qty": int(order.filled_qty)}
        except Exception as e:
            logger.error(f"Order failed for {ticker}: {e}")
            return {"id": None, "status": "failed", "filled_qty": 0}

    def close_position(self, ticker):
        if self.api is None:
            logger.info(f"[SIM] Close position: {ticker}")
            return True
        try:
            self.api.close_position(ticker)
            return True
        except Exception as e:
            logger.error(f"Failed to close {ticker}: {e}")
            return False

    def get_clock(self):
        if self.api is None:
            from market_hours import is_market_open, is_market_day
            now = datetime.now()
            return {"is_open": is_market_open(now), "is_market_day": is_market_day(now)}
        clock = self.api.get_clock()
        return {"is_open": clock.is_open, "next_open": clock.next_open}
