import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
from datetime import datetime

from signal_engine import SignalEngine
from position_manager import PositionManager
from alpaca_client import AlpacaClient
from market_hours import is_market_open, minutes_until_close
from logger import setup_logger

from holding_period_optimization import compute_features

logger = setup_logger("portfolio_manager")


class PortfolioManager:
    def __init__(self, config):
        self.config = config
        self.signal_engine = SignalEngine()
        self.alpaca = AlpacaClient(config)
        self.position_manager = PositionManager(config, self.alpaca)
        self.tickers = self._load_tickers()
        self.exit_bars = config.get("exit_bars", 1)
        self.data_cache = {}

    def _load_tickers(self):
        ticker_file = self.config.get("ticker_file", "tickers.txt")
        if not os.path.isabs(ticker_file):
            ticker_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ticker_file)
        try:
            with open(ticker_file, "r") as f:
                tickers = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(tickers)} tickers from {ticker_file}")
            return tickers
        except FileNotFoundError:
            logger.warning(f"Ticker file {ticker_file} not found. Using defaults.")
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ"]

    def refresh_data(self):
        bars = self.alpaca.get_bars(self.tickers, lookback_days=self.config.get("data_lookback_days", 60))
        self.data_cache = bars
        return bars

    def generate_signal_for_ticker(self, ticker, timestamp=None):
        if ticker not in self.data_cache:
            return None
        df = self.data_cache[ticker]
        if timestamp is not None:
            df = df[df.index <= timestamp]
        if len(df) < 30:
            return None
        features = compute_features(df).dropna()
        if timestamp is not None:
            feat_row = features.loc[timestamp] if timestamp in features.index else (
                features.loc[features.index[features.index <= timestamp][-1]]
                if len(features.index[features.index <= timestamp]) > 0 else features.iloc[-1])
        else:
            feat_row = features.iloc[-1]
        feat_dict = feat_row.to_dict()
        feat_dict["close"] = df.iloc[-1]["close"]
        signal = self.signal_engine.generate_signal(feat_dict)
        if signal == "BUY":
            return {
                "ticker": ticker, "price": feat_dict["close"],
                "score": feat_dict.get("atr_percentile", 0.5),
                "timestamp": df.index[-1] if timestamp is None else timestamp,
            }
        return None

    def collect_all_signals(self, timestamp=None):
        signals = []
        for ticker in self.tickers:
            if ticker not in self.data_cache:
                continue
            if ticker in self.position_manager.open_positions:
                continue
            sig = self.generate_signal_for_ticker(ticker, timestamp)
            if sig is not None:
                signals.append(sig)
        return signals

    def rank_signals(self, signals):
        signals.sort(key=lambda s: -s.get("score", 0))
        return signals

    def run_once(self):
        if not is_market_open():
            logger.info("Market is closed. Skipping cycle.")
            return
        mins_to_close = minutes_until_close()
        if mins_to_close < 5:
            logger.info(f"Market closing soon ({mins_to_close:.0f} min). Skipping.")
            return
        now = datetime.now()
        logger.info(f"=== Trading cycle: {now} ===")
        self.refresh_data()
        for ticker in self.position_manager.get_exit_targets(now, self.data_cache):
            logger.info(f"Exiting {ticker} (exit time reached)")
            self.alpaca.close_position(ticker)
            self.position_manager.exit_position(ticker)
        signals = self.collect_all_signals(now)
        ranked = self.rank_signals(signals)
        logger.info(f"  Signals: {len(signals)} total")
        account = self.alpaca.get_account_info()
        cash = account.get("cash", self.config.get("initial_capital", 10000))
        sizes = self.position_manager.compute_position_sizes(ranked, cash)
        slots = self.position_manager.get_available_slots()
        for i in range(min(len(ranked), slots)):
            if i >= len(sizes):
                break
            sig = ranked[i]
            size = sizes[i]
            if size < sig["price"]:
                continue
            qty = self.position_manager.get_qty_for_size(sig["ticker"], size, sig["price"])
            order = self.alpaca.submit_order(sig["ticker"], qty, "buy")
            if order.get("status") in ("accepted", "filled", "simulated"):
                self.position_manager.enter_position(
                    sig["ticker"], size, sig["price"],
                    sig.get("rule_idx", -1), sig.get("score", 0), now)
                logger.info(f"  Entered: {sig['ticker']} {qty} shares @ ${sig['price']:.2f}")
        total_positions = len(self.position_manager.open_positions)
        total_exposure = sum(p.get("capital", 0) for p in self.position_manager.open_positions.values())
        logger.info(f"  Positions: {total_positions}/{self.config.get('max_positions', 5)}, "
                    f"Exposure: ${total_exposure:.2f}, Cash: ${cash:.2f}")
