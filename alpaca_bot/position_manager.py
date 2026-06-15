import os
import pickle
from logger import setup_logger

logger = setup_logger("position_manager")


class PositionManager:
    def __init__(self, config, alpaca_client):
        self.config = config
        self.alpaca = alpaca_client
        self.state_file = config.get("state_file", "bot_state.pkl")
        self.max_positions = config.get("max_positions", 5)
        self.sizing_method = config.get("sizing_method", "rank_based")
        self.capital_utilization = config.get("capital_utilization", 1.0)
        self.exit_bars = config.get("exit_bars", 1)
        self.open_positions = {}
        self.entry_timestamps = {}
        self._load_state()

    def _load_state(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "rb") as f:
                    data = pickle.load(f)
                    self.open_positions = data.get("positions", {})
                    self.entry_timestamps = data.get("timestamps", {})
        except Exception:
            pass

    def _save_state(self):
        try:
            with open(self.state_file, "wb") as f:
                pickle.dump({
                    "positions": self.open_positions,
                    "timestamps": self.entry_timestamps,
                }, f)
        except Exception:
            pass

    def get_available_slots(self):
        return self.max_positions - len(self.open_positions)

    def compute_position_sizes(self, ranked_signals, cash):
        n = min(len(ranked_signals), self.get_available_slots())
        if n == 0:
            return []
        deployable = cash * self.capital_utilization
        if self.sizing_method == "equal":
            return [deployable / n] * n
        elif self.sizing_method == "rank_based":
            weights = [n + 1 - i for i in range(1, n + 1)]
            total_w = sum(weights)
            return [deployable * w / total_w for w in weights]
        elif self.sizing_method == "score_weighted":
            scores = [max(s.get("score", 1), 1) for s in ranked_signals[:n]]
            total_s = sum(scores)
            return [deployable * s / total_s for s in scores]
        return [deployable / n] * n

    def get_qty_for_size(self, ticker, size, price):
        return max(int(size / price), 1)

    def enter_position(self, ticker, size, price, rule_idx, score, timestamp):
        self.open_positions[ticker] = {
            "entry_price": price, "capital": size,
            "rule_idx": rule_idx, "score": score,
        }
        self.entry_timestamps[ticker] = timestamp
        self._save_state()

    def should_exit(self, ticker, current_timestamp, ticker_data):
        if ticker not in self.entry_timestamps:
            return True
        entry_ts = self.entry_timestamps[ticker]
        try:
            idx = ticker_data.index.get_loc(entry_ts)
            exit_idx = min(idx + self.exit_bars, len(ticker_data) - 1)
            return current_timestamp >= ticker_data.index[exit_idx]
        except (KeyError, ValueError):
            return True

    def exit_position(self, ticker):
        self.open_positions.pop(ticker, None)
        self.entry_timestamps.pop(ticker, None)
        self._save_state()

    def get_exit_targets(self, current_timestamp, all_data):
        return [t for t in list(self.open_positions.keys())
                if t in all_data and self.should_exit(t, current_timestamp, all_data[t])]
