import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from main_model import HardcodedTrader


class SignalEngine:
    def __init__(self):
        self.trader = HardcodedTrader()

    def generate_signal(self, feature_dict):
        return self.trader.generate_signal(feature_dict)
