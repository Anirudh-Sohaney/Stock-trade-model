"""
Stage 3 Trading Bot — Alpaca Deployment
=========================================
Usage:
    python main.py              # Run once
    python main.py --loop       # Run continuously (every 15 min)

Configuration: Edit config.json
"""

import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from portfolio_manager import PortfolioManager
from market_hours import is_market_open, now_eastern
from logger import setup_logger

logger = setup_logger("main")


def load_config(config_path="config.json"):
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path, "r") as f:
        return json.load(f)


def run_once(pm):
    logger.info("=" * 50)
    logger.info("STAGE 3 TRADING BOT — RUN CYCLE")
    logger.info("=" * 50)
    try:
        pm.run_once()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)


def run_loop(pm, interval_minutes=15):
    logger.info(f"Starting continuous loop (checking every {interval_minutes} min)")
    while True:
        if is_market_open(now_eastern()):
            run_once(pm)
        else:
            logger.info("Market closed. Waiting...")
        time.sleep(interval_minutes * 60)


def main():
    parser = argparse.ArgumentParser(description="Stage 3 Trading Bot")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--config", default="config.json", help="Config file path")
    args = parser.parse_args()
    config = load_config(args.config)
    pm = PortfolioManager(config)
    if args.loop:
        run_loop(pm)
    else:
        run_once(pm)


if __name__ == "__main__":
    main()
