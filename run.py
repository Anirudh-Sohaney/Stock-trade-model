"""Launches both the trading bot (loop) and dashboard."""

import sys
import os
import subprocess
import signal
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
BOT = [sys.executable, os.path.join(ROOT, "alpaca_bot", "main.py"), "--loop"]
DASHBOARD = [sys.executable, os.path.join(ROOT, "dashboard", "dashboard.py")]

processes = []


def cleanup(signum=None, frame=None):
    print("\nShutting down...")
    for p in processes:
        p.terminate()
    for p in processes:
        try:
            p.wait(timeout=5)
        except Exception:
            p.kill()
    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

print("Starting trading bot + dashboard...")
print("  Bot:       python alpaca_bot/main.py --loop")
print("  Dashboard: http://0.0.0.0:11434")
print("  Ctrl+C to stop both.\n")

processes.append(subprocess.Popen(BOT, cwd=ROOT))
processes.append(subprocess.Popen(DASHBOARD, cwd=ROOT))

try:
    for p in processes:
        p.wait()
except KeyboardInterrupt:
    cleanup()
