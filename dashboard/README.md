# Dashboard — Terminal Trading Bot Monitor

Minimal web dashboard for the Stage 3 trading bot. Black terminal-style UI with live SSE updates.

## Quick Start

```bash
pip install -r dashboard/requirements.txt
python dashboard/dashboard.py
```

Opens at `http://SERVER_IP:11434`

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Dashboard page |
| `GET /api/status` | Bot/market status |
| `GET /api/stats` | Performance stats |
| `GET /api/positions` | Open positions |
| `GET /api/logs?n=100` | Recent log events |
| `GET /api/trades` | Trade history |
| `GET /api/stream` | SSE live feed |

## Oracle Cloud Deployment

```bash
# Install deps
pip install -r dashboard/requirements.txt

# Open port
sudo firewall-cmd --add-port=11434/tcp --permanent && sudo firewall-cmd --reload

# Start (background)
nohup python dashboard/dashboard.py > dashboard.log 2>&1 &

# Or with tmux
tmux new -s trading
python dashboard/dashboard.py
# Ctrl+B D to detach
```

Dashboard runs independently. If the bot is offline it shows `OFFLINE` status.
