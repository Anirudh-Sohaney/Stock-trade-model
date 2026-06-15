import os
import re
import pickle
import json
from datetime import datetime, date, timedelta

BOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
ALPACA_DIR = os.path.join(BOT_DIR, 'alpaca_bot')
CONFIG_FILE = os.path.join(ALPACA_DIR, 'config.json')
STATE_FILE = os.path.join(ALPACA_DIR, 'bot_state.pkl')
LOG_FILE = os.path.join(ALPACA_DIR, 'bot.log')
US_HOLIDAYS = {
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16),
    date(2026, 4, 3), date(2026, 5, 25), date(2026, 6, 19),
    date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26),
    date(2026, 12, 25),
}

LOG_LINE_RE = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) - (\w+) - (.+)$')
ENTRY_RE = re.compile(r'Entered: (\w+) (\d+) shares @ \$(\d+\.?\d*)')
ORDER_RE = re.compile(r'\[SIM\] Order: (buy|sell) (\d+) (.+)')
CLOSE_RE = re.compile(r'\[SIM\] Close position: (.+)')
EXIT_RE = re.compile(r'Exiting (\w+)')
SIGNALS_RE = re.compile(r'Signals: (\d+) total')
SUMMARY_RE = re.compile(r'Positions: (\d+)/(\d+), Exposure: \$([\d,\.]+), Cash: \$([\d,\.]+)')
CYCLE_RE = re.compile(r'=== Trading cycle: (.+) ===')

_parsed_logs = []
_trade_history = []
_stats_cache = {}
_last_read_pos = 0


def _read_config():
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _read_state():
    try:
        with open(STATE_FILE, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return {'positions': {}, 'timestamps': {}}


def _read_log_lines(count=5000):
    global _last_read_pos
    try:
        with open(LOG_FILE, 'r') as f:
            f.seek(0, 2)
            file_size = f.tell()
            if file_size < _last_read_pos:
                _last_read_pos = 0
            start = max(0, file_size - 81920)
            f.seek(start)
            lines = f.readlines()
            return lines[-count:]
    except Exception:
        return []


def parse_log_lines(lines):
    events = []
    for line in lines:
        line = line.rstrip()
        m = LOG_LINE_RE.match(line)
        if not m:
            continue
        ts, level, msg = m.groups()
        ts_short = ts[:19] if len(ts) >= 19 else ts
        entry_m = ENTRY_RE.search(msg)
        if entry_m:
            events.append({'type': 'entry', 'ts': ts_short,
                           'ticker': entry_m.group(1),
                           'qty': int(entry_m.group(2)),
                           'price': float(entry_m.group(3))})
            continue
        order_m = ORDER_RE.search(msg)
        if order_m:
            events.append({'type': 'order', 'ts': ts_short,
                           'side': order_m.group(1).upper(),
                           'qty': int(order_m.group(2)),
                           'ticker': order_m.group(3).strip()})
            continue
        close_m = CLOSE_RE.search(msg)
        if close_m:
            events.append({'type': 'close', 'ts': ts_short,
                           'ticker': close_m.group(1).strip()})
            continue
        exit_m = EXIT_RE.search(msg)
        if exit_m:
            events.append({'type': 'exit', 'ts': ts_short,
                           'ticker': exit_m.group(1)})
            continue
        if 'Market is closed' in msg:
            events.append({'type': 'market', 'ts': ts_short, 'status': 'CLOSED'})
            continue
        if 'Market closing soon' in msg:
            events.append({'type': 'market', 'ts': ts_short, 'status': 'CLOSING_SOON'})
            continue
        cycle_m = CYCLE_RE.search(msg)
        if cycle_m:
            events.append({'type': 'cycle', 'ts': ts_short})
            continue
        sig_m = SIGNALS_RE.search(msg)
        if sig_m:
            events.append({'type': 'signals', 'ts': ts_short,
                           'count': int(sig_m.group(1))})
            continue
        summary_m = SUMMARY_RE.search(msg)
        if summary_m:
            events.append({'type': 'summary', 'ts': ts_short,
                           'positions': int(summary_m.group(1)),
                           'max_pos': int(summary_m.group(2)),
                           'exposure': float(summary_m.group(3).replace(',','')),
                           'cash': float(summary_m.group(4).replace(',',''))})
            continue
        if 'Bot started' in msg or 'Starting' in msg:
            events.append({'type': 'system', 'ts': ts_short, 'msg': msg.strip()})
            continue
        events.append({'type': 'log', 'ts': ts_short, 'msg': msg.strip(), 'level': level})
    return events


def get_logs(count=100):
    lines = _read_log_lines(count)
    return parse_log_lines(lines)


def get_recent_events(count=20):
    events = get_logs(200)
    return events[-count:]


def get_bot_status():
    state = _read_state()
    config = _read_config()
    has_keys = bool(config.get('alpaca_api_key') and config.get('alpaca_secret_key'))
    state_exists = os.path.exists(STATE_FILE)
    log_exists = os.path.exists(LOG_FILE)
    if state_exists or log_exists:
        return 'ONLINE' if has_keys else 'SIMULATION'
    return 'OFFLINE'


def is_market_open():
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    if now.date() in US_HOLIDAYS:
        return False
    t = now.time()
    return (t >= datetime.strptime('09:30', '%H:%M').time() and
            t < datetime.strptime('16:00', '%H:%M').time())


def is_trading():
    return get_bot_status() in ('ONLINE', 'SIMULATION') and is_market_open()


def get_positions():
    state = _read_state()
    positions = state.get('positions', {})
    return [{'ticker': t, 'entry_price': p.get('entry_price', 0),
             'capital': p.get('capital', 0), 'score': p.get('score', 0),
             'rule_idx': p.get('rule_idx', -1)}
            for t, p in sorted(positions.items())]


def compute_stats():
    config = _read_config()
    state = _read_state()
    initial = config.get('initial_capital', 10000.0)
    max_pos = config.get('max_positions', 5)
    hold_period = config.get('exit_bars', 1)
    positions = state.get('positions', {})
    timestamps = state.get('timestamps', {})
    open_count = len(positions)
    total_capital = sum(p.get('capital', 0) for p in positions.values())
    events = get_logs(2000)
    trades_today = 0
    trades_week = 0
    wins = 0
    losses = 0
    today = date.today()
    week_start = today - timedelta(days=today.weekday())
    realized_pnl = 0.0
    active_buys = {}
    for ev in reversed(events):
        if ev['type'] == 'entry':
            ts = ev['ts'][:10] if len(ev['ts']) >= 10 else ''
            if ts >= str(week_start):
                trades_week += 1
            if ts == str(today):
                trades_today += 1
            active_buys[ev['ticker']] = ev
        elif ev['type'] == 'close' or ev['type'] == 'exit':
            ticker = ev.get('ticker', '')
            if ticker in active_buys:
                del active_buys[ticker]
    for ev in events:
        if ev['type'] == 'summary':
            total_capital = ev.get('exposure', total_capital)
    current_balance = initial + realized_pnl
    deployed_pct = (total_capital / current_balance * 100) if current_balance > 0 else 0
    all_time_return = (current_balance - initial) / initial * 100 if initial > 0 else 0
    return {
        'current_balance': current_balance,
        'starting_balance': initial,
        'all_time_return': all_time_return,
        'open_positions': open_count,
        'max_positions': max_pos,
        'capital_deployed': total_capital,
        'capital_deployed_pct': deployed_pct,
        'buying_power': current_balance - total_capital,
        'trades_today': trades_today,
        'trades_week': trades_week,
        'wins': wins,
        'losses': losses,
        'win_rate': (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0,
        'strategy': 'STAGE3',
        'hold_period': f'{hold_period}c',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


def get_status():
    bot = get_bot_status()
    market = is_market_open()
    return {
        'bot_status': bot,
        'market': 'OPEN' if market else 'CLOSED',
        'trading': 'YES' if (bot in ('ONLINE', 'SIMULATION') and market) else 'NO',
    }


def get_trades():
    events = get_logs(5000)
    trades = []
    for ev in reversed(events):
        if ev['type'] in ('entry', 'order', 'close', 'exit'):
            trades.append(ev)
    return trades[-100:]


def get_all_data():
    return {
        **get_status(),
        'stats': compute_stats(),
        'positions': get_positions(),
        'events': get_recent_events(30),
    }
