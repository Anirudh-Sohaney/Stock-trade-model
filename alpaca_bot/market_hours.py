from datetime import datetime, time, date, timedelta
import pytz

EASTERN = pytz.timezone('US/Eastern')
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

US_HOLIDAYS_2026 = {
    date(2026, 1, 1),
    date(2026, 1, 19),
    date(2026, 2, 16),
    date(2026, 4, 3),
    date(2026, 5, 25),
    date(2026, 6, 19),
    date(2026, 7, 3),
    date(2026, 9, 7),
    date(2026, 11, 26),
    date(2026, 12, 25),
}


def now_eastern():
    return datetime.now(EASTERN)


def is_market_day(dt=None):
    if dt is None:
        dt = now_eastern()
    if isinstance(dt, datetime):
        dt = dt.date()
    if dt.weekday() >= 5:
        return False
    if dt in US_HOLIDAYS_2026:
        return False
    return True


def is_market_open(dt=None):
    if dt is None:
        dt = now_eastern()
    if isinstance(dt, datetime):
        dt_east = dt.astimezone(EASTERN)
    else:
        dt_east = dt
    if not is_market_day(dt_east):
        return False
    t = dt_east.time()
    return MARKET_OPEN <= t < MARKET_CLOSE


def minutes_until_close(dt=None):
    if dt is None:
        dt = now_eastern()
    dt_east = dt.astimezone(EASTERN)
    close = EASTERN.localize(datetime.combine(dt_east.date(), MARKET_CLOSE))
    delta = close - dt_east
    return max(0, delta.total_seconds() / 60)
