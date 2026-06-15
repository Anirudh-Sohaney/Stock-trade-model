"""
holding_period_optimization.py

7-phase investigation:
  1. Holding period optimization (exit_bars = 1..5)
  2. Rule-specific holding analysis
  3. Pattern-timeframe clustering
  4. Adaptive holding system
  5. Regime analysis
  6. Losing week investigation
  7. Regime filter simulation
"""

import pandas as pd
import numpy as np
import logging
import pickle
import os
import json
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = 'regression_checkpoints'
REPORT_FILE = 'holding_period_report.txt'

# =========================================================================
# Feature computation (identical to stage3_optimization.py)
# =========================================================================

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast = compute_ema(close, fast); ema_slow = compute_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line

def compute_bollinger_bands(close, period=20, std_dev=2):
    sma = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std()
    return sma + std_dev * std, sma - std_dev * std, (sma + std_dev * std - (sma - std_dev * std)) / sma, (close - (sma - std_dev * std)) / ((sma + std_dev * std) - (sma - std_dev * std)).replace(0, np.nan)

def compute_atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()

def compute_stochastic(high, low, close, k_period=14, d_period=3):
    low_min = low.rolling(window=k_period, min_periods=k_period).min()
    high_max = high.rolling(window=k_period, min_periods=k_period).max()
    k = 100 * (close - low_min) / (high_max - low_min).replace(0, np.nan)
    return k, k.rolling(window=d_period, min_periods=d_period).mean()

def compute_cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    sma = tp.rolling(window=period, min_periods=period).mean()
    mad = tp.rolling(window=period, min_periods=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))

def compute_obv(close, volume):
    return (volume * (close.diff() > 0).astype(int) - volume * (close.diff() < 0).astype(int)).cumsum()

def compute_cmf(high, low, close, volume, period=20):
    mf = volume * ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    return mf.rolling(window=period, min_periods=period).sum() / volume.rolling(window=period, min_periods=period).sum().replace(0, np.nan)

def linear_reg_slope(series, window=5):
    def _slope(y):
        if len(y) < 2: return np.nan
        return np.polyfit(np.arange(len(y)), y, 1)[0]
    return series.rolling(window=window, min_periods=window).apply(_slope, raw=True)

def compute_features(df):
    close = df['close']; high = df['high']; low = df['low']
    open_ = df['open']; volume = df['volume']
    cr = (high - low).replace(0, np.nan)
    body = (close - open_).abs()
    prev_close = close.shift(1)
    upper_wick = high - close.where(close >= open_, open_)
    lower_wick = close.where(close < open_, open_) - low

    features = pd.DataFrame(index=df.index)
    features['rsi_14'] = compute_rsi(close, 14)
    features['rsi_7'] = compute_rsi(close, 7)
    for p in [9, 21, 50]: features[f'ema_{p}'] = compute_ema(close, p)
    features['ema_9_above_21'] = (features['ema_9'] > features['ema_21']).astype(int)
    features['ema_9_above_50'] = (features['ema_9'] > features['ema_50']).astype(int)
    features['ema_21_above_50'] = (features['ema_21'] > features['ema_50']).astype(int)
    features['ema_cross_9_21'] = ((features['ema_9'] > features['ema_21']) & (features['ema_9'].shift(1) <= features['ema_21'].shift(1))).astype(int)
    macd_line, signal_line, hist = compute_macd(close)
    features['macd_line'] = macd_line; features['macd_signal'] = signal_line; features['macd_hist'] = hist
    features['macd_above_signal'] = (macd_line > signal_line).astype(int)
    bb_upper, bb_lower, bb_width, bb_pct = compute_bollinger_bands(close)
    features['bb_width'] = bb_width; features['bb_pct'] = bb_pct
    features['atr_14'] = compute_atr(high, low, close, 14)
    features['atr_pct'] = features['atr_14'] / close * 100
    features['volume_ratio'] = volume / volume.rolling(window=20, min_periods=20).mean().replace(0, np.nan)
    for p in [3, 5, 10]: features[f'roc_{p}'] = close.pct_change(periods=p) * 100
    features['candle_body_pct'] = body / cr * 100
    features['upper_wick_ratio'] = upper_wick / cr * 100
    features['lower_wick_ratio'] = lower_wick / cr * 100
    features['is_green'] = (close >= open_).astype(int)
    if hasattr(df.index, 'time') and len(df) > 1:
        features['hour'] = df.index.hour; features['minute'] = df.index.minute
        inferred_min = (df.index[1] - df.index[0]).total_seconds() / 60
        interval_min = max(int(inferred_min), 1)
        features['time_bar'] = ((features['hour'] - 9) * (60 // interval_min) + features['minute'] // interval_min).clip(lower=0) + 1
    daily_high = high.groupby(df.index.date).transform('max')
    daily_low = low.groupby(df.index.date).transform('min')
    daily_range = daily_high - daily_low
    features['price_position'] = (close - daily_low) / daily_range.replace(0, np.nan)
    vwap = (volume * (high + low + close) / 3).rolling(window=20).sum() / volume.rolling(window=20).sum().replace(0, np.nan)
    features['vwap_dev'] = (close - vwap) / vwap.replace(0, np.nan) * 100
    features['spread_proxy'] = (high - low) / close * 100
    features['dist_from_ema9'] = (close - features['ema_9']) / features['ema_9'] * 100
    features['dist_from_ema21'] = (close - features['ema_21']) / features['ema_21'] * 100
    features['body_to_range'] = body / cr
    features['upper_wick_pct'] = upper_wick / cr
    features['lower_wick_pct'] = lower_wick / cr
    features['consec_green'] = ((close > open_).astype(int).rolling(3).sum())
    features['consec_red'] = ((close < open_).astype(int).rolling(3).sum())
    features['gap_up_pct'] = ((open_ - prev_close) / prev_close.replace(0, np.nan) * 100).clip(lower=0)
    features['gap_down_pct'] = ((prev_close - open_) / prev_close.replace(0, np.nan) * 100).clip(lower=0)
    features['dist_prev_close'] = (close - prev_close) / prev_close.replace(0, np.nan) * 100
    features['rel_candle_size'] = (high - low) / (high - low).rolling(20).mean().replace(0, np.nan)
    features['atr_norm_candle'] = (high - low) / features['atr_14'].replace(0, np.nan)
    for p in [1, 2, 3, 5]: features[f'ret_{p}'] = close.pct_change(periods=p) * 100
    features['accel_3'] = features['ret_3'].diff(); features['accel_5'] = features['ret_5'].diff()
    for p in [5, 12, 20]: features[f'ema_{p}'] = compute_ema(close, p)
    for p in [5, 9, 12, 21]:
        ema_col = f'ema_{p}'
        features[f'ema{p}_slope'] = features[ema_col].diff() / features[ema_col].replace(0, np.nan) * 100
    features['ema_9_50_spread'] = (features['ema_9'] - features['ema_50']) / close * 100
    bull_cols = [c for c in features.columns if 'above' in c]
    features['trend_alignment'] = features[[c for c in bull_cols if c in features.columns]].sum(axis=1)
    features['trend_persistence'] = (close > features['ema_50']).astype(int).rolling(10).mean()
    features['lr_slope_5'] = linear_reg_slope(close, 5)
    features['lr_slope_10'] = linear_reg_slope(close, 10)
    for p in [7, 21]: features[f'atr_{p}'] = compute_atr(high, low, close, p)
    features['atr_percentile'] = features['atr_14'].rank(pct=True)
    features['atr_norm'] = features['atr_14'] / close * 100
    for p in [5, 10, 20]: features[f'hv_{p}'] = close.pct_change().rolling(p).std() * np.sqrt(252 / 26) * 100
    features['bb_expansion'] = features['bb_width'].diff()
    features['kc_mid'] = features['ema_20']
    features['kc_width'] = features['atr_14'] * 1.5
    features['kc_pct'] = (close - (features['kc_mid'] - features['kc_width'])) / (2 * features['kc_width']).replace(0, np.nan)
    features['volume_percentile'] = volume.rank(pct=True)
    features['obv'] = compute_obv(close, volume)
    features['obv_slope'] = (features['obv'].diff() / features['obv'].rolling(5).mean().replace(0, np.nan)) * 100
    features['cmf'] = compute_cmf(high, low, close, volume, 20)
    features['vwap_distance'] = (close - vwap) / vwap.replace(0, np.nan) * 100
    features['rsi_7_slope'] = features['rsi_7'].diff(); features['rsi_14_slope'] = features['rsi_14'].diff()
    stoch_k, stoch_d = compute_stochastic(high, low, close)
    features['stoch_k'] = stoch_k; features['stoch_d'] = stoch_d
    features['stoch_kd_diff'] = stoch_k - stoch_d
    features['macd_hist_slope'] = features['macd_hist'].diff(); features['macd_line_slope'] = features['macd_line'].diff()
    features['cci_20'] = compute_cci(high, low, close, 20)
    range_20_high = high.rolling(20).max()
    features['breakout_score'] = ((close - range_20_high.shift(1)) / range_20_high.shift(1).replace(0, np.nan) * 100).clip(lower=0)
    avg_range_20 = (high - low).rolling(20).mean()
    features['compression_score'] = (avg_range_20 / (high - low).replace(0, np.nan)).clip(upper=5)
    features['mean_rev_score'] = -((close - close.rolling(20).mean()) / close.rolling(20).std().replace(0, np.nan))
    return features

# =========================================================================
# Rule evaluation
# =========================================================================

def check_condition(feat_dict, feat, op, thresh):
    val = feat_dict.get(feat, np.nan)
    if np.isnan(val): return False
    if op == '<=': return val <= thresh
    return val > thresh

def evaluate_rules(feat_dict, rules):
    for i, rule in enumerate(rules):
        matched = True
        for feat, op, thresh in rule['conditions']:
            if not check_condition(feat_dict, feat, op, thresh):
                matched = False
                break
        if matched:
            return True, i, rule.get('risk_score', 0)
    return False, -1, 0

def filter_rules_by_score(rules, min_score=0):
    return [r for r in rules if r.get('risk_score', 0) >= min_score]

# =========================================================================
# Backtest engine (fixed exit)
# =========================================================================

def run_week_backtest(df, features_df, rules, exit_bars=3):
    """
    Returns (weekly_return, trades_list, rule_fire_counts, week_stats).
    """
    active_rules = filter_rules_by_score(rules, 0)
    if not active_rules:
        return 0.0, [], {}, {}

    cash = 100.0
    position = None
    trades = []
    rule_fire_counts = defaultdict(int)

    for i in range(len(features_df) - exit_bars):
        feat_row = features_df.iloc[i]
        price_row = df.iloc[i]
        feat_dict = feat_row.to_dict()
        feat_dict['close'] = price_row['close']

        if position is not None and i >= position['exit_bar_idx']:
            exit_price = df.iloc[i]['close']
            ret = (exit_price - position['entry_price']) / position['entry_price']
            trades.append({
                'entry_time': position['entry_time'], 'exit_time': df.index[i],
                'entry_price': position['entry_price'], 'exit_price': exit_price,
                'return_pct': ret * 100, 'rule_idx': position['rule_idx'],
                'bars_held': i - position['entry_idx'],
            })
            cash *= (1 + ret)
            position = None

        if position is None:
            matched, rule_idx, score = evaluate_rules(feat_dict, active_rules)
            if matched:
                rule_fire_counts[rule_idx] += 1
                position = {
                    'entry_price': price_row['close'], 'entry_idx': i,
                    'entry_time': df.index[i], 'rule_idx': rule_idx,
                    'rule_score': score, 'exit_bar_idx': i + exit_bars,
                }

    if position is not None:
        exit_price = df.iloc[-1]['close']
        ret = (exit_price - position['entry_price']) / position['entry_price']
        trades.append({
            'entry_time': position['entry_time'], 'exit_time': df.index[-1],
            'entry_price': position['entry_price'], 'exit_price': exit_price,
            'return_pct': ret * 100, 'rule_idx': position['rule_idx'],
            'bars_held': -1,
        })
        cash *= (1 + ret)

    weekly_return = (cash - 100.0) / 100.0 * 100

    feat_snapshot = features_df.iloc[:min(len(features_df), len(features_df)//2)]
    week_stats = {
        'n_trades': len(trades),
        'avg_vol_ratio': float(feat_snapshot['volume_ratio'].mean()) if len(feat_snapshot) > 0 else 0,
        'avg_atr_pct': float(feat_snapshot['atr_pct'].mean()) if len(feat_snapshot) > 0 else 0,
        'avg_hv': float(feat_snapshot['hv_20'].mean()) if len(feat_snapshot) > 0 else 0,
        'avg_rsi': float(feat_snapshot['rsi_14'].mean()) if len(feat_snapshot) > 0 else 0,
    }
    return weekly_return, trades, dict(rule_fire_counts), week_stats

# =========================================================================
# Run full validation across all tickers
# =========================================================================

def run_full_validation(all_data, rules, exit_bars=3):
    week_results = []
    trade_log = []
    rule_fire_total = defaultdict(int)

    for ticker, df in all_data.items():
        df_copy = df.copy()
        df_copy['weekday'] = df_copy.index.weekday
        df_copy['week_num'] = df_copy.index.isocalendar().week.astype(int)
        df_copy['year'] = df_copy.index.year

        week_groups = {}
        for (year, week), group in df_copy.groupby(['year', 'week_num'], sort=False):
            days = set(group['weekday'].unique())
            if days == {0, 1, 2, 3, 4}:
                week_groups[(year, week)] = group.index

        for week_key in sorted(week_groups.keys()):
            week_idx = week_groups[week_key]
            week_df = df.loc[week_idx].sort_index()
            features = compute_features(week_df).dropna()
            common_idx = features.index.intersection(week_df.index)
            if len(common_idx) < 20:
                continue

            features_aligned = features.loc[common_idx]
            prices_aligned = week_df.loc[common_idx]

            weekly_ret, trades, rule_fires, ws = run_week_backtest(
                prices_aligned, features_aligned, rules, exit_bars
            )

            week_results.append({
                'ticker': ticker, 'week': str(week_key),
                'weekly_return': weekly_ret, 'n_trades': ws['n_trades'],
                'avg_vol_ratio': ws['avg_vol_ratio'], 'avg_atr_pct': ws['avg_atr_pct'],
                'avg_hv': ws['avg_hv'], 'avg_rsi': ws['avg_rsi'],
            })
            for t in trades:
                t['ticker'] = ticker
                t['week'] = str(week_key)
                trade_log.append(t)
            for ridx, cnt in rule_fires.items():
                rule_fire_total[ridx] += cnt

    return week_results, trade_log, dict(rule_fire_total)

# =========================================================================
# Distribution stats
# =========================================================================

def compute_weekly_stats(week_results, label=""):
    returns = np.array([r['weekly_return'] for r in week_results])
    if len(returns) == 0:
        return {'label': label, 'n_weeks': 0}
    stats = {
        'label': label,
        'n_weeks': len(returns),
        'mean': float(np.mean(returns)),
        'median': float(np.median(returns)),
        'std': float(np.std(returns)),
        'min': float(np.min(returns)),
        'max': float(np.max(returns)),
        'p1': float(np.percentile(returns, 1)),
        'p5': float(np.percentile(returns, 5)),
        'p10': float(np.percentile(returns, 10)),
        'p25': float(np.percentile(returns, 25)),
        'p50': float(np.percentile(returns, 50)),
        'p75': float(np.percentile(returns, 75)),
        'p90': float(np.percentile(returns, 90)),
        'p95': float(np.percentile(returns, 95)),
        'p99': float(np.percentile(returns, 99)),
        'pct_positive': float((returns > 0).mean() * 100),
        'pct_above_1': float((returns > 1.0).mean() * 100),
        'pct_above_2': float((returns > 2.0).mean() * 100),
        'sharpe': float(np.mean(returns) / max(np.std(returns), 0.01)),
        'max_drawdown': float(np.min(returns)),
    }
    return stats

def print_weekly_stats(stats):
    if stats['n_weeks'] == 0:
        logger.info(f"  {stats['label']}: NO DATA")
        return
    logger.info(f"=== {stats['label']} ===")
    logger.info(f"  Weeks: {stats['n_weeks']}")
    logger.info(f"  Mean: {stats['mean']:.4f}% | Median: {stats['median']:.4f}%")
    logger.info(f"  Std: {stats['std']:.4f}% | Sharpe: {stats['sharpe']:.3f}")
    logger.info(f"  Min: {stats['min']:.2f}% | Max: {stats['max']:.2f}%")
    logger.info(f"  P5: {stats['p5']:.4f}% | P10: {stats['p10']:.4f}%")
    logger.info(f"  Positive: {stats['pct_positive']:.1f}% | >1%: {stats['pct_above_1']:.1f}%")
    logger.info(f"  Max DD: {stats['max_drawdown']:.2f}%")

# =========================================================================
# Regime classification
# =========================================================================

def classify_week_regime(week_df, features_df):
    feat_slice = features_df.iloc[:min(len(features_df), len(features_df)//2)]
    if len(feat_slice) < 5:
        return 'UNKNOWN', {}

    avg_trend_align = float(feat_slice['trend_alignment'].mean())
    avg_lr_slope = float(feat_slice['lr_slope_10'].mean())
    avg_price_pos = float(feat_slice['price_position'].mean())
    avg_hv = float(feat_slice['hv_20'].mean())
    avg_atr = float(feat_slice['atr_pct'].mean())
    avg_vol_ratio = float(feat_slice['volume_ratio'].mean())
    avg_rsi = float(feat_slice['rsi_14'].mean())
    atr_percentile = float(feat_slice['atr_percentile'].mean())
    avg_bb_pct = float(feat_slice['bb_pct'].mean())
    avg_compression = float(feat_slice['compression_score'].mean())

    is_high_vol = avg_hv > 2.5 or atr_percentile > 0.7
    is_low_vol = avg_hv < 1.0 and atr_percentile < 0.3
    is_trend_up = avg_trend_align > 1.5 and avg_lr_slope > 0 and avg_rsi > 50
    is_trend_down = avg_trend_align < -0.5 or (avg_lr_slope < -0.05 and avg_price_pos < 0.3)
    is_breakout = avg_vol_ratio > 1.3 and avg_bb_pct > 0.8 and avg_compression < 1.5
    is_mean_rev = avg_bb_pct < 0.2 and avg_rsi < 35

    if is_breakout:
        regime = 'BREAKOUT'
    elif is_mean_rev:
        regime = 'MEAN_REVERSION'
    elif is_high_vol:
        regime = 'HIGH_VOLATILITY'
    elif is_low_vol:
        regime = 'LOW_VOLATILITY'
    elif is_trend_up:
        regime = 'TRENDING_UP'
    elif is_trend_down:
        regime = 'TRENDING_DOWN'
    else:
        regime = 'RANGING'

    return regime, {
        'avg_trend_align': avg_trend_align, 'avg_lr_slope': avg_lr_slope,
        'avg_price_pos': avg_price_pos, 'avg_hv': avg_hv,
        'avg_atr': avg_atr, 'avg_vol_ratio': avg_vol_ratio,
        'avg_rsi': avg_rsi, 'atr_percentile': atr_percentile,
        'avg_bb_pct': avg_bb_pct, 'avg_compression': avg_compression,
    }

# =========================================================================
# Run with regime annotation
# =========================================================================

def run_full_validation_with_regime(all_data, rules, exit_bars=3):
    week_results = []
    trade_log = []
    rule_fire_total = defaultdict(int)

    for ticker, df in all_data.items():
        df_copy = df.copy()
        df_copy['weekday'] = df_copy.index.weekday
        df_copy['week_num'] = df_copy.index.isocalendar().week.astype(int)
        df_copy['year'] = df_copy.index.year

        week_groups = {}
        for (year, week), group in df_copy.groupby(['year', 'week_num'], sort=False):
            days = set(group['weekday'].unique())
            if days == {0, 1, 2, 3, 4}:
                week_groups[(year, week)] = group.index

        for week_key in sorted(week_groups.keys()):
            week_idx = week_groups[week_key]
            week_df = df.loc[week_idx].sort_index()
            features = compute_features(week_df).dropna()
            common_idx = features.index.intersection(week_df.index)
            if len(common_idx) < 20:
                continue

            features_aligned = features.loc[common_idx]
            prices_aligned = week_df.loc[common_idx]

            regime, regime_stats = classify_week_regime(prices_aligned, features_aligned)

            weekly_ret, trades, rule_fires, ws = run_week_backtest(
                prices_aligned, features_aligned, rules, exit_bars
            )

            week_results.append({
                'ticker': ticker, 'week': str(week_key),
                'weekly_return': weekly_ret, 'n_trades': ws['n_trades'],
                'regime': regime, **regime_stats,
            })
            for t in trades:
                t['ticker'] = ticker
                t['week'] = str(week_key)
                trade_log.append(t)
            for ridx, cnt in rule_fires.items():
                rule_fire_total[ridx] += cnt

    return week_results, trade_log, dict(rule_fire_total)


# =========================================================================
# PHASE 1: Holding Period Optimization
# =========================================================================

def phase1_holding_periods(all_data, rules, hold_periods=[1,2,3,4,5]):
    logger.info("\n" + "="*70)
    logger.info("PHASE 1: HOLDING PERIOD OPTIMIZATION")
    logger.info("="*70)

    results = {}
    for bars in hold_periods:
        logger.info(f"\n--- Testing exit_bars={bars} ({bars*15} min hold) ---")
        week_results, trade_log, _ = run_full_validation(all_data, rules, exit_bars=bars)
        stats = compute_weekly_stats(week_results, f"exit_bars={bars}")
        print_weekly_stats(stats)
        results[bars] = {
            'stats': stats,
            'week_results': week_results,
            'trade_log': trade_log,
            'n_trades': len(trade_log),
        }
    return results

# =========================================================================
# PHASE 2: Rule-Specific Holding Analysis
# =========================================================================

def phase2_rule_holding_analysis(all_data, rules, hold_periods=[1,2,3,4,5], hold_results=None):
    logger.info("\n" + "="*70)
    logger.info("PHASE 2: RULE-SPECIFIC HOLDING ANALYSIS")
    logger.info("="*70)

    per_rule_data = defaultdict(lambda: defaultdict(lambda: {'returns': [], 'trades': 0, 'wins': 0}))

    for bars in hold_periods:
        # Reuse Phase 1 trade_log if available
        if hold_results and bars in hold_results:
            trade_log = hold_results[bars]['trade_log']
            logger.info(f"  Using cached trade data for exit_bars={bars} ({len(trade_log)} trades)")
        else:
            logger.info(f"  Collecting trade data for exit_bars={bars}...")
            _, trade_log, _ = run_full_validation(all_data, rules, exit_bars=bars)

        for t in trade_log:
            ridx = t['rule_idx']
            per_rule_data[ridx][bars]['returns'].append(t['return_pct'])
            per_rule_data[ridx][bars]['trades'] += 1
            if t['return_pct'] > 0:
                per_rule_data[ridx][bars]['wins'] += 1

    rule_summaries = {}
    for ridx in sorted(per_rule_data.keys()):
        best_bars = None
        best_avg = -999
        bar_perf = {}

        for bars in hold_periods:
            d = per_rule_data[ridx][bars]
            if d['trades'] == 0:
                bar_perf[bars] = {'avg': 0, 'median': 0, 'wr': 0, 'pf': 0, 'n': 0}
                continue
            returns = np.array(d['returns'])
            avg_ret = float(np.mean(returns))
            med_ret = float(np.median(returns))
            wr = d['wins'] / d['trades']
            gross_profit = float(np.sum(returns[returns > 0])) if any(returns > 0) else 0
            gross_loss = float(abs(np.sum(returns[returns < 0]))) if any(returns < 0) else 0
            pf = gross_profit / max(gross_loss, 0.0001)
            bar_perf[bars] = {'avg': avg_ret, 'median': med_ret, 'wr': wr, 'pf': pf, 'n': d['trades']}
            if avg_ret > best_avg and d['trades'] >= 2:
                best_avg = avg_ret
                best_bars = bars

        rule_summaries[ridx] = {
            'bar_perf': bar_perf,
            'optimal_bars': best_bars if best_bars is not None else 3,
            'best_avg': best_avg,
        }

    # Summary counts
    optimal_counts = defaultdict(int)
    for ridx, rs in rule_summaries.items():
        if rs['optimal_bars'] is not None:
            optimal_counts[rs['optimal_bars']] += 1

    logger.info(f"\n--- Rule Optimal Holding Period Distribution ---")
    for bars in sorted(optimal_counts.keys()):
        logger.info(f"  {bars} candle{'s' if bars > 1 else ''}: {optimal_counts[bars]} rules")

    # Show examples
    logger.info(f"\n--- Example Rules by Optimal Period ---")
    for opt_bars in [1, 2, 3, 4, 5]:
        examples = [(ridx, rs) for ridx, rs in rule_summaries.items() if rs['optimal_bars'] == opt_bars]
        examples = sorted(examples, key=lambda x: x[1]['best_avg'], reverse=True)[:3]
        if examples:
            logger.info(f"\n  Optimal: {opt_bars} candle{'s' if opt_bars > 1 else ''}")
            for ridx, rs in examples:
                perfs = ' | '.join([f"{b}c: avg={rs['bar_perf'][b]['avg']:.2f}% n={rs['bar_perf'][b]['n']}" for b in sorted(rs['bar_perf'].keys())])
                logger.info(f"    Rule {ridx}: {perfs}")

    return rule_summaries

# =========================================================================
# PHASE 3: Pattern-Timeframe Clustering
# =========================================================================

def phase3_clustering(rule_summaries, rules):
    logger.info("\n" + "="*70)
    logger.info("PHASE 3: PATTERN-TIMEFRAME CLUSTERING")
    logger.info("="*70)

    # Classify rules by optimal holding period
    fast_rules = []   # opt=1
    medium_rules = [] # opt=2-3
    slow_rules = []   # opt=4-5

    for ridx, rs in rule_summaries.items():
        opt = rs['optimal_bars']
        if opt == 1:
            fast_rules.append(ridx)
        elif opt in [2, 3]:
            medium_rules.append(ridx)
        else:
            slow_rules.append(ridx)

    logger.info(f"\n  FAST patterns (opt=1): {len(fast_rules)} rules")
    logger.info(f"  MEDIUM patterns (opt=2-3): {len(medium_rules)} rules")
    logger.info(f"  SLOW patterns (opt=4-5): {len(slow_rules)} rules")

    # Analyze feature characteristics of each group
    # Extract condition features from rules
    group_features = {'fast': defaultdict(list), 'medium': defaultdict(list), 'slow': defaultdict(list)}

    for ridx, rule in enumerate(rules):
        if ridx not in rule_summaries:
            continue
        opt = rule_summaries[ridx]['optimal_bars']
        group = 'fast' if opt == 1 else ('medium' if opt in [2, 3] else 'slow')

        for feat, op, thresh in rule['conditions']:
            group_features[group][feat].append(thresh)

    for gname, gdict in [('FAST', group_features['fast']), ('MEDIUM', group_features['medium']), ('SLOW', group_features['slow'])]:
        logger.info(f"\n--- {gname} Pattern Feature Profile ---")
        feat_dirs = defaultdict(list)
        for feat, vals in gdict.items():
            if not vals: continue
            mean_val = np.mean(vals)
            # Determine direction: is the condition filtering for HIGH or LOW values?
            # Look at rules to see if feature is used with > or <=
            directions = []
            for ridx in (fast_rules if gname == 'FAST' else (medium_rules if gname == 'MEDIUM' else slow_rules)):
                if ridx >= len(rules): continue
                for f, op, t in rules[ridx]['conditions']:
                    if f == feat:
                        directions.append(op)
            if directions:
                pct_gt = sum(1 for d in directions if d == '>') / len(directions)
                feat_dirs[feat] = f"{(pct_gt*100):.0f}% > / {(100-pct_gt*100):.0f}% <="

        # Top features by frequency
        top_feats = sorted(gdict.keys(), key=lambda f: len(gdict[f]), reverse=True)[:10]
        logger.info(f"  Top features:")
        for f in top_feats:
            d_info = feat_dirs.get(f, "")
            logger.info(f"    {f}: {len(gdict[f])} rules {d_info}")

    return {'fast': fast_rules, 'medium': medium_rules, 'slow': slow_rules}


# =========================================================================
# PHASE 4: Adaptive Holding System
# =========================================================================

def run_week_backtest_adaptive(df, features_df, rules, rule_optimal_holds):
    """
    Each rule uses its own optimal holding period.
    """
    active_rules = filter_rules_by_score(rules, 0)
    if not active_rules:
        return 0.0, []

    cash = 100.0
    position = None
    trades = []

    # Determine max exit_bars needed
    max_hold = max(rule_optimal_holds.values()) if rule_optimal_holds else 5

    for i in range(len(features_df) - max_hold):
        feat_row = features_df.iloc[i]
        price_row = df.iloc[i]
        feat_dict = feat_row.to_dict()
        feat_dict['close'] = price_row['close']

        if position is not None and i >= position['exit_bar_idx']:
            exit_price = df.iloc[i]['close']
            ret = (exit_price - position['entry_price']) / position['entry_price']
            trades.append({
                'return_pct': ret * 100, 'rule_idx': position['rule_idx'],
                'bars_held': i - position['entry_idx'],
                'entry_time': position['entry_time'], 'exit_time': df.index[i],
            })
            cash *= (1 + ret)
            position = None

        if position is None:
            matched, rule_idx, score = evaluate_rules(feat_dict, active_rules)
            if matched:
                hold_bars = rule_optimal_holds.get(rule_idx, 3)
                position = {
                    'entry_price': price_row['close'], 'entry_idx': i,
                    'entry_time': df.index[i], 'rule_idx': rule_idx,
                    'exit_bar_idx': i + hold_bars,
                }

    if position is not None:
        exit_price = df.iloc[-1]['close']
        ret = (exit_price - position['entry_price']) / position['entry_price']
        trades.append({
            'return_pct': ret * 100, 'rule_idx': position['rule_idx'],
            'bars_held': -1, 'entry_time': position['entry_time'],
            'exit_time': df.index[-1],
        })
        cash *= (1 + ret)

    weekly_return = (cash - 100.0) / 100.0 * 100
    return weekly_return, trades

def run_full_validation_adaptive(all_data, rules, rule_optimal_holds):
    week_results = []
    trade_log = []

    for ticker, df in all_data.items():
        df_copy = df.copy()
        df_copy['weekday'] = df_copy.index.weekday
        df_copy['week_num'] = df_copy.index.isocalendar().week.astype(int)
        df_copy['year'] = df_copy.index.year

        week_groups = {}
        for (year, week), group in df_copy.groupby(['year', 'week_num'], sort=False):
            days = set(group['weekday'].unique())
            if days == {0, 1, 2, 3, 4}:
                week_groups[(year, week)] = group.index

        for week_key in sorted(week_groups.keys()):
            week_idx = week_groups[week_key]
            week_df = df.loc[week_idx].sort_index()
            features = compute_features(week_df).dropna()
            common_idx = features.index.intersection(week_df.index)
            if len(common_idx) < 20:
                continue

            features_aligned = features.loc[common_idx]
            prices_aligned = week_df.loc[common_idx]

            weekly_ret, trades = run_week_backtest_adaptive(
                prices_aligned, features_aligned, rules, rule_optimal_holds
            )

            week_results.append({
                'ticker': ticker, 'week': str(week_key),
                'weekly_return': weekly_ret, 'n_trades': len(trades),
            })
            for t in trades:
                t['ticker'] = ticker
                t['week'] = str(week_key)
                trade_log.append(t)

    return week_results, trade_log


def phase4_adaptive_holding(all_data, rules, rule_summaries, hold_periods=[1,2,3,4,5], hold_results=None):
    logger.info("\n" + "="*70)
    logger.info("PHASE 4: ADAPTIVE HOLDING SYSTEM")
    logger.info("="*70)

    # Build optimal hold map
    rule_optimal_holds = {}
    for ridx, rs in rule_summaries.items():
        opt = rs['optimal_bars']
        if opt is not None:
            rule_optimal_holds[ridx] = opt
        else:
            rule_optimal_holds[ridx] = 3  # default

    logger.info(f"  Adaptive holds assigned to {len(rule_optimal_holds)} rules")

    # Run adaptive backtest
    logger.info(f"  Running adaptive backtest...")
    week_results, trade_log = run_full_validation_adaptive(all_data, rules, rule_optimal_holds)
    stats_adaptive = compute_weekly_stats(week_results, "Adaptive Holding")
    print_weekly_stats(stats_adaptive)

    # Compare with best fixed (from cached Phase 1 results)
    best_fixed_bars = max(hold_periods, key=lambda b: hold_results[b]['stats']['sharpe'])
    best_fixed_stats = hold_results[best_fixed_bars]['stats']
    logger.info(f"  Using cached Phase 1 results — best fixed: {best_fixed_bars}c")

    logger.info(f"\n--- Adaptive vs Best Fixed Comparison ---")
    logger.info(f"  Best fixed: {best_fixed_bars} candles")
    logger.info(f"  Metric         | Fixed {best_fixed_bars}c | Adaptive | Delta")
    logger.info(f"  {'-'*55}")
    for metric in ['mean', 'median', 'p5', 'p10', 'pct_positive', 'sharpe', 'max_drawdown']:
        fv = best_fixed_stats.get(metric, 0)
        av = stats_adaptive.get(metric, 0)
        if isinstance(fv, float):
            delta = av - fv
            logger.info(f"  {metric:<16} | {fv:<10.4f} | {av:<9.4f} | {delta:<+9.4f}")
        else:
            logger.info(f"  {metric:<16} | {fv:<10} | {av:<9} | ")

    return {
        'adaptive_stats': stats_adaptive,
        'best_fixed_bars': best_fixed_bars,
        'best_fixed_stats': best_fixed_stats,
        'adaptive_week_results': week_results,
        'adaptive_trade_log': trade_log,
    }


# =========================================================================
# PHASE 5: Regime Analysis
# =========================================================================

def phase5_regime_analysis(all_data, rules, exit_bars=3):
    logger.info("\n" + "="*70)
    logger.info("PHASE 5: REGIME ANALYSIS")
    logger.info("="*70)

    week_results, trade_log, _ = run_full_validation_with_regime(all_data, rules, exit_bars)

    regimes = set(r['regime'] for r in week_results)
    regime_stats = {}
    for regime in sorted(regimes):
        rweeks = [r for r in week_results if r['regime'] == regime]
        returns = np.array([r['weekly_return'] for r in rweeks])
        stats = {
            'n_weeks': len(rweeks),
            'pct_of_total': len(rweeks) / max(len(week_results), 1) * 100,
            'mean': float(np.mean(returns)),
            'median': float(np.median(returns)),
            'std': float(np.std(returns)),
            'min': float(np.min(returns)),
            'max': float(np.max(returns)),
            'pct_positive': float((returns > 0).mean() * 100),
            'sharpe': float(np.mean(returns) / max(np.std(returns), 0.01)),
        }
        regime_stats[regime] = stats
        logger.info(f"\n  --- {regime} ({stats['n_weeks']} weeks, {stats['pct_of_total']:.0f}%) ---")
        logger.info(f"    Mean: {stats['mean']:.2f}% | Median: {stats['median']:.2f}%")
        logger.info(f"    Positive: {stats['pct_positive']:.1f}% | Sharpe: {stats['sharpe']:.2f}")
        logger.info(f"    Best: {stats['max']:.2f}% | Worst: {stats['min']:.2f}%")

    return regime_stats, week_results, trade_log

# =========================================================================
# PHASE 6: Losing Week Investigation
# =========================================================================

def phase6_losing_weeks(week_results, trade_log, rules):
    logger.info("\n" + "="*70)
    logger.info("PHASE 6: LOSING WEEK INVESTIGATION")
    logger.info("="*70)

    all_returns = np.array([r['weekly_return'] for r in week_results])
    loss_mask = all_returns < 0
    loss_big_mask = all_returns < -2
    loss_severe_mask = all_returns < -4

    losses = [r for r in week_results if r['weekly_return'] < 0]
    big_losses = [r for r in week_results if r['weekly_return'] < -2]
    severe_losses = [r for r in week_results if r['weekly_return'] < -4]

    logger.info(f"\n  All weeks: {len(week_results)}")
    logger.info(f"  Loss weeks (<0%): {len(losses)} ({len(losses)/len(week_results)*100:.1f}%)")
    logger.info(f"  Big loss weeks (<-2%): {len(big_losses)} ({len(big_losses)/len(week_results)*100:.1f}%)")
    logger.info(f"  Severe loss weeks (<-4%): {len(severe_losses)} ({len(severe_losses)/len(week_results)*100:.1f}%)")

    # Regime distribution in loss weeks
    if 'regime' in losses[0]:
        logger.info(f"\n  --- Regime Distribution of Loss Weeks ---")
        for label, week_set in [("All Losses (<0%)", losses), ("Big Losses (<-2%)", big_losses), ("Severe Losses (<-4%)", severe_losses)]:
            if not week_set: continue
            regime_counts = Counter(r['regime'] for r in week_set)
            logger.info(f"\n  {label}:")
            for regime, count in regime_counts.most_common():
                all_regime = sum(1 for r in week_results if r.get('regime') == regime)
                pct_loss = count / len(week_set) * 100
                pct_all = all_regime / max(len(week_results), 1) * 100
                enrichment = pct_loss / max(pct_all, 0.1)
                logger.info(f"    {regime}: {count} ({pct_loss:.0f}% of losses, {pct_all:.0f}% of all, {enrichment:.1f}x enriched)")

    # Average characteristics before loss weeks
    logger.info(f"\n  --- Average Characteristics: Loss vs All Weeks ---")
    for metric in ['avg_hv', 'atr_percentile', 'avg_vol_ratio', 'avg_trend_align', 'avg_rsi', 'avg_bb_pct', 'n_trades']:
        if metric not in week_results[0]:
            continue
        loss_vals = [r.get(metric, 0) for r in losses]
        all_vals = [r.get(metric, 0) for r in week_results]
        if loss_vals:
            logger.info(f"    {metric}: loss_mean={np.mean(loss_vals):.3f} vs all_mean={np.mean(all_vals):.3f}")

    # Rule fire patterns in loss weeks
    loss_weeks_set = set((r['ticker'], r['week']) for r in losses)

    # Find which rules fired in loss weeks
    loss_trades = [t for t in trade_log if (t['ticker'], t['week']) in loss_weeks_set]
    rule_loss_count = Counter(t['rule_idx'] for t in loss_trades)
    rule_loss_total = Counter(t['rule_idx'] for t in trade_log)

    logger.info(f"\n  --- Rules Most Active in Loss Weeks ---")
    for ridx, count in rule_loss_count.most_common(10):
        total = rule_loss_total.get(ridx, 1)
        ratio = count / max(len(losses), 1)
        logger.info(f"    Rule {ridx}: {count} trades in loss weeks ({ratio:.2f}/loss week), {total} total trades")

    return {
        'losses': losses, 'big_losses': big_losses, 'severe_losses': severe_losses,
        'loss_trades': loss_trades, 'rule_loss_count': rule_loss_count,
    }

# =========================================================================
# PHASE 7: Regime Filter Simulation
# =========================================================================

def run_week_backtest_with_filter(df, features_df, rules, exit_bars=3, filter_func=None):
    """
    filter_func(feat_dict, week_stats) -> True means ALLOW trade, False means SKIP
    """
    active_rules = filter_rules_by_score(rules, 0)
    if not active_rules:
        return 0.0, []

    cash = 100.0
    position = None
    trades = []
    skipped = 0

    # Compute week-level stats for filter
    feat_slice = features_df.iloc[:min(len(features_df), len(features_df)//2)]
    week_stats = {
        'avg_hv': float(feat_slice['hv_20'].mean()) if len(feat_slice) > 0 else 0,
        'avg_atr_pct': float(feat_slice['atr_pct'].mean()) if len(feat_slice) > 0 else 0,
        'avg_vol_ratio': float(feat_slice['volume_ratio'].mean()) if len(feat_slice) > 0 else 0,
        'avg_trend_align': float(feat_slice['trend_alignment'].mean()) if len(feat_slice) > 0 else 0,
        'avg_rsi': float(feat_slice['rsi_14'].mean()) if len(feat_slice) > 0 else 0,
        'atr_percentile': float(feat_slice['atr_percentile'].mean()) if len(feat_slice) > 0 else 0,
    }

    for i in range(len(features_df) - exit_bars):
        feat_row = features_df.iloc[i]
        price_row = df.iloc[i]
        feat_dict = feat_row.to_dict()
        feat_dict['close'] = price_row['close']

        if position is not None and i >= position['exit_bar_idx']:
            exit_price = df.iloc[i]['close']
            ret = (exit_price - position['entry_price']) / position['entry_price']
            trades.append({
                'return_pct': ret * 100, 'rule_idx': position['rule_idx'],
                'entry_time': position['entry_time'], 'exit_time': df.index[i],
            })
            cash *= (1 + ret)
            position = None

        if position is None:
            if filter_func is not None and not filter_func(feat_dict, week_stats):
                skipped += 1
                continue

            matched, rule_idx, score = evaluate_rules(feat_dict, active_rules)
            if matched:
                position = {
                    'entry_price': price_row['close'], 'entry_idx': i,
                    'entry_time': df.index[i], 'rule_idx': rule_idx,
                    'exit_bar_idx': i + exit_bars,
                }

    if position is not None:
        exit_price = df.iloc[-1]['close']
        ret = (exit_price - position['entry_price']) / position['entry_price']
        trades.append({'return_pct': ret * 100, 'rule_idx': position['rule_idx'],
                       'entry_time': position['entry_time'], 'exit_time': df.index[-1]})
        cash *= (1 + ret)

    weekly_return = (cash - 100.0) / 100.0 * 100
    return weekly_return, trades, skipped

def run_full_validation_with_filter(all_data, rules, exit_bars=3, filter_func=None, filter_name="no_filter"):
    week_results = []
    trade_log = []
    total_skipped = 0

    for ticker, df in all_data.items():
        df_copy = df.copy()
        df_copy['weekday'] = df_copy.index.weekday
        df_copy['week_num'] = df_copy.index.isocalendar().week.astype(int)
        df_copy['year'] = df_copy.index.year

        week_groups = {}
        for (year, week), group in df_copy.groupby(['year', 'week_num'], sort=False):
            days = set(group['weekday'].unique())
            if days == {0, 1, 2, 3, 4}:
                week_groups[(year, week)] = group.index

        for week_key in sorted(week_groups.keys()):
            week_idx = week_groups[week_key]
            week_df = df.loc[week_idx].sort_index()
            features = compute_features(week_df).dropna()
            common_idx = features.index.intersection(week_df.index)
            if len(common_idx) < 20:
                continue

            features_aligned = features.loc[common_idx]
            prices_aligned = week_df.loc[common_idx]

            weekly_ret, trades, skipped = run_week_backtest_with_filter(
                prices_aligned, features_aligned, rules, exit_bars, filter_func
            )
            total_skipped += skipped
            week_results.append({
                'ticker': ticker, 'week': str(week_key),
                'weekly_return': weekly_ret, 'n_trades': len(trades),
            })
            for t in trades:
                t['ticker'] = ticker
                t['week'] = str(week_key)
                trade_log.append(t)

    return week_results, trade_log, total_skipped


def phase7_regime_filters(all_data, rules, exit_bars=3):
    logger.info("\n" + "="*70)
    logger.info("PHASE 7: REGIME FILTER SIMULATION")
    logger.info("="*70)

    # Run baseline (no filter)
    logger.info(f"\n  Baseline (no filter)...")
    base_results, base_trades, _ = run_full_validation_with_filter(all_data, rules, exit_bars, None)
    base_stats = compute_weekly_stats(base_results, "No Filter (Baseline)")

    # Define filters
    filters = {}

    def filter_no_low_volume(feat_dict, week_stats):
        """Skip if volume is below 50% of 20-period average"""
        return feat_dict.get('volume_ratio', 1) >= 0.5
    filters['min_vol_0.5'] = filter_no_low_volume

    def filter_no_low_volume_strict(feat_dict, week_stats):
        return feat_dict.get('volume_percentile', 0) >= 0.2
    filters['min_vol_pct_20'] = filter_no_low_volume_strict

    def filter_no_high_atr(feat_dict, week_stats):
        """Skip if ATR percentile is above 80% (extreme volatility)"""
        return feat_dict.get('atr_percentile', 0) <= 0.8
    filters['max_atr_0.8'] = filter_no_high_atr

    def filter_no_high_atr_strict(feat_dict, week_stats):
        return feat_dict.get('atr_percentile', 0) <= 0.7
    filters['max_atr_0.7'] = filter_no_high_atr_strict

    def filter_trend_up_only(feat_dict, week_stats):
        """Only trade when market is trending up"""
        return week_stats.get('avg_trend_align', 0) >= 1.0
    filters['min_trend_1.0'] = filter_trend_up_only

    def filter_no_rsi_extreme(feat_dict, week_stats):
        """Skip if RSI is extreme (overbought or oversold)"""
        rsi = feat_dict.get('rsi_14', 50)
        return 25 <= rsi <= 75
    filters['rsi_25_75'] = filter_no_rsi_extreme

    def filter_combined(feat_dict, week_stats):
        """Combine multiple filters"""
        if feat_dict.get('atr_percentile', 0) > 0.8:
            return False
        if feat_dict.get('volume_percentile', 1) < 0.2:
            return False
        if not (25 <= feat_dict.get('rsi_14', 50) <= 75):
            return False
        return True
    filters['combined'] = filter_combined

    # Run each filter
    filter_results = {}
    for fname, ffunc in filters.items():
        logger.info(f"  Testing filter: {fname}...")
        try:
            f_results, f_trades, skipped = run_full_validation_with_filter(all_data, rules, exit_bars, ffunc)
            f_stats = compute_weekly_stats(f_results, fname)
            filter_results[fname] = {
                'stats': f_stats,
                'n_trades': len(f_trades),
                'skipped': skipped,
                'week_results': f_results,
            }
            logger.info(f"    Trades: {len(f_trades)} | Skipped: {skipped}")
            logger.info(f"    Mean: {f_stats['mean']:.2f}% | P5: {f_stats['p5']:.2f}% | Pos: {f_stats['pct_positive']:.1f}%")
        except Exception as e:
            logger.error(f"    Filter {fname} failed: {e}")

    # Comparison table
    logger.info(f"\n--- Filter Comparison ---")
    logger.info(f"{'Filter':<22} {'Mean':>8} {'Median':>8} {'P5':>8} {'P10':>8} {'Pos%':>7} {'Sharpe':>8} {'Trades':>8}")
    logger.info("-" * 80)
    bs = base_stats
    logger.info(f"{'NO FILTER':<22} {bs['mean']:>8.2f} {bs['median']:>8.2f} {bs['p5']:>8.2f} {bs['p10']:>8.2f} {bs['pct_positive']:>7.1f} {bs['sharpe']:>8.2f} {len(base_trades):>8}")
    for fname, fdata in filter_results.items():
        s = fdata['stats']
        logger.info(f"{fname:<22} {s['mean']:>8.2f} {s['median']:>8.2f} {s['p5']:>8.2f} {s['p10']:>8.2f} {s['pct_positive']:>7.1f} {s['sharpe']:>8.2f} {fdata['n_trades']:>8}")

    return filter_results, base_stats, base_trades


# =========================================================================
# Save Report
# =========================================================================

def save_report(hold_results, rule_summaries, clusters, adaptive_data,
                regime_stats, loss_data, filter_results, base_stats, base_trades):
    with open(REPORT_FILE, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("HOLDING PERIOD OPTIMIZATION + REGIME ANALYSIS\n")
        f.write("=" * 70 + "\n\n")

        # Section 1: Holding Period Results
        f.write("-" * 70 + "\n")
        f.write("SECTION 1: HOLDING PERIOD RESULTS\n")
        f.write("-" * 70 + "\n\n")
        f.write(f"{'Hold Period':<15} {'Mean':>8} {'Median':>8} {'P5':>8} {'P10':>8} {'Pos%':>7} {'Sharpe':>8} {'Trades':>8}\n")
        f.write("-" * 70 + "\n")
        for bars in sorted(hold_results.keys()):
            s = hold_results[bars]['stats']
            f.write(f"{bars} candles ({bars*15}min): {'Mean':>8} {s['mean']:>8.2f} {s['median']:>8.2f} {s['p5']:>8.2f} {s['p10']:>8.2f} {s['pct_positive']:>7.1f} {s['sharpe']:>8.2f} {hold_results[bars]['n_trades']:>8}\n".replace("Mean:", f"{bars:<15}"))
        f.write("\n")

        # Adaptive comparison
        f.write(f"\n--- Adaptive Holding ---\n")
        ad = adaptive_data
        f.write(f"  Adaptive: Mean={ad['adaptive_stats']['mean']:.2f}% Median={ad['adaptive_stats']['median']:.2f}% P5={ad['adaptive_stats']['p5']:.2f}% Pos={ad['adaptive_stats']['pct_positive']:.1f}% Sharpe={ad['adaptive_stats']['sharpe']:.2f}\n")
        f.write(f"  Best Fixed ({ad['best_fixed_bars']}c): Mean={ad['best_fixed_stats']['mean']:.2f}% P5={ad['best_fixed_stats']['p5']:.2f}%\n\n")

        # Section 2: Rule Timing
        f.write("-" * 70 + "\n")
        f.write("SECTION 2: RULE TIMING RESULTS\n")
        f.write("-" * 70 + "\n\n")

        optimal_counts = defaultdict(int)
        for ridx, rs in rule_summaries.items():
            if rs['optimal_bars'] is not None:
                optimal_counts[rs['optimal_bars']] += 1
        f.write(f"  Total rules analyzed: {len(rule_summaries)}\n")
        for bars in sorted(optimal_counts.keys()):
            f.write(f"  Optimal {bars}c hold: {optimal_counts[bars]} rules\n")

        f.write(f"\n  Quick (1c): {len(clusters['fast'])} rules\n")
        f.write(f"  Medium (2-3c): {len(clusters['medium'])} rules\n")
        f.write(f"  Slow (4-5c): {len(clusters['slow'])} rules\n\n")

        # Section 3: Regime Analysis
        f.write("-" * 70 + "\n")
        f.write("SECTION 3: REGIME ANALYSIS RESULTS\n")
        f.write("-" * 70 + "\n\n")
        f.write(f"{'Regime':<20} {'Weeks':>7} {'%Total':>8} {'Mean':>8} {'Median':>8} {'Pos%':>7} {'Sharpe':>8} {'Worst':>8}\n")
        f.write("-" * 80 + "\n")
        for regime, s in sorted(regime_stats.items()):
            f.write(f"{regime:<20} {s['n_weeks']:>7} {s['pct_of_total']:>7.1f}% {s['mean']:>8.2f} {s['median']:>8.2f} {s['pct_positive']:>7.1f} {s['sharpe']:>8.2f} {s['min']:>8.2f}\n")
        f.write("\n")

        # Section 4: Filter Results
        f.write("-" * 70 + "\n")
        f.write("SECTION 4: REGIME FILTER RESULTS\n")
        f.write("-" * 70 + "\n\n")
        f.write(f"{'Filter':<22} {'Mean':>8} {'Median':>8} {'P5':>8} {'Pos%':>7} {'Sharpe':>8} {'Trades':>8}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'NO FILTER':<22} {base_stats['mean']:>8.2f} {base_stats['median']:>8.2f} {base_stats['p5']:>8.2f} {base_stats['pct_positive']:>7.1f} {base_stats['sharpe']:>8.2f} {len(base_trades):>8}\n")
        for fname, fdata in sorted(filter_results.items()):
            s = fdata['stats']
            f.write(f"{fname:<22} {s['mean']:>8.2f} {s['median']:>8.2f} {s['p5']:>8.2f} {s['pct_positive']:>7.1f} {s['sharpe']:>8.2f} {fdata['n_trades']:>8}\n")
        f.write("\n")

        # Final recommendation
        f.write("=" * 70 + "\n")
        f.write("FINAL RECOMMENDATION\n")
        f.write("=" * 70 + "\n\n")

        best_fixed_bars = max(hold_results.keys(), key=lambda b: hold_results[b]['stats']['sharpe'])
        f.write(f"Phase 1 - Best fixed holding period: {best_fixed_bars} candle(s) ({best_fixed_bars*15}min)\n")
        f.write(f"  Sharpe={hold_results[best_fixed_bars]['stats']['sharpe']:.3f} vs baseline 3c={hold_results[3]['stats']['sharpe']:.3f}\n\n")

        f.write(f"Phase 4 - Adaptive vs Fixed {adaptive_data['best_fixed_bars']}c:\n")
        for m in ['mean', 'median', 'p5', 'sharpe']:
            f.write(f"  {m}: {adaptive_data['best_fixed_stats'].get(m,0):.4f} -> {adaptive_data['adaptive_stats'].get(m,0):.4f}\n")
        f.write("\n")

        # Recommended approach
        best_sharpe = hold_results[best_fixed_bars]['stats']['sharpe']
        adaptive_sharpe = adaptive_data['adaptive_stats']['sharpe']

        if adaptive_sharpe > best_sharpe * 1.05:
            f.write("RECOMMENDATION: C - Use adaptive rule-specific exits\n")
        elif best_fixed_bars != 3:
            f.write(f"RECOMMENDATION: B - Use fixed {best_fixed_bars}-candle exit\n")
        else:
            f.write("RECOMMENDATION: A - Keep fixed 3-candle exit\n")

        # Check if any filter helps
        best_filter = None
        best_filter_sharpe = 0
        for fname, fdata in filter_results.items():
            if fdata['stats']['sharpe'] > best_filter_sharpe:
                best_filter_sharpe = fdata['stats']['sharpe']
                best_filter = fname
        if best_filter and best_filter_sharpe > base_stats['sharpe'] * 1.05:
            f.write(f"  + Add regime filter: {best_filter} (Sharpe {base_stats['sharpe']:.3f} -> {best_filter_sharpe:.3f})\n")
            f.write(f"  RECOMMENDATION: E - Adaptive exits + regime filter\n")

    logger.info(f"\nReport saved to {REPORT_FILE}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    # Load data
    ckpt = os.path.join(CHECKPOINT_DIR, 'step1_data.pkl')
    if not os.path.exists(ckpt):
        logger.error("No cached data. Run regression_pipeline.py first.")
        return
    with open(ckpt, 'rb') as f:
        train_data, valid_data = pickle.load(f)
    all_data = {**train_data, **valid_data}
    logger.info(f"Loaded data: {len(all_data)} tickers")

    # Load optimized Stage 3 rules
    rules_path = 'regression_rules.json'
    if not os.path.exists(rules_path):
        logger.error("No regression_rules.json found.")
        return
    with open(rules_path, 'r') as f:
        rule_lib = json.load(f)
    rules = rule_lib.get('stage_3', [])
    logger.info(f"Loaded {len(rules)} Stage 3 rules")

    hold_periods = [1, 2, 3, 4, 5]

    # =============================================================
    # PHASE 1
    # =============================================================
    hp_cache = 'hp_phase1.pkl'
    if os.path.exists(hp_cache):
        logger.info("Loading cached Phase 1 results...")
        with open(hp_cache, 'rb') as f:
            hold_results = pickle.load(f)
    else:
        hold_results = phase1_holding_periods(all_data, rules, hold_periods)
        with open(hp_cache, 'wb') as f:
            pickle.dump(hold_results, f)

    # =============================================================
    # PHASE 2
    # =============================================================
    p2_cache = 'hp_phase2.pkl'
    if os.path.exists(p2_cache):
        logger.info("Loading cached Phase 2 results...")
        with open(p2_cache, 'rb') as f:
            rule_summaries = pickle.load(f)
    else:
        rule_summaries = phase2_rule_holding_analysis(all_data, rules, hold_periods, hold_results)
        with open(p2_cache, 'wb') as f:
            pickle.dump(rule_summaries, f)

    # =============================================================
    # PHASE 3
    # =============================================================
    clusters = phase3_clustering(rule_summaries, rules)

    # =============================================================
    # PHASE 4
    # =============================================================
    p4_cache = 'hp_phase4.pkl'
    if os.path.exists(p4_cache):
        logger.info("Loading cached Phase 4 results...")
        with open(p4_cache, 'rb') as f:
            adaptive_data = pickle.load(f)
    else:
        adaptive_data = phase4_adaptive_holding(all_data, rules, rule_summaries, hold_periods, hold_results)
        with open(p4_cache, 'wb') as f:
            pickle.dump(adaptive_data, f)

    # =============================================================
    # PHASE 5 + 6 (share data)
    # =============================================================
    p56_cache = 'hp_phase56.pkl'
    if os.path.exists(p56_cache):
        logger.info("Loading cached Phase 5+6 results...")
        with open(p56_cache, 'rb') as f:
            regime_stats, rw, tl, loss_data = pickle.load(f)
    else:
        regime_stats, rw, tl = phase5_regime_analysis(all_data, rules, 3)
        logger.info("\n" + "="*70)
        logger.info("PHASE 6: LOSING WEEK INVESTIGATION")
        logger.info("="*70)
        loss_data = phase6_losing_weeks(rw, tl, rules)
        with open(p56_cache, 'wb') as f:
            pickle.dump((regime_stats, rw, tl, loss_data), f)

    # =============================================================
    # PHASE 7
    # =============================================================
    p7_cache = 'hp_phase7.pkl'
    if os.path.exists(p7_cache):
        logger.info("Loading cached Phase 7 results...")
        with open(p7_cache, 'rb') as f:
            filter_results, base_stats, base_trades = pickle.load(f)
    else:
        filter_results, base_stats, base_trades = phase7_regime_filters(all_data, rules, 3)
        with open(p7_cache, 'wb') as f:
            pickle.dump((filter_results, base_stats, base_trades), f)

    # =============================================================
    # Save report
    # =============================================================
    save_report(hold_results, rule_summaries, clusters, adaptive_data,
                regime_stats, loss_data, filter_results, base_stats, base_trades)

    logger.info("\n" + "="*70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
