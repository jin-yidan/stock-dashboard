"""
Trading Strategy Service.

Implements sophisticated trading strategies from the InStock system.
Each strategy has specific conditions for entry signals.

Strategies:
1. Backtrace MA250 (回踩年线) - Pullback after breaking 250-day MA
2. Platform Breakthrough (平台突破) - Breakout from consolidation
3. Climax Limit Down (放量跌停) - Panic selling reversal
4. Low Backtrace Increase (无大幅回撤) - Quality uptrend with controlled drawdowns
5. Keep Increasing (持续增长) - Sustained MA progression
6. Turtle Trading (海龟法则) - Breakout above 60-day high
7. High Tight Flag (高紧旗形) - Continuation pattern with institutional support
8. Parking Apron (停机坪) - Post-breakout consolidation
9. Low ATR Growth (低波动增长) - Low volatility steady growth
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def _to_native(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_native(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, np.bool)):
        return bool(obj)
    return obj


def check_volume_surge(df, threshold=2.0, amount_min=200000000):
    """
    Check if there's a volume surge with minimum amount.

    Args:
        df: DataFrame with OHLCV data
        threshold: Volume ratio threshold (default 2x average)
        amount_min: Minimum trading amount (default 200M)

    Returns:
        bool: True if volume surge detected
    """
    if df.empty or len(df) < 6:
        return False

    latest = df.iloc[-1]
    volume = latest.get('volume', 0)
    close = latest.get('close', 0)

    # Check minimum amount
    if 'amount' in df.columns:
        amount = latest.get('amount', 0)
    else:
        amount = volume * close

    if amount < amount_min:
        return False

    # Calculate 5-day average volume
    vol_ma5 = df['volume'].tail(6).head(5).mean()

    if vol_ma5 <= 0:
        return False

    vol_ratio = volume / vol_ma5
    return vol_ratio >= threshold


def strategy_backtrace_ma250(df, threshold=60):
    """
    Backtrace MA250 Strategy (回踩年线).

    Catches pullbacks after price breaks above the 250-day MA.

    Conditions:
    1. Price crosses above MA250 in the front period
    2. Pullback stays ABOVE MA250 (support test)
    3. Pullback duration: 10-50 days from peak
    4. Volume contracts: peak_vol / trough_vol > 2
    5. Price retracement: trough / peak < 0.8

    Args:
        df: DataFrame with OHLCV data (needs 250+ days)
        threshold: Lookback period (default 60 days)

    Returns:
        dict: Strategy result with 'triggered' and 'details'
    """
    if df.empty or len(df) < 250:
        return {'triggered': False, 'reason': 'Insufficient data (need 250+ days)'}

    df = df.copy()

    # Calculate MA250
    df['ma250'] = df['close'].rolling(window=250).mean()

    # Get recent data
    recent = df.tail(threshold).copy()

    if recent.empty or recent['ma250'].isna().all():
        return {'triggered': False, 'reason': 'MA250 not available'}

    # Find highest and lowest points
    highest_idx = recent['close'].idxmax()
    highest_row = recent.loc[highest_idx]
    highest_close = highest_row['close']
    highest_vol = highest_row['volume']
    highest_date = highest_row.get('trade_date', highest_row.name)

    # Split into front (before peak) and end (after peak) periods
    front = recent.loc[:highest_idx]
    end = recent.loc[highest_idx:]

    if len(front) < 5:
        return {'triggered': False, 'reason': 'Peak too early in period'}

    # Check: Front period crosses from below to above MA250
    front_start_above = front.iloc[0]['close'] > front.iloc[0]['ma250']
    front_end_above = front.iloc[-1]['close'] > front.iloc[-1]['ma250']

    if not (not front_start_above and front_end_above):
        return {'triggered': False, 'reason': 'No MA250 breakout in front period'}

    # Check: End period stays above MA250
    for _, row in end.iterrows():
        if row['close'] < row['ma250']:
            return {'triggered': False, 'reason': 'Price fell below MA250 in pullback'}

    # Find lowest point in end period
    if len(end) < 2:
        return {'triggered': False, 'reason': 'Pullback period too short'}

    lowest_idx = end['close'].idxmin()
    lowest_row = end.loc[lowest_idx]
    lowest_close = lowest_row['close']
    lowest_vol = lowest_row['volume']
    lowest_date = lowest_row.get('trade_date', lowest_row.name)

    # Check duration: 10-50 days
    try:
        if isinstance(highest_date, str):
            highest_dt = datetime.strptime(highest_date, '%Y-%m-%d')
        else:
            highest_dt = pd.to_datetime(highest_date)

        if isinstance(lowest_date, str):
            lowest_dt = datetime.strptime(lowest_date, '%Y-%m-%d')
        else:
            lowest_dt = pd.to_datetime(lowest_date)

        date_diff = (lowest_dt - highest_dt).days

        if not (10 <= date_diff <= 50):
            return {'triggered': False, 'reason': f'Pullback duration {date_diff} days (need 10-50)'}
    except:
        pass  # Skip date check if parsing fails

    # Check volume contraction
    if lowest_vol <= 0:
        return {'triggered': False, 'reason': 'Invalid volume data'}

    vol_ratio = highest_vol / lowest_vol
    if vol_ratio < 2:
        return {'triggered': False, 'reason': f'Volume ratio {vol_ratio:.1f} < 2 (insufficient contraction)'}

    # Check price retracement
    back_ratio = lowest_close / highest_close
    if back_ratio >= 0.8:
        return {'triggered': False, 'reason': f'Price ratio {back_ratio:.2f} >= 0.8 (pullback too shallow)'}

    return {
        'triggered': True,
        'strategy': 'backtrace_ma250',
        'name_cn': '回踩年线',
        'details': {
            'highest_price': round(highest_close, 2),
            'lowest_price': round(lowest_close, 2),
            'retracement': round((1 - back_ratio) * 100, 1),
            'volume_ratio': round(vol_ratio, 1),
            'ma250': round(lowest_row['ma250'], 2)
        }
    }


def strategy_platform_breakthrough(df, threshold=60):
    """
    Platform Breakthrough Strategy (平台突破).

    Identifies breakout from tight consolidation near MA60.

    Conditions:
    1. Breakout candle: Open < MA60 <= Close
    2. Volume surge on breakout
    3. Prior consolidation: prices within -5% to +20% of MA60

    Args:
        df: DataFrame with OHLCV data
        threshold: Lookback period (default 60 days)

    Returns:
        dict: Strategy result
    """
    if df.empty or len(df) < threshold:
        return {'triggered': False, 'reason': 'Insufficient data'}

    df = df.copy()

    # Calculate MA60
    df['ma60'] = df['close'].rolling(window=60).mean()

    recent = df.tail(threshold).copy()

    if recent['ma60'].isna().all():
        return {'triggered': False, 'reason': 'MA60 not available'}

    # Find breakthrough candle
    breakthrough_date = None
    breakthrough_row = None

    for i in range(len(recent)):
        row = recent.iloc[i]
        if pd.isna(row['ma60']) or row['ma60'] <= 0:
            continue

        # Check: open < MA60 <= close (breakthrough candle)
        if row['open'] < row['ma60'] <= row['close']:
            # Check volume surge
            if i >= 5:
                vol_ma5 = recent.iloc[i-5:i]['volume'].mean()
                if vol_ma5 > 0 and row['volume'] / vol_ma5 >= 2:
                    breakthrough_date = row.get('trade_date', row.name)
                    breakthrough_row = row
                    break

    if breakthrough_date is None:
        return {'triggered': False, 'reason': 'No breakthrough candle found'}

    # Check prior consolidation
    pre_breakthrough = recent.loc[:breakthrough_date].iloc[:-1]

    if len(pre_breakthrough) < 10:
        return {'triggered': False, 'reason': 'Insufficient consolidation period'}

    for _, row in pre_breakthrough.iterrows():
        if pd.isna(row['ma60']) or row['ma60'] <= 0:
            continue
        deviation = (row['ma60'] - row['close']) / row['ma60']
        if not (-0.05 < deviation < 0.2):
            return {'triggered': False, 'reason': 'Prior consolidation violated'}

    return {
        'triggered': True,
        'strategy': 'platform_breakthrough',
        'name_cn': '平台突破',
        'details': {
            'breakthrough_date': str(breakthrough_date),
            'close': round(breakthrough_row['close'], 2),
            'ma60': round(breakthrough_row['ma60'], 2),
            'volume_ratio': round(breakthrough_row['volume'] / pre_breakthrough['volume'].mean(), 1)
        }
    }


def strategy_climax_limitdown(df, threshold=60):
    """
    Climax Limit Down Strategy (放量跌停).

    Identifies potential reversal after panic selling.

    Conditions:
    1. Drop > 9.5% (near limit down)
    2. Trading amount > 200M yuan
    3. Volume >= 4x 5-day average (capitulation volume)

    Args:
        df: DataFrame with OHLCV data
        threshold: Lookback period (default 60 days)

    Returns:
        dict: Strategy result
    """
    if df.empty or len(df) < threshold:
        return {'triggered': False, 'reason': 'Insufficient data'}

    latest = df.iloc[-1]

    # Calculate change percentage
    if 'change_pct' in df.columns:
        p_change = latest['change_pct']
    else:
        prev_close = df.iloc[-2]['close']
        p_change = (latest['close'] - prev_close) / prev_close * 100

    # Condition 1: Drop > 9.5%
    if p_change > -9.5:
        return {'triggered': False, 'reason': f'Drop {p_change:.1f}% < 9.5%'}

    # Condition 2: Amount > 200M
    if 'amount' in df.columns:
        amount = latest['amount']
    else:
        amount = latest['close'] * latest['volume']

    if amount < 200000000:
        return {'triggered': False, 'reason': f'Amount {amount/1e8:.1f}亿 < 2亿'}

    # Condition 3: Volume >= 4x average
    vol_ma5 = df.tail(6).head(5)['volume'].mean()

    if vol_ma5 <= 0:
        return {'triggered': False, 'reason': 'Invalid volume data'}

    vol_ratio = latest['volume'] / vol_ma5

    if vol_ratio < 4:
        return {'triggered': False, 'reason': f'Volume ratio {vol_ratio:.1f} < 4'}

    return {
        'triggered': True,
        'strategy': 'climax_limitdown',
        'name_cn': '放量跌停',
        'details': {
            'drop_pct': round(p_change, 1),
            'amount_yi': round(amount / 1e8, 2),
            'volume_ratio': round(vol_ratio, 1)
        }
    }


def strategy_low_backtrace_increase(df, threshold=60):
    """
    Low Backtrace Increase Strategy (无大幅回撤).

    Identifies quality uptrends with controlled drawdowns.

    Conditions:
    1. 60-day gain > 60%
    2. No single day drops > 7%
    3. No 2-day cumulative drops > 10%
    4. No high-open-low-close drops > 7%

    Args:
        df: DataFrame with OHLCV data
        threshold: Lookback period (default 60 days)

    Returns:
        dict: Strategy result
    """
    if df.empty or len(df) < threshold:
        return {'triggered': False, 'reason': 'Insufficient data'}

    recent = df.tail(threshold).copy()

    # Condition 1: 60-day gain > 60%
    start_close = recent.iloc[0]['close']
    end_close = recent.iloc[-1]['close']

    ratio_increase = (end_close - start_close) / start_close

    if ratio_increase < 0.6:
        return {'triggered': False, 'reason': f'Gain {ratio_increase*100:.1f}% < 60%'}

    # Calculate daily changes
    if 'change_pct' in recent.columns:
        recent['p_change'] = recent['change_pct']
    else:
        recent['p_change'] = recent['close'].pct_change() * 100

    # Check drawdown conditions
    prev_change = 100.0
    prev_open = recent.iloc[0]['open']

    for i in range(1, len(recent)):
        row = recent.iloc[i]
        p_change = row['p_change'] if not pd.isna(row['p_change']) else 0
        close = row['close']
        open_price = row['open']

        # Condition 2: No single day drop > 7%
        if p_change < -7:
            return {'triggered': False, 'reason': f'Single day drop {p_change:.1f}% > 7%'}

        # Condition 3: No high-open-low-close > 7%
        open_close_drop = (close - open_price) / open_price * 100
        if open_close_drop < -7:
            return {'triggered': False, 'reason': f'Open-close drop {open_close_drop:.1f}% > 7%'}

        # Condition 4: No 2-day cumulative drop > 10%
        if prev_change + p_change < -10:
            return {'triggered': False, 'reason': f'2-day cumulative drop > 10%'}

        # No 2-day high-open-low-close > 10%
        two_day_drop = (close - prev_open) / prev_open * 100
        if two_day_drop < -10:
            return {'triggered': False, 'reason': f'2-day open-close drop > 10%'}

        prev_change = p_change
        prev_open = open_price

    return {
        'triggered': True,
        'strategy': 'low_backtrace_increase',
        'name_cn': '无大幅回撤',
        'details': {
            'gain_pct': round(ratio_increase * 100, 1),
            'start_price': round(start_close, 2),
            'end_price': round(end_close, 2),
            'max_single_drop': round(recent['p_change'].min(), 1)
        }
    }


def strategy_keep_increasing(df, threshold=30):
    """
    Keep Increasing Strategy (持续增长).

    Identifies sustained MA progression (uptrend).

    Conditions:
    1. MA30 progression: 30 days ago < 20 days ago < 10 days ago < today
    2. MA30 today / MA30 30-days-ago > 1.2 (20% growth)

    Args:
        df: DataFrame with OHLCV data
        threshold: Lookback period (default 30 days)

    Returns:
        dict: Strategy result
    """
    if df.empty or len(df) < threshold + 30:
        return {'triggered': False, 'reason': 'Insufficient data'}

    df = df.copy()

    # Calculate MA30
    df['ma30'] = df['close'].rolling(window=30).mean()

    recent = df.tail(threshold + 1).copy()

    if recent['ma30'].isna().any():
        return {'triggered': False, 'reason': 'MA30 not available'}

    # Get MA30 at different points
    ma30_now = recent.iloc[-1]['ma30']
    ma30_10d = recent.iloc[-11]['ma30']
    ma30_20d = recent.iloc[-21]['ma30']
    ma30_30d = recent.iloc[0]['ma30']

    # Condition 1: MA30 progression
    if not (ma30_30d < ma30_20d < ma30_10d < ma30_now):
        return {'triggered': False, 'reason': 'MA30 not in consistent upward progression'}

    # Condition 2: 20% growth
    growth_ratio = ma30_now / ma30_30d

    if growth_ratio < 1.2:
        return {'triggered': False, 'reason': f'MA30 growth {(growth_ratio-1)*100:.1f}% < 20%'}

    return {
        'triggered': True,
        'strategy': 'keep_increasing',
        'name_cn': '持续增长',
        'details': {
            'ma30_growth': round((growth_ratio - 1) * 100, 1),
            'ma30_now': round(ma30_now, 2),
            'ma30_30d_ago': round(ma30_30d, 2)
        }
    }


def strategy_turtle_trade(df, threshold=60):
    """
    Turtle Trading Strategy (海龟法则).

    Simple breakout system based on Donchian channels.

    Condition:
    - Close >= 60-day high

    Args:
        df: DataFrame with OHLCV data
        threshold: Channel period (default 60 days)

    Returns:
        dict: Strategy result
    """
    if df.empty or len(df) < threshold:
        return {'triggered': False, 'reason': 'Insufficient data'}

    recent = df.tail(threshold + 1).copy()

    # Calculate 60-day high (excluding today)
    high_60d = recent.iloc[:-1]['high'].max()
    current_close = recent.iloc[-1]['close']

    if current_close >= high_60d:
        return {
            'triggered': True,
            'strategy': 'turtle_trade',
            'name_cn': '海龟突破',
            'details': {
                'close': round(current_close, 2),
                'high_60d': round(high_60d, 2),
                'breakout_pct': round((current_close - high_60d) / high_60d * 100, 2)
            }
        }

    return {
        'triggered': False,
        'reason': f'Close {current_close:.2f} < 60d high {high_60d:.2f}'
    }


def strategy_low_atr(df, threshold=10):
    """
    Low ATR Growth Strategy (低波动增长).

    Identifies stocks with low volatility but steady growth.

    Conditions:
    1. 10-day range ratio > 1.1 (high/low)
    2. Average daily change (ATR-like) < 10%
    3. Has 250+ days of history

    Args:
        df: DataFrame with OHLCV data
        threshold: Analysis period (default 10 days)

    Returns:
        dict: Strategy result
    """
    if df.empty or len(df) < 250:
        return {'triggered': False, 'reason': 'Insufficient data (need 250+ days)'}

    recent = df.tail(threshold).copy()

    # Condition 1: Range ratio > 1.1
    period_high = recent['high'].max()
    period_low = recent['low'].min()

    if period_low <= 0:
        return {'triggered': False, 'reason': 'Invalid price data'}

    range_ratio = period_high / period_low

    if range_ratio < 1.1:
        return {'triggered': False, 'reason': f'Range ratio {range_ratio:.2f} < 1.1'}

    # Condition 2: Average daily change < 10%
    if 'change_pct' in recent.columns:
        avg_change = recent['change_pct'].abs().mean()
    else:
        daily_changes = recent['close'].pct_change().abs() * 100
        avg_change = daily_changes.mean()

    if avg_change >= 10:
        return {'triggered': False, 'reason': f'Avg daily change {avg_change:.1f}% >= 10%'}

    return {
        'triggered': True,
        'strategy': 'low_atr',
        'name_cn': '低波动增长',
        'details': {
            'range_ratio': round(range_ratio, 3),
            'avg_daily_change': round(avg_change, 2),
            'period_high': round(period_high, 2),
            'period_low': round(period_low, 2)
        }
    }


def run_all_strategies(df):
    """
    Run all trading strategies on the given data.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        dict: Results from all strategies
    """
    results = {
        'triggered': [],
        'not_triggered': [],
        'strategies': {}
    }

    # Define strategies to run
    strategies = [
        ('backtrace_ma250', strategy_backtrace_ma250),
        ('platform_breakthrough', strategy_platform_breakthrough),
        ('climax_limitdown', strategy_climax_limitdown),
        ('low_backtrace_increase', strategy_low_backtrace_increase),
        ('keep_increasing', strategy_keep_increasing),
        ('turtle_trade', strategy_turtle_trade),
        ('low_atr', strategy_low_atr),
    ]

    for name, strategy_func in strategies:
        try:
            result = strategy_func(df)
            results['strategies'][name] = _to_native(result)

            if result.get('triggered', False):
                results['triggered'].append({
                    'strategy': name,
                    'name_cn': result.get('name_cn', name),
                    'details': _to_native(result.get('details', {}))
                })
            else:
                results['not_triggered'].append({
                    'strategy': name,
                    'reason': result.get('reason', 'Unknown')
                })
        except Exception as e:
            results['strategies'][name] = {
                'triggered': False,
                'error': str(e)
            }

    return results


def get_strategy_signal(df):
    """
    Generate a combined signal from all strategies.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        dict: Combined signal with score and details
    """
    results = run_all_strategies(df)

    triggered = results['triggered']
    score = 0

    # Weight different strategies
    strategy_weights = {
        'backtrace_ma250': 0.25,
        'platform_breakthrough': 0.25,
        'climax_limitdown': 0.30,  # Strong contrarian signal
        'low_backtrace_increase': 0.20,
        'keep_increasing': 0.15,
        'turtle_trade': 0.20,
        'low_atr': 0.10,
    }

    for item in triggered:
        strategy_name = item['strategy']
        weight = strategy_weights.get(strategy_name, 0.15)
        score += weight

    # Cap at 1.0
    score = min(1.0, score)

    signal = 'bullish' if score > 0.1 else 'neutral'

    return {
        'signal': signal,
        'score': score,
        'triggered_count': len(triggered),
        'triggered': triggered,
        'details': results
    }
