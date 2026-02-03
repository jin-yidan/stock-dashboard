import pandas as pd
import numpy as np

def calculate_ma(df, periods=[5, 10, 20, 60, 120, 250]):
    """Calculate Moving Averages including long-term (MA120, MA250)."""
    if df.empty:
        return df
    df = df.copy()
    for period in periods:
        if len(df) >= period:
            df[f'ma{period}'] = df['close'].rolling(window=period).mean()
    return df


def calculate_obv(df):
    """Calculate On-Balance Volume (OBV)."""
    if df.empty or len(df) < 2:
        return df
    df = df.copy()

    # Calculate OBV
    obv = [0]
    for i in range(1, len(df)):
        if df.iloc[i]['close'] > df.iloc[i-1]['close']:
            obv.append(obv[-1] + df.iloc[i]['volume'])
        elif df.iloc[i]['close'] < df.iloc[i-1]['close']:
            obv.append(obv[-1] - df.iloc[i]['volume'])
        else:
            obv.append(obv[-1])

    df['obv'] = obv
    # OBV moving average for trend
    df['obv_ma'] = df['obv'].rolling(window=20).mean()
    # OBV trend signal
    df['obv_trend'] = (df['obv'] > df['obv_ma']).astype(int)

    return df

def calculate_macd(df, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    if df.empty:
        return df
    df = df.copy()
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    df['macd_dif'] = ema_fast - ema_slow
    df['macd_dea'] = df['macd_dif'].ewm(span=signal, adjust=False).mean()
    df['macd_hist'] = (df['macd_dif'] - df['macd_dea']) * 2
    return df

def calculate_rsi(df, period=14):
    """Calculate RSI indicator."""
    if df.empty:
        return df
    df = df.copy()
    delta = df['close'].diff()
    gains = delta.where(delta > 0, 0)
    losses = (-delta).where(delta < 0, 0)
    avg_gains = gains.rolling(window=period).mean()
    avg_losses = losses.rolling(window=period).mean()
    rs = avg_gains / avg_losses.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def calculate_volume_ma(df, period=20):
    """Calculate volume moving average."""
    if df.empty:
        return df
    df = df.copy()
    df['volume_ma'] = df['volume'].rolling(window=period).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    return df

def calculate_atr(df, period=14):
    """Calculate Average True Range for volatility."""
    if df.empty:
        return df
    df = df.copy()
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=period).mean()
    return df


def calculate_kdj(df, n=9, m1=3, m2=3):
    """Calculate KDJ indicator (common in Chinese markets)."""
    if df.empty:
        return df
    df = df.copy()

    low_n = df['low'].rolling(window=n).min()
    high_n = df['high'].rolling(window=n).max()

    rsv = (df['close'] - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50)

    # K = 2/3 * prev_K + 1/3 * RSV
    df['kdj_k'] = rsv.ewm(alpha=1/m1, adjust=False).mean()
    # D = 2/3 * prev_D + 1/3 * K
    df['kdj_d'] = df['kdj_k'].ewm(alpha=1/m2, adjust=False).mean()
    # J = 3*K - 2*D
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']

    return df


def calculate_bollinger(df, period=20, std_dev=2):
    """Calculate Bollinger Bands."""
    if df.empty:
        return df
    df = df.copy()

    df['boll_mid'] = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    df['boll_upper'] = df['boll_mid'] + std_dev * std
    df['boll_lower'] = df['boll_mid'] - std_dev * std
    # Bandwidth: (upper - lower) / mid * 100
    df['boll_width'] = (df['boll_upper'] - df['boll_lower']) / df['boll_mid'] * 100
    # %B: (close - lower) / (upper - lower)
    df['boll_pct_b'] = (df['close'] - df['boll_lower']) / (df['boll_upper'] - df['boll_lower'])

    return df

def calculate_all(df):
    """Calculate all indicators."""
    if df.empty:
        return df
    df = df.copy()
    df = calculate_ma(df)
    df = calculate_macd(df)
    df = calculate_rsi(df)
    df = calculate_volume_ma(df)
    df = calculate_atr(df)
    df = calculate_kdj(df)
    df = calculate_bollinger(df)
    df = calculate_obv(df)
    return df

def get_ma_trend(df):
    """Analyze MA trend with multiple factors."""
    if df.empty or len(df) < 60:
        return {'trend': 'neutral', 'score': 0, 'details': {}}

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    ma5 = latest.get('ma5', 0)
    ma10 = latest.get('ma10', 0)
    ma20 = latest.get('ma20', 0)
    ma60 = latest.get('ma60', 0)
    close = latest.get('close', 0)

    if not all([ma5, ma10, ma20, ma60, close]):
        return {'trend': 'neutral', 'score': 0, 'details': {}}

    score = 0

    # 1. MA alignment (strongest signal)
    if ma5 > ma10 > ma20 > ma60:
        score += 0.4  # Perfect bullish alignment
    elif ma5 > ma10 > ma20:
        score += 0.25
    elif ma5 < ma10 < ma20 < ma60:
        score -= 0.4  # Perfect bearish alignment
    elif ma5 < ma10 < ma20:
        score -= 0.25

    # 2. Price position relative to MAs
    above_count = sum([close > ma for ma in [ma5, ma10, ma20, ma60]])
    score += (above_count - 2) * 0.1  # -0.2 to +0.2

    # 3. MA slope (trend strength)
    ma20_5d_ago = df.iloc[-5].get('ma20', ma20) if len(df) > 5 else ma20
    ma_slope = (ma20 - ma20_5d_ago) / ma20_5d_ago if ma20_5d_ago else 0
    if ma_slope > 0.02:
        score += 0.15
    elif ma_slope < -0.02:
        score -= 0.15

    # 4. Golden/Death cross detection
    prev_ma5 = prev.get('ma5', 0)
    prev_ma20 = prev.get('ma20', 0)
    if prev_ma5 and prev_ma20:
        if prev_ma5 <= prev_ma20 and ma5 > ma20:
            score += 0.2  # Golden cross
        elif prev_ma5 >= prev_ma20 and ma5 < ma20:
            score -= 0.2  # Death cross

    score = max(-1, min(1, score))
    trend = 'bullish' if score > 0.1 else ('bearish' if score < -0.1 else 'neutral')

    return {'trend': trend, 'score': score, 'details': {
        'alignment': 'bullish' if ma5 > ma10 > ma20 else ('bearish' if ma5 < ma10 < ma20 else 'mixed'),
        'price_position': above_count,
        'ma_slope': ma_slope
    }}

def get_macd_signal(df):
    """Analyze MACD with histogram momentum."""
    if df.empty or len(df) < 30:
        return {'signal': 'neutral', 'score': 0, 'details': {}}

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    prev2 = df.iloc[-3] if len(df) > 2 else prev

    dif = latest.get('macd_dif', 0)
    dea = latest.get('macd_dea', 0)
    hist = latest.get('macd_hist', 0)
    prev_hist = prev.get('macd_hist', 0)
    prev2_hist = prev2.get('macd_hist', 0)

    if dif is None or dea is None:
        return {'signal': 'neutral', 'score': 0, 'details': {}}

    score = 0

    # 1. DIF vs DEA position
    if dif > dea:
        score += 0.2
    else:
        score -= 0.2

    # 2. Histogram momentum (consecutive growth)
    if hist > 0:
        if hist > prev_hist > prev2_hist:
            score += 0.35  # Strong bullish momentum
        elif hist > prev_hist:
            score += 0.2
        else:
            score += 0.05  # Weakening
    else:
        if hist < prev_hist < prev2_hist:
            score -= 0.35  # Strong bearish momentum
        elif hist < prev_hist:
            score -= 0.2
        else:
            score -= 0.05  # Weakening

    # 3. Zero line position
    if dif > 0 and dea > 0:
        score += 0.15  # Both above zero
    elif dif < 0 and dea < 0:
        score -= 0.15  # Both below zero

    # 4. Crossover detection
    prev_dif = prev.get('macd_dif', 0)
    prev_dea = prev.get('macd_dea', 0)
    if prev_dif and prev_dea:
        if prev_dif <= prev_dea and dif > dea:
            score += 0.25  # Golden cross
        elif prev_dif >= prev_dea and dif < dea:
            score -= 0.25  # Death cross

    score = max(-1, min(1, score))
    signal = 'bullish' if score > 0.1 else ('bearish' if score < -0.1 else 'neutral')

    return {'signal': signal, 'score': score, 'details': {
        'histogram_trend': 'expanding' if abs(hist) > abs(prev_hist) else 'contracting',
        'above_zero': dif > 0
    }}

def get_rsi_signal(df):
    """Analyze RSI - IMPROVED based on backtesting.

    Key findings:
    - RSI < 30 oversold: 55% accuracy for buy signal
    - RSI > 70 overbought: Only 37.5% accuracy - NOT reliable for sell
    - Adjusted to asymmetric scoring
    """
    if df.empty or len(df) < 14:
        return {'signal': 'neutral', 'score': 0, 'rsi': 50, 'details': {}}

    latest = df.iloc[-1]
    rsi = latest.get('rsi', 50)

    if rsi is None or np.isnan(rsi):
        return {'signal': 'neutral', 'score': 0, 'rsi': 50, 'details': {}}

    # Get RSI trend (last 5 days)
    recent_rsi = [df.iloc[i].get('rsi', 50) for i in range(-5, 0) if len(df) > abs(i)]
    rsi_trend = 'rising' if len(recent_rsi) > 1 and recent_rsi[-1] > recent_rsi[0] else 'falling'

    score = 0

    # IMPROVED: Asymmetric scoring based on backtest results
    # Oversold signals work well, overbought signals don't
    if rsi < 25:
        score = 0.8  # Extreme oversold - strong buy
    elif rsi < 30:
        score = 0.6 if rsi_trend == 'rising' else 0.5  # Oversold - buy signal works
    elif rsi < 40:
        score = 0.2
    elif rsi > 75:
        score = -0.2  # REDUCED: Overbought signals unreliable
    elif rsi > 70:
        score = -0.15  # REDUCED: Don't trust overbought as much
    elif rsi > 60:
        score = -0.05  # Minimal bearish bias
    else:
        # Neutral zone 40-60
        if rsi_trend == 'rising' and rsi > 50:
            score = 0.1
        elif rsi_trend == 'falling' and rsi < 50:
            score = -0.05  # Reduced bearish bias

    signal = 'bullish' if score > 0.1 else ('bearish' if score < -0.1 else 'neutral')

    return {'signal': signal, 'score': score, 'rsi': rsi, 'details': {
        'level': 'oversold' if rsi < 30 else ('overbought' if rsi > 70 else 'normal'),
        'trend': rsi_trend
    }}

def get_volume_signal(df):
    """Analyze volume with price correlation."""
    if df.empty or len(df) < 20:
        return {'signal': 'neutral', 'score': 0, 'volume_ratio': 1, 'details': {}}

    latest = df.iloc[-1]
    volume_ratio = latest.get('volume_ratio', 1)
    change_pct = latest.get('change_pct', 0)

    if volume_ratio is None or np.isnan(volume_ratio):
        volume_ratio = 1

    score = 0

    # Volume-price relationship
    if volume_ratio > 2.0:
        # Very high volume
        score = 0.5 if change_pct > 1 else (-0.5 if change_pct < -1 else 0)
    elif volume_ratio > 1.5:
        score = 0.35 if change_pct > 0.5 else (-0.35 if change_pct < -0.5 else 0)
    elif volume_ratio > 1.2:
        score = 0.2 if change_pct > 0 else (-0.2 if change_pct < 0 else 0)
    elif volume_ratio < 0.5:
        # Very low volume - usually continuation of trend
        recent_changes = [df.iloc[i].get('change_pct', 0) for i in range(-5, 0) if len(df) > abs(i)]
        avg_change = sum(recent_changes) / len(recent_changes) if recent_changes else 0
        score = 0.1 if avg_change > 0 else (-0.1 if avg_change < 0 else 0)

    # Check volume trend (3 days)
    if len(df) >= 3:
        vol_trend = [df.iloc[i].get('volume_ratio', 1) for i in range(-3, 0)]
        if vol_trend[-1] > vol_trend[-2] > vol_trend[-3] and change_pct > 0:
            score += 0.15  # Increasing volume on up move

    signal = 'bullish' if score > 0.1 else ('bearish' if score < -0.1 else 'neutral')

    return {'signal': signal, 'score': score, 'volume_ratio': volume_ratio, 'details': {
        'level': 'high' if volume_ratio > 1.5 else ('low' if volume_ratio < 0.7 else 'normal')
    }}


def get_kdj_signal(df):
    """Analyze KDJ indicator."""
    if df.empty or len(df) < 9:
        return {'signal': 'neutral', 'score': 0, 'details': {}}

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    k = latest.get('kdj_k', 50)
    d = latest.get('kdj_d', 50)
    j = latest.get('kdj_j', 50)
    prev_k = prev.get('kdj_k', 50)
    prev_d = prev.get('kdj_d', 50)

    if k is None or np.isnan(k):
        return {'signal': 'neutral', 'score': 0, 'details': {}}

    score = 0

    # 1. Overbought/Oversold
    if j < 0:
        score += 0.4  # Extreme oversold
    elif k < 20 and d < 20:
        score += 0.3  # Oversold
    elif j > 100:
        score -= 0.4  # Extreme overbought
    elif k > 80 and d > 80:
        score -= 0.3  # Overbought

    # 2. Golden/Death cross
    if prev_k <= prev_d and k > d:
        score += 0.35  # Golden cross
    elif prev_k >= prev_d and k < d:
        score -= 0.35  # Death cross

    # 3. K/D position
    if k > d:
        score += 0.1
    else:
        score -= 0.1

    score = max(-1, min(1, score))
    signal = 'bullish' if score > 0.1 else ('bearish' if score < -0.1 else 'neutral')

    return {'signal': signal, 'score': score, 'details': {
        'k': round(k, 1),
        'd': round(d, 1),
        'j': round(j, 1),
        'level': 'oversold' if k < 20 else ('overbought' if k > 80 else 'normal')
    }}


def get_bollinger_signal(df):
    """Analyze Bollinger Bands."""
    if df.empty or len(df) < 20:
        return {'signal': 'neutral', 'score': 0, 'details': {}}

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    close = latest.get('close', 0)
    upper = latest.get('boll_upper', 0)
    lower = latest.get('boll_lower', 0)
    mid = latest.get('boll_mid', 0)
    pct_b = latest.get('boll_pct_b', 0.5)
    width = latest.get('boll_width', 10)
    prev_width = prev.get('boll_width', 10)

    if not all([close, upper, lower, mid]):
        return {'signal': 'neutral', 'score': 0, 'details': {}}

    score = 0

    # 1. Price position relative to bands
    if pct_b < 0:
        score += 0.4  # Below lower band - oversold
    elif pct_b < 0.2:
        score += 0.25  # Near lower band
    elif pct_b > 1:
        score -= 0.4  # Above upper band - overbought
    elif pct_b > 0.8:
        score -= 0.25  # Near upper band

    # 2. Band squeeze/expansion (volatility)
    if width < 5 and prev_width < 5:
        # Squeeze - potential breakout coming
        score *= 0.5  # Reduce confidence during squeeze
    elif width > prev_width * 1.2:
        # Expansion - trend starting
        if close > mid:
            score += 0.15
        else:
            score -= 0.15

    # 3. Price vs middle band (trend)
    if close > mid:
        score += 0.1
    else:
        score -= 0.1

    score = max(-1, min(1, score))
    signal = 'bullish' if score > 0.1 else ('bearish' if score < -0.1 else 'neutral')

    return {'signal': signal, 'score': score, 'details': {
        'pct_b': round(pct_b, 2),
        'width': round(width, 1),
        'position': 'below' if pct_b < 0.2 else ('above' if pct_b > 0.8 else 'middle'),
        'volatility': 'squeeze' if width < 5 else ('expanding' if width > 15 else 'normal')
    }}


def get_weekly_trend(df):
    """Analyze weekly trend for multi-timeframe confirmation."""
    if df.empty or len(df) < 20:
        return {'trend': 'neutral', 'score': 0, 'details': {}}

    # Resample daily to weekly (use last 5 days as a week approximation)
    # Compare current week vs previous weeks
    weeks = []
    for i in range(0, min(len(df), 20), 5):
        if i + 5 <= len(df):
            week_data = df.iloc[-(i+5):len(df)-i if i > 0 else len(df)]
            if not week_data.empty:
                weeks.append({
                    'close': week_data.iloc[-1]['close'],
                    'open': week_data.iloc[0]['open'],
                    'high': week_data['high'].max(),
                    'low': week_data['low'].min()
                })

    if len(weeks) < 2:
        return {'trend': 'neutral', 'score': 0, 'details': {}}

    current = weeks[0]
    prev = weeks[1]

    score = 0

    # 1. Weekly close vs previous close
    if current['close'] > prev['close']:
        score += 0.3
    else:
        score -= 0.3

    # 2. Weekly trend (close vs open)
    if current['close'] > current['open']:
        score += 0.2  # Bullish weekly candle
    else:
        score -= 0.2

    # 3. Higher highs / lower lows
    if len(weeks) >= 3:
        prev2 = weeks[2]
        if current['high'] > prev['high'] and prev['high'] > prev2['high']:
            score += 0.25  # Higher highs
        elif current['low'] < prev['low'] and prev['low'] < prev2['low']:
            score -= 0.25  # Lower lows

    score = max(-1, min(1, score))
    trend = 'bullish' if score > 0.1 else ('bearish' if score < -0.1 else 'neutral')

    return {'trend': trend, 'score': score, 'details': {
        'weekly_change': round((current['close'] - prev['close']) / prev['close'] * 100, 2) if prev['close'] else 0
    }}


def get_momentum_signal(df):
    """Analyze price momentum - ADAPTIVE based on volatility.

    Key findings from backtesting:
    - High volatility stocks: Mean reversion works (55.6%)
    - Low volatility stocks: Trend following works better for some
    - BYD: 60% accuracy on mean reversion
    - Ping An: 72% accuracy on trend following
    """
    if df.empty or len(df) < 20:
        return {'signal': 'neutral', 'score': 0, 'details': {}}

    current = df.iloc[-1]['close']
    past_5d = df.iloc[-6]['close'] if len(df) > 5 else current
    ma20 = df.iloc[-1].get('ma20', current)

    ret_5d = (current - past_5d) / past_5d * 100

    # Calculate volatility (20-day)
    returns = df['close'].pct_change().dropna().tail(20)
    volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 5 else 25

    score = 0
    strategy = 'neutral'

    # ADAPTIVE STRATEGY based on volatility
    if volatility > 30:
        # High volatility: Mean reversion (buy after drops)
        if ret_5d < -5:
            score = 0.5  # Strong mean reversion signal
            strategy = 'mean_reversion'
        elif ret_5d < -3:
            score = 0.3
            strategy = 'mean_reversion'
        elif ret_5d > 5:
            score = 0.2  # Some momentum also works
            strategy = 'momentum'
    else:
        # Lower volatility: Trend following
        in_uptrend = current > ma20

        if in_uptrend and ret_5d > 0:
            score = 0.4  # Trend following
            strategy = 'trend_follow'
        elif in_uptrend and ret_5d > -2:
            score = 0.2
            strategy = 'trend_follow'
        elif not in_uptrend and ret_5d < -5:
            score = 0.15  # Smaller mean reversion for low vol
            strategy = 'mean_reversion'
        elif not in_uptrend:
            score = -0.1  # Bearish when below MA20
            strategy = 'trend_follow'

    signal = 'bullish' if score > 0.1 else ('bearish' if score < -0.1 else 'neutral')

    return {'signal': signal, 'score': score, 'details': {
        'ret_5d': round(ret_5d, 2),
        'volatility': round(volatility, 1),
        'strategy': strategy,
        'in_uptrend': current > ma20
    }}


def get_stop_loss_suggestion(df):
    """Suggest stop-loss levels based on ATR."""
    if df.empty or len(df) < 14:
        return {'stop_loss': None, 'take_profit': None, 'details': {}}

    latest = df.iloc[-1]
    close = latest.get('close', 0)
    atr = latest.get('atr', 0)
    low = latest.get('low', 0)

    if not close or not atr:
        return {'stop_loss': None, 'take_profit': None, 'details': {}}

    # ATR-based stops
    atr_stop = close - 2 * atr  # 2x ATR below current price
    support_stop = low - atr * 0.5  # Just below recent low

    # Use the higher of the two (tighter stop)
    stop_loss = max(atr_stop, support_stop)

    # Take profit at 3:1 reward-risk ratio
    risk = close - stop_loss
    take_profit = close + 3 * risk

    # Calculate percentages
    stop_pct = (stop_loss - close) / close * 100
    profit_pct = (take_profit - close) / close * 100

    return {
        'stop_loss': round(stop_loss, 2),
        'take_profit': round(take_profit, 2),
        'details': {
            'stop_pct': round(stop_pct, 1),
            'profit_pct': round(profit_pct, 1),
            'risk_reward': '1:3',
            'atr': round(atr, 2),
            'atr_pct': round(atr / close * 100, 1)
        }
    }
