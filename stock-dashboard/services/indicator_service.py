import pandas as pd
import numpy as np

# =============================================================================
# MOVING AVERAGES
# =============================================================================

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

def calculate_adx(df, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    if df.empty or len(df) < period * 2:
        return df
    df = df.copy()

    # True Range
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        abs(high - prev_close),
        abs(low - prev_close)
    ], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Smoothed averages
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr

    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['adx'] = dx.rolling(window=period).mean()
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di

    return df


# =============================================================================
# NEW INDICATORS FROM INSTOCK
# =============================================================================

def calculate_supertrend(df, period=14, multiplier=3):
    """
    Calculate Supertrend indicator - superior trend-following indicator.

    Supertrend uses ATR to create dynamic support/resistance bands.
    When price closes above Supertrend, trend is bullish.
    When price closes below Supertrend, trend is bearish.

    Args:
        df: DataFrame with OHLCV data
        period: ATR period (default 14)
        multiplier: ATR multiplier for band width (default 3)

    Returns:
        DataFrame with supertrend, supertrend_ub, supertrend_lb, supertrend_direction
    """
    if df.empty or len(df) < period:
        return df
    df = df.copy()

    # Calculate ATR if not present
    if 'atr' not in df.columns:
        df = calculate_atr(df, period)

    # Basic bands
    hl_avg = (df['high'] + df['low']) / 2
    df['st_basic_ub'] = hl_avg + multiplier * df['atr']
    df['st_basic_lb'] = hl_avg - multiplier * df['atr']

    # Initialize arrays for stateful calculation
    size = len(df)
    ub = np.zeros(size)
    lb = np.zeros(size)
    st = np.zeros(size)
    direction = np.zeros(size)  # 1 = bullish, -1 = bearish

    for i in range(size):
        if i == 0:
            ub[i] = df['st_basic_ub'].iloc[i]
            lb[i] = df['st_basic_lb'].iloc[i]
            st[i] = ub[i] if df['close'].iloc[i] <= ub[i] else lb[i]
            direction[i] = -1 if df['close'].iloc[i] <= ub[i] else 1
            continue

        curr_close = df['close'].iloc[i]
        last_close = df['close'].iloc[i - 1]
        last_ub = ub[i - 1]
        last_lb = lb[i - 1]
        last_st = st[i - 1]
        curr_basic_ub = df['st_basic_ub'].iloc[i]
        curr_basic_lb = df['st_basic_lb'].iloc[i]

        # Upper band: can only move DOWN (tightens in uptrend)
        if curr_basic_ub < last_ub or last_close > last_ub:
            ub[i] = curr_basic_ub
        else:
            ub[i] = last_ub

        # Lower band: can only move UP (tightens in downtrend)
        if curr_basic_lb > last_lb or last_close < last_lb:
            lb[i] = curr_basic_lb
        else:
            lb[i] = last_lb

        # Determine Supertrend value and direction
        if last_st == last_ub:
            if curr_close <= ub[i]:
                st[i] = ub[i]
                direction[i] = -1  # Bearish
            else:
                st[i] = lb[i]
                direction[i] = 1   # Bullish flip
        else:  # last_st == last_lb
            if curr_close >= lb[i]:
                st[i] = lb[i]
                direction[i] = 1   # Bullish
            else:
                st[i] = ub[i]
                direction[i] = -1  # Bearish flip

    df['supertrend'] = st
    df['supertrend_ub'] = ub
    df['supertrend_lb'] = lb
    df['supertrend_direction'] = direction  # 1=bullish, -1=bearish

    # Cleanup temp columns
    df.drop(columns=['st_basic_ub', 'st_basic_lb'], inplace=True, errors='ignore')

    return df


def calculate_wave_trend(df, channel_period=10, avg_period=21):
    """
    Calculate Wave Trend (WT) indicator.

    Wave Trend is an advanced oscillator that measures the deviation of price
    from its smoothed average, normalized by volatility.

    WT1 crossing above WT2 = Buy signal
    WT1 crossing below WT2 = Sell signal

    Args:
        df: DataFrame with OHLCV and 'amount' column
        channel_period: EMA period for channel calculation (default 10)
        avg_period: EMA period for final smoothing (default 21)

    Returns:
        DataFrame with wt1, wt2
    """
    if df.empty or len(df) < avg_period:
        return df
    df = df.copy()

    # Use typical price or VWAP if amount available
    if 'amount' in df.columns and df['amount'].sum() > 0:
        # VWAP-like typical price
        typical_price = df['amount'] / df['volume'].replace(0, np.nan)
        typical_price = typical_price.fillna((df['high'] + df['low'] + df['close']) / 3)
    else:
        typical_price = (df['high'] + df['low'] + df['close']) / 3

    # EMA of typical price
    esa = typical_price.ewm(span=channel_period, adjust=False).mean()

    # EMA of absolute deviation
    d = abs(typical_price - esa).ewm(span=channel_period, adjust=False).mean()

    # Channel index (normalized deviation)
    ci = (typical_price - esa) / (0.015 * d.replace(0, np.nan))
    ci = ci.fillna(0)

    # Wave Trend lines
    df['wt1'] = ci.ewm(span=avg_period, adjust=False).mean()
    df['wt2'] = df['wt1'].rolling(window=4).mean()

    # Handle NaN
    df['wt1'] = df['wt1'].fillna(0)
    df['wt2'] = df['wt2'].fillna(0)

    return df


def calculate_vr(df, period=26):
    """
    Calculate VR (Volume Ratio) indicator.

    VR splits volume by up/down/flat days and calculates the ratio.
    VR > 150: Bullish accumulation
    VR < 70: Distribution/bearish

    Formula: VR = (Up Volume + Flat Volume/2) / (Down Volume + Flat Volume/2) * 100

    Args:
        df: DataFrame with OHLCV data
        period: Calculation period (default 26)

    Returns:
        DataFrame with vr, vr_ma
    """
    if df.empty or len(df) < period:
        return df
    df = df.copy()

    # Calculate price change
    change = df['close'].pct_change()

    # Classify volume by direction
    up_vol = np.where(change > 0, df['volume'], 0)
    down_vol = np.where(change < 0, df['volume'], 0)
    flat_vol = np.where(change == 0, df['volume'], 0)

    # Sum over period
    up_sum = pd.Series(up_vol).rolling(window=period).sum()
    down_sum = pd.Series(down_vol).rolling(window=period).sum()
    flat_sum = pd.Series(flat_vol).rolling(window=period).sum()

    # Calculate VR
    numerator = up_sum + flat_sum / 2
    denominator = down_sum + flat_sum / 2

    df['vr'] = (numerator / denominator.replace(0, np.nan)) * 100
    df['vr'] = df['vr'].fillna(100)
    df['vr'] = df['vr'].replace([np.inf, -np.inf], 100)

    # VR moving average
    df['vr_ma'] = df['vr'].rolling(window=6).mean()

    return df


def calculate_cr(df, period=26):
    """
    Calculate CR (Energy Indicator).

    CR measures buying pressure vs selling pressure relative to the mid-price.
    CR > 200: Overbought
    CR < 40: Oversold

    Args:
        df: DataFrame with OHLCV and 'amount' column
        period: Calculation period (default 26)

    Returns:
        DataFrame with cr, cr_ma1, cr_ma2, cr_ma3
    """
    if df.empty or len(df) < period:
        return df
    df = df.copy()

    # Calculate mid-price (VWAP if available)
    if 'amount' in df.columns and df['amount'].sum() > 0:
        mid_price = df['amount'] / df['volume'].replace(0, np.nan)
        mid_price = mid_price.fillna((df['high'] + df['low'] + df['close']) / 3)
    else:
        mid_price = (df['high'] + df['low'] + df['close']) / 3

    # Previous mid price
    prev_mid = mid_price.shift(1)

    # Buying pressure: high - min(prev_mid, high)
    h_m = df['high'] - np.minimum(prev_mid, df['high'])

    # Selling pressure: max(prev_mid, low) - low
    m_l = np.maximum(prev_mid, df['low']) - df['low']

    # Sum over period
    h_m_sum = h_m.rolling(window=period).sum()
    m_l_sum = m_l.rolling(window=period).sum()

    # Calculate CR
    df['cr'] = (h_m_sum / m_l_sum.replace(0, np.nan)) * 100
    df['cr'] = df['cr'].fillna(100)
    df['cr'] = df['cr'].replace([np.inf, -np.inf], 100)

    # CR moving averages
    df['cr_ma1'] = df['cr'].rolling(window=5).mean()
    df['cr_ma2'] = df['cr'].rolling(window=10).mean()
    df['cr_ma3'] = df['cr'].rolling(window=20).mean()

    return df


def calculate_trix(df, period=12, signal_period=20):
    """
    Calculate TRIX (Triple Exponential Moving Average).

    TRIX is a momentum indicator showing percentage rate of change
    of a triple-smoothed EMA. Good for filtering noise.

    Args:
        df: DataFrame with OHLCV data
        period: EMA period (default 12)
        signal_period: Signal line period (default 20)

    Returns:
        DataFrame with trix, trix_signal
    """
    if df.empty or len(df) < period * 3:
        return df
    df = df.copy()

    # Triple EMA
    ema1 = df['close'].ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()

    # TRIX = percentage change of triple EMA
    df['trix'] = ema3.pct_change() * 100
    df['trix'] = df['trix'].fillna(0)

    # Signal line
    df['trix_signal'] = df['trix'].rolling(window=signal_period).mean()

    return df


def calculate_psy(df, period=12):
    """
    Calculate PSY (Psychology Line).

    PSY measures the percentage of up days in a period.
    PSY > 75: Overbought (too optimistic)
    PSY < 25: Oversold (too pessimistic)

    Args:
        df: DataFrame with OHLCV data
        period: Calculation period (default 12)

    Returns:
        DataFrame with psy, psy_ma
    """
    if df.empty or len(df) < period:
        return df
    df = df.copy()

    # Mark up days
    up_day = (df['close'] > df['close'].shift(1)).astype(int)

    # Count up days in period
    up_count = up_day.rolling(window=period).sum()

    # Calculate PSY
    df['psy'] = (up_count / period) * 100
    df['psy'] = df['psy'].fillna(50)

    # PSY moving average
    df['psy_ma'] = df['psy'].rolling(window=6).mean()

    return df


def calculate_brar(df, period=26):
    """
    Calculate BRAR (Sentiment Indicators).

    AR (Atmoshere): Buying/selling strength based on open price
    BR (Belief): Buying/selling strength based on previous close

    AR > 150: Strong buying pressure
    BR > AR: Bullish sentiment

    Args:
        df: DataFrame with OHLCV data
        period: Calculation period (default 26)

    Returns:
        DataFrame with ar, br
    """
    if df.empty or len(df) < period:
        return df
    df = df.copy()

    # AR calculation: (High - Open) / (Open - Low)
    h_o = df['high'] - df['open']
    o_l = df['open'] - df['low']

    h_o_sum = h_o.rolling(window=period).sum()
    o_l_sum = o_l.rolling(window=period).sum()

    df['ar'] = (h_o_sum / o_l_sum.replace(0, np.nan)) * 100
    df['ar'] = df['ar'].fillna(100)
    df['ar'] = df['ar'].replace([np.inf, -np.inf], 100)

    # BR calculation: (High - Prev_Close) / (Prev_Close - Low)
    prev_close = df['close'].shift(1)
    h_cy = df['high'] - prev_close
    cy_l = prev_close - df['low']

    h_cy_sum = h_cy.rolling(window=period).sum()
    cy_l_sum = cy_l.rolling(window=period).sum()

    df['br'] = (h_cy_sum / cy_l_sum.replace(0, np.nan)) * 100
    df['br'] = df['br'].fillna(100)
    df['br'] = df['br'].replace([np.inf, -np.inf], 100)

    return df


def calculate_bias(df, periods=[6, 12, 24]):
    """
    Calculate BIAS (Deviation from Moving Average).

    BIAS measures how far the price has deviated from its MA.
    Extreme BIAS values suggest mean reversion.

    Args:
        df: DataFrame with OHLCV data
        periods: List of MA periods to calculate BIAS for

    Returns:
        DataFrame with bias columns for each period
    """
    if df.empty or len(df) < max(periods):
        return df
    df = df.copy()

    for period in periods:
        ma = df['close'].rolling(window=period).mean()
        df[f'bias_{period}'] = ((df['close'] - ma) / ma) * 100
        df[f'bias_{period}'] = df[f'bias_{period}'].fillna(0)

    return df


def calculate_force_index(df, short_period=2, long_period=13):
    """
    Calculate Force Index (FI).

    Force Index combines price change and volume to measure
    the force behind price movements.

    Positive FI: Bulls in control
    Negative FI: Bears in control

    Args:
        df: DataFrame with OHLCV data
        short_period: Short EMA period (default 2)
        long_period: Long EMA period (default 13)

    Returns:
        DataFrame with fi, force_2, force_13
    """
    if df.empty or len(df) < long_period:
        return df
    df = df.copy()

    # Raw Force Index = price change * volume
    price_change = df['close'].diff()
    df['fi'] = price_change * df['volume']

    # Smoothed versions
    df['force_2'] = df['fi'].ewm(span=short_period, adjust=False).mean()
    df['force_13'] = df['fi'].ewm(span=long_period, adjust=False).mean()

    df['fi'] = df['fi'].fillna(0)
    df['force_2'] = df['force_2'].fillna(0)
    df['force_13'] = df['force_13'].fillna(0)

    return df


def calculate_emv(df, period=14, signal_period=9):
    """
    Calculate EMV (Ease of Movement).

    EMV measures how easily price moves relative to volume.
    High EMV: Price moves easily on low volume
    Low EMV: Price struggles even with high volume

    Args:
        df: DataFrame with OHLCV and 'amount' column
        period: EMV period (default 14)
        signal_period: Signal line period (default 9)

    Returns:
        DataFrame with emv, emv_signal
    """
    if df.empty or len(df) < period:
        return df
    df = df.copy()

    # Distance moved
    hl_avg = (df['high'] + df['low']) / 2
    prev_hl_avg = hl_avg.shift(1)
    distance = hl_avg - prev_hl_avg

    # Box ratio (range / amount)
    h_l = df['high'] - df['low']
    if 'amount' in df.columns:
        box_ratio = h_l / df['amount'].replace(0, np.nan)
    else:
        box_ratio = h_l / (df['volume'] * df['close']).replace(0, np.nan)

    # EMV
    emv_raw = distance * box_ratio
    df['emv'] = emv_raw.rolling(window=period).sum()
    df['emv'] = df['emv'].fillna(0)

    # Signal line
    df['emv_signal'] = df['emv'].rolling(window=signal_period).mean()

    return df


def calculate_dpo(df, period=11):
    """
    Calculate DPO (Detrended Price Oscillator).

    DPO removes trend from price to show cycles more clearly.

    Args:
        df: DataFrame with OHLCV data
        period: Lookback period (default 11)

    Returns:
        DataFrame with dpo, dpo_ma
    """
    if df.empty or len(df) < period + 1:
        return df
    df = df.copy()

    # DPO = Close - MA shifted by (period/2 + 1)
    ma = df['close'].rolling(window=period).mean()
    shift_periods = period // 2 + 1
    df['dpo'] = df['close'] - ma.shift(shift_periods)
    df['dpo'] = df['dpo'].fillna(0)

    # DPO moving average
    df['dpo_ma'] = df['dpo'].rolling(window=6).mean()

    return df


def calculate_vhf(df, period=28):
    """
    Calculate VHF (Vertical Horizontal Filter).

    VHF determines whether prices are trending or ranging.
    VHF > 0.4: Strong trend
    VHF < 0.3: Ranging/consolidation

    Args:
        df: DataFrame with OHLCV data
        period: Calculation period (default 28)

    Returns:
        DataFrame with vhf
    """
    if df.empty or len(df) < period:
        return df
    df = df.copy()

    # Highest and lowest close in period
    highest = df['close'].rolling(window=period).max()
    lowest = df['close'].rolling(window=period).min()

    # Numerator: range
    numerator = abs(highest - lowest)

    # Denominator: sum of daily changes
    daily_change = abs(df['close'].diff())
    denominator = daily_change.rolling(window=period).sum()

    df['vhf'] = numerator / denominator.replace(0, np.nan)
    df['vhf'] = df['vhf'].fillna(0)

    return df


def calculate_rvi(df, period=10):
    """
    Calculate RVI (Relative Vigor Index).

    RVI measures the conviction of a price move by comparing
    close-open range to high-low range.

    RVI > 0: Bullish vigor
    RVI < 0: Bearish vigor

    Args:
        df: DataFrame with OHLCV data
        period: Smoothing period (default 10)

    Returns:
        DataFrame with rvi, rvi_signal
    """
    if df.empty or len(df) < period + 3:
        return df
    df = df.copy()

    # Numerator: (Close - Open) weighted
    close_open = df['close'] - df['open']
    num = (
        close_open +
        2 * close_open.shift(1) +
        2 * close_open.shift(2) +
        close_open.shift(3)
    ) / 6

    # Denominator: (High - Low) weighted
    high_low = df['high'] - df['low']
    den = (
        high_low +
        2 * high_low.shift(1) +
        2 * high_low.shift(2) +
        high_low.shift(3)
    ) / 6

    # RVI
    num_ma = num.rolling(window=period).mean()
    den_ma = den.rolling(window=period).mean()

    df['rvi'] = num_ma / den_ma.replace(0, np.nan)
    df['rvi'] = df['rvi'].fillna(0)

    # Signal line (weighted average of RVI)
    df['rvi_signal'] = (
        df['rvi'] +
        2 * df['rvi'].shift(1) +
        2 * df['rvi'].shift(2) +
        df['rvi'].shift(3)
    ) / 6
    df['rvi_signal'] = df['rvi_signal'].fillna(0)

    return df


def calculate_ene(df, period=10, upper_pct=11, lower_pct=9):
    """
    Calculate ENE (Envelope Channels).

    ENE creates upper and lower bands as fixed percentages from MA.

    Args:
        df: DataFrame with OHLCV data
        period: MA period (default 10)
        upper_pct: Upper band percentage (default 11%)
        lower_pct: Lower band percentage (default 9%)

    Returns:
        DataFrame with ene_upper, ene_lower, ene_mid
    """
    if df.empty or len(df) < period:
        return df
    df = df.copy()

    ma = df['close'].rolling(window=period).mean()

    df['ene_upper'] = ma * (1 + upper_pct / 100)
    df['ene_lower'] = ma * (1 - lower_pct / 100)
    df['ene_mid'] = (df['ene_upper'] + df['ene_lower']) / 2

    return df


def calculate_stoch_rsi(df, rsi_period=14, stoch_period=14, k_period=3):
    """
    Calculate Stochastic RSI.

    StochRSI applies Stochastic formula to RSI values.
    More sensitive than regular RSI.

    Args:
        df: DataFrame with 'rsi' column (or will calculate)
        rsi_period: RSI period (default 14)
        stoch_period: Stochastic period (default 14)
        k_period: %K smoothing period (default 3)

    Returns:
        DataFrame with stochrsi_k, stochrsi_d
    """
    if df.empty or len(df) < rsi_period + stoch_period:
        return df
    df = df.copy()

    # Calculate RSI if not present
    if 'rsi' not in df.columns:
        df = calculate_rsi(df, rsi_period)

    rsi = df['rsi']

    # Stochastic of RSI
    rsi_min = rsi.rolling(window=stoch_period).min()
    rsi_max = rsi.rolling(window=stoch_period).max()

    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
    df['stochrsi_k'] = stoch_rsi.rolling(window=k_period).mean() * 100
    df['stochrsi_k'] = df['stochrsi_k'].fillna(50)

    # %D line
    df['stochrsi_d'] = df['stochrsi_k'].rolling(window=3).mean()

    return df


def calculate_ppo(df, fast=12, slow=26, signal=9):
    """
    Calculate PPO (Percentage Price Oscillator).

    Similar to MACD but expressed as percentage.
    PPO > 0: Bullish momentum
    PPO < 0: Bearish momentum

    Args:
        df: DataFrame with OHLCV data
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)

    Returns:
        DataFrame with ppo, ppo_signal, ppo_hist
    """
    if df.empty or len(df) < slow:
        return df
    df = df.copy()

    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()

    df['ppo'] = ((ema_fast - ema_slow) / ema_slow) * 100
    df['ppo_signal'] = df['ppo'].ewm(span=signal, adjust=False).mean()
    df['ppo_hist'] = df['ppo'] - df['ppo_signal']

    return df


def calculate_williams_r(df, periods=[6, 10, 14]):
    """
    Calculate Williams %R.

    Williams %R shows overbought/oversold levels.
    %R < -80: Oversold
    %R > -20: Overbought

    Args:
        df: DataFrame with OHLCV data
        periods: List of periods to calculate

    Returns:
        DataFrame with wr columns for each period
    """
    if df.empty or len(df) < max(periods):
        return df
    df = df.copy()

    for period in periods:
        highest = df['high'].rolling(window=period).max()
        lowest = df['low'].rolling(window=period).min()

        df[f'wr_{period}'] = ((highest - df['close']) / (highest - lowest).replace(0, np.nan)) * -100
        df[f'wr_{period}'] = df[f'wr_{period}'].fillna(-50)

    return df


def calculate_cci(df, periods=[14, 84]):
    """
    Calculate CCI (Commodity Channel Index).

    CCI measures the current price level relative to an average.
    CCI > 100: Overbought
    CCI < -100: Oversold

    Args:
        df: DataFrame with OHLCV data
        periods: List of periods to calculate

    Returns:
        DataFrame with cci columns for each period
    """
    if df.empty or len(df) < max(periods):
        return df
    df = df.copy()

    typical_price = (df['high'] + df['low'] + df['close']) / 3

    for period in periods:
        ma = typical_price.rolling(window=period).mean()
        md = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )

        col_name = 'cci' if period == 14 else f'cci_{period}'
        df[col_name] = (typical_price - ma) / (0.015 * md.replace(0, np.nan))
        df[col_name] = df[col_name].fillna(0)

    return df


def calculate_dma(df, fast=10, slow=50, signal=10):
    """
    Calculate DMA (Differential Moving Average).

    DMA is the difference between short and long MAs.

    Args:
        df: DataFrame with OHLCV data
        fast: Fast MA period (default 10)
        slow: Slow MA period (default 50)
        signal: Signal line period (default 10)

    Returns:
        DataFrame with dma, dma_signal
    """
    if df.empty or len(df) < slow:
        return df
    df = df.copy()

    ma_fast = df['close'].rolling(window=fast).mean()
    ma_slow = df['close'].rolling(window=slow).mean()

    df['dma'] = ma_fast - ma_slow
    df['dma_signal'] = df['dma'].rolling(window=signal).mean()

    return df


def calculate_mfi(df, period=14):
    """
    Calculate MFI (Money Flow Index).

    MFI is volume-weighted RSI.
    MFI > 80: Overbought
    MFI < 20: Oversold

    Args:
        df: DataFrame with OHLCV data
        period: Calculation period (default 14)

    Returns:
        DataFrame with mfi
    """
    if df.empty or len(df) < period:
        return df
    df = df.copy()

    # Typical price
    typical_price = (df['high'] + df['low'] + df['close']) / 3

    # Raw money flow
    raw_mf = typical_price * df['volume']

    # Direction
    tp_change = typical_price.diff()

    # Positive and negative money flow
    pos_mf = np.where(tp_change > 0, raw_mf, 0)
    neg_mf = np.where(tp_change < 0, raw_mf, 0)

    # Sum over period
    pos_sum = pd.Series(pos_mf).rolling(window=period).sum()
    neg_sum = pd.Series(neg_mf).rolling(window=period).sum()

    # Money flow ratio
    mfr = pos_sum / neg_sum.replace(0, np.nan)

    # MFI
    df['mfi'] = 100 - (100 / (1 + mfr))
    df['mfi'] = df['mfi'].fillna(50)

    return df


def calculate_vwma(df, period=14):
    """
    Calculate VWMA (Volume-Weighted Moving Average).

    VWMA weights prices by volume, giving more weight to
    high-volume periods.

    Args:
        df: DataFrame with OHLCV data
        period: Calculation period (default 14)

    Returns:
        DataFrame with vwma
    """
    if df.empty or len(df) < period:
        return df
    df = df.copy()

    if 'amount' in df.columns:
        tpv_sum = df['amount'].rolling(window=period).sum()
    else:
        tpv_sum = (df['close'] * df['volume']).rolling(window=period).sum()

    vol_sum = df['volume'].rolling(window=period).sum()

    df['vwma'] = tpv_sum / vol_sum.replace(0, np.nan)
    df['vwma'] = df['vwma'].fillna(df['close'])

    return df


def calculate_advanced_features(df):
    """Calculate advanced features for ML and analysis."""
    if df.empty or len(df) < 20:
        return df
    df = df.copy()

    # Price patterns
    df['gap'] = df['open'] / df['close'].shift(1) - 1  # Gap up/down
    df['body_size'] = abs(df['close'] - df['open']) / df['open']
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']

    # Trend strength
    df['trend_strength'] = abs(df['close'] - df['close'].shift(20)) / df['close'].shift(20)

    # Returns at different periods
    df['ret_1d'] = df['close'].pct_change(1) * 100
    df['ret_5d'] = df['close'].pct_change(5) * 100
    df['ret_10d'] = df['close'].pct_change(10) * 100
    df['ret_20d'] = df['close'].pct_change(20) * 100

    # Volatility
    df['volatility_20d'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252) * 100

    # Volume features
    df['volume_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
    df['price_volume_corr'] = df['close'].pct_change().rolling(10).corr(
        df['volume'].pct_change()
    )

    # Support/Resistance proximity
    df['dist_to_high_20'] = (df['close'] / df['high'].rolling(20).max() - 1) * 100
    df['dist_to_low_20'] = (df['close'] / df['low'].rolling(20).min() - 1) * 100

    # 52-week high/low position
    if len(df) >= 250:
        high_52w = df['high'].rolling(250).max()
        low_52w = df['low'].rolling(250).min()
        df['position_52w'] = (df['close'] - low_52w) / (high_52w - low_52w) * 100

    # Consecutive up/down days
    df['up_day'] = (df['close'] > df['close'].shift(1)).astype(int)
    df['consecutive_up'] = df['up_day'].groupby(
        (df['up_day'] != df['up_day'].shift()).cumsum()
    ).cumsum() * df['up_day']
    df['consecutive_down'] = (1 - df['up_day']).groupby(
        ((1 - df['up_day']) != (1 - df['up_day']).shift()).cumsum()
    ).cumsum() * (1 - df['up_day'])

    return df


def calculate_all(df, include_advanced=True):
    """
    Calculate all indicators including new indicators from InStock.

    Args:
        df: DataFrame with OHLCV data
        include_advanced: Whether to include advanced/expensive indicators

    Returns:
        DataFrame with all indicators calculated
    """
    if df.empty:
        return df
    df = df.copy()

    # Core indicators (always calculated)
    df = calculate_ma(df)
    df = calculate_macd(df)
    df = calculate_rsi(df)
    df = calculate_volume_ma(df)
    df = calculate_atr(df)
    df = calculate_kdj(df)
    df = calculate_bollinger(df)
    df = calculate_obv(df)
    df = calculate_adx(df)

    # New indicators from InStock
    df = calculate_supertrend(df)      # Trend following
    df = calculate_wave_trend(df)       # Advanced oscillator
    df = calculate_vr(df)               # Volume ratio
    df = calculate_cr(df)               # Energy indicator
    df = calculate_psy(df)              # Psychology line
    df = calculate_brar(df)             # Sentiment
    df = calculate_bias(df)             # MA deviation
    df = calculate_mfi(df)              # Money flow
    df = calculate_vwma(df)             # Volume-weighted MA

    if include_advanced:
        # Additional indicators (more expensive to compute)
        df = calculate_trix(df)             # Triple EMA momentum
        df = calculate_force_index(df)      # Volume-weighted momentum
        df = calculate_emv(df)              # Ease of movement
        df = calculate_dpo(df)              # Detrended price
        df = calculate_vhf(df)              # Trend strength filter
        df = calculate_rvi(df)              # Relative vigor
        df = calculate_ene(df)              # Envelope channels
        df = calculate_stoch_rsi(df)        # Stochastic RSI
        df = calculate_ppo(df)              # Percentage price oscillator
        df = calculate_williams_r(df)       # Williams %R
        df = calculate_cci(df)              # Commodity channel index
        df = calculate_dma(df)              # Differential MA
        df = calculate_advanced_features(df)

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


# =============================================================================
# NEW SIGNAL ANALYSIS FUNCTIONS
# =============================================================================

def get_supertrend_signal(df):
    """
    Analyze Supertrend indicator for trend direction.

    Supertrend is one of the most reliable trend-following indicators.
    """
    if df.empty or 'supertrend_direction' not in df.columns:
        return {'signal': 'neutral', 'score': 0, 'details': {}}

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    direction = latest.get('supertrend_direction', 0)
    prev_direction = prev.get('supertrend_direction', 0)
    close = latest.get('close', 0)
    supertrend = latest.get('supertrend', 0)

    score = 0

    # Direction gives primary signal
    if direction == 1:
        score = 0.5  # Bullish
    elif direction == -1:
        score = -0.5  # Bearish

    # Check for trend flip (strong signal)
    if prev_direction == -1 and direction == 1:
        score = 0.8  # Bullish flip
    elif prev_direction == 1 and direction == -1:
        score = -0.8  # Bearish flip

    # Distance from supertrend line
    if close > 0 and supertrend > 0:
        distance_pct = (close - supertrend) / close * 100
        if direction == 1 and distance_pct > 5:
            score += 0.1  # Strong uptrend
        elif direction == -1 and distance_pct < -5:
            score -= 0.1  # Strong downtrend

    score = max(-1, min(1, score))
    signal = 'bullish' if score > 0.1 else ('bearish' if score < -0.1 else 'neutral')

    return {
        'signal': signal,
        'score': score,
        'details': {
            'direction': 'bullish' if direction == 1 else 'bearish',
            'supertrend': round(supertrend, 2) if supertrend else 0,
            'flipped': prev_direction != direction
        }
    }


def get_wave_trend_signal(df):
    """
    Analyze Wave Trend indicator.

    WT is an advanced oscillator combining price smoothing with deviation analysis.
    """
    if df.empty or 'wt1' not in df.columns:
        return {'signal': 'neutral', 'score': 0, 'details': {}}

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    wt1 = latest.get('wt1', 0)
    wt2 = latest.get('wt2', 0)
    prev_wt1 = prev.get('wt1', 0)
    prev_wt2 = prev.get('wt2', 0)

    score = 0

    # Overbought/Oversold levels
    if wt1 < -60:
        score = 0.6  # Extreme oversold
    elif wt1 < -40:
        score = 0.3  # Oversold
    elif wt1 > 60:
        score = -0.6  # Extreme overbought
    elif wt1 > 40:
        score = -0.3  # Overbought

    # Crossover signals
    if prev_wt1 <= prev_wt2 and wt1 > wt2:
        score += 0.3  # Bullish crossover
    elif prev_wt1 >= prev_wt2 and wt1 < wt2:
        score -= 0.3  # Bearish crossover

    # WT1 vs WT2 position
    if wt1 > wt2:
        score += 0.1
    else:
        score -= 0.1

    score = max(-1, min(1, score))
    signal = 'bullish' if score > 0.1 else ('bearish' if score < -0.1 else 'neutral')

    return {
        'signal': signal,
        'score': score,
        'details': {
            'wt1': round(wt1, 1),
            'wt2': round(wt2, 1),
            'level': 'oversold' if wt1 < -40 else ('overbought' if wt1 > 40 else 'normal')
        }
    }


def get_vr_signal(df):
    """
    Analyze VR (Volume Ratio) indicator.

    VR > 150: Bullish accumulation
    VR < 70: Distribution
    """
    if df.empty or 'vr' not in df.columns:
        return {'signal': 'neutral', 'score': 0, 'details': {}}

    latest = df.iloc[-1]
    vr = latest.get('vr', 100)

    score = 0

    if vr > 250:
        score = 0.6  # Strong accumulation
    elif vr > 150:
        score = 0.4  # Accumulation
    elif vr > 120:
        score = 0.2  # Mild bullish
    elif vr < 50:
        score = -0.6  # Strong distribution
    elif vr < 70:
        score = -0.4  # Distribution
    elif vr < 80:
        score = -0.2  # Mild bearish

    signal = 'bullish' if score > 0.1 else ('bearish' if score < -0.1 else 'neutral')

    return {
        'signal': signal,
        'score': score,
        'details': {
            'vr': round(vr, 1),
            'level': 'accumulation' if vr > 150 else ('distribution' if vr < 70 else 'normal')
        }
    }


def get_cr_signal(df):
    """
    Analyze CR (Energy Indicator).

    CR > 200: Overbought (selling pressure likely)
    CR < 40: Oversold (buying opportunity)
    """
    if df.empty or 'cr' not in df.columns:
        return {'signal': 'neutral', 'score': 0, 'details': {}}

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    cr = latest.get('cr', 100)
    cr_ma1 = latest.get('cr_ma1', cr)
    prev_cr = prev.get('cr', 100)

    score = 0

    # Overbought/Oversold
    if cr < 20:
        score = 0.7  # Extreme oversold
    elif cr < 40:
        score = 0.5  # Oversold
    elif cr > 300:
        score = -0.5  # Extreme overbought
    elif cr > 200:
        score = -0.3  # Overbought

    # CR momentum
    if cr > cr_ma1 and prev_cr < cr:
        score += 0.15
    elif cr < cr_ma1 and prev_cr > cr:
        score -= 0.15

    score = max(-1, min(1, score))
    signal = 'bullish' if score > 0.1 else ('bearish' if score < -0.1 else 'neutral')

    return {
        'signal': signal,
        'score': score,
        'details': {
            'cr': round(cr, 1),
            'level': 'oversold' if cr < 40 else ('overbought' if cr > 200 else 'normal')
        }
    }


def get_psy_signal(df):
    """
    Analyze PSY (Psychology Line).

    PSY > 75: Market too optimistic (contrarian sell)
    PSY < 25: Market too pessimistic (contrarian buy)
    """
    if df.empty or 'psy' not in df.columns:
        return {'signal': 'neutral', 'score': 0, 'details': {}}

    latest = df.iloc[-1]
    psy = latest.get('psy', 50)

    score = 0

    # Contrarian signals
    if psy < 17:
        score = 0.6  # Extreme pessimism - buy
    elif psy < 25:
        score = 0.4  # Pessimism - buy
    elif psy > 83:
        score = -0.6  # Extreme optimism - sell
    elif psy > 75:
        score = -0.4  # Optimism - sell
    elif psy > 58:
        score = 0.1  # Slight bullish bias
    elif psy < 42:
        score = -0.1  # Slight bearish bias

    signal = 'bullish' if score > 0.1 else ('bearish' if score < -0.1 else 'neutral')

    return {
        'signal': signal,
        'score': score,
        'details': {
            'psy': round(psy, 1),
            'level': 'pessimistic' if psy < 25 else ('optimistic' if psy > 75 else 'normal')
        }
    }


def get_brar_signal(df):
    """
    Analyze BRAR (Sentiment Indicators).

    AR measures buying strength, BR measures sentiment.
    """
    if df.empty or 'ar' not in df.columns:
        return {'signal': 'neutral', 'score': 0, 'details': {}}

    latest = df.iloc[-1]
    ar = latest.get('ar', 100)
    br = latest.get('br', 100)

    score = 0

    # AR analysis
    if ar > 180:
        score -= 0.2  # Too much buying pressure - might reverse
    elif ar > 120:
        score += 0.2  # Good buying strength
    elif ar < 50:
        score += 0.3  # Oversold, potential reversal

    # BR vs AR
    if br > ar:
        score += 0.15  # Bullish sentiment
    elif br < ar * 0.7:
        score -= 0.15  # Bearish sentiment

    # Extreme BR
    if br < 40:
        score += 0.3  # Extreme pessimism
    elif br > 400:
        score -= 0.3  # Extreme optimism

    score = max(-1, min(1, score))
    signal = 'bullish' if score > 0.1 else ('bearish' if score < -0.1 else 'neutral')

    return {
        'signal': signal,
        'score': score,
        'details': {
            'ar': round(ar, 1),
            'br': round(br, 1),
            'sentiment': 'bullish' if br > ar else 'bearish'
        }
    }


def get_bias_signal(df):
    """
    Analyze BIAS (MA Deviation).

    Large positive BIAS: Price too far above MA - pullback likely
    Large negative BIAS: Price too far below MA - bounce likely
    """
    if df.empty or 'bias_6' not in df.columns:
        return {'signal': 'neutral', 'score': 0, 'details': {}}

    latest = df.iloc[-1]
    bias_6 = latest.get('bias_6', 0)
    bias_12 = latest.get('bias_12', 0)
    bias_24 = latest.get('bias_24', 0)

    score = 0

    # Mean reversion based on BIAS
    if bias_6 < -8:
        score = 0.5  # Strong bounce potential
    elif bias_6 < -5:
        score = 0.3  # Bounce potential
    elif bias_6 > 10:
        score = -0.3  # Pullback likely
    elif bias_6 > 15:
        score = -0.5  # Strong pullback likely

    # Confirm with longer-term BIAS
    if bias_24 < -10:
        score += 0.2  # Long-term oversold
    elif bias_24 > 15:
        score -= 0.2  # Long-term overbought

    score = max(-1, min(1, score))
    signal = 'bullish' if score > 0.1 else ('bearish' if score < -0.1 else 'neutral')

    return {
        'signal': signal,
        'score': score,
        'details': {
            'bias_6': round(bias_6, 1),
            'bias_12': round(bias_12, 1),
            'bias_24': round(bias_24, 1)
        }
    }


def get_mfi_signal(df):
    """
    Analyze MFI (Money Flow Index).

    MFI is volume-weighted RSI.
    MFI > 80: Overbought
    MFI < 20: Oversold
    """
    if df.empty or 'mfi' not in df.columns:
        return {'signal': 'neutral', 'score': 0, 'details': {}}

    latest = df.iloc[-1]
    mfi = latest.get('mfi', 50)

    score = 0

    if mfi < 15:
        score = 0.7  # Extreme oversold
    elif mfi < 20:
        score = 0.5  # Oversold
    elif mfi < 30:
        score = 0.25  # Mildly oversold
    elif mfi > 85:
        score = -0.5  # Extreme overbought
    elif mfi > 80:
        score = -0.3  # Overbought
    elif mfi > 70:
        score = -0.15  # Mildly overbought

    signal = 'bullish' if score > 0.1 else ('bearish' if score < -0.1 else 'neutral')

    return {
        'signal': signal,
        'score': score,
        'details': {
            'mfi': round(mfi, 1),
            'level': 'oversold' if mfi < 20 else ('overbought' if mfi > 80 else 'normal')
        }
    }


def get_force_index_signal(df):
    """
    Analyze Force Index.

    Positive Force: Bulls in control
    Negative Force: Bears in control
    """
    if df.empty or 'force_13' not in df.columns:
        return {'signal': 'neutral', 'score': 0, 'details': {}}

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    force_2 = latest.get('force_2', 0)
    force_13 = latest.get('force_13', 0)
    prev_force_13 = prev.get('force_13', 0)

    score = 0

    # Force direction
    if force_13 > 0:
        score = 0.3  # Bullish force
    else:
        score = -0.3  # Bearish force

    # Force momentum
    if force_13 > prev_force_13:
        score += 0.2  # Increasing force
    else:
        score -= 0.2  # Decreasing force

    # Short vs long-term force
    if force_2 > 0 and force_13 > 0:
        score += 0.15  # Aligned bullish
    elif force_2 < 0 and force_13 < 0:
        score -= 0.15  # Aligned bearish

    score = max(-1, min(1, score))
    signal = 'bullish' if score > 0.1 else ('bearish' if score < -0.1 else 'neutral')

    return {
        'signal': signal,
        'score': score,
        'details': {
            'force_2': round(force_2, 0),
            'force_13': round(force_13, 0),
            'direction': 'bullish' if force_13 > 0 else 'bearish'
        }
    }


def get_vhf_signal(df):
    """
    Analyze VHF (Vertical Horizontal Filter).

    VHF determines if market is trending or ranging.
    VHF > 0.4: Strong trend
    VHF < 0.3: Ranging
    """
    if df.empty or 'vhf' not in df.columns:
        return {'signal': 'neutral', 'score': 0, 'details': {}}

    latest = df.iloc[-1]
    vhf = latest.get('vhf', 0.35)

    # VHF doesn't give direction, just trend strength
    trending = vhf > 0.4
    ranging = vhf < 0.3

    return {
        'signal': 'neutral',  # VHF doesn't give direction
        'score': 0,
        'details': {
            'vhf': round(vhf, 3),
            'market_state': 'trending' if trending else ('ranging' if ranging else 'moderate'),
            'use_trend_strategy': trending,
            'use_mean_reversion': ranging
        }
    }


def get_rvi_signal(df):
    """
    Analyze RVI (Relative Vigor Index).

    RVI > 0: Bullish vigor
    RVI < 0: Bearish vigor
    """
    if df.empty or 'rvi' not in df.columns:
        return {'signal': 'neutral', 'score': 0, 'details': {}}

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    rvi = latest.get('rvi', 0)
    rvi_signal = latest.get('rvi_signal', 0)
    prev_rvi = prev.get('rvi', 0)
    prev_rvi_signal = prev.get('rvi_signal', 0)

    score = 0

    # RVI direction
    if rvi > 0.1:
        score = 0.3  # Bullish vigor
    elif rvi < -0.1:
        score = -0.3  # Bearish vigor

    # Crossover
    if prev_rvi <= prev_rvi_signal and rvi > rvi_signal:
        score += 0.3  # Bullish crossover
    elif prev_rvi >= prev_rvi_signal and rvi < rvi_signal:
        score -= 0.3  # Bearish crossover

    score = max(-1, min(1, score))
    signal = 'bullish' if score > 0.1 else ('bearish' if score < -0.1 else 'neutral')

    return {
        'signal': signal,
        'score': score,
        'details': {
            'rvi': round(rvi, 3),
            'rvi_signal': round(rvi_signal, 3)
        }
    }


def get_trix_signal(df):
    """
    Analyze TRIX (Triple EMA Momentum).

    TRIX > 0: Bullish momentum
    TRIX crossing signal line: Entry/exit signal
    """
    if df.empty or 'trix' not in df.columns:
        return {'signal': 'neutral', 'score': 0, 'details': {}}

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    trix = latest.get('trix', 0)
    trix_signal = latest.get('trix_signal', 0)
    prev_trix = prev.get('trix', 0)
    prev_trix_signal = prev.get('trix_signal', 0)

    score = 0

    # TRIX direction
    if trix > 0:
        score = 0.2  # Bullish
    else:
        score = -0.2  # Bearish

    # Crossover
    if prev_trix <= prev_trix_signal and trix > trix_signal:
        score += 0.4  # Bullish crossover
    elif prev_trix >= prev_trix_signal and trix < trix_signal:
        score -= 0.4  # Bearish crossover

    # Momentum
    if trix > prev_trix:
        score += 0.1
    else:
        score -= 0.1

    score = max(-1, min(1, score))
    signal = 'bullish' if score > 0.1 else ('bearish' if score < -0.1 else 'neutral')

    return {
        'signal': signal,
        'score': score,
        'details': {
            'trix': round(trix, 4),
            'trix_signal': round(trix_signal, 4) if trix_signal else 0
        }
    }
