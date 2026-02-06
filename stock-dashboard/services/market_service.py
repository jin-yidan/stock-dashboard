"""
Market regime detection and market-wide signals.
Provides context for individual stock analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    import adata
    HAS_ADATA = True
except ImportError:
    HAS_ADATA = False

# Cache
_cache = {}
_cache_time = {}
CACHE_TTL = 1800  # 30 minutes


def _get_cached(key):
    if key in _cache and key in _cache_time:
        if (datetime.now() - _cache_time[key]).seconds < CACHE_TTL:
            return _cache[key]
    return None


def _set_cached(key, value):
    _cache[key] = value
    _cache_time[key] = datetime.now()


def get_index_data(index_code='000001', days=60):
    """Get index historical data (default: Shanghai Composite)."""
    if not HAS_ADATA:
        return pd.DataFrame()

    cache_key = f"index_{index_code}_{days}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    try:
        df = adata.stock.market.get_market_index(index_code=index_code)
        if df is not None and not df.empty:
            df = df.sort_values('trade_date').tail(days)
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['open'] = pd.to_numeric(df['open'], errors='coerce')
            df['high'] = pd.to_numeric(df['high'], errors='coerce')
            df['low'] = pd.to_numeric(df['low'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            _set_cached(cache_key, df)
            return df
    except Exception as e:
        print(f"Error getting index data: {e}")

    return pd.DataFrame()


def detect_market_regime():
    """
    Detect current market regime: strong_bull, bull, sideways, bear, strong_bear.
    Uses Shanghai Composite Index.
    """
    cache_key = "market_regime"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    df = get_index_data('000001', days=120)
    if df.empty or len(df) < 60:
        return {'regime': 'unknown', 'confidence': 0, 'details': {}}

    close = df['close'].values

    # Calculate indicators
    ma5 = pd.Series(close).rolling(5).mean().iloc[-1]
    ma10 = pd.Series(close).rolling(10).mean().iloc[-1]
    ma20 = pd.Series(close).rolling(20).mean().iloc[-1]
    ma60 = pd.Series(close).rolling(60).mean().iloc[-1]

    current = close[-1]

    # Volatility (annualized)
    returns = pd.Series(close).pct_change().dropna()
    volatility = returns.std() * np.sqrt(252)
    recent_volatility = returns.tail(10).std() * np.sqrt(252)

    # Trend strength
    ret_5d = (close[-1] / close[-5] - 1) * 100 if len(close) >= 5 else 0
    ret_20d = (close[-1] / close[-20] - 1) * 100 if len(close) >= 20 else 0
    ret_60d = (close[-1] / close[-60] - 1) * 100 if len(close) >= 60 else 0

    # MA alignment score
    ma_score = 0
    if current > ma5:
        ma_score += 1
    if current > ma10:
        ma_score += 1
    if current > ma20:
        ma_score += 1
    if current > ma60:
        ma_score += 1
    if ma5 > ma10:
        ma_score += 1
    if ma10 > ma20:
        ma_score += 1
    if ma20 > ma60:
        ma_score += 1

    # Determine regime
    if ma_score >= 6 and ret_20d > 5:
        regime = 'strong_bull'
        confidence = min(90, 60 + ret_20d)
    elif ma_score >= 5 and ret_20d > 0:
        regime = 'bull'
        confidence = min(80, 50 + ma_score * 5)
    elif ma_score <= 1 and ret_20d < -5:
        regime = 'strong_bear'
        confidence = min(90, 60 + abs(ret_20d))
    elif ma_score <= 2 and ret_20d < 0:
        regime = 'bear'
        confidence = min(80, 50 + (7 - ma_score) * 5)
    else:
        regime = 'sideways'
        confidence = 50 + (3.5 - abs(ma_score - 3.5)) * 10

    result = {
        'regime': regime,
        'confidence': round(confidence),
        'details': {
            'ma_score': ma_score,
            'ret_5d': round(ret_5d, 2),
            'ret_20d': round(ret_20d, 2),
            'ret_60d': round(ret_60d, 2),
            'volatility': round(volatility * 100, 1),
            'recent_volatility': round(recent_volatility * 100, 1),
            'index_close': round(current, 2)
        }
    }

    _set_cached(cache_key, result)
    return result


def get_regime_weights(regime):
    """
    Get indicator weights adjusted for market regime.

    OPTIMIZED based on backtest accuracy:
    - supertrend: 62.4% (best)
    - bollinger: 60.5%
    - volume: 58.4%
    - wave_trend: 55.4%
    - weekly: 53.8%
    - momentum: 40.4% (worst - minimized)
    """
    # Base optimized weights - high accuracy indicators get more weight
    base = {
        'supertrend': 0.18,    # 62.4% - best predictor
        'bollinger': 0.14,     # 60.5%
        'volume': 0.12,        # 58.4%
        'wave_trend': 0.10,    # 55.4%
        'weekly': 0.10,        # 53.8%
        'ma': 0.08,            # 50.1%
        'macd': 0.06,          # 49.8%
        'rsi': 0.04,           # 47.8%
        'kdj': 0.04,           # 46.4%
        'momentum': 0.02,      # 40.4% - worst
        'capital_flow': 0.04,
        'trend_strength': 0.02,
        'vr': 0.02,
        'cr': 0.02,
        'cyq': 0.02,
        'strategy': 0.00,
    }

    # Slight regime adjustments (keep base weights dominant)
    if regime == 'strong_bull':
        base['supertrend'] = 0.20  # Boost trend in bull
        base['weekly'] = 0.12
        base['bollinger'] = 0.12
    elif regime == 'bull':
        base['supertrend'] = 0.19
        base['weekly'] = 0.11
    elif regime == 'strong_bear':
        base['bollinger'] = 0.16  # Boost mean reversion in bear
        base['wave_trend'] = 0.12
        base['supertrend'] = 0.16
    elif regime == 'bear':
        base['bollinger'] = 0.15
        base['wave_trend'] = 0.11
    # sideways uses base weights

    return base


def get_northbound_flow(days=20):
    """
    Get northbound capital flow (沪股通+深股通).
    This is "smart money" from foreign institutions.

    Fetches data from East Money datacenter API.
    """
    import requests

    cache_key = f"northbound_{days}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
        url = 'https://datacenter-web.eastmoney.com/api/data/v1/get'
        params = {
            'reportName': 'RPT_MUTUAL_QUOTA',
            'columns': 'ALL',
            'pageSize': days,
            'sortColumns': 'TRADE_DATE',
            'sortTypes': -1,
            'source': 'WEB',
            'client': 'WEB',
            'filter': '(MUTUAL_TYPE="001")'  # 001 = northbound total (沪股通+深股通)
        }

        r = requests.get(url, params=params, headers=headers, timeout=15)
        if r.status_code != 200:
            return pd.DataFrame()

        data = r.json()
        if 'result' not in data or not data['result'] or 'data' not in data['result']:
            return pd.DataFrame()

        records = data['result']['data']
        rows = []
        for rec in records:
            rows.append({
                'trade_date': rec.get('TRADE_DATE', '')[:10],
                'net_flow': rec.get('NET_DEAL_AMT', 0) or 0,  # 净买入额
                'buy_amt': rec.get('BUY_AMT', 0) or 0,
                'sell_amt': rec.get('SELL_AMT', 0) or 0,
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values('trade_date').reset_index(drop=True)
            _set_cached(cache_key, df)

        return df

    except Exception as e:
        print(f"Error fetching northbound flow: {e}")
        return pd.DataFrame()


def get_northbound_signal():
    """
    Generate trading signal from northbound flow.

    Returns score from -1 (bearish) to +1 (bullish)
    """
    cache_key = "northbound_signal"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    df = get_northbound_flow(days=20)
    if df.empty:
        return {'score': 0, 'signal': 'neutral', 'details': {}}

    try:
        # Get net flow column (might be named differently)
        flow_col = None
        for col in ['north_money', 'net_flow', 'value', 'net_buy']:
            if col in df.columns:
                flow_col = col
                break

        if flow_col is None:
            return {'score': 0, 'signal': 'neutral', 'details': {}}

        flows = pd.to_numeric(df[flow_col], errors='coerce').fillna(0)

        # Calculate signals
        total_5d = flows.tail(5).sum()
        total_10d = flows.tail(10).sum()
        total_20d = flows.sum()

        # Recent trend
        recent_avg = flows.tail(5).mean()
        older_avg = flows.tail(20).head(15).mean()

        # Score calculation
        score = 0

        # 5-day flow strength
        if total_5d > 10000000000:  # >100亿
            score += 0.4
        elif total_5d > 5000000000:  # >50亿
            score += 0.25
        elif total_5d > 0:
            score += 0.1
        elif total_5d > -5000000000:
            score -= 0.1
        elif total_5d > -10000000000:
            score -= 0.25
        else:
            score -= 0.4

        # Trend acceleration
        if recent_avg > older_avg * 1.5:
            score += 0.2
        elif recent_avg < older_avg * 0.5:
            score -= 0.2

        # Consistency
        positive_days = (flows.tail(10) > 0).sum()
        if positive_days >= 8:
            score += 0.15
        elif positive_days <= 2:
            score -= 0.15

        score = max(-1, min(1, score))

        signal = 'bullish' if score > 0.2 else ('bearish' if score < -0.2 else 'neutral')

        result = {
            'score': round(score, 3),
            'signal': signal,
            'details': {
                'total_5d_yi': round(total_5d / 100000000, 1),
                'total_10d_yi': round(total_10d / 100000000, 1),
                'total_20d_yi': round(total_20d / 100000000, 1),
                'positive_days_10d': int(positive_days),
                'trend': 'accelerating' if recent_avg > older_avg else 'decelerating'
            }
        }

        _set_cached(cache_key, result)
        return result

    except Exception as e:
        print(f"Error calculating northbound signal: {e}")
        return {'score': 0, 'signal': 'neutral', 'details': {}}


def get_margin_data(stock_code, days=20):
    """Get margin trading data for a stock (融资融券)."""
    if not HAS_ADATA:
        return pd.DataFrame()

    cache_key = f"margin_{stock_code}_{days}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    try:
        df = adata.stock.market.get_margin_detail(stock_code=stock_code)
        if df is not None and not df.empty:
            df = df.sort_values('trade_date').tail(days)
            _set_cached(cache_key, df)
            return df
    except Exception as e:
        print(f"Error getting margin data for {stock_code}: {e}")

    return pd.DataFrame()


def get_margin_signal(stock_code):
    """
    Generate signal from margin trading data.

    融资余额增加 = 看多情绪
    融券余额增加 = 看空情绪
    """
    df = get_margin_data(stock_code, days=20)
    if df.empty:
        return {'score': 0, 'signal': 'neutral', 'has_data': False}

    try:
        # Find relevant columns
        buy_col = None
        sell_col = None
        for col in df.columns:
            if '融资' in col and '余额' in col:
                buy_col = col
            elif '融券' in col and '余额' in col:
                sell_col = col

        if buy_col is None:
            return {'score': 0, 'signal': 'neutral', 'has_data': False}

        buy_balance = pd.to_numeric(df[buy_col], errors='coerce').fillna(0)

        # Calculate trend
        if len(buy_balance) >= 10:
            recent = buy_balance.tail(5).mean()
            older = buy_balance.tail(10).head(5).mean()

            change_pct = (recent / older - 1) * 100 if older > 0 else 0

            if change_pct > 5:
                score = 0.4
            elif change_pct > 2:
                score = 0.2
            elif change_pct > 0:
                score = 0.1
            elif change_pct > -2:
                score = -0.1
            elif change_pct > -5:
                score = -0.2
            else:
                score = -0.4

            signal = 'bullish' if score > 0.15 else ('bearish' if score < -0.15 else 'neutral')

            return {
                'score': round(score, 3),
                'signal': signal,
                'has_data': True,
                'details': {
                    'margin_change_pct': round(change_pct, 2),
                    'trend': 'increasing' if change_pct > 0 else 'decreasing'
                }
            }

    except Exception as e:
        print(f"Error calculating margin signal: {e}")

    return {'score': 0, 'signal': 'neutral', 'has_data': False}


def get_market_breadth():
    """
    Calculate market breadth - how many stocks are rising vs falling.
    Good for confirming market direction.
    """
    if not HAS_ADATA:
        return {'score': 0, 'rising': 0, 'falling': 0, 'total': 0}

    cache_key = "market_breadth"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    try:
        # Get current market data for all stocks
        df = adata.stock.market.list_market_current()
        if df is not None and not df.empty:
            change_pct = pd.to_numeric(df['change_pct'], errors='coerce')

            rising = (change_pct > 0).sum()
            falling = (change_pct < 0).sum()
            flat = (change_pct == 0).sum()
            total = len(change_pct)

            # Limit up/down
            limit_up = (change_pct >= 9.9).sum()
            limit_down = (change_pct <= -9.9).sum()

            advance_decline_ratio = rising / falling if falling > 0 else 10

            # Score
            if advance_decline_ratio > 3:
                score = 0.8
            elif advance_decline_ratio > 2:
                score = 0.5
            elif advance_decline_ratio > 1.2:
                score = 0.2
            elif advance_decline_ratio > 0.8:
                score = 0
            elif advance_decline_ratio > 0.5:
                score = -0.2
            elif advance_decline_ratio > 0.33:
                score = -0.5
            else:
                score = -0.8

            result = {
                'score': round(score, 3),
                'rising': int(rising),
                'falling': int(falling),
                'flat': int(flat),
                'total': int(total),
                'advance_decline_ratio': round(advance_decline_ratio, 2),
                'limit_up': int(limit_up),
                'limit_down': int(limit_down),
                'rising_pct': round(rising / total * 100, 1) if total > 0 else 0
            }

            _set_cached(cache_key, result)
            return result

    except Exception as e:
        print(f"Error getting market breadth: {e}")

    return {'score': 0, 'rising': 0, 'falling': 0, 'total': 0}


def get_market_overview():
    """Get comprehensive market overview for dashboard."""
    regime = detect_market_regime()
    northbound = get_northbound_signal()
    breadth = get_market_breadth()

    # Combined market score
    market_score = (
        regime['details'].get('ret_5d', 0) / 10 * 0.3 +
        northbound['score'] * 0.4 +
        breadth['score'] * 0.3
    )

    return {
        'regime': regime,
        'northbound': northbound,
        'breadth': breadth,
        'market_score': round(max(-1, min(1, market_score)), 3),
        'market_signal': 'bullish' if market_score > 0.2 else ('bearish' if market_score < -0.2 else 'neutral')
    }
