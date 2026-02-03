import pandas as pd
import numpy as np
from services import data_service, indicator_service, ml_service, fundamental_service, czsc_service
from services import market_service
from services import cyq_service, strategy_service
from services import education_service


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
    elif hasattr(obj, 'dtype'):  # Catch any other numpy types
        return obj.item() if hasattr(obj, 'item') else float(obj)
    return obj

# Dynamic weights - adjusted based on market regime
# See market_service.get_regime_weights() for regime-specific weights

# Default weights - OPTIMIZED based on backtest accuracy
# Higher weights for indicators with >55% accuracy
DEFAULT_WEIGHTS = {
    # High accuracy indicators (>58%)
    'supertrend': 0.18,      # 62.4% accuracy - best predictor
    'bollinger': 0.14,       # 60.5% accuracy
    'volume': 0.12,          # 58.4% accuracy
    # Medium accuracy (53-56%)
    'wave_trend': 0.10,      # 55.4% accuracy
    'weekly': 0.10,          # 53.8% accuracy
    # Lower accuracy (<52%)
    'ma': 0.08,              # 50.1% accuracy
    'macd': 0.06,            # 49.8% accuracy
    'rsi': 0.04,             # 47.8% accuracy
    'kdj': 0.04,             # 46.4% accuracy
    'momentum': 0.02,        # 40.4% accuracy - worst predictor
    # Other indicators
    'capital_flow': 0.04,
    'trend_strength': 0.02,
    'vr': 0.02,
    'cr': 0.02,
    'cyq': 0.02,
    'strategy': 0.00,        # Disabled - too rare to trigger
}


def get_current_weights():
    """Get indicator weights adjusted for current market regime."""
    try:
        regime_info = market_service.detect_market_regime()
        regime = regime_info.get('regime', 'sideways')
        weights = market_service.get_regime_weights(regime)

        # Add CZSC weight if available
        if czsc_service.is_czsc_available():
            # Reduce other weights proportionally to add CZSC
            factor = 0.88  # Leave 12% for CZSC
            weights = {k: v * factor for k, v in weights.items()}
            weights['czsc'] = 0.12

        return weights, regime_info
    except Exception as e:
        print(f"Error getting regime weights: {e}")
        return DEFAULT_WEIGHTS, {'regime': 'unknown'}


# For backward compatibility
WEIGHTS = DEFAULT_WEIGHTS

def analyze_ma(df):
    """Analyze MA trend."""
    result = indicator_service.get_ma_trend(df)
    score = result['score']
    details = result.get('details', {})

    if df.empty or len(df) < 60:
        return {
            'name': '均线',
            'indicator_key': 'ma',
            'signal': 'neutral',
            'score': 0,
            'weight': WEIGHTS['ma'],
            'weighted_score': 0,
            'plain_explanation': '数据不足'
        }

    alignment = details.get('alignment', 'mixed')
    price_pos = details.get('price_position', 2)
    slope = details.get('ma_slope', 0)

    if alignment == 'bullish':
        if slope > 0.02:
            exp = '均线多头排列且向上发散，趋势向上明确'
        else:
            exp = '均线多头排列，短期趋势偏多'
    elif alignment == 'bearish':
        if slope < -0.02:
            exp = '均线空头排列且向下发散，趋势向下明确'
        else:
            exp = '均线空头排列，短期趋势偏空'
    else:
        if price_pos >= 3:
            exp = '均线交织但股价在多数均线上方，方向待确认'
        elif price_pos <= 1:
            exp = '均线交织但股价在多数均线下方，方向待确认'
        else:
            exp = '均线交织，趋势不明'

    return {
        'name': '均线',
        'indicator_key': 'ma',
        'signal': result['trend'],
        'score': score,
        'weight': WEIGHTS['ma'],
        'weighted_score': score * WEIGHTS['ma'],
        'plain_explanation': exp
    }

def analyze_macd(df):
    """Analyze MACD."""
    result = indicator_service.get_macd_signal(df)
    score = result['score']
    details = result.get('details', {})

    if df.empty or len(df) < 30:
        return {
            'name': 'MACD',
            'indicator_key': 'macd',
            'signal': 'neutral',
            'score': 0,
            'weight': WEIGHTS['macd'],
            'weighted_score': 0,
            'plain_explanation': '数据不足'
        }

    hist_trend = details.get('histogram_trend', 'contracting')
    above_zero = details.get('above_zero', False)

    if score > 0.5:
        exp = 'MACD金叉或红柱持续放大，短期动能强'
    elif score > 0.2:
        if above_zero:
            exp = 'MACD在零轴上方，多头动能占优'
        else:
            exp = 'MACD虽在零轴下方但动能转强'
    elif score < -0.5:
        exp = 'MACD死叉或绿柱持续放大，短期动能弱'
    elif score < -0.2:
        if not above_zero:
            exp = 'MACD在零轴下方，空头动能占优'
        else:
            exp = 'MACD虽在零轴上方但动能转弱'
    else:
        exp = 'MACD动能平衡，方向不明'

    return {
        'name': 'MACD',
        'indicator_key': 'macd',
        'signal': result['signal'],
        'score': score,
        'weight': WEIGHTS['macd'],
        'weighted_score': score * WEIGHTS['macd'],
        'plain_explanation': exp
    }

def analyze_kdj(df):
    """Analyze KDJ indicator."""
    result = indicator_service.get_kdj_signal(df)
    score = result['score']
    details = result.get('details', {})

    if df.empty or len(df) < 9:
        return {
            'name': 'KDJ',
            'indicator_key': 'kdj',
            'signal': 'neutral',
            'score': 0,
            'weight': WEIGHTS['kdj'],
            'weighted_score': 0,
            'plain_explanation': '数据不足'
        }

    k = details.get('k', 50)
    d = details.get('d', 50)
    j = details.get('j', 50)
    level = details.get('level', 'normal')

    if score > 0.5:
        exp = f'KDJ金叉，J值{j:.0f}从超卖区回升，买入信号'
    elif score > 0.2:
        if level == 'oversold':
            exp = f'KDJ处于超卖区(K={k:.0f})，可能反弹'
        else:
            exp = f'KDJ偏多(K={k:.0f},D={d:.0f})，短期向上'
    elif score < -0.5:
        exp = f'KDJ死叉，J值{j:.0f}从超买区回落，卖出信号'
    elif score < -0.2:
        if level == 'overbought':
            exp = f'KDJ处于超买区(K={k:.0f})，注意回调风险'
        else:
            exp = f'KDJ偏空(K={k:.0f},D={d:.0f})，短期向下'
    else:
        exp = f'KDJ中性(K={k:.0f},D={d:.0f})，方向不明'

    return {
        'name': 'KDJ',
        'indicator_key': 'kdj',
        'signal': result['signal'],
        'score': score,
        'weight': WEIGHTS['kdj'],
        'weighted_score': score * WEIGHTS['kdj'],
        'plain_explanation': exp
    }

def analyze_bollinger(df):
    """Analyze Bollinger Bands."""
    result = indicator_service.get_bollinger_signal(df)
    score = result['score']
    details = result.get('details', {})

    if df.empty or len(df) < 20:
        return {
            'name': '布林带',
            'signal': 'neutral',
            'score': 0,
            'weight': WEIGHTS['bollinger'],
            'weighted_score': 0,
            'plain_explanation': '数据不足'
        }

    pct_b = details.get('pct_b', 0.5)
    position = details.get('position', 'middle')
    volatility = details.get('volatility', 'normal')

    if position == 'below':
        if volatility == 'squeeze':
            exp = '股价触及下轨且波动收窄，等待突破方向'
        else:
            exp = '股价触及下轨，可能超卖反弹'
    elif position == 'above':
        if volatility == 'squeeze':
            exp = '股价触及上轨且波动收窄，等待突破方向'
        else:
            exp = '股价触及上轨，注意获利回吐'
    else:
        if volatility == 'squeeze':
            exp = '布林带收窄，即将选择方向突破'
        elif volatility == 'expanding':
            exp = '布林带扩张，趋势正在形成'
        else:
            exp = '股价在布林带中轨附近震荡'

    return {
        'name': '布林带',
        'signal': result['signal'],
        'score': score,
        'weight': WEIGHTS['bollinger'],
        'weighted_score': score * WEIGHTS['bollinger'],
        'plain_explanation': exp
    }

def analyze_rsi(df):
    """Analyze RSI."""
    result = indicator_service.get_rsi_signal(df)
    score = result['score']
    rsi = result.get('rsi', 50)
    details = result.get('details', {})

    if df.empty or len(df) < 14:
        return {
            'name': 'RSI',
            'signal': 'neutral',
            'score': 0,
            'weight': WEIGHTS['rsi'],
            'weighted_score': 0,
            'plain_explanation': '数据不足'
        }

    level = details.get('level', 'normal')
    trend = details.get('trend', 'flat')

    if level == 'oversold':
        if trend == 'rising':
            exp = f'RSI {rsi:.0f} 超卖后回升，可能反弹'
        else:
            exp = f'RSI {rsi:.0f} 处于超卖区，但尚未企稳'
    elif level == 'overbought':
        if trend == 'falling':
            exp = f'RSI {rsi:.0f} 超买后回落，可能调整'
        else:
            exp = f'RSI {rsi:.0f} 处于超买区，注意风险'
    else:
        exp = f'RSI {rsi:.0f} 处于正常区间'

    return {
        'name': 'RSI',
        'signal': result['signal'],
        'score': score,
        'weight': WEIGHTS['rsi'],
        'weighted_score': score * WEIGHTS['rsi'],
        'plain_explanation': exp
    }

def analyze_volume(df):
    """Analyze volume."""
    result = indicator_service.get_volume_signal(df)
    score = result['score']
    vol_ratio = result.get('volume_ratio', 1)
    details = result.get('details', {})

    if df.empty or len(df) < 20:
        return {
            'name': '成交量',
            'signal': 'neutral',
            'score': 0,
            'weight': WEIGHTS['volume'],
            'weighted_score': 0,
            'plain_explanation': '数据不足'
        }

    latest = df.iloc[-1]
    change_pct = latest.get('change_pct', 0)
    level = details.get('level', 'normal')

    if level == 'high':
        if change_pct > 0:
            exp = f'放量上涨（量比{vol_ratio:.1f}），资金流入'
        else:
            exp = f'放量下跌（量比{vol_ratio:.1f}），资金流出'
    elif level == 'low':
        exp = f'缩量（量比{vol_ratio:.1f}），观望情绪浓'
    else:
        exp = f'成交量正常（量比{vol_ratio:.1f}）'

    return {
        'name': '成交量',
        'signal': result['signal'],
        'score': score,
        'weight': WEIGHTS['volume'],
        'weighted_score': score * WEIGHTS['volume'],
        'plain_explanation': exp
    }

def analyze_capital_flow(stock_code):
    """Analyze capital flow."""
    flows = data_service.get_capital_flow(stock_code, days=5)

    if not flows:
        return {
            'name': '资金',
            'signal': 'neutral',
            'score': 0,
            'weight': WEIGHTS['capital_flow'],
            'weighted_score': 0,
            'plain_explanation': '暂无数据'
        }

    total = sum(f.get('main_net_inflow', 0) for f in flows)
    total_yi = total / 100000000

    if total > 100000000:
        score = 0.7
        exp = f'5日主力净流入{total_yi:.1f}亿'
    elif total > 30000000:
        score = 0.4
        exp = f'5日主力净流入{total/10000:.0f}万'
    elif total > 0:
        score = 0.15
        exp = f'5日主力小幅流入'
    elif total > -30000000:
        score = -0.15
        exp = f'5日主力小幅流出'
    elif total > -100000000:
        score = -0.4
        exp = f'5日主力净流出{abs(total)/10000:.0f}万'
    else:
        score = -0.7
        exp = f'5日主力净流出{abs(total_yi):.1f}亿'

    signal = 'bullish' if score > 0.1 else ('bearish' if score < -0.1 else 'neutral')

    return {
        'name': '资金',
        'signal': signal,
        'score': score,
        'weight': WEIGHTS['capital_flow'],
        'weighted_score': score * WEIGHTS['capital_flow'],
        'plain_explanation': exp
    }

def analyze_momentum(df):
    """Analyze price momentum - ADAPTIVE strategy based on volatility."""
    result = indicator_service.get_momentum_signal(df)
    score = result['score']
    details = result.get('details', {})

    if df.empty or len(df) < 20:
        return {
            'name': '动量',
            'signal': 'neutral',
            'score': 0,
            'weight': WEIGHTS['momentum'],
            'weighted_score': 0,
            'plain_explanation': '数据不足'
        }

    ret_5d = details.get('ret_5d', 0)
    strategy = details.get('strategy', 'neutral')
    volatility = details.get('volatility', 25)
    in_uptrend = details.get('in_uptrend', False)

    vol_desc = '高波动' if volatility > 30 else '低波动'

    if strategy == 'mean_reversion':
        if ret_5d < -5:
            exp = f'{vol_desc}股，近5日跌{abs(ret_5d):.1f}%，超跌反弹机会'
        else:
            exp = f'{vol_desc}股，近5日跌{abs(ret_5d):.1f}%，关注反弹'
    elif strategy == 'trend_follow':
        if in_uptrend:
            exp = f'{vol_desc}股，处于上升趋势，顺势看多'
        else:
            exp = f'{vol_desc}股，处于下降趋势，谨慎观望'
    elif strategy == 'momentum':
        exp = f'近5日涨{ret_5d:.1f}%，上涨动量持续'
    else:
        if ret_5d > 0:
            exp = f'近5日涨{ret_5d:.1f}%，走势平稳'
        else:
            exp = f'近5日跌{abs(ret_5d):.1f}%，走势偏弱'

    return {
        'name': '动量',
        'signal': result['signal'],
        'score': score,
        'weight': WEIGHTS['momentum'],
        'weighted_score': score * WEIGHTS['momentum'],
        'plain_explanation': exp
    }


def analyze_weekly_trend(df):
    """Analyze weekly trend for multi-timeframe confirmation."""
    result = indicator_service.get_weekly_trend(df)
    score = result['score']
    details = result.get('details', {})

    if df.empty or len(df) < 20:
        return {
            'name': '周线',
            'signal': 'neutral',
            'score': 0,
            'weight': WEIGHTS['weekly'],
            'weighted_score': 0,
            'plain_explanation': '数据不足'
        }

    weekly_change = details.get('weekly_change', 0)

    if score > 0.3:
        exp = f'周线趋势向上（周涨{weekly_change:.1f}%），中期看多'
    elif score > 0:
        exp = f'周线偏多（周涨{weekly_change:.1f}%）'
    elif score < -0.3:
        exp = f'周线趋势向下（周跌{abs(weekly_change):.1f}%），中期看空'
    elif score < 0:
        exp = f'周线偏空（周跌{abs(weekly_change):.1f}%）'
    else:
        exp = '周线方向不明'

    signal = 'bullish' if score > 0.1 else ('bearish' if score < -0.1 else 'neutral')

    return {
        'name': '周线',
        'signal': signal,
        'score': score,
        'weight': WEIGHTS['weekly'],
        'weighted_score': score * WEIGHTS['weekly'],
        'plain_explanation': exp
    }


def analyze_czsc(stock_code, df):
    """Analyze using CZSC (Chan Theory) indicators."""
    if not czsc_service.is_czsc_available():
        return None  # Return None when CZSC not available

    result = czsc_service.get_czsc_combined_signal(stock_code, df)

    if not result.get('available'):
        return {
            'name': '缠论',
            'signal': 'neutral',
            'score': 0,
            'weight': 0,  # Zero weight when unavailable
            'weighted_score': 0,
            'plain_explanation': result.get('error', '缠论分析不可用')
        }

    score = result.get('combined_score', 0)
    bi_count = result.get('bi_count', 0)
    bi_direction_cn = result.get('bi_direction_cn', '未知')
    buy_point = result.get('buy_point_type')
    sell_point = result.get('sell_point_type')

    # Build explanation
    if buy_point:
        exp = f'{buy_point}，当前笔{bi_direction_cn}，共{bi_count}笔'
    elif sell_point:
        exp = f'{sell_point}，当前笔{bi_direction_cn}，共{bi_count}笔'
    else:
        exp = f'当前笔{bi_direction_cn}，共{bi_count}笔'

    signal = 'bullish' if score > 0.1 else ('bearish' if score < -0.1 else 'neutral')

    return {
        'name': '缠论',
        'signal': signal,
        'score': score,
        'weight': WEIGHTS.get('czsc', 0),
        'weighted_score': score * WEIGHTS.get('czsc', 0),
        'plain_explanation': exp,
        'czsc_details': {
            'bi_count': bi_count,
            'bi_direction': result.get('bi_direction'),
            'buy_point_type': buy_point,
            'sell_point_type': sell_point,
            'analysis': result.get('analysis', [])
        }
    }

def analyze_trend_strength(df):
    """Analyze trend strength using ADX."""
    if df.empty or len(df) < 30 or 'adx' not in df.columns:
        return {
            'name': '趋势强度',
            'signal': 'neutral',
            'score': 0,
            'weight': 0.04,
            'weighted_score': 0,
            'plain_explanation': '数据不足'
        }

    latest = df.iloc[-1]
    adx = latest.get('adx', 25)
    plus_di = latest.get('plus_di', 50)
    minus_di = latest.get('minus_di', 50)

    if adx is None or pd.isna(adx):
        adx = 25

    # ADX interpretation
    # > 25: Strong trend
    # < 20: Weak/no trend
    score = 0

    if adx > 40:
        # Very strong trend - follow the direction
        if plus_di > minus_di:
            score = 0.6
            exp = f'ADX {adx:.0f}，强势上涨趋势'
        else:
            score = -0.6
            exp = f'ADX {adx:.0f}，强势下跌趋势'
    elif adx > 25:
        # Moderate trend
        if plus_di > minus_di:
            score = 0.3
            exp = f'ADX {adx:.0f}，上涨趋势中'
        else:
            score = -0.3
            exp = f'ADX {adx:.0f}，下跌趋势中'
    else:
        # Weak trend - mean reversion more likely
        score = 0
        exp = f'ADX {adx:.0f}，趋势不明显'

    signal = 'bullish' if score > 0.1 else ('bearish' if score < -0.1 else 'neutral')

    return {
        'name': '趋势强度',
        'signal': signal,
        'score': score,
        'weight': 0.04,
        'weighted_score': score * 0.04,
        'plain_explanation': exp
    }


def analyze_northbound(stock_code):
    """Analyze northbound flow signal."""
    try:
        nb_signal = market_service.get_northbound_signal()
        score = nb_signal.get('score', 0)
        details = nb_signal.get('details', {})

        total_5d = details.get('total_5d_yi', 0)

        if score > 0.3:
            exp = f'北向资金5日净流入{total_5d}亿，外资看多'
        elif score > 0:
            exp = f'北向资金小幅流入，外资偏多'
        elif score < -0.3:
            exp = f'北向资金5日净流出{abs(total_5d)}亿，外资看空'
        elif score < 0:
            exp = f'北向资金小幅流出，外资偏空'
        else:
            exp = '北向资金流向中性'

        signal = nb_signal.get('signal', 'neutral')

        return {
            'name': '北向资金',
            'signal': signal,
            'score': score,
            'weight': 0.10,
            'weighted_score': score * 0.10,
            'plain_explanation': exp
        }
    except Exception as e:
        return {
            'name': '北向资金',
            'signal': 'neutral',
            'score': 0,
            'weight': 0.10,
            'weighted_score': 0,
            'plain_explanation': '暂无数据'
        }


# =============================================================================
# NEW INDICATOR ANALYSIS FUNCTIONS (from InStock)
# =============================================================================

def analyze_supertrend(df):
    """Analyze Supertrend indicator - superior trend following."""
    result = indicator_service.get_supertrend_signal(df)
    score = result['score']
    details = result.get('details', {})

    if df.empty or 'supertrend_direction' not in df.columns:
        return {
            'name': '超级趋势',
            'signal': 'neutral',
            'score': 0,
            'weight': DEFAULT_WEIGHTS.get('supertrend', 0.08),
            'weighted_score': 0,
            'plain_explanation': '数据不足'
        }

    direction = details.get('direction', 'neutral')
    flipped = details.get('flipped', False)

    if flipped:
        if direction == 'bullish':
            exp = '超级趋势翻多，趋势反转向上'
        else:
            exp = '超级趋势翻空，趋势反转向下'
    else:
        if direction == 'bullish':
            exp = '超级趋势看多，保持上涨趋势'
        else:
            exp = '超级趋势看空，保持下跌趋势'

    return {
        'name': '超级趋势',
        'signal': result['signal'],
        'score': score,
        'weight': DEFAULT_WEIGHTS.get('supertrend', 0.08),
        'weighted_score': score * DEFAULT_WEIGHTS.get('supertrend', 0.08),
        'plain_explanation': exp
    }


def analyze_wave_trend(df):
    """Analyze Wave Trend indicator - advanced oscillator."""
    result = indicator_service.get_wave_trend_signal(df)
    score = result['score']
    details = result.get('details', {})

    if df.empty or 'wt1' not in df.columns:
        return {
            'name': '波浪趋势',
            'signal': 'neutral',
            'score': 0,
            'weight': DEFAULT_WEIGHTS.get('wave_trend', 0.05),
            'weighted_score': 0,
            'plain_explanation': '数据不足'
        }

    wt1 = details.get('wt1', 0)
    level = details.get('level', 'normal')

    if level == 'oversold':
        exp = f'WT={wt1:.0f}，处于超卖区，可能反弹'
    elif level == 'overbought':
        exp = f'WT={wt1:.0f}，处于超买区，注意回调'
    else:
        if score > 0.2:
            exp = f'WT={wt1:.0f}，金叉向上，短期看多'
        elif score < -0.2:
            exp = f'WT={wt1:.0f}，死叉向下，短期看空'
        else:
            exp = f'WT={wt1:.0f}，方向不明'

    return {
        'name': '波浪趋势',
        'signal': result['signal'],
        'score': score,
        'weight': DEFAULT_WEIGHTS.get('wave_trend', 0.05),
        'weighted_score': score * DEFAULT_WEIGHTS.get('wave_trend', 0.05),
        'plain_explanation': exp
    }


def analyze_vr(df):
    """Analyze VR (Volume Ratio) - buying vs selling volume."""
    result = indicator_service.get_vr_signal(df)
    score = result['score']
    details = result.get('details', {})

    if df.empty or 'vr' not in df.columns:
        return {
            'name': '量比',
            'signal': 'neutral',
            'score': 0,
            'weight': DEFAULT_WEIGHTS.get('vr', 0.04),
            'weighted_score': 0,
            'plain_explanation': '数据不足'
        }

    vr = details.get('vr', 100)
    level = details.get('level', 'normal')

    if level == 'accumulation':
        exp = f'VR={vr:.0f}，多方成交量占优，资金吸筹'
    elif level == 'distribution':
        exp = f'VR={vr:.0f}，空方成交量占优，资金派发'
    else:
        exp = f'VR={vr:.0f}，多空成交量平衡'

    return {
        'name': '量比',
        'signal': result['signal'],
        'score': score,
        'weight': DEFAULT_WEIGHTS.get('vr', 0.04),
        'weighted_score': score * DEFAULT_WEIGHTS.get('vr', 0.04),
        'plain_explanation': exp
    }


def analyze_cr(df):
    """Analyze CR (Energy Indicator) - buying vs selling pressure."""
    result = indicator_service.get_cr_signal(df)
    score = result['score']
    details = result.get('details', {})

    if df.empty or 'cr' not in df.columns:
        return {
            'name': '能量',
            'signal': 'neutral',
            'score': 0,
            'weight': DEFAULT_WEIGHTS.get('cr', 0.03),
            'weighted_score': 0,
            'plain_explanation': '数据不足'
        }

    cr = details.get('cr', 100)
    level = details.get('level', 'normal')

    if level == 'oversold':
        exp = f'CR={cr:.0f}，能量超卖，买入机会'
    elif level == 'overbought':
        exp = f'CR={cr:.0f}，能量超买，注意减仓'
    else:
        exp = f'CR={cr:.0f}，能量正常'

    return {
        'name': '能量',
        'signal': result['signal'],
        'score': score,
        'weight': DEFAULT_WEIGHTS.get('cr', 0.03),
        'weighted_score': score * DEFAULT_WEIGHTS.get('cr', 0.03),
        'plain_explanation': exp
    }


def analyze_cyq(df):
    """Analyze CYQ (Chip Distribution) - institutional analysis."""
    result = cyq_service.get_cyq_analysis(df)

    if not result.get('available', False):
        return {
            'name': '筹码',
            'signal': 'neutral',
            'score': 0,
            'weight': DEFAULT_WEIGHTS.get('cyq', 0.06),
            'weighted_score': 0,
            'plain_explanation': '筹码数据不足'
        }

    signal_data = result.get('signal', {})
    score = signal_data.get('score', 0)
    details = signal_data.get('details', {})

    benefit_part = details.get('benefit_part', 50)
    avg_cost = details.get('avg_cost', 0)

    if benefit_part < 20:
        exp = f'仅{benefit_part:.0f}%筹码盈利，超卖可能反弹'
    elif benefit_part < 40:
        exp = f'{benefit_part:.0f}%筹码盈利，平均成本{avg_cost:.2f}'
    elif benefit_part > 90:
        exp = f'{benefit_part:.0f}%筹码盈利，获利盘压力大'
    elif benefit_part > 80:
        exp = f'{benefit_part:.0f}%筹码盈利，注意减仓'
    else:
        exp = f'{benefit_part:.0f}%筹码盈利，平均成本{avg_cost:.2f}'

    signal = signal_data.get('signal', 'neutral')

    return {
        'name': '筹码',
        'signal': signal,
        'score': score,
        'weight': DEFAULT_WEIGHTS.get('cyq', 0.06),
        'weighted_score': score * DEFAULT_WEIGHTS.get('cyq', 0.06),
        'plain_explanation': exp,
        'cyq_details': details
    }


def analyze_strategies(df):
    """Analyze trading strategies - pattern-based signals."""
    result = strategy_service.get_strategy_signal(df)
    score = result.get('score', 0)
    triggered = result.get('triggered', [])
    triggered_count = result.get('triggered_count', 0)

    if triggered_count == 0:
        return {
            'name': '策略',
            'signal': 'neutral',
            'score': 0,
            'weight': DEFAULT_WEIGHTS.get('strategy', 0.06),
            'weighted_score': 0,
            'plain_explanation': '无策略触发'
        }

    # Build explanation from triggered strategies
    strategy_names = [s.get('name_cn', s.get('strategy', '')) for s in triggered[:3]]
    exp = f"触发{triggered_count}个策略：{'、'.join(strategy_names)}"

    signal = result.get('signal', 'neutral')

    return {
        'name': '策略',
        'signal': signal,
        'score': score,
        'weight': DEFAULT_WEIGHTS.get('strategy', 0.06),
        'weighted_score': score * DEFAULT_WEIGHTS.get('strategy', 0.06),
        'plain_explanation': exp,
        'triggered_strategies': triggered
    }


def generate_signal(stock_code):
    """Generate comprehensive signal using ensemble approach with market regime detection.

    Key improvements:
    1. Dynamic weights based on market regime (bull/bear/sideways)
    2. Northbound flow signal (smart money indicator)
    3. ADX trend strength signal
    4. Ensemble combining technical + market signals
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    df = data_service.get_stock_kline(stock_code)

    if df.empty:
        return {
            'stock_code': stock_code,
            'signal': 'hold',
            'signal_cn': '无数据',
            'score': 0,
            'confidence': 0,
            'indicators': [],
            'simple_summary': '无法获取数据',
            'stop_loss': None,
            'take_profit': None
        }

    df = indicator_service.calculate_all(df)

    # Save to database in background (don't block)
    from services import db_service
    import threading
    threading.Thread(target=db_service.save_daily_data, args=(stock_code, df), daemon=True).start()

    # Get market regime and adjusted weights
    weights, regime_info = get_current_weights()
    regime = regime_info.get('regime', 'unknown')

    # Analyze all technical indicators (core + new from InStock)
    indicators = [
        # Core indicators
        analyze_ma(df),
        analyze_macd(df),
        analyze_momentum(df),
        analyze_kdj(df),
        analyze_bollinger(df),
        analyze_rsi(df),
        analyze_volume(df),
        analyze_weekly_trend(df),
        analyze_trend_strength(df),
        # New indicators from InStock
        analyze_supertrend(df),      # Superior trend following
        analyze_wave_trend(df),      # Advanced oscillator
        analyze_vr(df),              # Volume ratio by direction
        analyze_cr(df),              # Energy indicator
        analyze_cyq(df),             # Chip distribution
        analyze_strategies(df),      # Trading strategies
    ]

    # Update weights based on regime and add indicator keys
    key_map = {
        # Core indicators
        '均线': 'ma', 'MACD': 'macd', '动量': 'momentum',
        'KDJ': 'kdj', '布林带': 'bollinger', 'RSI': 'rsi',
        '成交量': 'volume', '周线': 'weekly', '趋势强度': 'adx',
        # New indicators from InStock
        '超级趋势': 'supertrend',
        '波浪趋势': 'wave_trend',
        '量比': 'vr',
        '能量': 'cr',
        '筹码': 'cyq',
        '策略': 'strategy',
        # External signals
        '资金流向': 'capital_flow',
        '北向资金': 'northbound',
        '缠论': 'czsc',
    }

    for ind in indicators:
        name_key = ind['name']
        indicator_key = key_map.get(name_key, name_key.lower())

        # Add indicator_key if not present
        if 'indicator_key' not in ind:
            ind['indicator_key'] = indicator_key

        # Add education content
        edu = education_service.get_indicator_education(indicator_key)
        if edu:
            ind['education'] = {
                'what_it_measures': edu.get('what_it_measures', ''),
                'how_to_read': edu.get('how_to_read', []),
                'tips': edu.get('tips', []),
                'accuracy_note': edu.get('accuracy_note', ''),
            }

        # Update weights
        weight_key = key_map.get(name_key)
        if weight_key and weight_key in weights:
            ind['weight'] = weights[weight_key]
            ind['weighted_score'] = ind['score'] * weights[weight_key]

    # Fetch external signals in parallel
    capital_result = None
    czsc_result = None
    northbound_result = None

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(analyze_capital_flow, stock_code): 'capital',
            executor.submit(analyze_northbound, stock_code): 'northbound',
        }
        if czsc_service.is_czsc_available():
            futures[executor.submit(analyze_czsc, stock_code, df)] = 'czsc'

        for future in as_completed(futures, timeout=8):
            name = futures[future]
            try:
                result = future.result(timeout=5)
                if name == 'capital':
                    capital_result = result
                elif name == 'czsc':
                    czsc_result = result
                elif name == 'northbound':
                    northbound_result = result
            except Exception:
                pass

    # Update capital flow weight
    if capital_result:
        capital_result['weight'] = weights.get('capital_flow', 0.08)
        capital_result['weighted_score'] = capital_result['score'] * capital_result['weight']
        indicators.append(capital_result)

    # Add northbound signal
    if northbound_result:
        indicators.append(northbound_result)

    # Add CZSC with regime-adjusted weight
    if czsc_result:
        czsc_result['weight'] = weights.get('czsc', 0.12)
        czsc_result['weighted_score'] = czsc_result['score'] * czsc_result['weight']
        indicators.append(czsc_result)

    # Calculate ensemble score
    technical_score = sum(ind['weighted_score'] for ind in indicators)

    # Regime adjustment: boost/dampen signal based on market condition
    regime_multiplier = 1.0
    if regime == 'strong_bull' and technical_score > 0:
        regime_multiplier = 1.15  # Boost bullish signals in bull market
    elif regime == 'strong_bear' and technical_score < 0:
        regime_multiplier = 1.15  # Boost bearish signals in bear market
    elif regime == 'strong_bull' and technical_score < 0:
        regime_multiplier = 0.85  # Dampen bearish signals in bull market
    elif regime == 'strong_bear' and technical_score > 0:
        regime_multiplier = 0.85  # Dampen bullish signals in bear market

    total_score = technical_score * regime_multiplier

    # Get stop-loss suggestion
    stop_info = indicator_service.get_stop_loss_suggestion(df)

    # Determine signal with regime context
    # Thresholds: 0.15 for buy/sell (~54% accuracy), 0.35 for strong signals
    if total_score >= 0.35:
        signal, signal_cn = 'strong_buy', '强烈看多'
    elif total_score >= 0.15:
        signal, signal_cn = 'buy', '看多'
    elif total_score <= -0.35:
        signal, signal_cn = 'strong_sell', '强烈看空'
    elif total_score <= -0.15:
        signal, signal_cn = 'sell', '看空'
    else:
        signal, signal_cn = 'hold', '中性'

    bullish = [i['name'] for i in indicators if i['signal'] == 'bullish']
    bearish = [i['name'] for i in indicators if i['signal'] == 'bearish']

    # Generate summary with regime context
    regime_cn = {
        'strong_bull': '强势市场',
        'bull': '多头市场',
        'sideways': '震荡市场',
        'bear': '空头市场',
        'strong_bear': '弱势市场'
    }.get(regime, '')

    # Add comma separator only if regime_cn is not empty
    prefix = f"{regime_cn}，" if regime_cn else ""

    if total_score >= 0.35:
        summary = f"{prefix}{', '.join(bullish[:3])}看多，技术面强势"
    elif total_score >= 0.15:
        summary = f"{prefix}{', '.join(bullish[:3])}偏多"
    elif total_score <= -0.35:
        summary = f"{prefix}{', '.join(bearish[:3])}看空，技术面弱势"
    elif total_score <= -0.15:
        summary = f"{prefix}{', '.join(bearish[:3])}偏空"
    else:
        summary = f"{prefix}多空分歧"

    # Confidence based on signal agreement + regime confirmation
    agreement_pct = max(len(bullish), len(bearish)) / len(indicators) * 100
    regime_boost = 10 if (
        (regime in ['strong_bull', 'bull'] and total_score > 0) or
        (regime in ['strong_bear', 'bear'] and total_score < 0)
    ) else 0
    confidence = min(95, int(agreement_pct + regime_boost))

    # Generate education context for the overall signal
    signal_education = education_service.generate_signal_context(
        {'signal': signal, 'signal_cn': signal_cn, 'score': total_score, 'confidence': confidence, 'indicators': indicators},
        regime
    )

    # Get market regime education
    regime_education = education_service.get_market_regime_education(regime)

    return _to_native({
        'stock_code': stock_code,
        'signal': signal,
        'signal_cn': signal_cn,
        'score': round(total_score, 3),
        'confidence': confidence,
        'indicators': indicators,
        'simple_summary': summary,
        'stop_loss': stop_info.get('stop_loss'),
        'take_profit': stop_info.get('take_profit'),
        'risk_info': stop_info.get('details', {}),
        'market_regime': regime,
        'regime_details': regime_info.get('details', {}),
        'ml_prediction': None,
        'fundamental': None,
        # Education content
        'education': {
            'signal_context': signal_education,
            'regime_education': regime_education,
            'key_factors': signal_education.get('key_factors', []),
            'risk_factors': signal_education.get('risk_factors', []),
            'suggested_action': signal_education.get('suggested_action', ''),
        }
    })


def backtest_signal(stock_code, lookback_days=60):
    """Backtesting - check historical signal accuracy using full algorithm."""
    df = data_service.get_stock_kline(stock_code, days=lookback_days + 30)

    if df.empty or len(df) < lookback_days:
        return {'error': 'Insufficient data', 'signals': []}

    df = indicator_service.calculate_all(df)
    results = []

    # Generate signals for each day and check forward returns
    for i in range(30, len(df) - 5):  # Need 5 days forward
        day_df = df.iloc[:i+1].copy()

        # Use all indicators with proper weights
        ma = indicator_service.get_ma_trend(day_df)['score'] * WEIGHTS['ma']
        macd = indicator_service.get_macd_signal(day_df)['score'] * WEIGHTS['macd']
        momentum = indicator_service.get_momentum_signal(day_df)['score'] * WEIGHTS['momentum']
        kdj = indicator_service.get_kdj_signal(day_df)['score'] * WEIGHTS['kdj']
        boll = indicator_service.get_bollinger_signal(day_df)['score'] * WEIGHTS['bollinger']
        rsi = indicator_service.get_rsi_signal(day_df)['score'] * WEIGHTS['rsi']
        vol = indicator_service.get_volume_signal(day_df)['score'] * WEIGHTS['volume']
        weekly = indicator_service.get_weekly_trend(day_df)['score'] * WEIGHTS['weekly']

        score = ma + macd + momentum + kdj + boll + rsi + vol + weekly

        # Check forward returns
        entry_price = df.iloc[i]['close']
        future_prices = df.iloc[i+1:i+6]['close'].values

        if len(future_prices) >= 5:
            return_5d = (future_prices[-1] - entry_price) / entry_price * 100
            max_return = (max(future_prices) - entry_price) / entry_price * 100
            min_return = (min(future_prices) - entry_price) / entry_price * 100

            # Use same thresholds as generate_signal
            if score > 0.18 and return_5d > 0:
                correct = True
            elif score < -0.18 and return_5d < 0:
                correct = True
            elif -0.18 <= score <= 0.18:
                correct = None  # Neutral, not counted
            else:
                correct = False

            results.append({
                'date': df.iloc[i]['trade_date'],
                'score': round(score, 3),
                'signal': 'buy' if score > 0.18 else ('sell' if score < -0.18 else 'hold'),
                'return_5d': round(return_5d, 2),
                'max_return': round(max_return, 2),
                'min_return': round(min_return, 2),
                'correct': correct
            })

    # Calculate stats
    buy_signals = [r for r in results if r['signal'] == 'buy']
    sell_signals = [r for r in results if r['signal'] == 'sell']

    buy_correct = len([r for r in buy_signals if r['correct'] == True])
    sell_correct = len([r for r in sell_signals if r['correct'] == True])

    return {
        'stock_code': stock_code,
        'period': f'{lookback_days}天',
        'total_signals': len(buy_signals) + len(sell_signals),
        'buy_signals': len(buy_signals),
        'buy_accuracy': round(buy_correct / len(buy_signals) * 100, 1) if buy_signals else 0,
        'buy_avg_return': round(sum(r['return_5d'] for r in buy_signals) / len(buy_signals), 2) if buy_signals else 0,
        'sell_signals': len(sell_signals),
        'sell_accuracy': round(sell_correct / len(sell_signals) * 100, 1) if sell_signals else 0,
        'sell_avg_return': round(sum(r['return_5d'] for r in sell_signals) / len(sell_signals), 2) if sell_signals else 0,
        'recent_signals': results[-10:] if results else []
    }


def get_signal_color(signal):
    colors = {
        'strong_buy': '#d1242f',
        'buy': '#d1242f',
        'hold': '#666',
        'sell': '#1a7f37',
        'strong_sell': '#1a7f37'
    }
    return colors.get(signal, '#666')
