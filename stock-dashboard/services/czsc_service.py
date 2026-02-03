# -*- coding: utf-8 -*-
"""
CZSC (Chan Theory) Integration Service

Provides Chan Theory technical analysis as an optional enhancement layer.
All functions gracefully handle cases where CZSC is not installed.
"""
import pandas as pd
from datetime import datetime

# Try to import CZSC library
CZSC_AVAILABLE = False
CZSC_VERSION = None

try:
    from czsc.core import CZSC, RawBar, Freq, Direction
    CZSC_AVAILABLE = True
    try:
        import czsc
        CZSC_VERSION = getattr(czsc, '__version__', 'unknown')
    except:
        CZSC_VERSION = 'unknown'
except ImportError:
    pass


def is_czsc_available():
    """Check if CZSC library is installed and available."""
    return CZSC_AVAILABLE


def get_czsc_version():
    """Get CZSC library version."""
    return CZSC_VERSION if CZSC_AVAILABLE else None


def convert_df_to_rawbars(df, symbol, freq=Freq.D if CZSC_AVAILABLE else None):
    """
    Convert pandas DataFrame to List[RawBar] for CZSC analysis.

    :param df: DataFrame with columns: trade_date, open, close, high, low, volume, amount
    :param symbol: Stock symbol/code
    :param freq: Frequency (default: Daily)
    :return: List of RawBar objects
    """
    if not CZSC_AVAILABLE:
        return []

    if df.empty:
        return []

    bars = []
    for i, row in df.iterrows():
        try:
            # Handle date conversion
            dt = row.get('trade_date')
            if isinstance(dt, str):
                dt = pd.to_datetime(dt)
            elif not isinstance(dt, datetime):
                dt = pd.to_datetime(str(dt))

            bar = RawBar(
                symbol=symbol,
                id=i if isinstance(i, int) else len(bars),
                dt=dt,
                freq=freq if freq else Freq.D,
                open=float(row['open']),
                close=float(row['close']),
                high=float(row['high']),
                low=float(row['low']),
                vol=float(row.get('volume', 0) or 0),
                amount=float(row.get('amount', 0) or 0)
            )
            bars.append(bar)
        except Exception as e:
            continue

    return bars


class CzscAnalyzer:
    """
    CZSC Analyzer class wrapping Chan Theory analysis functionality.
    """

    def __init__(self, stock_code, df):
        """
        Initialize CZSC analyzer with stock data.

        :param stock_code: Stock code (e.g., '600519')
        :param df: DataFrame with OHLCV data
        """
        self.stock_code = stock_code
        self.df = df
        self.czsc = None
        self.error = None

        if not CZSC_AVAILABLE:
            self.error = 'CZSC library not installed'
            return

        if df is None or df.empty:
            self.error = 'No data provided'
            return

        if len(df) < 100:
            self.error = f'Insufficient data: {len(df)} bars (need at least 100)'
            return

        try:
            bars = convert_df_to_rawbars(df, stock_code)
            if len(bars) < 100:
                self.error = f'Insufficient valid bars: {len(bars)}'
                return
            self.czsc = CZSC(bars)
        except Exception as e:
            self.error = str(e)

    def get_chan_analysis(self):
        """
        Get Chan Theory analysis: fractals (分型), strokes (笔), and current direction.

        :return: dict with analysis results
        """
        if not self.czsc:
            return {
                'has_data': False,
                'error': self.error or 'CZSC not initialized'
            }

        try:
            bi_list = self.czsc.bi_list

            result = {
                'has_data': True,
                'bi_count': len(bi_list),
                'bi_list': [],
                'current_bi_direction': None,
                'current_bi_change': None,
                'fx_count': 0
            }

            # Extract last 10 strokes info
            for bi in bi_list[-10:]:
                bi_info = {
                    'direction': bi.direction.value if hasattr(bi.direction, 'value') else str(bi.direction),
                    'start_dt': bi.sdt.strftime('%Y-%m-%d') if bi.sdt else None,
                    'end_dt': bi.edt.strftime('%Y-%m-%d') if bi.edt else None,
                    'high': round(bi.high, 2),
                    'low': round(bi.low, 2),
                    'change': round(bi.change * 100, 2) if hasattr(bi, 'change') else None
                }
                result['bi_list'].append(bi_info)

            # Current stroke direction
            if bi_list:
                last_bi = bi_list[-1]
                direction = last_bi.direction
                if hasattr(direction, 'value'):
                    result['current_bi_direction'] = direction.value
                else:
                    result['current_bi_direction'] = str(direction)

                if hasattr(last_bi, 'change'):
                    result['current_bi_change'] = round(last_bi.change * 100, 2)

            # Count fractals from bars_ubi
            if hasattr(self.czsc, 'bars_ubi'):
                result['ubi_count'] = len(self.czsc.bars_ubi)

            return result

        except Exception as e:
            return {
                'has_data': False,
                'error': str(e)
            }

    def get_technical_signals(self):
        """
        Get technical signals from CZSC analysis.
        Note: CZSC signal functions require ta-lib which may not be installed.

        :return: dict with technical signals
        """
        if not self.czsc:
            return {
                'has_data': False,
                'error': self.error or 'CZSC not initialized'
            }

        signals = {
            'has_data': True,
            'signals': {}
        }

        try:
            # Try to get MACD signal from CZSC
            from czsc.signals import tas as czsc_tas

            # These require ta-lib, so we wrap in try-catch
            try:
                czsc_tas.update_macd_cache(self.czsc)
                last_bar = self.czsc.bars_raw[-1]
                if last_bar.cache:
                    macd_key = 'MACD12#26#9'
                    if macd_key in last_bar.cache:
                        macd_data = last_bar.cache[macd_key]
                        signals['signals']['macd'] = {
                            'dif': round(macd_data.get('dif', 0), 4),
                            'dea': round(macd_data.get('dea', 0), 4),
                            'macd': round(macd_data.get('macd', 0), 4)
                        }
            except:
                pass

            # Try MA signals
            try:
                for period in [5, 10, 20]:
                    czsc_tas.update_ma_cache(self.czsc, timeperiod=period, ma_type='SMA')
                    last_bar = self.czsc.bars_raw[-1]
                    if last_bar.cache:
                        ma_key = f'SMA#{period}'
                        if ma_key in last_bar.cache:
                            signals['signals'][f'ma{period}'] = round(last_bar.cache[ma_key], 2)
            except:
                pass

        except ImportError:
            signals['ta_lib_available'] = False
        except Exception as e:
            signals['error'] = str(e)

        return signals

    def get_buy_sell_points(self):
        """
        Get Chan Theory buy/sell points (买卖点) based on stroke patterns.

        Buy signals:
        - Bottom fractal after down stroke (第一类买点)
        - Pullback above previous low (第二类买点)

        Sell signals:
        - Top fractal after up stroke (第一类卖点)
        - Rally below previous high (第二类卖点)

        :return: dict with buy/sell point analysis
        """
        if not self.czsc:
            return {
                'has_data': False,
                'error': self.error or 'CZSC not initialized'
            }

        try:
            bi_list = self.czsc.bi_list

            result = {
                'has_data': True,
                'current_signal': 'neutral',
                'signal_strength': 0,
                'buy_point_type': None,
                'sell_point_type': None,
                'analysis': []
            }

            if len(bi_list) < 3:
                result['analysis'].append('笔数量不足，无法判断买卖点')
                return result

            # Analyze last few strokes
            last_bi = bi_list[-1]
            prev_bi = bi_list[-2]
            prev_prev_bi = bi_list[-3] if len(bi_list) >= 3 else None

            direction = last_bi.direction
            is_up = (direction == Direction.Up if CZSC_AVAILABLE else str(direction) == 'Up')

            # Check for buy signals
            if not is_up:  # Last stroke is down
                # Type 1 buy point: current low is higher than previous down stroke's low
                if prev_prev_bi and last_bi.low > prev_prev_bi.low:
                    result['current_signal'] = 'bullish'
                    result['buy_point_type'] = '二类买点'
                    result['signal_strength'] = 0.6
                    result['analysis'].append(f'当前下跌笔低点({last_bi.low:.2f})高于前低({prev_prev_bi.low:.2f})，二类买点特征')

                # Check if the down stroke is weakening
                if hasattr(last_bi, 'power_price') and hasattr(prev_bi, 'power_price'):
                    if last_bi.power_price < prev_bi.power_price * 0.8:
                        result['signal_strength'] = min(result['signal_strength'] + 0.2, 1.0)
                        result['analysis'].append('下跌力度减弱')

            else:  # Last stroke is up
                # Check for sell signals
                # Type 1 sell point: current high is lower than previous up stroke's high
                if prev_prev_bi and last_bi.high < prev_prev_bi.high:
                    result['current_signal'] = 'bearish'
                    result['sell_point_type'] = '二类卖点'
                    result['signal_strength'] = -0.6
                    result['analysis'].append(f'当前上涨笔高点({last_bi.high:.2f})低于前高({prev_prev_bi.high:.2f})，二类卖点特征')

                # Check if the up stroke is weakening
                if hasattr(last_bi, 'power_price') and hasattr(prev_bi, 'power_price'):
                    if last_bi.power_price < prev_bi.power_price * 0.8:
                        result['signal_strength'] = min(result['signal_strength'] - 0.2, -1.0)
                        result['analysis'].append('上涨力度减弱')

            # Add stroke count info
            result['analysis'].append(f'共识别{len(bi_list)}笔')

            return result

        except Exception as e:
            return {
                'has_data': False,
                'error': str(e)
            }


def get_czsc_combined_signal(stock_code, df):
    """
    Get combined CZSC signal for dashboard integration.

    :param stock_code: Stock code
    :param df: DataFrame with OHLCV data
    :return: dict with combined signal data
    """
    if not CZSC_AVAILABLE:
        return {
            'available': False,
            'error': 'CZSC library not installed'
        }

    analyzer = CzscAnalyzer(stock_code, df)

    chan_analysis = analyzer.get_chan_analysis()
    buy_sell = analyzer.get_buy_sell_points()

    if not chan_analysis.get('has_data') or not buy_sell.get('has_data'):
        return {
            'available': False,
            'error': chan_analysis.get('error') or buy_sell.get('error') or 'Analysis failed'
        }

    # Combine signals into a score
    combined_score = 0

    # Use buy/sell point signal
    if buy_sell.get('current_signal') == 'bullish':
        combined_score = buy_sell.get('signal_strength', 0.3)
    elif buy_sell.get('current_signal') == 'bearish':
        combined_score = buy_sell.get('signal_strength', -0.3)

    # Adjust based on current stroke direction
    bi_direction = chan_analysis.get('current_bi_direction')
    # Handle both English and Chinese direction values
    is_up = bi_direction in ('Up', '向上')
    is_down = bi_direction in ('Down', '向下')

    if is_up:
        combined_score += 0.1
    elif is_down:
        combined_score -= 0.1

    # Clamp score to [-1, 1]
    combined_score = max(-1, min(1, combined_score))

    # Translate direction to Chinese if needed
    if is_up:
        bi_direction_cn = '向上'
    elif is_down:
        bi_direction_cn = '向下'
    else:
        bi_direction_cn = bi_direction or '未知'

    return {
        'available': True,
        'combined_score': round(combined_score, 3),
        'bi_count': chan_analysis.get('bi_count', 0),
        'bi_direction': bi_direction,
        'bi_direction_cn': bi_direction_cn,
        'signal': buy_sell.get('current_signal', 'neutral'),
        'buy_point_type': buy_sell.get('buy_point_type'),
        'sell_point_type': buy_sell.get('sell_point_type'),
        'analysis': buy_sell.get('analysis', []),
        'version': CZSC_VERSION
    }


def get_czsc_status():
    """
    Get CZSC integration status.

    :return: dict with status info
    """
    status = {
        'available': CZSC_AVAILABLE,
        'version': CZSC_VERSION
    }

    if CZSC_AVAILABLE:
        # Check if ta-lib is available
        try:
            import talib
            status['talib_available'] = True
        except ImportError:
            status['talib_available'] = False
            status['talib_note'] = 'ta-lib not installed, some CZSC signals unavailable'

    return status
