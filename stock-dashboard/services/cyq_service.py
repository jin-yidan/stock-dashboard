"""
CYQ (Chip Distribution / 筹码分布) Service.

This is a sophisticated cost distribution algorithm that calculates
where most shareholders acquired their shares. This is institutional-level
analysis rarely found in retail trading platforms.

Key Outputs:
- benefit_part: % of chips profitable at current price
- avg_cost: Average cost of all shareholders (50th percentile)
- percent_chips: Price ranges containing 70% and 90% of chips
- concentration: How spread out the chips are

Use Cases:
- benefit_part > 90%: Most holders profitable, resistance likely
- benefit_part < 20%: Most underwater, potential capitulation
- Low concentration: Chips spread out, breakout possible
"""

import numpy as np
import pandas as pd


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


class CYQCalculator:
    """
    Chip Distribution Calculator.

    Calculates the distribution of cost basis among shareholders
    based on historical volume and price data.

    Args:
        kdata: DataFrame with columns: date, open, close, high, low, volume, turnover
               (turnover is turnover rate as percentage, e.g., 5.2 for 5.2%)
        accuracy_factor: Number of price buckets (default 150)
        calc_range: Number of recent days to analyze (default 120)
        cyq_days: Trading days to consider for chip distribution (default 210)
    """

    def __init__(self, kdata, accuracy_factor=150, calc_range=120, cyq_days=210):
        self.klinedata = kdata.copy()
        self.factor = accuracy_factor
        self.range = calc_range
        self.tradingdays = cyq_days

    def calc(self, index=None):
        """
        Calculate chip distribution for a given index.

        Args:
            index: K-line index to calculate for. If None, uses the latest.

        Returns:
            CYQData object with distribution results
        """
        if index is None:
            index = len(self.klinedata)

        # Determine data range
        end = index - self.range + 1 if index >= self.range else 0
        start = max(0, end - self.tradingdays)

        if end <= 0:
            kdata = self.klinedata.tail(min(self.tradingdays, len(self.klinedata)))
        else:
            kdata = self.klinedata.iloc[start:end]

        if len(kdata) < 10:
            return None

        # Find price range
        max_price = kdata['high'].max()
        min_price = kdata['low'].min()

        if max_price <= min_price:
            return None

        # Accuracy (price step)
        accuracy = max(0.01, (max_price - min_price) / (self.factor - 1))
        current_price = kdata.iloc[-1]['close']

        # Build price range (y-axis)
        y_range = []
        boundary = -1

        for i in range(self.factor):
            price = round(min_price + accuracy * i, 2)
            y_range.append(price)
            if boundary == -1 and price >= current_price:
                boundary = i

        # Initialize chip distribution (x-axis)
        x_data = [0.0] * self.factor

        # Process each candle
        for _, row in kdata.iterrows():
            open_price = row['open']
            close = row['close']
            high = row['high']
            low = row['low']

            # Get turnover rate (normalize to 0-1)
            turnover = row.get('turnover', 1.0)
            if pd.isna(turnover) or turnover <= 0:
                turnover = 1.0
            turnover_rate = min(1.0, turnover / 100)

            # Average price for the candle
            avg = (open_price + close + high + low) / 4

            # Price level indices
            h_idx = int((high - min_price) / accuracy)
            l_idx = int((low - min_price) / accuracy + 0.99)
            h_idx = min(h_idx, self.factor - 1)
            l_idx = max(l_idx, 0)

            # G-point (peak of triangular distribution)
            if high == low:
                g_height = self.factor - 1
            else:
                g_height = 2 / (high - low)
            g_idx = int((avg - min_price) / accuracy)
            g_idx = max(0, min(g_idx, self.factor - 1))

            # Decay existing chips based on turnover
            for n in range(len(x_data)):
                x_data[n] *= (1 - turnover_rate)

            # Distribute new chips using triangular distribution
            if high == low:
                # Limit-up/down: all chips at one price
                x_data[g_idx] += g_height * turnover_rate / 2
            else:
                for j in range(l_idx, min(h_idx + 1, self.factor)):
                    cur_price = min_price + accuracy * j

                    if cur_price <= avg:
                        # Lower triangle
                        if abs(avg - low) < 1e-8:
                            weight = g_height
                        else:
                            weight = (cur_price - low) / (avg - low) * g_height
                    else:
                        # Upper triangle
                        if abs(high - avg) < 1e-8:
                            weight = g_height
                        else:
                            weight = (high - cur_price) / (high - avg) * g_height

                    x_data[j] += weight * turnover_rate

        # Total chips for normalization
        total_chips = sum(x_data)

        if total_chips <= 0:
            return None

        # Helper function: get cost at given percentile
        def get_cost_by_chip(chip_target):
            """Find price level at given chip accumulation."""
            result = min_price
            sum_chips = 0
            for i in range(self.factor):
                if sum_chips + x_data[i] > chip_target:
                    result = min_price + i * accuracy
                    break
                sum_chips += x_data[i]
            return result

        # Helper function: get benefit ratio at price
        def get_benefit_part(price):
            """Calculate % of chips profitable at given price."""
            below = 0
            for i in range(self.factor):
                if price >= min_price + i * accuracy:
                    below += x_data[i]
            return below / total_chips if total_chips > 0 else 0

        # Calculate percentile chip ranges
        def compute_percent_chips(percent):
            """Calculate price range containing given % of chips."""
            if percent > 1 or percent < 0:
                return None
            ps = [(1 - percent) / 2, (1 + percent) / 2]
            pr = [
                get_cost_by_chip(total_chips * ps[0]),
                get_cost_by_chip(total_chips * ps[1])
            ]
            concentration = 0 if (pr[0] + pr[1]) == 0 else (pr[1] - pr[0]) / (pr[0] + pr[1])
            return {
                'price_range': [round(pr[0], 2), round(pr[1], 2)],
                'concentration': round(concentration, 4)
            }

        # Build result
        result = CYQData()
        result.x = x_data
        result.y = y_range
        result.boundary = boundary + 1
        result.date = kdata.iloc[-1].get('trade_date', kdata.iloc[-1].get('date', ''))
        result.trading_days = len(kdata)
        result.benefit_part = round(get_benefit_part(current_price), 4)
        result.avg_cost = round(get_cost_by_chip(total_chips * 0.5), 2)
        result.percent_chips = {
            '90': compute_percent_chips(0.9),
            '70': compute_percent_chips(0.7)
        }
        result.current_price = current_price

        return result


class CYQData:
    """Container for CYQ calculation results."""

    def __init__(self):
        self.x = None               # Chip distribution (horizontal axis)
        self.y = None               # Price levels (vertical axis)
        self.benefit_part = None    # % of chips profitable
        self.avg_cost = None        # Average cost (50th percentile)
        self.percent_chips = None   # Price ranges for 70%/90% chips
        self.boundary = None        # Index separating profit/loss
        self.date = None            # Date of calculation
        self.trading_days = None    # Number of trading days analyzed
        self.current_price = None   # Current price

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'benefit_part': self.benefit_part,
            'avg_cost': self.avg_cost,
            'percent_chips': self.percent_chips,
            'boundary': self.boundary,
            'date': self.date,
            'trading_days': self.trading_days,
            'current_price': self.current_price,
            # Don't include x/y arrays - too large for API response
        }


def calculate_cyq(df, accuracy_factor=150, calc_range=120, cyq_days=210):
    """
    Calculate CYQ chip distribution for a stock.

    Args:
        df: DataFrame with OHLCV and turnover data
        accuracy_factor: Number of price buckets
        calc_range: Recent days to analyze
        cyq_days: Total trading days for distribution

    Returns:
        CYQData object or None if insufficient data
    """
    if df is None or df.empty or len(df) < 30:
        return None

    # Ensure required columns exist
    required = ['open', 'close', 'high', 'low', 'volume']
    if not all(col in df.columns for col in required):
        return None

    # Add turnover if not present (estimate from volume)
    if 'turnover' not in df.columns:
        # Rough estimate: assume 1% turnover per day as default
        df = df.copy()
        df['turnover'] = 1.0

    calculator = CYQCalculator(
        df,
        accuracy_factor=accuracy_factor,
        calc_range=min(calc_range, len(df)),
        cyq_days=min(cyq_days, len(df))
    )

    return calculator.calc()


def get_cyq_signal(cyq_data):
    """
    Generate trading signal from CYQ data.

    Args:
        cyq_data: CYQData object from calculate_cyq

    Returns:
        Signal dictionary with score and details
    """
    if cyq_data is None:
        return {'signal': 'neutral', 'score': 0, 'details': {}}

    benefit_part = cyq_data.benefit_part
    avg_cost = cyq_data.avg_cost
    current_price = cyq_data.current_price

    score = 0

    # Benefit ratio analysis
    if benefit_part < 0.1:
        # Less than 10% profitable - extreme capitulation potential
        score = 0.7
    elif benefit_part < 0.2:
        # Less than 20% profitable - oversold
        score = 0.5
    elif benefit_part < 0.3:
        # Less than 30% profitable - mild oversold
        score = 0.3
    elif benefit_part > 0.95:
        # Over 95% profitable - extreme resistance
        score = -0.5
    elif benefit_part > 0.90:
        # Over 90% profitable - strong resistance
        score = -0.3
    elif benefit_part > 0.80:
        # Over 80% profitable - some resistance
        score = -0.15

    # Price vs average cost
    if avg_cost and current_price:
        price_vs_cost = (current_price - avg_cost) / avg_cost
        if price_vs_cost < -0.15:
            score += 0.2  # Price well below avg cost
        elif price_vs_cost > 0.20:
            score -= 0.15  # Price well above avg cost

    # Concentration analysis
    if cyq_data.percent_chips:
        conc_70 = cyq_data.percent_chips.get('70', {}).get('concentration', 0.5)
        if conc_70 < 0.08:
            # Very concentrated chips - potential breakout
            score += 0.1
        elif conc_70 > 0.20:
            # Widely spread chips - choppy action likely
            score *= 0.8

    score = max(-1, min(1, score))
    signal = 'bullish' if score > 0.1 else ('bearish' if score < -0.1 else 'neutral')

    return {
        'signal': signal,
        'score': score,
        'details': {
            'benefit_part': round(benefit_part * 100, 1) if benefit_part else 0,
            'avg_cost': avg_cost,
            'current_price': current_price,
            'percent_chips': cyq_data.percent_chips
        }
    }


def get_cyq_analysis(df):
    """
    Complete CYQ analysis with signal generation.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        Dictionary with CYQ data and signal
    """
    cyq_data = calculate_cyq(df)

    if cyq_data is None:
        return {
            'available': False,
            'error': 'Insufficient data for CYQ calculation',
            'signal': get_cyq_signal(None)
        }

    signal = get_cyq_signal(cyq_data)

    return {
        'available': True,
        'data': _to_native(cyq_data.to_dict()),
        'signal': _to_native(signal)
    }
