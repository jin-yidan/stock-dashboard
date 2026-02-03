from flask import Flask, render_template, request, jsonify
from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATABASE_PATH
from services import db_service, data_service, signal_service, indicator_service
from services import ml_service, fundamental_service, czsc_service, market_service

app = Flask(__name__)

os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
db_service.init_db()
data_service.init_stock_list()  # Load stock list in background


@app.route('/')
def index():
    """Home page with stock input."""
    hot_stocks = data_service.get_hot_stocks_fast()
    return render_template('index.html', hot_stocks=hot_stocks)


@app.route('/stock/<code>')
def stock_detail(code):
    """Stock analysis page - loads data via AJAX for faster initial render."""
    stock_name = data_service.get_stock_name(code)
    return render_template(
        'stock_detail.html',
        stock_code=code,
        stock_name=stock_name
    )


@app.route('/api/stock/<code>/signal')
def api_signal(code):
    """API endpoint for signal data."""
    signal_data = signal_service.generate_signal(code)
    return jsonify(signal_data)


@app.route('/api/search')
def api_search():
    """Search stocks by code or name."""
    query = request.args.get('q', '').strip()
    if len(query) < 1:
        return jsonify([])
    results = data_service.search_stocks(query, limit=8)
    return jsonify(results)


@app.route('/api/stock/<code>/info')
def api_stock_info(code):
    """Get basic stock info quickly."""
    name = data_service.get_stock_name(code)
    return jsonify({'stock_code': code, 'short_name': name})


@app.route('/api/stock/<code>/preload')
def api_preload(code):
    """Preload stock data in background (call on hover)."""
    import threading

    def _preload():
        try:
            data_service.get_stock_kline(code)
        except Exception:
            pass

    threading.Thread(target=_preload, daemon=True).start()
    return jsonify({'status': 'loading'})


@app.route('/api/stock/<code>/quote')
def api_quote(code):
    """Get realtime quote."""
    quote = data_service.get_realtime_quote(code)
    if quote:
        return jsonify(quote)
    return jsonify({'error': 'No data'})


@app.route('/api/stock/<code>/backtest')
def api_backtest(code):
    """Get backtest results for signal accuracy."""
    days = request.args.get('days', 60, type=int)
    result = signal_service.backtest_signal(code, lookback_days=days)
    return jsonify(result)


@app.route('/api/stock/<code>/ml/train')
def api_ml_train(code):
    """Train ML model for a specific stock."""
    df = data_service.get_stock_kline(code, days=500)  # More data for training
    if df.empty:
        return jsonify({'error': 'No data'})

    df = indicator_service.calculate_all(df)
    model, scaler, stats = ml_service.train_model(code, df, threshold=1.0)  # Lower threshold

    if model is None:
        return jsonify({'error': stats})

    return jsonify({
        'status': 'success',
        'stock_code': code,
        'accuracy': stats['cv_accuracy'],
        'accuracy_std': stats['cv_std'],
        'samples': stats['samples'],
        'positive_rate': stats['positive_rate'],
        'top_features': [{'name': f[0], 'importance': round(f[1], 3)} for f in stats['top_features']]
    })


@app.route('/api/stock/<code>/ml/predict')
def api_ml_predict(code):
    """Get ML prediction for a stock."""
    df = data_service.get_stock_kline(code, days=200)  # Need more data for features
    if df.empty:
        return jsonify({'error': 'No data'})

    df = indicator_service.calculate_all(df)

    # Try stock-specific model first, then general model
    result, error = ml_service.predict(code, df, threshold=1.0)
    if error:
        result, error = ml_service.predict_with_general_model(df, threshold=1.0)

    if error:
        return jsonify({'error': error})

    return jsonify(result)


@app.route('/api/ml/train-general')
def api_ml_train_general():
    """Train a general ML model using multiple stocks."""
    stocks = ['600519', '000858', '002594', '300750', '601318', '600036', '000001', '000333']

    stocks_data = {}
    for code in stocks:
        df = data_service.get_stock_kline(code, days=200)
        if not df.empty:
            df = indicator_service.calculate_all(df)
            stocks_data[code] = df

    if not stocks_data:
        return jsonify({'error': 'No data'})

    model, scaler, stats = ml_service.train_general_model(stocks_data)

    if model is None:
        return jsonify({'error': stats})

    return jsonify({
        'status': 'success',
        'accuracy': stats['cv_accuracy'],
        'accuracy_std': stats['cv_std'],
        'samples': stats['samples'],
        'stocks_used': stats['stocks_used'],
        'positive_rate': stats['positive_rate'],
        'top_features': [{'name': f[0], 'importance': round(f[1], 3)} for f in stats['top_features']]
    })


@app.route('/api/stock/<code>/fundamental')
def api_fundamental(code):
    """Get fundamental data for a stock."""
    quote = data_service.get_realtime_quote(code)
    price = quote.get('price', 0) if quote else 0

    summary = fundamental_service.get_fundamental_summary(code, price)
    score = fundamental_service.get_fundamental_score(code, price)

    return jsonify({
        **summary,
        'fundamental_score': score['score'],
        'score_reasons': score['reasons']
    })


def safe_round(val, decimals=2):
    """Safely round a value, return None if NaN."""
    import math
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    return round(val, decimals)


@app.route('/api/stock/<code>/kline')
def api_kline(code):
    """Get K-line data for charting."""
    import math
    days = request.args.get('days', 60, type=int)
    df = data_service.get_stock_kline(code, days=days)

    if df.empty:
        return jsonify({'error': 'No data'})

    df = indicator_service.calculate_all(df)

    # Format data for ECharts
    kline_data = []
    ma5_data = []
    ma10_data = []
    ma20_data = []
    volume_data = []
    dates = []

    for _, row in df.iterrows():
        date_str = str(row.get('trade_date', ''))[:10]
        dates.append(date_str)

        # K-line: [open, close, low, high]
        kline_data.append([
            safe_round(row['open']),
            safe_round(row['close']),
            safe_round(row['low']),
            safe_round(row['high'])
        ])

        # MAs - handle NaN values
        ma5_val = row.get('ma5')
        ma10_val = row.get('ma10')
        ma20_val = row.get('ma20')
        ma5_data.append(safe_round(ma5_val))
        ma10_data.append(safe_round(ma10_val))
        ma20_data.append(safe_round(ma20_val))

        # Volume with color
        is_up = row['close'] >= row['open']
        vol = row.get('volume', 0)
        vol = int(vol) if vol and not math.isnan(vol) else 0
        volume_data.append({
            'value': vol,
            'itemStyle': {'color': '#ef5350' if is_up else '#26a69a'}
        })

    return jsonify({
        'dates': dates,
        'kline': kline_data,
        'ma5': ma5_data,
        'ma10': ma10_data,
        'ma20': ma20_data,
        'volume': volume_data
    })


@app.route('/api/ml/info')
def api_ml_info():
    """Get ML model information."""
    info = ml_service.get_model_info()
    return jsonify(info)


# CZSC (Chan Theory) API Endpoints
@app.route('/api/czsc/status')
def api_czsc_status():
    """Check CZSC availability and version."""
    return jsonify(czsc_service.get_czsc_status())


@app.route('/api/stock/<code>/czsc')
def api_czsc_combined(code):
    """Get combined CZSC analysis for a stock."""
    if not czsc_service.is_czsc_available():
        return jsonify({
            'available': False,
            'error': 'CZSC library not installed'
        })

    df = data_service.get_stock_kline(code, days=200)
    if df.empty:
        return jsonify({
            'available': False,
            'error': 'No stock data available'
        })

    df = indicator_service.calculate_all(df)
    result = czsc_service.get_czsc_combined_signal(code, df)
    return jsonify(result)


@app.route('/api/stock/<code>/czsc/chan')
def api_czsc_chan(code):
    """Get detailed Chan analysis (strokes, fractals)."""
    if not czsc_service.is_czsc_available():
        return jsonify({
            'has_data': False,
            'error': 'CZSC library not installed'
        })

    df = data_service.get_stock_kline(code, days=200)
    if df.empty:
        return jsonify({
            'has_data': False,
            'error': 'No stock data available'
        })

    analyzer = czsc_service.CzscAnalyzer(code, df)
    result = analyzer.get_chan_analysis()
    return jsonify(result)


@app.route('/api/stock/<code>/czsc/signals')
def api_czsc_signals(code):
    """Get CZSC technical signals."""
    if not czsc_service.is_czsc_available():
        return jsonify({
            'has_data': False,
            'error': 'CZSC library not installed'
        })

    df = data_service.get_stock_kline(code, days=200)
    if df.empty:
        return jsonify({
            'has_data': False,
            'error': 'No stock data available'
        })

    analyzer = czsc_service.CzscAnalyzer(code, df)
    result = analyzer.get_technical_signals()
    return jsonify(result)


# ============== NEW FEATURES ==============

# Feature 1: Stock Comparison Page
@app.route('/compare/<code1>/<code2>')
def compare_stocks(code1, code2):
    """Compare two stocks side by side."""
    name1 = data_service.get_stock_name(code1)
    name2 = data_service.get_stock_name(code2)
    return render_template(
        'compare.html',
        stock1={'code': code1, 'name': name1},
        stock2={'code': code2, 'name': name2}
    )


@app.route('/api/compare/<code1>/<code2>')
def api_compare(code1, code2):
    """API for comparison data."""
    from concurrent.futures import ThreadPoolExecutor

    def get_stock_data(code):
        df = data_service.get_stock_kline(code, days=250)
        if df.empty:
            return None
        df = indicator_service.calculate_all(df)
        quote = data_service.get_realtime_quote(code)

        # Calculate 52-week high/low
        high_52w = df['high'].max()
        low_52w = df['low'].min()
        current = quote.get('price', df.iloc[-1]['close']) if quote else df.iloc[-1]['close']

        # Performance metrics
        if len(df) >= 5:
            ret_5d = (df.iloc[-1]['close'] / df.iloc[-5]['close'] - 1) * 100
        else:
            ret_5d = 0
        if len(df) >= 20:
            ret_20d = (df.iloc[-1]['close'] / df.iloc[-20]['close'] - 1) * 100
        else:
            ret_20d = 0
        if len(df) >= 60:
            ret_60d = (df.iloc[-1]['close'] / df.iloc[-60]['close'] - 1) * 100
        else:
            ret_60d = 0

        return {
            'code': code,
            'name': data_service.get_stock_name(code),
            'price': current,
            'change_pct': quote.get('change_pct', 0) if quote else 0,
            'high_52w': round(high_52w, 2),
            'low_52w': round(low_52w, 2),
            'pct_from_high': round((current / high_52w - 1) * 100, 1),
            'pct_from_low': round((current / low_52w - 1) * 100, 1),
            'ret_5d': round(ret_5d, 2),
            'ret_20d': round(ret_20d, 2),
            'ret_60d': round(ret_60d, 2),
            'volume': quote.get('volume', 0) if quote else 0,
            'rsi': round(df.iloc[-1].get('rsi', 50), 1) if 'rsi' in df.columns else 50,
        }

    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(get_stock_data, code1)
        future2 = executor.submit(get_stock_data, code2)
        data1 = future1.result(timeout=15)
        data2 = future2.result(timeout=15)

    if not data1 or not data2:
        return jsonify({'error': 'Failed to fetch data'})

    return jsonify({'stock1': data1, 'stock2': data2})


# Market Analysis APIs
@app.route('/api/market/regime')
def api_market_regime():
    """Get current market regime (bull/bear/sideways)."""
    regime = market_service.detect_market_regime()
    return jsonify(regime)


@app.route('/api/market/overview')
def api_market_overview():
    """Get comprehensive market overview."""
    overview = market_service.get_market_overview()
    return jsonify(overview)


@app.route('/api/market/northbound')
def api_northbound():
    """Get northbound capital flow signal."""
    signal = market_service.get_northbound_signal()
    return jsonify(signal)


@app.route('/api/market/breadth')
def api_market_breadth():
    """Get market breadth (advance/decline)."""
    breadth = market_service.get_market_breadth()
    return jsonify(breadth)


@app.route('/api/stock/<code>/margin')
def api_margin(code):
    """Get margin trading signal for a stock."""
    signal = market_service.get_margin_signal(code)
    return jsonify(signal)


# =============================================================================
# NEW ENDPOINTS: CYQ Chip Distribution & Trading Strategies
# =============================================================================

@app.route('/api/stock/<code>/cyq')
def api_cyq(code):
    """
    Get CYQ (Chip Distribution / 筹码分布) analysis for a stock.

    Returns:
    - benefit_part: % of chips profitable at current price
    - avg_cost: Average cost of all shareholders
    - percent_chips: Price ranges containing 70%/90% of chips
    - signal: Trading signal based on chip distribution
    """
    from services import cyq_service
    df = data_service.get_stock_kline(code, days=250)
    if df.empty:
        return jsonify({'error': 'No data', 'available': False})

    result = cyq_service.get_cyq_analysis(df)
    return jsonify(result)


@app.route('/api/stock/<code>/strategies')
def api_strategies(code):
    """
    Run all trading strategies on a stock.

    Strategies include:
    - backtrace_ma250: Pullback after breaking 250-day MA
    - platform_breakthrough: Breakout from consolidation
    - climax_limitdown: Panic selling reversal
    - low_backtrace_increase: Quality uptrend with controlled drawdowns
    - keep_increasing: Sustained MA progression
    - turtle_trade: Breakout above 60-day high
    - low_atr: Low volatility steady growth
    """
    from services import strategy_service
    df = data_service.get_stock_kline(code, days=300)
    if df.empty:
        return jsonify({'error': 'No data'})

    result = strategy_service.run_all_strategies(df)
    result['stock_code'] = code
    return jsonify(result)


@app.route('/api/stock/<code>/indicators/new')
def api_new_indicators(code):
    """
    Get all new indicator values from InStock integration.

    Includes:
    - supertrend: Trend direction and levels
    - wave_trend: WT1 and WT2 values
    - vr: Volume ratio
    - cr: Energy indicator
    - psy: Psychology line
    - brar: AR and BR sentiment
    - bias: MA deviation
    - trix: Triple EMA momentum
    - mfi: Money flow index
    - vhf: Trend vs ranging filter
    """
    df = data_service.get_stock_kline(code)
    if df.empty:
        return jsonify({'error': 'No data'})

    df = indicator_service.calculate_all(df)
    latest = df.iloc[-1]

    # Extract new indicators
    indicators = {}

    # Supertrend
    if 'supertrend' in df.columns:
        indicators['supertrend'] = {
            'value': round(latest.get('supertrend', 0), 2),
            'upper_band': round(latest.get('supertrend_ub', 0), 2),
            'lower_band': round(latest.get('supertrend_lb', 0), 2),
            'direction': 'bullish' if latest.get('supertrend_direction', 0) == 1 else 'bearish',
            'signal': indicator_service.get_supertrend_signal(df)
        }

    # Wave Trend
    if 'wt1' in df.columns:
        indicators['wave_trend'] = {
            'wt1': round(latest.get('wt1', 0), 2),
            'wt2': round(latest.get('wt2', 0), 2),
            'signal': indicator_service.get_wave_trend_signal(df)
        }

    # Volume Ratio
    if 'vr' in df.columns:
        indicators['vr'] = {
            'value': round(latest.get('vr', 100), 2),
            'ma': round(latest.get('vr_ma', 100), 2),
            'signal': indicator_service.get_vr_signal(df)
        }

    # CR Energy
    if 'cr' in df.columns:
        indicators['cr'] = {
            'value': round(latest.get('cr', 100), 2),
            'ma1': round(latest.get('cr_ma1', 100), 2),
            'ma2': round(latest.get('cr_ma2', 100), 2),
            'signal': indicator_service.get_cr_signal(df)
        }

    # PSY Psychology
    if 'psy' in df.columns:
        indicators['psy'] = {
            'value': round(latest.get('psy', 50), 2),
            'ma': round(latest.get('psy_ma', 50), 2),
            'signal': indicator_service.get_psy_signal(df)
        }

    # BRAR Sentiment
    if 'ar' in df.columns:
        indicators['brar'] = {
            'ar': round(latest.get('ar', 100), 2),
            'br': round(latest.get('br', 100), 2),
            'signal': indicator_service.get_brar_signal(df)
        }

    # BIAS Deviation
    if 'bias_6' in df.columns:
        indicators['bias'] = {
            'bias_6': round(latest.get('bias_6', 0), 2),
            'bias_12': round(latest.get('bias_12', 0), 2),
            'bias_24': round(latest.get('bias_24', 0), 2),
            'signal': indicator_service.get_bias_signal(df)
        }

    # TRIX
    if 'trix' in df.columns:
        indicators['trix'] = {
            'value': round(latest.get('trix', 0), 4),
            'signal_line': round(latest.get('trix_signal', 0), 4),
            'signal': indicator_service.get_trix_signal(df)
        }

    # MFI Money Flow
    if 'mfi' in df.columns:
        indicators['mfi'] = {
            'value': round(latest.get('mfi', 50), 2),
            'signal': indicator_service.get_mfi_signal(df)
        }

    # VHF Trend Filter
    if 'vhf' in df.columns:
        indicators['vhf'] = {
            'value': round(latest.get('vhf', 0.35), 4),
            'signal': indicator_service.get_vhf_signal(df)
        }

    # Force Index
    if 'force_13' in df.columns:
        indicators['force'] = {
            'force_2': round(latest.get('force_2', 0), 0),
            'force_13': round(latest.get('force_13', 0), 0),
            'signal': indicator_service.get_force_index_signal(df)
        }

    # RVI
    if 'rvi' in df.columns:
        indicators['rvi'] = {
            'value': round(latest.get('rvi', 0), 4),
            'signal_line': round(latest.get('rvi_signal', 0), 4),
            'signal': indicator_service.get_rvi_signal(df)
        }

    return jsonify({
        'stock_code': code,
        'indicators': indicators,
        'count': len(indicators)
    })


# Feature 2: Sector Heat Map
@app.route('/heatmap')
def sector_heatmap():
    """Sector heat map page."""
    return render_template('heatmap.html')


@app.route('/api/heatmap')
def api_heatmap():
    """API for sector heat map data."""
    # Representative stocks for each sector
    sectors = {
        '白酒': ['600519', '000858', '000568'],
        '银行': ['601398', '600036', '000001'],
        '新能源': ['300750', '002594', '600438'],
        '医药': ['600276', '000538', '300760'],
        '科技': ['002415', '000725', '603986'],
        '地产': ['000002', '001979', '600048'],
        '消费': ['000333', '000651', '600887'],
        '证券': ['600030', '601688', '000776'],
    }

    result = []
    for sector_name, codes in sectors.items():
        sector_data = {'name': sector_name, 'stocks': [], 'avg_change': 0}
        total_change = 0
        count = 0

        for code in codes:
            try:
                quote = data_service.get_realtime_quote(code)
                if quote and 'price' in quote:
                    stock_name = data_service.get_stock_name(code)
                    change = quote.get('change_pct', 0)
                    sector_data['stocks'].append({
                        'code': code,
                        'name': stock_name,
                        'price': quote['price'],
                        'change_pct': change
                    })
                    total_change += change
                    count += 1
            except Exception:
                pass

        if count > 0:
            sector_data['avg_change'] = round(total_change / count, 2)
        result.append(sector_data)

    # Sort by average change
    result.sort(key=lambda x: x['avg_change'], reverse=True)
    return jsonify(result)


# Feature 3: 52-week High/Low API
@app.route('/api/stock/<code>/52week')
def api_52week(code):
    """Get 52-week high/low data."""
    df = data_service.get_stock_kline(code, days=250)
    if df.empty:
        return jsonify({'error': 'No data'})

    quote = data_service.get_realtime_quote(code)
    current = quote.get('price', df.iloc[-1]['close']) if quote else df.iloc[-1]['close']

    high_52w = df['high'].max()
    low_52w = df['low'].min()
    high_date = df.loc[df['high'].idxmax(), 'trade_date']
    low_date = df.loc[df['low'].idxmin(), 'trade_date']

    return jsonify({
        'current': round(current, 2),
        'high_52w': round(high_52w, 2),
        'low_52w': round(low_52w, 2),
        'high_date': str(high_date)[:10],
        'low_date': str(low_date)[:10],
        'pct_from_high': round((current / high_52w - 1) * 100, 1),
        'pct_from_low': round((current / low_52w - 1) * 100, 1),
        'position': round((current - low_52w) / (high_52w - low_52w) * 100, 1) if high_52w != low_52w else 50
    })


# Feature 4: Price Alerts
@app.route('/alerts')
def alerts_page():
    """Price alerts management page."""
    return render_template('alerts.html')


if __name__ == '__main__':
    print("Stock Analysis App")
    print("Open http://localhost:8080")
    app.run(debug=True, host='0.0.0.0', port=8080)
