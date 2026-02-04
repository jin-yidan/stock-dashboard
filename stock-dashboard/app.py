from flask import Flask, render_template, request, jsonify, abort
from datetime import datetime
import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATABASE_PATH
from services import db_service, data_service, signal_service, indicator_service
from services import ml_service, fundamental_service, czsc_service, market_service

app = Flask(__name__)


def validate_stock_code(code):
    """Validate stock code format (6 digits only). Returns None if invalid."""
    if not code or not isinstance(code, str):
        return None
    # Stock codes are exactly 6 digits
    if not re.match(r'^\d{6}$', code):
        return None
    return code


def require_valid_code(code):
    """Validate stock code or abort with 400 error."""
    if not validate_stock_code(code):
        abort(400, description='Invalid stock code format')
    return code

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
    code = require_valid_code(code)
    stock_name = data_service.get_stock_name(code)
    return render_template(
        'stock_detail.html',
        stock_code=code,
        stock_name=stock_name or code
    )


@app.route('/api/stock/<code>/signal')
def api_signal(code):
    """API endpoint for signal data."""
    code = require_valid_code(code)
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
    code = require_valid_code(code)
    name = data_service.get_stock_name(code)
    return jsonify({'stock_code': code, 'short_name': name})


@app.route('/api/stock/<code>/preload')
def api_preload(code):
    """Preload stock data in background (call on hover)."""
    if not validate_stock_code(code):
        return jsonify({'status': 'invalid'})
    import threading

    def _preload():
        try:
            data_service.get_stock_kline(code)
        except Exception as e:
            print(f"Preload error for {code}: {e}")

    threading.Thread(target=_preload, daemon=True).start()
    return jsonify({'status': 'loading'})


@app.route('/api/stock/<code>/quote')
def api_quote(code):
    """Get realtime quote."""
    code = require_valid_code(code)
    quote = data_service.get_realtime_quote(code)
    if quote:
        return jsonify(quote)
    return jsonify({'error': 'No data'})


@app.route('/api/quotes/batch')
def api_quotes_batch():
    """
    Get quotes for multiple stocks at once.
    Usage: /api/quotes/batch?codes=600519,000858,002594
    Returns: { "quotes": { "600519": {...}, "000858": {...} } }
    """
    from concurrent.futures import ThreadPoolExecutor

    codes_param = request.args.get('codes', '')
    if not codes_param:
        return jsonify({'error': 'Provide codes via ?codes=600519,000858,...'})

    # Validate and limit codes (max 20 to prevent abuse)
    codes = [c.strip() for c in codes_param.split(',') if validate_stock_code(c.strip())][:20]
    if not codes:
        return jsonify({'error': 'No valid stock codes provided'})

    def get_quote_safe(code):
        try:
            quote = data_service.get_realtime_quote(code)
            return code, quote if quote else {'error': 'No data'}
        except Exception:
            return code, {'error': 'Failed to fetch'}

    quotes = {}
    with ThreadPoolExecutor(max_workers=min(len(codes), 10)) as executor:
        futures = {executor.submit(get_quote_safe, code): code for code in codes}
        for future in futures:
            try:
                code, quote = future.result(timeout=8)
                quotes[code] = quote
            except Exception:
                quotes[futures[future]] = {'error': 'Timeout'}

    return jsonify({'quotes': quotes})


@app.route('/api/stock/<code>/backtest')
def api_backtest(code):
    """Get backtest results for signal accuracy."""
    code = require_valid_code(code)
    days = min(request.args.get('days', 60, type=int), 365)  # Cap at 1 year
    result = signal_service.backtest_signal(code, lookback_days=days)
    return jsonify(result)


@app.route('/api/stock/<code>/ml/train')
def api_ml_train(code):
    """Train ML model for a specific stock."""
    code = require_valid_code(code)
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
    code = require_valid_code(code)
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
    code = require_valid_code(code)
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
    code = require_valid_code(code)
    import math
    days = min(request.args.get('days', 60, type=int), 500)  # Cap at 500 days
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

        # Use change_pct to determine color (based on daily change vs previous close)
        change_pct = row.get('change_pct', 0)
        if change_pct is None or (isinstance(change_pct, float) and math.isnan(change_pct)):
            change_pct = 0
        is_up = change_pct >= 0

        # K-line with color based on daily change_pct (not close vs open)
        kline_data.append({
            'value': [
                safe_round(row['open']),
                safe_round(row['close']),
                safe_round(row['low']),
                safe_round(row['high'])
            ],
            'itemStyle': {
                'color': '#ef5350' if is_up else '#26a69a',
                'color0': '#ef5350' if is_up else '#26a69a',
                'borderColor': '#ef5350' if is_up else '#26a69a',
                'borderColor0': '#ef5350' if is_up else '#26a69a'
            }
        })

        # MAs - handle NaN values
        ma5_val = row.get('ma5')
        ma10_val = row.get('ma10')
        ma20_val = row.get('ma20')
        ma5_data.append(safe_round(ma5_val))
        ma10_data.append(safe_round(ma10_val))
        ma20_data.append(safe_round(ma20_val))

        # Volume with color based on daily change_pct
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
    code = require_valid_code(code)
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
    code = require_valid_code(code)
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
    code = require_valid_code(code)
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
    code = require_valid_code(code)
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
    code = require_valid_code(code)
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
    code = require_valid_code(code)
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
    code = require_valid_code(code)
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



# Feature 3: 52-week High/Low API
@app.route('/api/stock/<code>/52week')
def api_52week(code):
    """Get 52-week high/low data."""
    code = require_valid_code(code)
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


# AI Analysis with Web Search
@app.route('/api/stock/<code>/ai-analysis', methods=['POST'])
def api_ai_analysis(code):
    """
    Generate AI-powered analysis using user's API key.

    Combines:
    - Our technical analysis data (indicators, CYQ, strategies)
    - LLM analysis for comprehensive insights
    - Investment advice

    Request body:
    {
        "api_key": "..."  // User's API key (OpenAI, Claude, etc.)
    }
    """
    code = require_valid_code(code)
    from services import ai_service, cyq_service, strategy_service

    # Get API key from request
    data = request.get_json()
    if not data or not data.get('api_key'):
        return jsonify({'success': False, 'error': '请提供 API Key'})

    api_key = data['api_key']

    # Get stock name
    stock_name = data_service.get_stock_name(code)
    if not stock_name:
        stock_name = code

    # Gather all our existing data
    try:
        # 1. Main signal data (includes all indicators)
        signal_data = signal_service.generate_signal(code)

        # 2. CYQ chip distribution
        df = data_service.get_stock_kline(code, days=300)
        cyq_data = cyq_service.get_cyq_analysis(df) if not df.empty else None

        # 3. Trading strategies
        strategies_data = strategy_service.run_all_strategies(df) if not df.empty else {'triggered': []}

        # 4. 52-week data
        df_52w = data_service.get_stock_kline(code, days=250)
        week52_data = None
        if not df_52w.empty:
            quote = data_service.get_realtime_quote(code)
            current = quote.get('price', df_52w.iloc[-1]['close']) if quote else df_52w.iloc[-1]['close']
            high_52w = df_52w['high'].max()
            low_52w = df_52w['low'].min()
            week52_data = {
                'current': round(current, 2),
                'high_52w': round(high_52w, 2),
                'low_52w': round(low_52w, 2),
                'pct_from_high': round((current / high_52w - 1) * 100, 1),
                'pct_from_low': round((current / low_52w - 1) * 100, 1),
                'position': round((current - low_52w) / (high_52w - low_52w) * 100, 1) if high_52w != low_52w else 50
            }

        # Generate AI analysis
        result = ai_service.generate_ai_analysis(
            api_key=api_key,
            stock_code=code,
            stock_name=stock_name,
            signal_data=signal_data,
            cyq_data=cyq_data,
            strategies_data=strategies_data,
            week52_data=week52_data
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': f'分析过程出错: {str(e)}'})


# Feature 4: Price Alerts
@app.route('/alerts')
def alerts_page():
    """Price alerts management page."""
    return render_template('alerts.html')


# =============================================================================
# NEW FEATURES: Market Status, Cache, Patterns, Sector Rotation, Correlation
# =============================================================================

@app.route('/api/market/status')
def api_market_status():
    """Get current market status (trading hours, holidays)."""
    from services import market_status_service
    return jsonify(market_status_service.get_market_status())


@app.route('/api/cache/stats')
def api_cache_stats():
    """Get cache statistics."""
    from services.cache_service import cache
    return jsonify(cache.get_stats())


@app.route('/api/stock/<code>/patterns')
def api_candlestick_patterns(code):
    """Get candlestick pattern analysis for a stock."""
    code = require_valid_code(code)
    df = data_service.get_stock_kline(code, days=60)
    if df.empty:
        return jsonify({'error': 'No data'})
    df = indicator_service.detect_candlestick_patterns(df)
    result = indicator_service.get_candlestick_patterns(df)
    result['stock_code'] = code
    return jsonify(result)


@app.route('/api/stock/<code>/data-freshness')
def api_data_freshness(code):
    """Get data freshness information for a stock."""
    code = require_valid_code(code)
    from services import market_status_service
    df = data_service.get_stock_kline(code, days=5)
    if df.empty:
        return jsonify({'fresh': False, 'status': 'no_data', 'status_cn': '无数据'})
    latest_date = df.iloc[-1]['trade_date']
    freshness = market_status_service.get_data_freshness(latest_date)
    freshness['last_data_date'] = str(latest_date)[:10]
    freshness['last_trading_date'] = market_status_service.get_last_trading_date()
    return jsonify(freshness)


@app.route('/api/sector/rotation')
def api_sector_rotation():
    """Get sector rotation analysis - which sectors are gaining/losing momentum."""
    from concurrent.futures import ThreadPoolExecutor

    sector_representatives = {
        '白酒': ['600519', '000858'],
        '银行': ['601398', '600036'],
        '新能源': ['300750', '002594'],
        '医药': ['600276', '000538'],
        '科技': ['002415', '300059'],
        '地产': ['000002', '001979'],
        '消费': ['000651', '000333'],
        '证券': ['600030', '601688'],
    }

    def get_sector_performance(sector, codes):
        total_change = 0
        count = 0
        for code in codes:
            try:
                df = data_service.get_stock_kline(code, days=30)
                if not df.empty and len(df) >= 20:
                    ret_20d = (df.iloc[-1]['close'] / df.iloc[-20]['close'] - 1) * 100
                    total_change += ret_20d
                    count += 1
            except Exception:
                pass
        return {'sector': sector, 'change_20d': round(total_change / count, 2) if count > 0 else 0}

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(get_sector_performance, s, c): s for s, c in sector_representatives.items()}
        results = []
        for future in futures:
            try:
                results.append(future.result(timeout=15))
            except Exception:
                pass

    results.sort(key=lambda x: x['change_20d'], reverse=True)
    return jsonify({'sectors': results, 'leaders': results[:3], 'laggards': results[-3:]})


@app.route('/api/watchlist/correlation')
def api_watchlist_correlation():
    """Calculate correlation matrix for watchlist stocks."""
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    codes_param = request.args.get('codes', '')
    if not codes_param:
        return jsonify({'error': 'Provide codes via ?codes=600519,000858,...'})

    # Validate and filter stock codes
    codes = [c.strip() for c in codes_param.split(',') if validate_stock_code(c.strip())][:10]
    if len(codes) < 2:
        return jsonify({'error': 'Need at least 2 valid stock codes'})

    def get_returns(code):
        df = data_service.get_stock_kline(code, days=120)
        if df.empty or len(df) < 60:
            return None
        returns = df['close'].pct_change().dropna().values
        return {'code': code, 'name': data_service.get_stock_name(code), 'returns': returns}

    with ThreadPoolExecutor(max_workers=len(codes)) as executor:
        futures = {executor.submit(get_returns, code): code for code in codes}
        stock_data = {}
        for future in futures:
            try:
                result = future.result(timeout=10)
                if result:
                    stock_data[futures[future]] = result
            except Exception:
                pass

    if len(stock_data) < 2:
        return jsonify({'error': 'Not enough data'})

    valid_codes = list(stock_data.keys())
    min_len = min(len(stock_data[c]['returns']) for c in valid_codes)
    returns_matrix = np.array([stock_data[c]['returns'][-min_len:] for c in valid_codes])
    corr_matrix = np.corrcoef(returns_matrix)

    matrix = [[round(corr_matrix[i, j], 3) for j in range(len(valid_codes))] for i in range(len(valid_codes))]
    stocks_info = [{'code': c, 'name': stock_data[c]['name']} for c in valid_codes]

    high_corr = []
    for i in range(len(valid_codes)):
        for j in range(i + 1, len(valid_codes)):
            if abs(corr_matrix[i, j]) > 0.7:
                high_corr.append({'stock1': stocks_info[i], 'stock2': stocks_info[j], 'correlation': round(corr_matrix[i, j], 3)})

    return jsonify({'stocks': stocks_info, 'matrix': matrix, 'high_correlations': high_corr, 'period_days': min_len})


@app.route('/api/stock/<code>/historical-accuracy')
def api_historical_accuracy(code):
    """Get historical signal accuracy for a stock."""
    code = require_valid_code(code)
    result = signal_service.backtest_signal(code, lookback_days=60)
    if result.get('error'):
        return jsonify(result)

    # Calculate overall accuracy from buy and sell accuracy
    buy_signals = result.get('buy_signals', 0)
    sell_signals = result.get('sell_signals', 0)
    total = buy_signals + sell_signals

    if total > 0:
        # Weighted average of buy and sell accuracy
        buy_acc = result.get('buy_accuracy', 0)
        sell_acc = result.get('sell_accuracy', 0)
        accuracy = (buy_acc * buy_signals + sell_acc * sell_signals) / total
    else:
        accuracy = 0

    return jsonify({
        'stock_code': code,
        'accuracy': round(accuracy, 1),
        'total_signals': total,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'buy_accuracy': result.get('buy_accuracy', 0),
        'sell_accuracy': result.get('sell_accuracy', 0),
        'period': result.get('period', ''),
        'reliability': 'high' if accuracy > 55 else ('moderate' if accuracy > 45 else 'low')
    })


if __name__ == '__main__':
    print("Stock Analysis App")
    print("Open http://localhost:8080")
    app.run(debug=True, host='0.0.0.0', port=8080)
