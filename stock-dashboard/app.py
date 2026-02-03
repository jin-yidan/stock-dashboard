from flask import Flask, render_template, request, jsonify
from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATABASE_PATH
from services import db_service, data_service, signal_service, indicator_service
from services import ml_service, fundamental_service, czsc_service

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


if __name__ == '__main__':
    print("Stock Analysis App")
    print("Open http://localhost:8080")
    app.run(debug=True, host='0.0.0.0', port=8080)
