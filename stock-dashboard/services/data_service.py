import pandas as pd
from datetime import datetime, timedelta
from config import DEFAULT_KLINE_DAYS
from services import db_service
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Try to import adata - may not be available on Vercel
try:
    import adata
    HAS_ADATA = True
except ImportError:
    HAS_ADATA = False
    print("Warning: adata not available, using fallback data")

# Simple in-memory cache
_cache = {}
_cache_expiry = {}
CACHE_TTL = 1800  # 30 minutes - kline data doesn't change frequently
QUOTE_TTL = 60    # 1 minute for realtime-ish quotes
NAME_TTL = 86400  # 24 hours for stock names
API_TIMEOUT_SECONDS = 30  # Allow more time for slow external APIs

def _get_cached(key):
    """Get cached value if not expired."""
    if key in _cache and key in _cache_expiry:
        if datetime.now() <= _cache_expiry[key]:
            return _cache[key]
        # Expired - cleanup
        _cache.pop(key, None)
        _cache_expiry.pop(key, None)
    return None

def _set_cached(key, value, ttl=CACHE_TTL):
    """Set cache value with TTL."""
    _cache[key] = value
    _cache_expiry[key] = datetime.now() + timedelta(seconds=ttl)

def _call_with_timeout(fn, timeout_seconds, *args, **kwargs):
    """Call a function with timeout. Returns (result, error_str)."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds), None
        except FuturesTimeoutError:
            return None, f"timeout after {timeout_seconds}s"
        except Exception as e:
            return None, str(e)

def _get_kline_from_tencent(stock_code, days=120):
    """Fallback: fetch kline data from Tencent API (more reliable for some stocks)."""
    import requests
    import json
    try:
        # Determine market prefix: sz for 0/3, sh for 6
        prefix = 'sh' if stock_code.startswith('6') else 'sz'
        url = 'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get'
        params = {
            '_var': 'kline_dayqfq',
            'param': f'{prefix}{stock_code},day,,,{days},qfq'
        }
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}

        r = requests.get(url, params=params, headers=headers, timeout=30)
        if r.status_code != 200:
            return pd.DataFrame()

        text = r.text
        if 'kline_dayqfq=' not in text:
            return pd.DataFrame()

        json_str = text.split('kline_dayqfq=')[1]
        data = json.loads(json_str)

        stock_key = f'{prefix}{stock_code}'
        if 'data' not in data or stock_key not in data['data']:
            return pd.DataFrame()

        stock_data = data['data'][stock_key]
        klines = stock_data.get('qfqday', stock_data.get('day', []))

        if not klines:
            return pd.DataFrame()

        # Parse kline data: [date, open, close, high, low, volume]
        rows = []
        for k in klines:
            if len(k) >= 6:
                rows.append({
                    'trade_date': k[0],
                    'open': float(k[1]),
                    'close': float(k[2]),
                    'high': float(k[3]),
                    'low': float(k[4]),
                    'volume': float(k[5]),
                    'amount': 0,  # Tencent API doesn't provide amount
                    'change_pct': 0  # Will be calculated later if needed
                })

        df = pd.DataFrame(rows)
        # Calculate change_pct
        if len(df) > 1:
            df['change_pct'] = df['close'].pct_change() * 100
            df['change_pct'] = df['change_pct'].fillna(0)

        return df
    except Exception as e:
        print(f"Tencent API error for {stock_code}: {e}")
        return pd.DataFrame()

def _get_kline_from_db(stock_code, days):
    """Fallback: load kline data from local DB cache."""
    rows = db_service.get_daily_data(stock_code, days=days)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Normalize types
    df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce').dt.strftime('%Y-%m-%d')
    for col in ['open', 'close', 'high', 'low', 'volume', 'amount', 'change_pct']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # DB returns latest first; sort ascending for indicator calculations
    df = df.sort_values('trade_date').reset_index(drop=True)
    return df

def get_all_stocks():
    """Get list of all A-share stocks."""
    if not HAS_ADATA:
        return [{'stock_code': c[0], 'short_name': c[1], 'exchange': ''} for c in POPULAR_STOCKS]
    try:
        df = adata.stock.info.all_code()
        stocks = []
        for _, row in df.iterrows():
            stock = {
                'stock_code': row.get('stock_code', ''),
                'short_name': row.get('short_name', ''),
                'exchange': row.get('exchange', '')
            }
            stocks.append(stock)
            # Cache to database
            db_service.save_stock_info(
                stock['stock_code'],
                stock['short_name'],
                stock['exchange']
            )
        return stocks
    except Exception as e:
        print(f"Error fetching all stocks: {e}")
        return []

def get_stock_kline(stock_code, days=DEFAULT_KLINE_DAYS):
    """Get historical K-line data for a stock using East Money API."""
    if not HAS_ADATA:
        return _get_kline_from_db(stock_code, days)

    cache_key = f"kline_{stock_code}_{days}"
    cached = _get_cached(cache_key)
    if cached is not None and not cached.empty:
        return cached

    try:
        # Try baidu_market first, fall back to east_market if empty or timeout
        df, err = _call_with_timeout(
            adata.stock.market.baidu_market.get_market,
            API_TIMEOUT_SECONDS,
            stock_code=stock_code
        )
        if df is None or df.empty:
            df, err = _call_with_timeout(
                adata.stock.market.east_market.get_market,
                API_TIMEOUT_SECONDS,
                stock_code=stock_code
            )

        # Third fallback: Tencent API (more reliable for some stocks)
        if df is None or df.empty:
            df = _get_kline_from_tencent(stock_code, days=max(days, 120))

        if df is None or df.empty:
            return _get_kline_from_db(stock_code, days)

        # Convert columns to proper types
        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
        df['open'] = pd.to_numeric(df['open'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['high'] = pd.to_numeric(df['high'], errors='coerce')
        df['low'] = pd.to_numeric(df['low'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df['change_pct'] = pd.to_numeric(df['change_pct'], errors='coerce')

        # Sort by date and get recent days
        df = df.sort_values('trade_date').reset_index(drop=True)
        # Save full data to DB cache before trimming
        db_service.save_daily_data(stock_code, df)
        df = df.tail(days)

        _set_cached(cache_key, df)
        return df

    except Exception as e:
        print(f"Error fetching kline for {stock_code}: {e}")
        return _get_kline_from_db(stock_code, days)

def get_realtime_quote(stock_code):
    """Get realtime quote for a stock.

    Tries realtime API first, falls back to latest kline close if unavailable.
    """
    cache_key = f"quote_{stock_code}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    # Try realtime API first (if available)
    quote = _get_quote_from_realtime(stock_code)
    if quote:
        _set_cached(cache_key, quote, ttl=QUOTE_TTL)
        return quote

    # Fallback to latest kline close (not realtime)
    quote = _get_quote_from_kline(stock_code)
    if quote:
        _set_cached(cache_key, quote, ttl=QUOTE_TTL)
    return quote

def _get_quote_from_realtime(stock_code):
    """Try to get realtime quote from adata list_market_current."""
    if not HAS_ADATA:
        return None
    try:
        df, err = _call_with_timeout(
            adata.stock.market.list_market_current,
            API_TIMEOUT_SECONDS,
            code_list=[stock_code]
        )
        if df is None or df.empty:
            return None
        row = df.iloc[0]
        result = {
            'stock_code': stock_code,
            'price': _safe_float(row.get('price', 0)),
            'change_pct': _safe_float(row.get('change_pct', 0)),
            'open': _safe_float(row.get('open', 0)),
            'high': _safe_float(row.get('high', 0)),
            'low': _safe_float(row.get('low', 0)),
            'volume': _safe_int(row.get('volume', 0)),
            'amount': _safe_float(row.get('amount', 0)),
            'pre_close': _safe_float(row.get('pre_close', 0)),
            'is_realtime': True,
            'data_source': 'realtime'
        }
        return result
    except Exception as e:
        print(f"Error getting realtime quote for {stock_code}: {e}")
        return None

def _get_quote_from_kline(stock_code):
    """Fallback: get quote from latest kline data."""
    try:
        df = get_stock_kline(stock_code, days=10)
        if df.empty:
            return None

        row = df.iloc[-1]
        prev_row = df.iloc[-2] if len(df) > 1 else row

        result = {
            'stock_code': stock_code,
            'price': _safe_float(row.get('close', 0)),
            'change_pct': _safe_float(row.get('change_pct', 0)),
            'open': _safe_float(row.get('open', 0)),
            'high': _safe_float(row.get('high', 0)),
            'low': _safe_float(row.get('low', 0)),
            'volume': _safe_int(row.get('volume', 0)),
            'amount': _safe_float(row.get('amount', 0)),
            'pre_close': _safe_float(prev_row.get('close', 0)),
            'is_realtime': False,
            'data_source': 'kline',
            'last_trade_date': str(row.get('trade_date', ''))[:10]
        }
        return result
    except Exception as e:
        print(f"Error getting quote from kline for {stock_code}: {e}")
        return None

def _safe_float(val, default=0.0):
    try:
        if val is None or pd.isna(val):
            return default
        return float(val)
    except Exception:
        return default

def _safe_int(val, default=0):
    try:
        if val is None or pd.isna(val):
            return default
        return int(val)
    except Exception:
        return default

def get_capital_flow(stock_code, days=5):
    """Get capital flow data for a stock."""
    if not HAS_ADATA:
        return []

    cache_key = f"capital_{stock_code}_{days}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    try:
        df = adata.stock.market.get_capital_flow(stock_code=stock_code)
        if df is None or df.empty:
            return []

        # Get recent days
        df = df.sort_values('trade_date', ascending=False).head(days)

        flows = []
        for _, row in df.iterrows():
            flow = {
                'trade_date': str(row.get('trade_date', '')),
                'main_net_inflow': float(row.get('main_net_inflow', 0) or 0)
            }
            flows.append(flow)
            # Cache to database
            db_service.save_capital_flow(
                stock_code,
                flow['trade_date'],
                flow['main_net_inflow']
            )

        _set_cached(cache_key, flows)
        return flows

    except Exception as e:
        print(f"Error fetching capital flow for {stock_code}: {e}")
        return []

# Popular stocks list
POPULAR_STOCKS = [
    ('600519', '贵州茅台'), ('000858', '五粮液'), ('000333', '美的集团'),
    ('002594', '比亚迪'), ('300750', '宁德时代'), ('601318', '中国平安'),
    ('600036', '招商银行'), ('000001', '平安银行'), ('600030', '中信证券'),
    ('000651', '格力电器'), ('600276', '恒瑞医药'), ('601398', '工商银行'),
]

def init_stock_list():
    """Initialize stock list in database (runs in background)."""
    import threading
    def _load():
        try:
            # Save popular stocks first
            for code, name in POPULAR_STOCKS:
                db_service.save_stock_info(code, name, '')
            if not HAS_ADATA:
                return
            # Then load full list
            df = adata.stock.info.all_code()
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    db_service.save_stock_info(
                        row.get('stock_code', ''),
                        row.get('short_name', ''),
                        row.get('exchange', '')
                    )
                print(f"Loaded {len(df)} stocks to database")
        except Exception as e:
            print(f"Error loading stock list: {e}")
    threading.Thread(target=_load, daemon=True).start()

def get_hot_stocks_fast(limit=12):
    """Get hot stocks list instantly (no API calls)."""
    return [{'stock_code': c[0], 'short_name': c[1], 'change_pct': 0}
            for c in POPULAR_STOCKS[:limit]]

def get_stock_name(stock_code):
    """Get stock name by code."""
    cache_key = f"name_{stock_code}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    # First check database
    info = db_service.get_stock_info(stock_code)
    if info:
        name = info.get('short_name', '')
        _set_cached(cache_key, name, ttl=NAME_TTL)
        return name

    if HAS_ADATA:
        # Check current market data
        try:
            df = adata.stock.market.list_market_current(code_list=[stock_code])
            if df is not None and not df.empty:
                name = df.iloc[0].get('short_name', '')
                if name:
                    db_service.save_stock_info(stock_code, name, '')
                    _set_cached(cache_key, name, ttl=NAME_TTL)
                    return name
        except Exception as e:
            print(f"Error from list_market_current: {e}")

        # Fetch from all_code (slower but comprehensive)
        try:
            df = adata.stock.info.all_code()
            match = df[df['stock_code'] == stock_code]
            if not match.empty:
                name = match.iloc[0].get('short_name', '')
                exchange = match.iloc[0].get('exchange', '')
                db_service.save_stock_info(stock_code, name, exchange)
                _set_cached(cache_key, name, ttl=NAME_TTL)
                return name
        except Exception as e:
            print(f"Error getting stock name from all_code: {e}")

    _set_cached(cache_key, stock_code, ttl=NAME_TTL)
    return stock_code

def search_stocks(query, limit=10):
    """Search stocks by code or name - uses database first for speed."""
    # First try database (fast)
    db_results = db_service.search_stocks(query)
    if db_results:
        return [{'stock_code': r['stock_code'], 'short_name': r['short_name']}
                for r in db_results[:limit]]

    # Fallback to API (slow, but populates database)
    if not HAS_ADATA:
        # Return matching popular stocks as fallback
        query_upper = query.upper()
        return [{'stock_code': c[0], 'short_name': c[1]}
                for c in POPULAR_STOCKS
                if query_upper in c[0] or query in c[1]][:limit]

    cache_key = f"search_{query}_{limit}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    results = []
    try:
        df = adata.stock.info.all_code()
        if df is None or df.empty:
            return results

        # Save all stocks to database for future searches
        for _, row in df.iterrows():
            db_service.save_stock_info(
                row.get('stock_code', ''),
                row.get('short_name', ''),
                row.get('exchange', '')
            )

        query_upper = query.upper()
        # Match by code (prefix) or name (contains)
        mask = (df['stock_code'].str.startswith(query_upper) |
                df['short_name'].str.contains(query, case=False, na=False))
        matches = df[mask].head(limit)

        for _, row in matches.iterrows():
            results.append({
                'stock_code': row.get('stock_code', ''),
                'short_name': row.get('short_name', '')
            })

        _set_cached(cache_key, results)
    except Exception as e:
        print(f"Error searching stocks: {e}")

    return results

def get_index_quotes():
    """Get major index quotes (Shanghai, Shenzhen)."""
    indices = [
        {'code': '000001', 'name': '上证指数'},
        {'code': '399001', 'name': '深证成指'},
        {'code': '399006', 'name': '创业板指'}
    ]

    if not HAS_ADATA:
        return [{'code': i['code'], 'name': i['name'], 'price': 0, 'change_pct': 0} for i in indices]

    results = []
    for idx in indices:
        try:
            # Try current index API
            df = adata.stock.market.get_market_index_current(index_code=idx['code'])
            if df is not None and not df.empty:
                row = df.iloc[0]
                results.append({
                    'code': idx['code'],
                    'name': idx['name'],
                    'price': float(row.get('price', 0) or row.get('close', 0) or 0),
                    'change_pct': float(row.get('change_pct', 0) or 0)
                })
                continue
        except Exception as e:
            print(f"Error with get_market_index_current for {idx['code']}: {e}")

        try:
            # Fallback to historical index data
            df = adata.stock.market.get_market_index(index_code=idx['code'])
            if df is not None and not df.empty:
                row = df.iloc[-1]
                results.append({
                    'code': idx['code'],
                    'name': idx['name'],
                    'price': float(row.get('close', 0) or 0),
                    'change_pct': float(row.get('change_pct', 0) or 0)
                })
                continue
        except Exception as e:
            print(f"Error with get_market_index for {idx['code']}: {e}")

        # If all fails, add placeholder
        results.append({
            'code': idx['code'],
            'name': idx['name'],
            'price': 0,
            'change_pct': 0
        })

    return results
