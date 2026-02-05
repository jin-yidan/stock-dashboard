import pandas as pd
from datetime import datetime, timedelta
from config import DEFAULT_KLINE_DAYS
from services import db_service

# Try to import adata - may not be available on Vercel
try:
    import adata
    HAS_ADATA = True
except ImportError:
    HAS_ADATA = False
    print("Warning: adata not available, using fallback data")

# Simple in-memory cache
_cache = {}
_cache_time = {}
CACHE_TTL = 1800  # 30 minutes - kline data doesn't change frequently

def _get_cached(key):
    """Get cached value if not expired."""
    if key in _cache and key in _cache_time:
        if (datetime.now() - _cache_time[key]).seconds < CACHE_TTL:
            return _cache[key]
    return None

def _set_cached(key, value):
    """Set cache value."""
    _cache[key] = value
    _cache_time[key] = datetime.now()

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
        return pd.DataFrame()

    cache_key = f"kline_{stock_code}_{days}"
    cached = _get_cached(cache_key)
    if cached is not None and not cached.empty:
        return cached

    try:
        # Try baidu_market first, fall back to east_market if empty
        df = adata.stock.market.baidu_market.get_market(stock_code=stock_code)
        if df is None or df.empty:
            df = adata.stock.market.east_market.get_market(stock_code=stock_code)

        if df is None or df.empty:
            return pd.DataFrame()

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
        df = df.tail(days)

        _set_cached(cache_key, df)
        return df

    except Exception as e:
        print(f"Error fetching kline for {stock_code}: {e}")
        return pd.DataFrame()

def get_realtime_quote(stock_code):
    """Get realtime quote for a stock.

    Optimized: Uses kline data directly instead of slow list_market_current API.
    """
    cache_key = f"quote_{stock_code}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    # Use kline data directly - it's faster and already has all the data we need
    return _get_quote_from_kline(stock_code)

def _get_quote_from_kline(stock_code):
    """Fallback: get quote from latest kline data."""
    cache_key = f"quote_{stock_code}"
    try:
        df = get_stock_kline(stock_code, days=10)
        if df.empty:
            return None

        row = df.iloc[-1]
        prev_row = df.iloc[-2] if len(df) > 1 else row

        result = {
            'stock_code': stock_code,
            'price': float(row.get('close', 0)),
            'change_pct': float(row.get('change_pct', 0)),
            'open': float(row.get('open', 0)),
            'high': float(row.get('high', 0)),
            'low': float(row.get('low', 0)),
            'volume': int(row.get('volume', 0)),
            'amount': float(row.get('amount', 0)),
            'pre_close': float(prev_row.get('close', 0))
        }
        _set_cached(cache_key, result)
        return result
    except Exception as e:
        print(f"Error getting quote from kline for {stock_code}: {e}")
        return None

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
        _set_cached(cache_key, name)
        return name

    if HAS_ADATA:
        # Check current market data
        try:
            df = adata.stock.market.list_market_current(code_list=[stock_code])
            if df is not None and not df.empty:
                name = df.iloc[0].get('short_name', '')
                if name:
                    db_service.save_stock_info(stock_code, name, '')
                    _set_cached(cache_key, name)
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
                _set_cached(cache_key, name)
                return name
        except Exception as e:
            print(f"Error getting stock name from all_code: {e}")

    _set_cached(cache_key, stock_code)
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
