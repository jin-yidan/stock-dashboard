"""
Fundamental data service.
Provides basic fundamental analysis using available data.
"""

import pandas as pd
from datetime import datetime
from services import db_service

try:
    import adata
    HAS_ADATA = True
except ImportError:
    HAS_ADATA = False

# Cache
_cache = {}
_cache_time = {}
CACHE_TTL = 3600  # 1 hour for fundamental data


def _get_cached(key):
    if key in _cache and key in _cache_time:
        if (datetime.now() - _cache_time[key]).seconds < CACHE_TTL:
            return _cache[key]
    return None


def _set_cached(key, value):
    _cache[key] = value
    _cache_time[key] = datetime.now()


def get_stock_shares(stock_code):
    """Get total shares for a stock."""
    if not HAS_ADATA:
        return None

    cache_key = f"shares_{stock_code}"
    cached = _get_cached(cache_key)
    if cached:
        return cached

    try:
        df = adata.stock.info.get_stock_shares(stock_code=stock_code)
        if df is not None and not df.empty:
            # Get latest share count
            df = df.sort_values('change_date', ascending=False)
            latest = df.iloc[0]
            result = {
                'total_shares': int(latest.get('total_shares', 0)),
                'list_a_shares': int(latest.get('list_a_shares', 0)),
                'change_date': str(latest.get('change_date', ''))
            }
            _set_cached(cache_key, result)
            return result
    except Exception as e:
        print(f"Error getting shares for {stock_code}: {e}")

    return None


def get_dividend_info(stock_code):
    """Get dividend information."""
    if not HAS_ADATA:
        return {'has_dividend': False, 'recent_dividends': 0, 'total_dividends': 0}

    cache_key = f"dividend_{stock_code}"
    cached = _get_cached(cache_key)
    if cached:
        return cached

    try:
        df = adata.stock.market.get_dividend(stock_code=stock_code)
        if df is not None and not df.empty:
            df = df.sort_values('report_date', ascending=False)

            # Count total dividends
            total_count = len(df)

            # Get recent dividends (last 3 years based on date)
            from datetime import datetime, timedelta
            three_years_ago = datetime.now() - timedelta(days=3*365)
            recent_df = df[pd.to_datetime(df['report_date']) > three_years_ago]
            recent_count = len(recent_df)

            result = {
                'has_dividend': total_count > 0,
                'recent_dividends': recent_count,
                'total_dividends': total_count,
                'latest_dividend': str(df.iloc[0]['dividend_plan']) if total_count > 0 else None,
                'latest_date': str(df.iloc[0]['report_date']) if total_count > 0 else None,
            }
            _set_cached(cache_key, result)
            return result
    except Exception as e:
        print(f"Error getting dividend for {stock_code}: {e}")

    return {'has_dividend': False, 'recent_dividends': 0, 'total_dividends': 0}


def get_market_cap(stock_code, current_price):
    """Calculate market cap from shares and price."""
    shares = get_stock_shares(stock_code)
    if shares and shares.get('list_a_shares'):
        market_cap = shares['list_a_shares'] * current_price
        return {
            'market_cap': market_cap,
            'market_cap_yi': round(market_cap / 100000000, 1),  # In 亿
            'size': 'large' if market_cap > 100000000000 else ('medium' if market_cap > 10000000000 else 'small')
        }
    return None


def get_industry(stock_code):
    """Get stock industry classification."""
    if not HAS_ADATA:
        return None

    cache_key = f"industry_{stock_code}"
    cached = _get_cached(cache_key)
    if cached:
        return cached

    try:
        df = adata.stock.info.get_industry_sw(stock_code=stock_code)
        if df is not None and not df.empty:
            row = df.iloc[0]
            result = {
                'industry_name': str(row.get('industry_name', '')),
                'industry_code': str(row.get('industry_code', ''))
            }
            _set_cached(cache_key, result)
            return result
    except Exception as e:
        print(f"Error getting industry for {stock_code}: {e}")

    return None


def get_concept(stock_code):
    """Get stock concepts/themes."""
    if not HAS_ADATA:
        return {'concepts': []}

    cache_key = f"concept_{stock_code}"
    cached = _get_cached(cache_key)
    if cached:
        return cached

    try:
        df = adata.stock.info.get_concept_east(stock_code=stock_code)
        if df is not None and not df.empty:
            # Column might be 'name' or 'concept_name' depending on adata version
            col = 'name' if 'name' in df.columns else 'concept_name'
            concepts = df[col].tolist()[:5]  # Top 5 concepts
            result = {'concepts': concepts}
            _set_cached(cache_key, result)
            return result
    except Exception as e:
        pass  # Silent fail - concepts are optional

    return {'concepts': []}


def get_fundamental_summary(stock_code, current_price):
    """Get comprehensive fundamental summary.

    Optimized to fetch all data in parallel.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    summary = {
        'stock_code': stock_code,
        'has_data': False
    }

    # Fetch all data in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(get_stock_shares, stock_code): 'shares',
            executor.submit(get_dividend_info, stock_code): 'dividend',
            executor.submit(get_industry, stock_code): 'industry',
            executor.submit(get_concept, stock_code): 'concept',
        }

        for future in as_completed(futures, timeout=10):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception:
                results[name] = None

    # Market cap (uses shares result)
    shares = results.get('shares')
    if shares and shares.get('list_a_shares') and current_price > 0:
        market_cap = shares['list_a_shares'] * current_price
        summary['market_cap_yi'] = round(market_cap / 100000000, 1)
        summary['size'] = 'large' if market_cap > 100000000000 else ('medium' if market_cap > 10000000000 else 'small')
        summary['has_data'] = True

    # Dividend
    dividend = results.get('dividend')
    if dividend:
        summary['has_dividend'] = dividend.get('has_dividend', False)
        summary['dividend_count'] = dividend.get('recent_dividends', 0)

    # Industry
    industry = results.get('industry')
    if industry:
        summary['industry'] = industry.get('industry_name', '')

    # Concepts
    concept = results.get('concept')
    if concept:
        summary['concepts'] = concept.get('concepts', [])

    return summary


def get_fundamental_score(stock_code, current_price):
    """Calculate a simple fundamental score."""
    score = 0
    reasons = []

    # Market cap - prefer large caps (more stable)
    market_cap = get_market_cap(stock_code, current_price)
    if market_cap:
        if market_cap['size'] == 'large':
            score += 0.2
            reasons.append('大盘股，稳定性较好')
        elif market_cap['size'] == 'medium':
            score += 0.1
            reasons.append('中盘股')
        else:
            score += 0
            reasons.append('小盘股，波动较大')

    # Dividend - companies with dividends tend to be more stable
    dividend = get_dividend_info(stock_code)
    if dividend and dividend['has_dividend']:
        if dividend['recent_dividends'] >= 3:
            score += 0.2
            reasons.append(f'近期分红{dividend["recent_dividends"]}次，回报股东')
        else:
            score += 0.1
            reasons.append('有分红记录')
    else:
        reasons.append('无近期分红')

    return {
        'score': round(score, 2),
        'max_score': 0.4,
        'reasons': reasons
    }
