"""
Fundamental data service.
Provides basic fundamental analysis using available data.
"""

import adata
import pandas as pd
from datetime import datetime
from services import db_service

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
    cache_key = f"concept_{stock_code}"
    cached = _get_cached(cache_key)
    if cached:
        return cached

    try:
        df = adata.stock.info.get_concept_east(stock_code=stock_code)
        if df is not None and not df.empty:
            concepts = df['concept_name'].tolist()[:5]  # Top 5 concepts
            result = {'concepts': concepts}
            _set_cached(cache_key, result)
            return result
    except Exception as e:
        print(f"Error getting concept for {stock_code}: {e}")

    return {'concepts': []}


def get_fundamental_summary(stock_code, current_price):
    """Get comprehensive fundamental summary."""
    summary = {
        'stock_code': stock_code,
        'has_data': False
    }

    # Market cap
    market_cap = get_market_cap(stock_code, current_price)
    if market_cap:
        summary['market_cap_yi'] = market_cap['market_cap_yi']
        summary['size'] = market_cap['size']
        summary['has_data'] = True

    # Dividend
    dividend = get_dividend_info(stock_code)
    if dividend:
        summary['has_dividend'] = dividend['has_dividend']
        summary['dividend_count'] = dividend['recent_dividends']

    # Industry
    industry = get_industry(stock_code)
    if industry:
        summary['industry'] = industry['industry_name']

    # Concepts
    concept = get_concept(stock_code)
    if concept:
        summary['concepts'] = concept['concepts']

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
