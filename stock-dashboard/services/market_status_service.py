"""
Market status service - tracks trading hours, holidays, and data freshness.
"""

from datetime import datetime, time, timedelta
import pytz

# China timezone
CHINA_TZ = pytz.timezone('Asia/Shanghai')

# Trading hours (China market)
MORNING_OPEN = time(9, 30)
MORNING_CLOSE = time(11, 30)
AFTERNOON_OPEN = time(13, 0)
AFTERNOON_CLOSE = time(15, 0)
PRE_OPEN = time(9, 15)
PRE_CLOSE = time(9, 25)

# 2024-2025 Chinese market holidays (approximate - update as needed)
HOLIDAYS_2024 = [
    # New Year
    '2024-01-01',
    # Spring Festival
    '2024-02-09', '2024-02-10', '2024-02-11', '2024-02-12', '2024-02-13',
    '2024-02-14', '2024-02-15', '2024-02-16', '2024-02-17',
    # Qingming
    '2024-04-04', '2024-04-05', '2024-04-06',
    # Labor Day
    '2024-05-01', '2024-05-02', '2024-05-03', '2024-05-04', '2024-05-05',
    # Dragon Boat
    '2024-06-08', '2024-06-09', '2024-06-10',
    # Mid-Autumn
    '2024-09-15', '2024-09-16', '2024-09-17',
    # National Day
    '2024-10-01', '2024-10-02', '2024-10-03', '2024-10-04', '2024-10-05',
    '2024-10-06', '2024-10-07',
]

HOLIDAYS_2025 = [
    # New Year
    '2025-01-01',
    # Spring Festival
    '2025-01-28', '2025-01-29', '2025-01-30', '2025-01-31', '2025-02-01',
    '2025-02-02', '2025-02-03', '2025-02-04',
    # Qingming
    '2025-04-04', '2025-04-05', '2025-04-06',
    # Labor Day
    '2025-05-01', '2025-05-02', '2025-05-03', '2025-05-04', '2025-05-05',
    # Dragon Boat
    '2025-05-31', '2025-06-01', '2025-06-02',
    # Mid-Autumn + National Day
    '2025-10-01', '2025-10-02', '2025-10-03', '2025-10-04', '2025-10-05',
    '2025-10-06', '2025-10-07', '2025-10-08',
]

HOLIDAYS = set(HOLIDAYS_2024 + HOLIDAYS_2025)


def get_china_now():
    """Get current time in China timezone."""
    return datetime.now(CHINA_TZ)


def is_holiday(date=None):
    """Check if a date is a market holiday."""
    if date is None:
        date = get_china_now().date()
    return date.strftime('%Y-%m-%d') in HOLIDAYS


def is_weekend(date=None):
    """Check if a date is weekend."""
    if date is None:
        date = get_china_now().date()
    return date.weekday() >= 5


def is_trading_day(date=None):
    """Check if a date is a trading day."""
    if date is None:
        date = get_china_now().date()
    return not is_weekend(date) and not is_holiday(date)


def get_market_status():
    """
    Get comprehensive market status.

    Returns dict with:
    - status: 'trading', 'pre_open', 'lunch_break', 'closed', 'holiday', 'weekend'
    - is_open: boolean
    - current_time: formatted current time
    - next_open: when market opens next (if closed)
    - session: 'morning', 'afternoon', None
    - message: human-readable status message
    """
    now = get_china_now()
    current_time = now.time()
    current_date = now.date()

    # Get local time (server/user timezone)
    local_now = datetime.now()

    result = {
        'current_time': now.strftime('%Y-%m-%d %H:%M:%S'),
        'current_time_cn': now.strftime('%H:%M'),
        'beijing_time': now.strftime('%H:%M'),
        'local_time': local_now.strftime('%H:%M'),
        'timezone': 'Asia/Shanghai',
    }

    # Check weekend
    if is_weekend(current_date):
        result.update({
            'status': 'weekend',
            'status_cn': '周末休市',
            'is_open': False,
            'message': '周末休市',
            'next_open': _get_next_trading_day(current_date)
        })
        return result

    # Check holiday
    if is_holiday(current_date):
        result.update({
            'status': 'holiday',
            'status_cn': '节假日休市',
            'is_open': False,
            'message': '节假日休市',
            'next_open': _get_next_trading_day(current_date)
        })
        return result

    # Check trading hours
    if current_time < PRE_OPEN:
        result.update({
            'status': 'pre_market',
            'status_cn': '盘前',
            'is_open': False,
            'message': f'距开盘 {_time_until(current_time, MORNING_OPEN)}',
            'session': None
        })
    elif PRE_OPEN <= current_time < MORNING_OPEN:
        result.update({
            'status': 'pre_open',
            'status_cn': '集合竞价',
            'is_open': False,
            'message': '集合竞价中',
            'session': None
        })
    elif MORNING_OPEN <= current_time < MORNING_CLOSE:
        result.update({
            'status': 'trading',
            'status_cn': '交易中',
            'is_open': True,
            'message': '上午交易时段',
            'session': 'morning'
        })
    elif MORNING_CLOSE <= current_time < AFTERNOON_OPEN:
        result.update({
            'status': 'lunch_break',
            'status_cn': '午间休市',
            'is_open': False,
            'message': f'午休，{_time_until(current_time, AFTERNOON_OPEN)}后开盘',
            'session': None
        })
    elif AFTERNOON_OPEN <= current_time < AFTERNOON_CLOSE:
        result.update({
            'status': 'trading',
            'status_cn': '交易中',
            'is_open': True,
            'message': '下午交易时段',
            'session': 'afternoon'
        })
    else:
        result.update({
            'status': 'closed',
            'status_cn': '已收盘',
            'is_open': False,
            'message': '今日已收盘',
            'next_open': _get_next_trading_day(current_date)
        })

    return result


def _time_until(current, target):
    """Calculate time until target time."""
    current_minutes = current.hour * 60 + current.minute
    target_minutes = target.hour * 60 + target.minute
    diff = target_minutes - current_minutes

    if diff <= 0:
        return "0分钟"

    hours = diff // 60
    minutes = diff % 60

    if hours > 0:
        return f"{hours}小时{minutes}分钟"
    return f"{minutes}分钟"


def _get_next_trading_day(from_date):
    """Get the next trading day after from_date."""
    next_day = from_date + timedelta(days=1)
    attempts = 0

    while attempts < 30:  # Max 30 days ahead
        if is_trading_day(next_day):
            return next_day.strftime('%Y-%m-%d')
        next_day += timedelta(days=1)
        attempts += 1

    return None


def get_data_freshness(data_timestamp):
    """
    Evaluate data freshness based on timestamp and market status.

    Args:
        data_timestamp: datetime or string timestamp of the data

    Returns dict with:
    - fresh: boolean - is data considered fresh
    - age: age description
    - status: 'live', 'recent', 'delayed', 'stale'
    """
    if data_timestamp is None:
        return {'fresh': False, 'age': '未知', 'status': 'unknown', 'status_cn': '未知'}

    if isinstance(data_timestamp, str):
        try:
            data_timestamp = datetime.fromisoformat(data_timestamp.replace('Z', '+00:00'))
        except ValueError:
            try:
                data_timestamp = datetime.strptime(data_timestamp[:10], '%Y-%m-%d')
            except ValueError:
                return {'fresh': False, 'age': '无效时间', 'status': 'unknown', 'status_cn': '未知'}

    # Make timezone-aware if needed
    now = get_china_now()
    if data_timestamp.tzinfo is None:
        data_timestamp = CHINA_TZ.localize(data_timestamp)

    age = now - data_timestamp
    age_seconds = age.total_seconds()
    age_minutes = age_seconds / 60
    age_hours = age_seconds / 3600
    age_days = age_seconds / 86400

    # Format age string
    if age_seconds < 60:
        age_str = '刚刚更新'
    elif age_minutes < 60:
        age_str = f'{int(age_minutes)}分钟前'
    elif age_hours < 24:
        age_str = f'{int(age_hours)}小时前'
    else:
        age_str = f'{int(age_days)}天前'

    # Determine freshness based on market status
    market = get_market_status()

    if market['is_open']:
        # During trading, data older than 5 minutes is delayed
        if age_seconds < 60:
            status = 'live'
            status_cn = '实时'
            fresh = True
        elif age_seconds < 300:
            status = 'recent'
            status_cn = '较新'
            fresh = True
        elif age_seconds < 1800:
            status = 'delayed'
            status_cn = '延迟'
            fresh = False
        else:
            status = 'stale'
            status_cn = '过期'
            fresh = False
    else:
        # Market closed - data from today's close is fresh
        if age_hours < 20:  # Within today
            status = 'recent'
            status_cn = '最新收盘'
            fresh = True
        elif age_days < 3:
            status = 'delayed'
            status_cn = '近期数据'
            fresh = True
        else:
            status = 'stale'
            status_cn = '历史数据'
            fresh = False

    return {
        'fresh': fresh,
        'age': age_str,
        'status': status,
        'status_cn': status_cn,
        'age_seconds': int(age_seconds)
    }


def get_last_trading_date():
    """Get the most recent trading date (could be today if market is/was open)."""
    now = get_china_now()
    current_date = now.date()
    current_time = now.time()

    # If market hasn't opened yet today, use previous trading day
    if current_time < MORNING_OPEN:
        current_date = current_date - timedelta(days=1)

    # Find last trading day
    attempts = 0
    while attempts < 30:
        if is_trading_day(current_date):
            return current_date.strftime('%Y-%m-%d')
        current_date = current_date - timedelta(days=1)
        attempts += 1

    return None
