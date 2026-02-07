import sqlite3
from contextlib import contextmanager
from config import DATABASE_PATH

def init_db():
    """Initialize database with required tables."""
    with get_connection() as conn:
        cursor = conn.cursor()

        # Stock info cache
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stocks (
                stock_code TEXT PRIMARY KEY,
                short_name TEXT,
                exchange TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Daily data with indicators
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_daily (
                stock_code TEXT,
                trade_date TEXT,
                open REAL,
                close REAL,
                high REAL,
                low REAL,
                volume INTEGER,
                amount REAL,
                change_pct REAL,
                ma5 REAL,
                ma10 REAL,
                ma20 REAL,
                ma60 REAL,
                macd_dif REAL,
                macd_dea REAL,
                macd_hist REAL,
                rsi REAL,
                PRIMARY KEY (stock_code, trade_date)
            )
        ''')

        # Capital flow
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS capital_flow (
                stock_code TEXT,
                trade_date TEXT,
                main_net_inflow REAL,
                PRIMARY KEY (stock_code, trade_date)
            )
        ''')

        conn.commit()

@contextmanager
def get_connection():
    """Get database connection context manager."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def save_stock_info(stock_code, short_name, exchange):
    """Save stock info to database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO stocks (stock_code, short_name, exchange, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ''', (stock_code, short_name, exchange))
        conn.commit()

def get_stock_info(stock_code):
    """Get stock info from database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM stocks WHERE stock_code = ?', (stock_code,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

def save_daily_data(stock_code, df):
    """Save daily data with indicators to database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        rows = []
        for _, row in df.iterrows():
            rows.append((
                stock_code,
                str(row.get('trade_date', '')),
                row.get('open'),
                row.get('close'),
                row.get('high'),
                row.get('low'),
                row.get('volume'),
                row.get('amount'),
                row.get('change_pct'),
                row.get('ma5'),
                row.get('ma10'),
                row.get('ma20'),
                row.get('ma60'),
                row.get('macd_dif'),
                row.get('macd_dea'),
                row.get('macd_hist'),
                row.get('rsi')
            ))
        cursor.executemany('''
            INSERT OR REPLACE INTO stock_daily
            (stock_code, trade_date, open, close, high, low, volume, amount, change_pct,
             ma5, ma10, ma20, ma60, macd_dif, macd_dea, macd_hist, rsi)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', rows)
        conn.commit()

def get_daily_data(stock_code, days=120):
    """Get daily data from database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM stock_daily
            WHERE stock_code = ?
            ORDER BY trade_date DESC
            LIMIT ?
        ''', (stock_code, days))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def save_capital_flow(stock_code, trade_date, main_net_inflow):
    """Save capital flow data."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO capital_flow (stock_code, trade_date, main_net_inflow)
            VALUES (?, ?, ?)
        ''', (stock_code, trade_date, main_net_inflow))
        conn.commit()

def get_capital_flow(stock_code, days=5):
    """Get recent capital flow data."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM capital_flow
            WHERE stock_code = ?
            ORDER BY trade_date DESC
            LIMIT ?
        ''', (stock_code, days))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def search_stocks(query):
    """Search stocks by code or name."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM stocks
            WHERE stock_code LIKE ? OR short_name LIKE ?
            LIMIT 20
        ''', (f'%{query}%', f'%{query}%'))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
