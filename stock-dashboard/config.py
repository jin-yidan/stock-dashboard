import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Database
DATABASE_PATH = os.path.join(BASE_DIR, 'data', 'stock.db')

# Cache settings
CACHE_EXPIRE_HOURS = 4  # Re-fetch data after 4 hours

# Default settings
DEFAULT_KLINE_DAYS = 120  # Days of historical data to fetch
