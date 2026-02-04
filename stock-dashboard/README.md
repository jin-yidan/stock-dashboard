# Stock Dashboard - A股技术分析工具

A Chinese A-share stock analysis dashboard with technical indicators, trading signals, and AI-powered analysis.

## Features

- Real-time stock quotes and K-line charts
- 20+ technical indicators (MACD, RSI, KDJ, Bollinger Bands, etc.)
- Trading signal generation with confidence scores
- Chip distribution (筹码分布) analysis
- Multiple trading strategies detection
- AI analysis powered by Claude API (optional)
- Watchlist with local storage

## Requirements

- Python 3.9+
- pip

## Quick Start

```bash
# Clone the repo
git clone https://github.com/jin-yidan/stock-dashboard.git
cd stock-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Then open http://localhost:8080 in your browser.

## Dependencies

- **adata** - Chinese stock market data API
- **Flask** - Web framework
- **pandas/numpy** - Data processing
- **scikit-learn** - ML models (optional)
- **anthropic** - Claude AI analysis (optional, requires API key)

## Usage

1. Enter a stock code (e.g., `600519` for 贵州茅台) or search by name
2. View technical analysis, indicators, and trading signals
3. Add stocks to your watchlist (saved in browser)
4. Use AI analysis with your own Claude API key (optional)

## Notes

- Data is fetched from Chinese market APIs (Baidu, Eastmoney)
- Best used within China or with good network access to Chinese servers
- Some features require the `czsc` library for Chan Theory analysis (optional)

## License

MIT
