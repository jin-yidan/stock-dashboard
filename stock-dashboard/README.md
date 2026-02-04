# Stock Dashboard

A modern technical analysis tool for Chinese A-share stocks.

## What It Does

This dashboard helps you analyze Chinese stocks with:

- **K-Line Charts** - Interactive candlestick charts with MA lines
- **Technical Indicators** - MACD, RSI, KDJ, Bollinger Bands, and 20+ more
- **Trading Signals** - Buy/sell signals with confidence scores
- **AI Analysis** - Optional Claude-powered stock analysis
- **Watchlist** - Save your favorite stocks locally

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/jin-yidan/stock-dashboard.git
cd stock-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py
```

Open **http://localhost:8080** in your browser.

## How to Use

1. **Search** - Enter a stock code (e.g., `600519`) or name (e.g., `茅台`)
2. **Analyze** - View charts, indicators, and trading signals
3. **Save** - Add stocks to your watchlist (stored in browser)
4. **AI Analysis** - Click "AI 分析" and enter your Claude API key (optional)

## Features

| Feature | Description |
|---------|-------------|
| K-Line Chart | Daily candlestick chart with MA5/10/20 |
| Signal Score | Overall buy/sell signal (0-100) |
| Technical Indicators | MACD, RSI, KDJ, BOLL, etc. |
| 52-Week Range | Current price position in yearly range |
| Chip Distribution | 筹码分布 analysis |
| Trading Strategies | Turtle trading, platform breakout, etc. |
| AI Analysis | Claude-powered stock analysis |

## Requirements

- Python 3.9+
- Internet connection (for market data)

## Data Sources

Market data is fetched from:
- Baidu Stock (百度股市通)
- East Money (东方财富)

**Note:** Works best with access to Chinese network.

## Tech Stack

- **Backend:** Flask, pandas, numpy
- **Frontend:** ECharts, vanilla JavaScript
- **Data:** adata library
- **AI:** Anthropic Claude API (optional)

## License

MIT
