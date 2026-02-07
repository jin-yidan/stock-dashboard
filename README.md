# Stock Dashboard

A modern technical analysis and signal dashboard for Chinese A-share stocks.

## Dashboard （可以添加自选股)
<img width="1512" height="982" alt="Screenshot 2026-02-04 at 12 21 38" src="https://github.com/user-attachments/assets/4ae99863-6614-49ad-80b6-f9a99c1698ef" />


## 个股分析
<img width="1512" height="982" alt="Screenshot 2026-02-04 at 12 23 37" src="https://github.com/user-attachments/assets/531f3752-1c8a-45c5-b94f-d1a6c19c0fd6" />
<img width="1512" height="982" alt="Screenshot 2026-02-04 at 12 24 04" src="https://github.com/user-attachments/assets/efb19fbd-1491-4e7e-b839-44ae7a7d77c7" />


## 实时k线
<img width="1512" height="982" alt="Screenshot 2026-02-04 at 12 24 16" src="https://github.com/user-attachments/assets/d0e2a6fc-bf7a-446c-ab0a-5bbf70359e8b" />


## 基本面与技术指标
<img width="1512" height="982" alt="Screenshot 2026-02-04 at 12 24 27" src="https://github.com/user-attachments/assets/b91f12d9-9e69-4ca2-be33-07298ff783f6" />
<img width="1512" height="982" alt="Screenshot 2026-02-04 at 12 24 42" src="https://github.com/user-attachments/assets/e2643beb-f48c-4df7-8edf-8b63ed6779d3" />

(截图仅展示了部分功能）

## What It Does

This dashboard helps you analyze Chinese stocks with:

- **Realtime Quote (when available)** - Tries realtime API first, falls back to latest daily close
- **Data Freshness** - Clearly shows whether data is live, recent, or stale
- **K-Line Charts** - Interactive candlestick charts with MA lines
- **Technical Indicators** - MACD, RSI, KDJ, Bollinger Bands, and 20+ more
- **Trading Signals** - Market-regime-aware signals with confidence scores
- **Historical Accuracy** - Win rate, expected return, MFE/MAE, and calibration buckets
- **ML Prediction (5-day horizon)** - ATR-scaled labeling with walk-forward validation
- **AI Analysis** - Optional Claude-powered stock analysis
- **Watchlist** - Save your favorite stocks locally

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/jin-yidan/stock-dashboard.git
cd stock-dashboard/stock-dashboard

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
| Realtime Quote | Live quote when available, with fallback to latest daily close |
| Data Freshness | Indicates if data is live / recent / stale |
| K-Line Chart | Daily candlestick chart with MA5/10/20 |
| Signal Score | Overall buy/sell signal with confidence |
| Historical Accuracy | Win rate, expected return, MFE/MAE, score buckets |
| Technical Indicators | MACD, RSI, KDJ, BOLL, and more |
| 52-Week Range | Current price position in yearly range |
| Chip Distribution | 筹码分布 analysis |
| Trading Strategies | Turtle trading, platform breakout, etc. |
| ML Prediction | 5-day horizon, ATR-scaled labels, walk-forward validation |
| AI Analysis | Claude-powered stock analysis |

## Requirements

- Python 3.9+
- Internet connection (for market data)

## Data Sources

Market data is fetched from:
- Baidu Stock (百度股市通) and East Money (东方财富) for daily K-line (前复权)
- Sina / Tencent realtime quotes via `adata.stock.market.list_market_current`
- Tencent K-line as a fallback

**Note:** Works best with access to Chinese network.

## Tech Stack

- **Backend:** Flask, pandas, numpy
- **Frontend:** ECharts, vanilla JavaScript
- **Data:** adata library
- **AI:** Anthropic Claude API (optional)

## License

MIT
