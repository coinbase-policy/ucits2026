# UCITS Figures

Scatter plots comparing EURO STOXX 50 constituents with top crypto assets.
All figures plot Market Cap (EUR) on the x-axis.
Black dots = EURO STOXX 50 stocks | Red dots = top crypto assets.

## Figures

| File | Description |
|------|-------------|
| `figure1_volume_vs_mktcap.png` | Average daily trading volume (EUR) vs market cap |
| `figure2_amihud.png` | Amihud (2002) illiquidity ratio (`|daily return| / daily EUR volume`) vs market cap |
| `figure3_volatility.png` | Daily volatility (std of daily returns) vs market cap |
| `figure4_correlation.png` | Correlation of daily returns with the EURO STOXX 50 index vs market cap |

## Sample Period

August 1, 2023 – May 31, 2026

## Data Sources

- **Prices and volumes**: [yfinance](https://github.com/ranaroussi/yfinance) (Yahoo Finance)
- **Crypto market caps and volumes**: [CoinGecko public API](https://www.coingecko.com/en/api)

Top-10 crypto assets by market cap as of June 1, 2026, excluding exchange tokens and stablecoins:
BTC, ETH, XRP, SOL, LINK, DOGE, ADA, XLM, ZEC, HYPE.

> **Note:** HYPE (Hyperliquid) has no yfinance ticker and appears only in Figure 1 (volume/market cap from CoinGecko).

## How to Run

```bash
pip install -r requirements.txt
python figures.py
```

A `cache/` directory is created on the first run to avoid re-downloading data. Subsequent runs use the cached data (prices cached indefinitely as Parquet files; market cap and CoinGecko data cached for 24 hours).
