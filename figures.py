"""
4 figures comparing EURO STOXX 50 constituents with top crypto assets.
All figures are scatter plots with Market Cap (EUR) on the x-axis.
Black dots = EURO STOXX 50 stocks  |  Red dots = top crypto assets

Figure 1: Daily Trading Volume (€) vs Market Cap (€)
Figure 2: Amihud (2002) Illiquidity Ratio vs Market Cap (€)
Figure 3: Daily Volatility (std of daily returns) vs Market Cap (€)
Figure 4: 1-year Correlation with EURO STOXX 50 vs Market Cap (€)

Sample period: Aug 1, 2023 – May 31, 2026
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import yfinance as yf
import requests
from datetime import datetime

# ── colours ───────────────────────────────────────────────────────────────────
CRYPTO_COLOR = "red"
STOCK_COLOR  = "black"
FIG_DPI      = 150
SAVE_DIR     = "."

# ── sample period ─────────────────────────────────────────────────────────────
EXT_START   = "2023-08-01"
EXT_END     = "2026-05-31"

# ── asset universes ───────────────────────────────────────────────────────────
# Top-10 non-exchange crypto assets by market cap as of June 1, 2026
# yfinance EUR tickers; HYPE has no yfinance listing so it appears only in Fig 1
CRYPTO_TICKERS = {
    "BTC-EUR":  "BTC",
    "ETH-EUR":  "ETH",
    "XRP-EUR":  "XRP",
    "SOL-EUR":  "SOL",
    "LINK-EUR": "LINK",
    "DOGE-EUR": "DOGE",
    "ADA-EUR":  "ADA",
    "XLM-EUR":  "XLM",
    "ZEC-EUR":  "ZEC",
}
# HYPE (Hyperliquid) has no yfinance EUR ticker; handled via CoinGecko only
CRYPTO_FOOTNOTE = ("Top-10 crypto assets by market capitalization as of June 1, 2026,"
                   " excluding exchange tokens and stablecoins.")


# EURO STOXX 50 constituents (EUR-listed where possible)
STOXX50_TICKERS = {
    "AIR.PA":"AIR", "ALV.DE":"ALV", "ASML.AS":"ASML", "BBVA.MC":"BBVA",
    "BMW.DE":"BMW", "BNP.PA":"BNP", "CS.PA":"CS",   "DTE.DE":"DTE",
    "ENEL.MI":"ENEL","ENI.MI":"ENI", "IBE.MC":"IBE", "IFX.DE":"IFX",
    "INGA.AS":"ING", "ISP.MI":"ISP", "KER.PA":"KER", "MC.PA":"MC",
    "MUV2.DE":"MUV", "OR.PA":"LOR",  "ORA.PA":"ORA", "PHIA.AS":"PHI",
    "PRX.AS":"PRX",  "RMS.PA":"HER", "SAF.PA":"SAF", "SAN.MC":"SAN",
    "SAP.DE":"SAP",  "SIE.DE":"SIE", "SU.PA":"SCH",  "TTE.PA":"TTE",
    "UCG.MI":"UCG",  "VIV.PA":"VIV", "VOW3.DE":"VW",  "STLAM.MI":"STL",
    "ABI.BR":"ABI",  "AD.AS":"ADYEN","DBK.DE":"DBK",  "MBG.DE":"MER",
    "BAS.DE":"BAS",  "BAYN.DE":"BAY","AI.PA":"AI",    "SGO.PA":"SGO",
    "DHL.DE":"DHL",  "ENGI.PA":"ENGI","EL.PA":"EL",   "FLTR.L":"FLTR",
    "CRH.L":"CRH",
}

INDEX_TICKER = "EXS1.DE"   # iShares Core EURO STOXX 50 ETF


# ─────────────────────────────────────────────────────────────────────────────
# DATA HELPERS  (disk cache: prices → parquet, snapshots → json with 24h TTL)
# ─────────────────────────────────────────────────────────────────────────────

import os as _os, json as _json, hashlib as _hashlib, time as _time
_CACHE_DIR = "./cache"
_os.makedirs(_CACHE_DIR, exist_ok=True)

def _cache_path(tickers, start, end, tag="prices"):
    key = "_".join(sorted(tickers)) + f"_{start}_{end}_{tag}"
    h = _hashlib.md5(key.encode()).hexdigest()[:10]
    return f"{_CACHE_DIR}/{h}.parquet"

def _json_cache_load(path, ttl_hours=24):
    """Load JSON cache if it exists and is fresher than ttl_hours."""
    if _os.path.exists(path):
        age = _time.time() - _os.path.getmtime(path)
        if age < ttl_hours * 3600:
            with open(path) as f:
                return _json.load(f)
    return None

def _json_cache_save(path, data):
    with open(path, "w") as f:
        _json.dump(data, f)

def fetch_prices(tickers, start, end):
    path = _cache_path(tickers, start, end, "prices")
    if _os.path.exists(path):
        return pd.read_parquet(path)
    raw = yf.download(list(tickers), start=start, end=end,
                      auto_adjust=True, progress=False, threads=True)
    if isinstance(raw.columns, pd.MultiIndex):
        result = raw["Close"].dropna(how="all")
    else:
        result = raw[["Close"]].dropna(how="all")
    result.to_parquet(path)
    return result


def fetch_ohlc(tickers, start, end):
    path = _cache_path(tickers, start, end, "ohlc")
    if _os.path.exists(path):
        stored = pd.read_parquet(path)
        return {col: stored[col] for col in stored.columns.get_level_values(0).unique()}
    raw = yf.download(list(tickers), start=start, end=end,
                      auto_adjust=True, progress=False, threads=True)
    if not isinstance(raw.columns, pd.MultiIndex):
        out = {}
        for col in ["High","Low","Close","Volume"]:
            if col in raw.columns:
                out[col] = raw[[col]].dropna(how="all")
        return out
    available = raw.columns.get_level_values(0).unique()
    result = {col: raw[col].dropna(how="all")
              for col in ["High","Low","Close","Volume"] if col in available}
    pd.concat(result.values(), axis=1, keys=result.keys()).to_parquet(path)
    return result


def get_market_cap_eur(tickers_dict):
    """Return {label: market_cap_EUR} using today's yfinance fast_info.market_cap.
    Cached to disk for 24 hours. LSE tickers (.L) are in GBX; divide by 100.
    """
    cache_key = "_".join(sorted(tickers_dict.keys()))
    h = _hashlib.md5(cache_key.encode()).hexdigest()[:10]
    path = f"{_CACHE_DIR}/mc_{h}.json"
    cached = _json_cache_load(path)
    if cached:
        return cached
    result = {}
    for ticker, label in tickers_dict.items():
        try:
            fi = yf.Ticker(ticker).fast_info
            mc = getattr(fi, "market_cap", None)
            if mc and mc > 0:
                if ticker.endswith(".L"):
                    mc = mc / 100
                result[label] = mc
        except Exception:
            pass
    _json_cache_save(path, result)
    return result


def get_avg_daily_volume_eur(tickers_dict, start, end):
    """Return {label: avg_daily_volume_EUR} = avg(Close × Volume) over period.
    LSE tickers (.L) return Close in GBX (pence); divide by 100 to get GBP.
    """
    result = {}
    ohlc = fetch_ohlc(list(tickers_dict.keys()), start, end)
    close  = ohlc.get("Close", pd.DataFrame())
    volume = ohlc.get("Volume", pd.DataFrame())
    for ticker, label in tickers_dict.items():
        try:
            if ticker in close.columns and ticker in volume.columns:
                dv = (close[ticker] * volume[ticker]).dropna()
                if ticker.endswith(".L"):
                    dv = dv / 100  # GBX → GBP
                if len(dv) > 20:
                    result[label] = dv.mean()
        except Exception:
            pass
    return result


COINGECKO_IDS = "bitcoin,ethereum,ripple,solana,chainlink,dogecoin,cardano,stellar,zcash,hyperliquid"

COINGECKO_ID_MAP = {
    "bitcoin":     "BTC",
    "ethereum":    "ETH",
    "ripple":      "XRP",
    "solana":      "SOL",
    "chainlink":   "LINK",
    "dogecoin":    "DOGE",
    "cardano":     "ADA",
    "stellar":     "XLM",
    "zcash":       "ZEC",
    "hyperliquid": "HYPE",
}

def coingecko_eur():
    """Return DataFrame [label, market_cap, total_volume] in EUR. Cached 24h."""
    path = f"{_CACHE_DIR}/coingecko_eur.json"
    cached = _json_cache_load(path)
    if cached:
        return pd.DataFrame(cached)

    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": "eur", "ids": COINGECKO_IDS,
              "order": "market_cap_desc", "per_page": 10, "sparkline": False}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        cg_data = {row["symbol"].lower(): row for row in r.json()}
    except Exception as e:
        print(f"  CoinGecko unavailable: {e}")
        return None

    label_map = {"btc":"BTC","eth":"ETH","xrp":"XRP","sol":"SOL",
                 "link":"LINK","doge":"DOGE","ada":"ADA","xlm":"XLM",
                 "zec":"ZEC","hype":"HYPE"}
    rows = []
    for sym, label in label_map.items():
        if sym not in cg_data:
            continue
        rows.append({"label":        label,
                     "market_cap":   cg_data[sym].get("market_cap") or 0,
                     "total_volume": cg_data[sym].get("total_volume") or 0})
    if rows:
        _json_cache_save(path, rows)
    return pd.DataFrame(rows) if rows else None


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Daily Trading Volume vs Market Cap  (EUR)
# ─────────────────────────────────────────────────────────────────────────────

def make_figure1(label_suffix, start, end, period_note):
    print(f"  Figure 1{label_suffix} …")

    cg = coingecko_eur()

    # STOXX 50: average daily EUR volume from yfinance
    stoxx_vol = get_avg_daily_volume_eur(STOXX50_TICKERS, start, end)
    stoxx_mc  = get_market_cap_eur(STOXX50_TICKERS)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=FIG_DPI)

    # STOXX 50 black dots
    for label in set(stoxx_vol) & set(stoxx_mc):
        ax.scatter(stoxx_mc[label], stoxx_vol[label],
                   color=STOCK_COLOR, alpha=0.7, s=40, zorder=3)

    # Crypto red dots — exactly the 7 assets in the paper
    if cg is not None:
        for _, row in cg.iterrows():
            ax.scatter(row["market_cap"], row["total_volume"],
                       color=CRYPTO_COLOR, s=50, zorder=5)
            ax.annotate(row["label"], (row["market_cap"], row["total_volume"]),
                        fontsize=8, color=CRYPTO_COLOR,
                        xytext=(4, 3), textcoords="offset points")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Market Cap (Euro)", fontsize=11)
    ax.set_ylabel("Daily Trading Volume (Euro)", fontsize=11)
    ax.set_title(f"Trading Volume: EURO STOXX 50 Constit. Vs Top Crypto\n"
                 f"Black dots = EURO STOXX 50  |  Red dots = top crypto-assets\n"
                 f"Data is from {start} to {end}",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    fig.text(0.5, 0.02, CRYPTO_FOOTNOTE, ha="center", fontsize=7, color="gray",
             style="italic", wrap=True)
    path = f"{SAVE_DIR}/figure1_volume_vs_mktcap{label_suffix}.png"
    fig.savefig(path, dpi=FIG_DPI); plt.close(fig)
    print(f"    Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Amihud (2002) Illiquidity Ratio vs Market Cap (EUR)
# Amihud = avg(|daily return| / daily EUR volume)
# ─────────────────────────────────────────────────────────────────────────────

def make_figure2(label_suffix, start, end, period_note):
    print(f"  Figure 2{label_suffix} …")

    stoxx_tickers  = list(STOXX50_TICKERS.keys())
    crypto_tickers = list(CRYPTO_TICKERS.keys())

    stoxx_ohlc  = fetch_ohlc(stoxx_tickers,  start, end)
    crypto_ohlc = fetch_ohlc(crypto_tickers, start, end)
    stoxx_mc    = get_market_cap_eur(STOXX50_TICKERS)
    cg          = coingecko_eur()
    cg_mc       = dict(zip(cg["label"], cg["market_cap"])) if cg is not None else {}

    def amihud(close, volume, gbx=False, vol_in_eur=False):
        """Avg(|return| / EUR_volume).
        gbx=True: close is in pence, divide by 100.
        vol_in_eur=True: volume is already in EUR (crypto), don't multiply by close.
        """
        c = close.dropna()
        v = volume.reindex(c.index).dropna()
        c = c.reindex(v.index)
        if gbx:
            c = c / 100
        if vol_in_eur:
            eur_vol = v.replace(0, np.nan)
        else:
            eur_vol = (c * v).replace(0, np.nan)
        ret = c.pct_change().abs()
        ratio = (ret / eur_vol).dropna()
        return ratio.mean() if len(ratio) > 20 else None

    fig, ax = plt.subplots(figsize=(9, 6), dpi=FIG_DPI)

    s_close  = stoxx_ohlc.get("Close",  pd.DataFrame())
    s_volume = stoxx_ohlc.get("Volume", pd.DataFrame())
    for ticker, label in STOXX50_TICKERS.items():
        if ticker not in s_close.columns or label not in stoxx_mc:
            continue
        val = amihud(s_close[ticker], s_volume[ticker], gbx=ticker.endswith(".L"))
        if val and np.isfinite(val) and val > 0:
            ax.scatter(stoxx_mc[label], val, color=STOCK_COLOR, alpha=0.7, s=40, zorder=3)

    c_close  = crypto_ohlc.get("Close",  pd.DataFrame())
    c_volume = crypto_ohlc.get("Volume", pd.DataFrame())
    for ticker, label in CRYPTO_TICKERS.items():
        if ticker not in c_close.columns or label not in cg_mc:
            continue
        val = amihud(c_close[ticker], c_volume[ticker], vol_in_eur=True)
        if val and np.isfinite(val) and val > 0:
            ax.scatter(cg_mc[label], val, color=CRYPTO_COLOR, s=60, zorder=5)
            ax.annotate(label, (cg_mc[label], val), fontsize=8, color=CRYPTO_COLOR,
                        xytext=(4, 3), textcoords="offset points")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Market Cap (Euro)", fontsize=11)
    ax.set_ylabel("Amihud Illiquidity Ratio (|ret| / EUR volume)", fontsize=11)
    ax.set_title(f"Illiquidity: EURO STOXX 50 Constit. Vs Top Crypto\n"
                 f"Black dots = EURO STOXX 50  |  Red dots = top crypto-assets\n"
                 f"Data is from {start} to {end}",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    fig.text(0.5, 0.02, CRYPTO_FOOTNOTE, ha="center", fontsize=7, color="gray",
             style="italic", wrap=True)
    path = f"{SAVE_DIR}/figure2_amihud{label_suffix}.png"
    fig.savefig(path, dpi=FIG_DPI); plt.close(fig)
    print(f"    Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Daily Volatility (std of daily returns) vs Market Cap (EUR)
# ─────────────────────────────────────────────────────────────────────────────

def make_figure3(label_suffix, start, end, period_note):
    print(f"  Figure 3{label_suffix} …")

    stoxx_tickers = list(STOXX50_TICKERS.keys())
    crypto_tickers = list(CRYPTO_TICKERS.keys())

    stoxx_p  = fetch_prices(stoxx_tickers,  start, end)
    crypto_p = fetch_prices(crypto_tickers, start, end)
    stoxx_mc = get_market_cap_eur(STOXX50_TICKERS)

    # CoinGecko for crypto market cap in EUR
    cg = coingecko_eur()

    fig, ax = plt.subplots(figsize=(9, 6), dpi=FIG_DPI)

    # STOXX 50: daily vol = std(pct_change), in decimal form
    for ticker, label in STOXX50_TICKERS.items():
        if ticker not in stoxx_p.columns or label not in stoxx_mc:
            continue
        ret = stoxx_p[ticker].pct_change().dropna()
        if len(ret) < 20:
            continue
        daily_vol = ret.std()
        ax.scatter(stoxx_mc[label], daily_vol,
                   color=STOCK_COLOR, alpha=0.7, s=40, zorder=3)

    # Crypto: daily vol — use label column from coingecko_eur()
    if cg is not None:
        cg_mc = dict(zip(cg["label"], cg["market_cap"]))
    else:
        cg_mc = {}

    for ticker, label in CRYPTO_TICKERS.items():
        if ticker not in crypto_p.columns or label not in cg_mc:
            continue
        ret = crypto_p[ticker].pct_change().dropna()
        if len(ret) < 20:
            continue
        daily_vol = ret.std()
        ax.scatter(cg_mc[label], daily_vol, color=CRYPTO_COLOR, s=60, zorder=5)
        ax.annotate(label, (cg_mc[label], daily_vol), fontsize=8, color=CRYPTO_COLOR,
                    xytext=(4, 3), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Market Cap (Euro)", fontsize=11)
    ax.set_ylabel("Daily Volatility (std of daily returns)", fontsize=11)
    ax.set_title(f"1-year Volatility: EURO STOXX 50 Constit. Vs Top Crypto\n"
                 f"Black dots = EURO STOXX 50  |  Red dots = top crypto-assets\n"
                 f"Data is from {start} to {end}",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    fig.text(0.5, 0.02, CRYPTO_FOOTNOTE, ha="center", fontsize=7, color="gray",
             style="italic", wrap=True)
    path = f"{SAVE_DIR}/figure3_volatility{label_suffix}.png"
    fig.savefig(path, dpi=FIG_DPI); plt.close(fig)
    print(f"    Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — 1-year Correlation with EURO STOXX 50 vs Market Cap (EUR)
# ─────────────────────────────────────────────────────────────────────────────

def make_figure4(label_suffix, start, end, period_note):
    print(f"  Figure 4{label_suffix} …")

    stoxx_tickers  = list(STOXX50_TICKERS.keys())
    crypto_tickers = list(CRYPTO_TICKERS.keys())

    stoxx_p  = fetch_prices(stoxx_tickers,  start, end)
    crypto_p = fetch_prices(crypto_tickers, start, end)
    index_p  = fetch_prices([INDEX_TICKER], start, end)
    stoxx_mc = get_market_cap_eur(STOXX50_TICKERS)

    if INDEX_TICKER not in index_p.columns:
        print(f"    Index {INDEX_TICKER} unavailable — skipping."); return
    idx_ret = index_p[INDEX_TICKER].pct_change().dropna()

    cg = coingecko_eur()
    cg_mc = dict(zip(cg["label"], cg["market_cap"])) if cg is not None else {}

    fig, ax = plt.subplots(figsize=(9, 6), dpi=FIG_DPI)

    # STOXX 50
    for ticker, label in STOXX50_TICKERS.items():
        if ticker not in stoxx_p.columns or label not in stoxx_mc:
            continue
        ret = stoxx_p[ticker].pct_change().dropna()
        common = ret.index.intersection(idx_ret.index)
        if len(common) < 30:
            continue
        corr = ret.loc[common].corr(idx_ret.loc[common])
        if pd.notna(corr):
            ax.scatter(stoxx_mc[label], corr,
                       color=STOCK_COLOR, alpha=0.7, s=40, zorder=3)

    # Crypto
    for ticker, label in CRYPTO_TICKERS.items():
        if ticker not in crypto_p.columns or label not in cg_mc:
            continue
        ret = crypto_p[ticker].pct_change().dropna()
        common = ret.index.intersection(idx_ret.index)
        if len(common) < 30:
            continue
        corr = ret.loc[common].corr(idx_ret.loc[common])
        mc   = cg_mc[label]
        if pd.notna(corr):
            ax.scatter(mc, corr, color=CRYPTO_COLOR, s=60, zorder=5)
            ax.annotate(label, (mc, corr), fontsize=8, color=CRYPTO_COLOR,
                        xytext=(4, 3), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_xlabel("Market Cap (Euro)", fontsize=11)
    ax.set_ylabel("Correlation Coefficient", fontsize=11)
    ax.set_title(f"1-year Correlation with EURO STOXX 50\n"
                 f"Black dots = EURO STOXX 50  |  Red dots = top crypto-assets\n"
                 f"Data is from {start} to {end}",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    fig.text(0.5, 0.02, CRYPTO_FOOTNOTE, ha="center", fontsize=7, color="gray",
             style="italic", wrap=True)
    path = f"{SAVE_DIR}/figure4_correlation{label_suffix}.png"
    fig.savefig(path, dpi=FIG_DPI); plt.close(fig)
    print(f"    Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.chdir(SAVE_DIR)

    print("=" * 60)
    print(f"Generating figures  ({EXT_START} – {EXT_END})")
    print("=" * 60)
    make_figure1("", EXT_START, EXT_END, f"Data is from {EXT_START} to {EXT_END}")
    make_figure2("", EXT_START, EXT_END, f"Data is from {EXT_START} to {EXT_END}")
    make_figure3("", EXT_START, EXT_END, f"Data is from {EXT_START} to {EXT_END}")
    make_figure4("", EXT_START, EXT_END, f"Data is from {EXT_START} to {EXT_END}")

    print()
    print("Done. Output files:")
    for f in sorted(os.listdir(SAVE_DIR)):
        if f.startswith("figure") and f.endswith(".png"):
            print(f"  {f}")
