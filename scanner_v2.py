# ============================================================
# KORIVA CRYPTO SCANNER V2.1
# Binance USDT-M Perpetual Futures
# Public API - No API key required
# ============================================================
#
# SCORE SEMANTICS
# ---------------
# The live score (calculate_live_score) is an ACTIVITY /
# CONFLUENCE score, not a directional prediction score.
# Momentum enters the score as abs(momentum_1h), so +5% and
# -5% contribute the same activity points. Direction is
# handled separately in classify_market() and, for the
# backtest, in classify_direction(). This is intentional and
# left unchanged so as not to silently redefine the scanner.
#
# The backtest score (calculate_backtest_score) is a
# DIFFERENT scale because historical OI and funding are not
# available via the Binance public API. Only momentum,
# relative volume, trend and RSI are used. The maximum is
# normalized to 100 for comparability but must not be
# interpreted as identical to the live score.
# ============================================================

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests


# ============================================================
# CONSTANTS
# ============================================================

# BASE_URL resolution order:
#   1. BINANCE_API_BASE  (a reverse proxy in front of Binance,
#      e.g. a Cloudflare Worker, useful when the runner's IP is
#      geo-blocked by Binance with HTTP 451)
#   2. https://fapi.binance.com (default, direct connection)
#
# A reverse-proxy base URL is different from an HTTP forward
# proxy: here the *hostname itself* changes, and the proxy
# server is expected to forward the request path/query as-is
# to fapi.binance.com and return the raw response body.
BASE_URL = os.environ.get("BINANCE_API_BASE", "https://fapi.binance.com").rstrip("/")

TOP_N = 20
MIN_VOLUME_USDT = 5_000_000
MARKET_ANALYSIS_LIMIT = 60
KLINES_LIMIT = 120

BACKTEST_KLINES = 1000   # ~10 days of 15m candles
BACKTEST_HORIZON = 96    # 96 * 15m = 24h forward
BACKTEST_STEP = 4        # 1h between samples

# OI history window: 5m period x 6 samples.
# first-to-last span = (6 - 1) * 5m = 25 minutes (not 30m).
OI_WINDOW_PERIOD = "5m"
OI_WINDOW_LIMIT = 6
OI_WINDOW_SPAN_MIN = (OI_WINDOW_LIMIT - 1) * 5   # 25

# Global verbosity (set by argparse)
VERBOSE = True


def log(msg="", end="\n"):
    """Progress / info messages -> stderr (keeps --json stdout clean)."""
    if VERBOSE:
        print(msg, end=end, file=sys.stderr, flush=True)


# ============================================================
# HTTP SESSION
# ============================================================
#
# NOTE ON REVERSE-PROXY MODE:
# When BINANCE_API_BASE is set (e.g. to a Cloudflare Worker
# URL), requests are sent directly to that host instead of
# fapi.binance.com, and no HTTP forward-proxy is configured on
# the session. This is the correct setup for a Worker-style
# reverse proxy, which listens on its own hostname and forwards
# the request server-side.
#
# A classic HTTP/SOCKS forward proxy (BINANCE_PROXY /
# HTTPS_PROXY / HTTP_PROXY) is still supported as an
# alternative and takes effect only when BINANCE_API_BASE is
# NOT set, since the two approaches are mutually exclusive.

session = requests.Session()
session.headers.update({"User-Agent": "KORIVA-Crypto-Scanner/2.1"})

USING_API_BASE_OVERRIDE = "BINANCE_API_BASE" in os.environ

if USING_API_BASE_OVERRIDE:
    log(f"[KORIVA] Using reverse-proxy base URL -> {BASE_URL}")
else:
    PROXY_URL = (
        os.environ.get("BINANCE_PROXY")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("http_proxy")
    )
    if PROXY_URL:
        session.proxies = {"http": PROXY_URL, "https": PROXY_URL}
        # Never print credentials: keep only the part after '@' if present.
        _safe_proxy = PROXY_URL.split("@")[-1]
        log(f"[KORIVA] Proxy enabled -> {_safe_proxy}")
    else:
        log("[KORIVA] No proxy configured (using direct connection).")


def get_json(endpoint, params=None, max_retries=5):
    """
    GET request with automatic retry on:
      - network errors (Timeout / ConnectionError)
      - HTTP 429 (rate limit) -> respects Retry-After header
      - HTTP 5xx (server errors)

    HTTP 418 (IP ban) and HTTP 451 (geo-block) are fatal and
    raised immediately with an explicit message.

    On final failure, raises HTTPError with a clear description
    of the last observed error (never 'last_exc = None').
    """
    url = BASE_URL + endpoint
    last_error_desc = None

    for attempt in range(max_retries):

        # ----- Network layer -----
        try:
            response = session.get(url, params=params, timeout=15)
        except (requests.Timeout, requests.ConnectionError) as e:
            last_error_desc = f"{type(e).__name__}: {e}"
            wait = min(2 ** attempt, 30)
            log(f"\n[retry] {endpoint}: network error -> wait {wait}s")
            time.sleep(wait)
            continue

        # ----- Rate limit -----
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else min(2 ** attempt, 30)
            last_error_desc = "HTTP 429 Too Many Requests"
            log(f"\n[retry] {endpoint}: 429 rate limit -> wait {wait:.1f}s")
            time.sleep(wait)
            continue

        # ----- IP ban (fatal) -----
        if response.status_code == 418:
            raise requests.HTTPError(
                f"418 IP banned by Binance for {endpoint}: "
                f"{response.text[:200]}"
            )

        # ----- Geo-block (fatal) -----
        # Binance returns 451 from US IPs (GitHub Actions runners
        # are US-based). A reverse-proxy base URL (BINANCE_API_BASE)
        # or a classic proxy is required.
        if response.status_code == 451:
            raise requests.HTTPError(
                f"451 Geo-blocked by Binance for {endpoint}: "
                f"this IP is in a restricted region (HTTP 451). "
                f"Set BINANCE_API_BASE to a reverse proxy (e.g. a "
                f"Cloudflare Worker) in a non-US region, or configure "
                f"BINANCE_PROXY with a non-US proxy. "
                f"Body: {response.text[:200]}"
            )

        # ----- Server errors -----
        if 500 <= response.status_code < 600:
            last_error_desc = f"HTTP {response.status_code} {response.reason}"
            wait = min(2 ** attempt, 30)
            log(f"\n[retry] {endpoint}: {response.status_code} -> wait {wait}s")
            time.sleep(wait)
            continue

        # ----- Other 4xx -----
        if not response.ok:
            raise requests.HTTPError(
                f"{endpoint} failed with HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )

        return response.json()

    # If we get here, we exhausted retries on transient errors.
    raise requests.HTTPError(
        f"Max retries ({max_retries}) exceeded for {endpoint}; "
        f"last error: {last_error_desc or 'unknown'}"
    )


# ============================================================
# BINANCE DATA
# ============================================================

def get_futures_tickers():
    return get_json("/fapi/v1/ticker/24hr")


def get_exchange_info():
    return get_json("/fapi/v1/exchangeInfo")


def get_klines(symbol, interval="15m", limit=KLINES_LIMIT):
    return get_json(
        "/fapi/v1/klines",
        {"symbol": symbol, "interval": interval, "limit": limit}
    )


def get_funding_rate(symbol):
    """
    Latest funding rate (%) or None if unavailable.
    Returning None (not 0.0) is important: missing data must
    contribute 0 score points, not the 'low funding' bonus.
    """
    try:
        data = get_json(
            "/fapi/v1/fundingRate",
            {"symbol": symbol, "limit": 1}
        )
        if not data:
            return None
        return float(data[-1]["fundingRate"]) * 100
    except Exception:
        return None


def get_open_interest(symbol):
    """Open interest or None if unavailable."""
    try:
        data = get_json("/fapi/v1/openInterest", {"symbol": symbol})
        return float(data["openInterest"])
    except Exception:
        return None


def get_open_interest_history(symbol):
    """
    Recent OI % change over the fetched window.

    NOTE ON WINDOW SPAN:
        period = "5m", limit = 6.
        The first-to-last observation therefore spans
        (6 - 1) * 5 = 25 minutes, NOT 30 minutes.
        The internal key is still "oi_change"; the display
        label makes the ~25m span explicit.

    Returns None (not 0.0) when unavailable so missing data
    contributes 0 score points instead of a neutral value.
    """
    try:
        data = get_json(
            "/futures/data/openInterestHist",
            {"symbol": symbol, "period": OI_WINDOW_PERIOD,
             "limit": OI_WINDOW_LIMIT}
        )
        if len(data) < 2:
            return None
        old_oi = float(data[0]["sumOpenInterest"])
        new_oi = float(data[-1]["sumOpenInterest"])
        if old_oi <= 0:
            return None
        return (new_oi - old_oi) / old_oi * 100
    except Exception:
        return None


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calculate_rsi(close, period=14):
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.rolling(period).mean()
    avg_loss = losses.rolling(period).mean()

    # Replace 0 with NaN so we don't divide by zero; then mask
    # the case where there were only gains (RSI should be 100).
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)

    v = rsi.iloc[-1]
    if pd.isna(v):
        return 50.0
    return float(v)


def calculate_ema(close, period):
    return float(close.ewm(span=period, adjust=False).mean().iloc[-1])


def calculate_atr(df, period=14):
    """
    ATR is computed and displayed for context, but is NOT used
    in either live or backtest scoring. Rationale:
      - ATR measures volatility magnitude, not direction or
        confluence.
      - Momentum and relative volume already implicitly reward
        volatility expansion.
      - ATR thresholds are regime-dependent (a 'high' ATR% for
        BTC is normal for a small altcoin), making fixed
        thresholds arbitrary.
    Adding ATR to the score would double-count and add weight
    without a clear theoretical justification.
    """
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    v = atr.iloc[-1]
    if pd.isna(v):
        return 0.0
    return float(v)


# ============================================================
# KLINES PARSING + SHARED FEATURE ENGINEERING
# ============================================================

KLINES_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]

NUMERIC_COLUMNS = [
    "open", "high", "low", "close", "volume", "quote_volume",
]


def parse_klines(raw):
    df = pd.DataFrame(raw, columns=KLINES_COLUMNS)
    for c in NUMERIC_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def compute_features(df, closed_idx):
    """
    Compute all kline-based features using ONLY candles up to
    and including index `closed_idx`.

    Time-alignment contract (single source of truth for both
    the live scanner AND the backtest):

      * `closed_idx` is the position (into `df`) of the LAST
        FULLY CLOSED candle available at signal time.
      * No data after `closed_idx` is ever used to compute any
        indicator. No look-ahead bias.

      * Relative Volume (identical rule in live and backtest):
            latest_volume  = quote_volume at `closed_idx`
            baseline       = mean of the 20 CLOSED candles
                             strictly BEFORE `closed_idx`
                             (the measured candle is NOT
                              included in its own average).

      * Momentum:
            1H = close[closed_idx] vs close[closed_idx - 4]
            4H = close[closed_idx] vs close[closed_idx - 16]

    Live scanner call:
        df    = raw klines (last row = in-progress candle)
        closed_idx = -2      (last fully closed candle)

    Backtest call at signal index i:
        closed_idx = i       (candle i is the last closed one)
    """
    n = len(df)

    # Allow negative indexing for the live path.
    if closed_idx < 0:
        closed_idx = n + closed_idx

    # Need at least 60 candles for meaningful EMA50 and RSI, and
    # closed_idx must point to an existing candle in df.
    if closed_idx < 60 or closed_idx >= n:
        return None

    # Window includes the closed candle as its last row.
    window = df.iloc[: closed_idx + 1]
    close = window["close"]

    current_price = float(close.iloc[-1])

    ema20 = calculate_ema(close, 20)
    ema50 = calculate_ema(close, 50)
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(window, 14)
    atr_percent = atr / current_price * 100 if current_price > 0 else 0.0

    # 1H momentum (4 x 15m candles)
    price_1h = float(close.iloc[-5])
    momentum_1h = (
        (current_price - price_1h) / price_1h * 100
        if price_1h > 0 else 0.0
    )

    # 4H momentum (16 x 15m candles)
    price_4h = float(close.iloc[-17])
    momentum_4h = (
        (current_price - price_4h) / price_4h * 100
        if price_4h > 0 else 0.0
    )

    # Relative Volume: last closed candle vs the 20 closed
    # candles strictly BEFORE it.
    latest_vol = float(window["quote_volume"].iloc[-1])
    baseline_vol = float(window["quote_volume"].iloc[-21:-1].mean())

    if baseline_vol > 0:
        relative_volume = latest_vol / baseline_vol
    else:
        relative_volume = 0.0

    # 20-candle price range / volatility (informational)
    recent_high = float(window["high"].iloc[-20:].max())
    recent_low = float(window["low"].iloc[-20:].min())
    range_percent = (
        (recent_high - recent_low) / recent_low * 100
        if recent_low > 0 else 0.0
    )

    return {
        "price": current_price,
        "ema20": ema20,
        "ema50": ema50,
        "rsi": rsi,
        "atr": atr,
        "atr_percent": atr_percent,
        "momentum_1h": momentum_1h,
        "momentum_4h": momentum_4h,
        "relative_volume": relative_volume,
        "range_percent": range_percent,
    }


def analyze_klines(symbol):
    """Live-scan wrapper: features on the last fully closed candle."""
    try:
        raw = get_klines(symbol)
        if len(raw) < 60:
            return None
        df = parse_klines(raw)
        return compute_features(df, closed_idx=-2)
    except Exception as e:
        log(f"\nTechnical analysis failed for {symbol}: {e}")
        return None


# ============================================================
# TRADABLE SYMBOLS
# ============================================================

def get_tradable_symbols():
    exchange_info = get_exchange_info()
    symbols = set()

    excluded = {
        "USDC", "FDUSD", "TUSD", "USDP",
        "DAI", "USDE", "USDS", "BUSD",
    }

    for item in exchange_info["symbols"]:
        if item.get("quoteAsset") != "USDT":
            continue
        if item.get("status") != "TRADING":
            continue
        if item.get("contractType") != "PERPETUAL":
            continue
        if item.get("baseAsset") in excluded:
            continue
        symbols.add(item["symbol"])

    return symbols


# ============================================================
# SHARED SUBSCORES (momentum / RV / trend / RSI)
# ============================================================
#
# These four components are computed identically for the live
# score and for the backtest score. Only the OI and funding
# components differ (available live, not available in the
# current backtest).

def _subscore_momentum(row):
    # NOTE: abs(momentum_1h) on purpose. This is an ACTIVITY
    # score, not a direction score.
    m = abs(row["momentum_1h"])
    if m >= 8:   return 20
    if m >= 5:   return 16
    if m >= 3:   return 12
    if m >= 1.5: return 8
    if m >= 0.5: return 4
    return 0


def _subscore_relative_volume(row):
    rv = row["relative_volume"]
    if rv >= 3:   return 20
    if rv >= 2:   return 16
    if rv >= 1.5: return 12
    if rv >= 1.2: return 8
    if rv >= 1.0: return 4
    return 0


def _subscore_trend(row):
    price = row["price"]
    ema20 = row["ema20"]
    ema50 = row["ema50"]

    bullish = price > ema20 and ema20 > ema50
    bearish = price < ema20 and ema20 < ema50

    if bullish or bearish:
        return 20
    if ema20 > 0 and abs(price - ema20) / ema20 > 0.005:
        return 8
    return 0


def _subscore_rsi(row):
    rsi = row["rsi"]
    if 45 <= rsi <= 65:  return 15
    if 35 <= rsi < 45:   return 10
    if 65 < rsi <= 75:   return 10
    if 30 <= rsi < 35:   return 6
    if 75 < rsi <= 80:   return 5
    return 2


# Max achievable from these four: 20 + 20 + 20 + 15 = 75
BACKTEST_SUBSCORE_MAX = 75


# ============================================================
# LIVE SCORE  (activity / confluence)
# ============================================================

def calculate_live_score(row):
    """
    Live score /100:
        momentum (20) + RV (20) + trend (20) + RSI (15)
        + OI (15) + funding (10)

    Missing OI or funding contributes 0 points (NOT a neutral
    value that would trigger the 'low funding' bonus).
    """
    score = 0
    score += _subscore_momentum(row)
    score += _subscore_relative_volume(row)
    score += _subscore_trend(row)
    score += _subscore_rsi(row)

    # ----- OI component (max 15) -----
    oi_change = row.get("oi_change")
    if oi_change is not None:
        a = abs(oi_change)
        if a >= 5:   score += 15
        elif a >= 3: score += 12
        elif a >= 2: score += 9
        elif a >= 1: score += 5

    # ----- Funding component (max 10) -----
    funding = row.get("funding")
    if funding is not None:
        f = abs(funding)
        if f < 0.01:   score += 10
        elif f < 0.03: score += 8
        elif f < 0.05: score += 5
        elif f < 0.08: score += 3
        else:          score += 1

    return min(round(score), 100)


# ============================================================
# BACKTEST SCORE  (no OI, no funding)
# ============================================================

def calculate_backtest_score(row):
    """
    Backtest score on a 0-100 scale, built ONLY from features
    genuinely available historically:
        momentum (20) + RV (20) + trend (20) + RSI (15)
    The 75-point sum is rescaled to 100.

    This is NOT the same as the live score:
      - Historical OI is not available via the public API.
      - Historical funding is not available via the public API.
    So the backtest intentionally omits those two components
    rather than faking them with 0.0, which would have given
    every sample the 'low funding' bonus.
    """
    sub = (
        _subscore_momentum(row)
        + _subscore_relative_volume(row)
        + _subscore_trend(row)
        + _subscore_rsi(row)
    )
    return int(round(sub * 100 / BACKTEST_SUBSCORE_MAX))


# ============================================================
# DIRECTION CLASSIFICATION
# ============================================================

def classify_direction(features):
    """
    Conservative historical direction, using only features
    available at signal time.

    BULLISH: price > EMA20 > EMA50, 1H mom > 0, 4H mom >= 0
    BEARISH: price < EMA20 < EMA50, 1H mom < 0, 4H mom <= 0
    else   : NEUTRAL
    """
    price = features["price"]
    ema20 = features["ema20"]
    ema50 = features["ema50"]
    m1 = features["momentum_1h"]
    m4 = features["momentum_4h"]

    bullish = (
        price > ema20 and ema20 > ema50
        and m1 > 0 and m4 >= 0
    )
    bearish = (
        price < ema20 and ema20 < ema50
        and m1 < 0 and m4 <= 0
    )

    if bullish:
        return "BULLISH"
    if bearish:
        return "BEARISH"
    return "NEUTRAL"


def classify_market(row):
    """
    Live market classification. Unchanged from V2.1 semantics,
    except OI checks guard against None.
    """
    change = row["change"]
    momentum = row["momentum_1h"]
    rsi = row["rsi"]
    oi = row.get("oi_change") or 0.0
    rv = row["relative_volume"]
    price = row["price"]
    ema20 = row["ema20"]
    ema50 = row["ema50"]

    if abs(change) >= 20:
        return "EXTREME MOVE"

    if momentum >= 3 and rv >= 1.5 and oi >= 1:
        return "BULLISH EXPANSION"

    if momentum <= -3 and rv >= 1.5 and oi >= 1:
        return "BEARISH EXPANSION"

    if momentum > 2 and oi < -1:
        return "RALLY + OI FALL"

    if momentum < -2 and oi < -1:
        return "SELL-OFF + OI FALL"

    if price > ema20 and ema20 > ema50:
        if rsi > 75:
            return "UPTREND / OVERBOUGHT"
        return "UPTREND"

    if price < ema20 and ema20 < ema50:
        if rsi < 30:
            return "DOWNTREND / OVERSOLD"
        return "DOWNTREND"

    if rsi > 80:
        return "RSI EXTREME HIGH"

    if rsi < 25:
        return "RSI EXTREME LOW"

    if rv >= 2:
        return "VOLUME EXPANSION"

    return "NEUTRAL"


# ============================================================
# LIVE SCANNER
# ============================================================

def scan():
    log()
    log("=" * 120)
    log("             KORIVA CRYPTO MARKET SCANNER V2.1")
    log("             Binance USDT-M PERPETUAL FUTURES")
    log("=" * 120)
    log("Scan time: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log("\nLoading Binance Futures markets...")

    tickers = get_futures_tickers()
    tradable = get_tradable_symbols()

    candidates = []
    for ticker in tickers:
        symbol = ticker["symbol"]
        if symbol not in tradable:
            continue
        try:
            volume = float(ticker["quoteVolume"])
            change = float(ticker["priceChangePercent"])
            if volume < MIN_VOLUME_USDT:
                continue
            candidates.append({
                "symbol": symbol,
                "volume": volume,
                "change": change,
            })
        except Exception:
            continue

    candidates.sort(key=lambda x: x["volume"], reverse=True)
    candidates = candidates[:MARKET_ANALYSIS_LIMIT]

    log(f"Liquid markets selected: {len(candidates)}")
    log("Calculating technical + Futures data...")

    results = []
    for index, candidate in enumerate(candidates, start=1):
        symbol = candidate["symbol"]
        log(f"\rAnalyzing {index:02d}/{len(candidates)} {symbol:<18}", end="")

        technical = analyze_klines(symbol)
        if technical is None:
            continue

        funding = get_funding_rate(symbol)
        oi = get_open_interest(symbol)
        oi_change = get_open_interest_history(symbol)

        row = {
            "symbol": symbol,
            "change": candidate["change"],
            "volume": candidate["volume"],
            "funding": funding,
            "open_interest": oi,
            "oi_change": oi_change,
            **technical,
        }

        row["score"] = calculate_live_score(row)
        row["classification"] = classify_market(row)
        results.append(row)

        time.sleep(0.08)

    log("\n")

    df = pd.DataFrame(results)
    if df.empty:
        log("No suitable markets found.")
        return None

    df.sort_values("score", ascending=False, inplace=True)
    return df.head(TOP_N)


# ============================================================
# BACKTEST
# ============================================================

def backtest(symbols, horizon=BACKTEST_HORIZON, step=BACKTEST_STEP):
    """
    Walk-forward backtest with strict time alignment.

    For each symbol:
      for signal index i in [60, n - horizon - 1]:
          features  = compute_features(df, closed_idx=i)
              (uses df[0..i] only; candle i is the last closed one)
          signal_time = df["close_time"].iloc[i]
              (moment candle i closes, i.e. the signal moment)
          entry     = close[i]           (signal candle close)
          exit      = close[i+horizon]   (exactly `horizon` candles later)
          forward   = (exit - entry) / entry * 100
          direction = classify_direction(features)
          signal    = forward   if BULLISH
                      -forward  if BEARISH
                      0         if NEUTRAL
          score     = calculate_backtest_score(features)

    No future data is used to compute indicators, direction or
    score. Entry stays at the signal candle close: this is a
    research baseline, not a conservative execution simulator
    (no slippage, no fees, no next-open entry).
    """
    records = []

    log()
    log("=" * 100)
    log(f"  BACKTEST - {len(symbols)} symbols, "
        f"horizon {horizon} candles (~{horizon * 15 / 60:.0f}h), "
        f"step {step} candles")
    log("=" * 100)

    for idx, symbol in enumerate(symbols, start=1):
        log(f"\rBacktesting {idx:02d}/{len(symbols)} {symbol:<18}", end="")

        try:
            raw = get_klines(symbol, interval="15m", limit=BACKTEST_KLINES)
            if len(raw) < 100:
                continue

            df = parse_klines(raw)
            n = len(df)

            for i in range(60, n - horizon, step):

                features = compute_features(df, closed_idx=i)
                if features is None:
                    continue

                # ----- Signal timestamp: candle i close -----
                signal_time = int(df["close_time"].iloc[i])

                # ----- Entry: signal candle close (unchanged) -----
                entry_price = float(df["close"].iloc[i])

                # ----- Exit: exactly `horizon` candles later -----
                exit_price = float(df["close"].iloc[i + horizon])
                if entry_price <= 0:
                    continue

                forward_return = (exit_price - entry_price) / entry_price * 100

                direction = classify_direction(features)
                if direction == "BULLISH":
                    signal_return = forward_return
                elif direction == "BEARISH":
                    signal_return = -forward_return
                else:
                    signal_return = 0.0

                score = calculate_backtest_score(features)

                records.append({
                    "symbol": symbol,
                    "signal_time": signal_time,
                    "score": score,
                    "direction": direction,
                    "forward_return": forward_return,
                    "signal_return": signal_return,
                    "momentum_1h": features["momentum_1h"],
                    "relative_volume": features["relative_volume"],
                    "rsi": features["rsi"],
                })

            time.sleep(0.1)

        except Exception as e:
            log(f"\nBacktest failed for {symbol}: {e}")
            continue

    log("\n")
    return pd.DataFrame(records)


def _bucket_stats(sub, label):
    if sub.empty:
        return {
            "bucket": label,
            "count": 0,
            "avg_signal_return": None,
            "median_signal_return": None,
            "win_rate": None,
            "std": None,
        }
    return {
        "bucket": label,
        "count": int(len(sub)),
        "avg_signal_return": float(sub["signal_return"].mean()),
        "median_signal_return": float(sub["signal_return"].median()),
        "win_rate": float((sub["signal_return"] > 0).mean() * 100),
        "std": (
            float(sub["signal_return"].std()) if len(sub) > 1 else 0.0
        ),
    }


def summarize_backtest(bt_df, horizon):
    """Return a dict with directional and bucket statistics."""
    if bt_df.empty:
        return None

    df = bt_df.copy()

    bull = df[df["direction"] == "BULLISH"]
    bear = df[df["direction"] == "BEARISH"]
    neut = df[df["direction"] == "NEUTRAL"]
    directional = df[df["direction"].isin(["BULLISH", "BEARISH"])]

    def _stats(sub, col="signal_return"):
        if sub.empty:
            return {
                "count": 0,
                "avg": None,
                "median": None,
                "win_rate": None,
                "std": None,
            }
        return {
            "count": int(len(sub)),
            "avg": float(sub[col].mean()),
            "median": float(sub[col].median()),
            "win_rate": float((sub[col] > 0).mean() * 100),
            "std": float(sub[col].std()) if len(sub) > 1 else 0.0,
        }

    # Score buckets on directional samples only (neutral samples
    # have signal_return = 0 by construction and would bias buckets).
    bins = [0, 20, 40, 60, 80, 100]
    labels = ["0-20", "20-40", "40-60", "60-80", "80-100"]

    buckets = []
    if not directional.empty:
        directional = directional.copy()
        directional["bucket"] = pd.cut(
            directional["score"],
            bins=bins,
            labels=labels,
            include_lowest=True,
        )
        for label in labels:
            buckets.append(
                _bucket_stats(directional[directional["bucket"] == label],
                              label)
            )
    else:
        for label in labels:
            buckets.append(_bucket_stats(pd.DataFrame(), label))

    # Correlation between score and directional signal return.
    corr = None
    if len(directional) >= 3:
        c = directional["score"].corr(directional["signal_return"])
        if pd.notna(c):
            corr = float(c)

    return {
        "horizon_candles": horizon,
        "horizon_hours": horizon * 15 / 60,
        "total_samples": int(len(df)),
        "bullish_samples": int(len(bull)),
        "bearish_samples": int(len(bear)),
        "neutral_samples": int(len(neut)),
        "bullish_stats": _stats(bull),
        "bearish_stats": _stats(bear),
        "directional_stats": _stats(directional),
        "correlation_score_vs_directional_return": corr,
        "score_buckets": buckets,
        "signal_time_field": "close_time",
        "notes": (
            "Backtest score is NOT the live score: historical OI "
            "and funding are not available via the public Binance "
            "API, so those two components are intentionally omitted. "
            "Entry is taken at the signal candle close (close[i]); "
            "no slippage, fees, spread or latency are modelled."
        ),
    }


# ============================================================
# DISPLAY
# ============================================================

def _fmt(v, width, prec=2, default="n/a"):
    if v is None:
        return f"{default:>{width}}"
    return f"{v:>{width}.{prec}f}"


def display(df):
    print("=" * 145)
    print("                         TOP MARKET CANDIDATES")
    print("=" * 145)

    for index, (_, row) in enumerate(df.iterrows(), start=1):
        change_arrow = "^" if row["change"] >= 0 else "v"
        momentum_arrow = "^" if row["momentum_1h"] >= 0 else "v"
        oi_str = (
            f"{row['oi_change']:>6.2f}%"
            if row.get("oi_change") is not None else f"{'n/a':>7}"
        )
        f_str = (
            f"{row['funding']:>7.3f}%"
            if row.get("funding") is not None else f"{'n/a':>8}"
        )

        print(
            f"{index:02d}. "
            f"{row['symbol']:<16} "
            f"{change_arrow}{row['change']:>7.2f}%  "
            f"1H {momentum_arrow}{row['momentum_1h']:>6.2f}%  "
            f"4H {row['momentum_4h']:>6.2f}%  "
            f"RSI {row['rsi']:>5.1f}  "
            f"RV {row['relative_volume']:>4.1f}x  "
            f"OI \u0394~{OI_WINDOW_SPAN_MIN}m {oi_str}  "
            f"F {f_str}  "
            f"S {row['score']:>3.0f}  "
            f"{row['classification']}"
        )

    print("=" * 145)
    print()
    print("LEGEND")
    print("-" * 50)
    print("24H      = Binance Futures 24-hour price change")
    print("1H       = Approximate 1-hour momentum")
    print("4H       = Approximate 4-hour momentum")
    print("RSI      = 14-period RSI on 15-minute candles")
    print("RV       = Latest CLOSED 15m candle / avg of previous 20 closed")
    print(f"OI \u0394~{OI_WINDOW_SPAN_MIN}m = Approx. {OI_WINDOW_SPAN_MIN}-min Open Interest change "
          f"({OI_WINDOW_PERIOD} x {OI_WINDOW_LIMIT} samples), n/a if unavailable")
    print("F        = Latest Funding Rate (n/a if unavailable)")
    print("S        = ACTIVITY score /100 (not a directional prediction)")
    print("=" * 145)
    print("This scanner is for market research.")
    print("It does not predict price direction and does not execute trades.")
    print("=" * 145)


def display_backtest(summary):
    if summary is None:
        print("\nBacktest: no data.")
        return

    print()
    print("=" * 100)
    print(f"  BACKTEST - score vs directional forward return "
          f"(horizon {summary['horizon_candles']} candles "
          f"= {summary['horizon_hours']:.0f}h)")
    print("=" * 100)

    print(f"Total samples     : {summary['total_samples']}")
    print(f"  Bullish samples : {summary['bullish_samples']}")
    print(f"  Bearish samples : {summary['bearish_samples']}")
    print(f"  Neutral samples : {summary['neutral_samples']}")
    print(f"Signal timestamp  : {summary['signal_time_field']} "
          f"(candle close)")

    print()
    print("DIRECTIONAL STATISTICS")
    print("-" * 100)
    print(f"{'Group':<14} {'N':>7} {'Avg ret%':>10} {'Med ret%':>10} "
          f"{'Win%':>7} {'Std%':>7}")
    print("-" * 100)

    for name, key in [
        ("Bullish", "bullish_stats"),
        ("Bearish", "bearish_stats"),
        ("Directional", "directional_stats"),
    ]:
        s = summary[key]
        print(
            f"{name:<14} "
            f"{s['count']:>7} "
            f"{_fmt(s['avg'], 10)} "
            f"{_fmt(s['median'], 10)} "
            f"{_fmt(s['win_rate'], 7, 1)} "
            f"{_fmt(s['std'], 7)}"
        )

    print()
    print("BY SCORE BUCKET (directional samples only)")
    print("-" * 100)
    print(f"{'Score bucket':<15} {'N':>7} {'Avg ret%':>10} "
          f"{'Med ret%':>10} {'Win%':>7} {'Std%':>7}")
    print("-" * 100)

    for b in summary["score_buckets"]:
        print(
            f"{b['bucket']:<15} "
            f"{b['count']:>7} "
            f"{_fmt(b['avg_signal_return'], 10)} "
            f"{_fmt(b['median_signal_return'], 10)} "
            f"{_fmt(b['win_rate'], 7, 1)} "
            f"{_fmt(b['std'], 7)}"
        )

    corr = summary["correlation_score_vs_directional_return"]
    print()
    if corr is None:
        print("Pearson correlation (score vs directional return): n/a")
    else:
        print(f"Pearson correlation (score vs directional return): "
              f"{corr:.3f}")

    print()
    print("=" * 100)
    print("NOTE: backtest score is NOT the live score. Historical OI")
    print("and funding are not available via the public Binance API,")
    print("so those two components are intentionally omitted.")
    print("Entry is the signal candle close (close[i]); no slippage,")
    print("fees, spread or latency are modelled.")
    print("=" * 100)
    print("A high score is not proof of future profitability.")
    print("This backtest is a research tool, not a trading system.")
    print("=" * 100)


# ============================================================
# JSON SANITIZATION
# ============================================================

def sanitize(obj):
    """Recursively convert numpy / pandas / NaN / Inf to JSON-safe values."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if obj is None or obj is pd.NA or obj is pd.NaT:
        return None
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def build_json_output(df, bt_summary=None):
    output = {
        "scan_time": datetime.now().isoformat(timespec="seconds"),
        "top_n": TOP_N,
        "min_volume_usdt": MIN_VOLUME_USDT,
        "score_type": "activity_confluence",
        "oi_change_window": {
            "period": OI_WINDOW_PERIOD,
            "samples": OI_WINDOW_LIMIT,
            "approx_span_minutes": OI_WINDOW_SPAN_MIN,
            "note": (
                "first-to-last observation span = "
                "(samples - 1) * period = 25 minutes"
            ),
        },
        "results": df.to_dict(orient="records"),
    }
    if bt_summary is not None:
        output["backtest"] = bt_summary
    return output


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="KORIVA Crypto Scanner V2.1 - Binance USDT-M Perpetuals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scanner_v2.py\n"
            "  python scanner_v2.py --json | jq .\n"
            "  python scanner_v2.py --backtest\n"
            "  python scanner_v2.py --json --backtest --backtest-top 10 "
            "--horizon 96\n"
        ),
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON (progress goes to stderr).",
    )
    parser.add_argument(
        "--backtest", action="store_true",
        help="Run a directional forward-return backtest on the top symbols.",
    )
    parser.add_argument(
        "--backtest-top", type=int, default=10,
        help="Number of top scanned symbols to backtest (default: 10).",
    )
    parser.add_argument(
        "--horizon", type=int, default=BACKTEST_HORIZON,
        help="Forward horizon in 15m candles (default: 96 = 24h).",
    )
    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    global VERBOSE
    args = parse_args()
    VERBOSE = not args.json

    try:
        dataframe = scan()

        if dataframe is None:
            if args.json:
                print(json.dumps(
                    {"error": "No suitable markets found."},
                    indent=2,
                ))
            sys.exit(1)

        bt_summary = None
        if args.backtest:
            n_top = max(1, min(args.backtest_top, len(dataframe)))
            symbols = dataframe["symbol"].head(n_top).tolist()

            bt_df = backtest(
                symbols,
                horizon=args.horizon,
                step=BACKTEST_STEP,
            )
            bt_summary = summarize_backtest(bt_df, args.horizon)

        if args.json:
            payload = build_json_output(dataframe, bt_summary)
            print(json.dumps(
                sanitize(payload),
                indent=2,
                allow_nan=False,
                default=str,
            ))
        else:
            display(dataframe)
            if bt_summary is not None:
                display_backtest(bt_summary)

    except requests.RequestException as error:
        if args.json:
            print(json.dumps({"error": str(error)}, indent=2))
        else:
            print()
            print("Binance API connection error:")
            print(error)
        sys.exit(2)

    except KeyboardInterrupt:
        if not args.json:
            print()
            print("Scan interrupted by user.")
        sys.exit(130)

    except Exception as error:
        if args.json:
            print(json.dumps(
                {"error": f"{type(error).__name__}: {error}"},
                indent=2,
            ))
        else:
            print()
            print("Unexpected error:")
            print(type(error).__name__, error)
        sys.exit(1)


if __name__ == "__main__":
    main()
