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
#
# Two scores, two names:
#   - live score      -> LIVE ACTIVITY score      (0-100)
#   - backtest score  -> BACKTEST TECHNICAL score (0-100)
# The JSON field is still named "score" for backward
# compatibility; the top-level key "score_field_description"
# clarifies which score it is in each context.
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

BASE_URL = "https://fapi.binance.com"

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

# Minimum number of directional samples required before the
# score/return correlation is reported. Below this threshold
# the coefficient is statistically meaningless.
MIN_SAMPLES_FOR_CORRELATION = 30

# Global verbosity (set by argparse)
VERBOSE = True


def log(msg="", end="\n"):
    """Progress / info messages -> stderr (keeps --json stdout clean)."""
    if VERBOSE:
        print(msg, end=end, file=sys.stderr, flush=True)


# ============================================================
# HTTP SESSION + PROXY + RETRY
# ============================================================

# Proxy resolution order:
#   1. BINANCE_PROXY  (dedicated, recommended on GitHub Actions)
#   2. HTTPS_PROXY    (standard env var, often auto-set)
#   3. HTTP_PROXY     (standard env var, often auto-set)
#
# Accepted formats:
#   http://user:pass@host:port
#   socks5://user:pass@host:port   (requires: pip install requests[socks])
PROXY_URL = (
    os.environ.get("BINANCE_PROXY")
    or os.environ.get("HTTPS_PROXY")
    or os.environ.get("HTTP_PROXY")
    or os.environ.get("https_proxy")
    or os.environ.get("http_proxy")
)

session = requests.Session()
session.headers.update({"User-Agent": "KORIVA-Crypto-Scanner/2.1"})

if PROXY_URL:
    session.proxies = {"http": PROXY_URL, "https": PROXY_URL}
    # Never print credentials: keep only the part after '@' if present.
    _safe_proxy = PROXY_URL.split("@")[-1]
    log(f"[KORIVA] Proxy enabled -> {_safe_proxy}")
else:
    log("[KORIVA] No proxy configured (using direct connection).")


def _is_fatal_http_error(err):
    """
    True if the HTTPError corresponds to a fatal deployment
    condition (IP ban or geo-block) that must not be silently
    swallowed by high-level helpers.

    A silent None on these errors would hide a whole-scan
    failure (e.g. every OI / funding value missing) behind a
    'working' scan output.
    """
    s = str(err)
    return ("418" in s) or ("451" in s)


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
        # Binance may return HTTP 451 when the runner's IP/region
        # is restricted for the requested endpoint. A proxy or a
        # non-restricted runner region is the documented workaround.
        if response.status_code == 451:
            raise requests.HTTPError(
                f"451 Geo-blocked by Binance for {endpoint}: "
                f"this IP/region is restricted for the requested "
                f"endpoint (HTTP 451). Configure BINANCE_PROXY with "
                f"a permitted egress, or use a self-hosted runner in "
                f"a non-restricted region. Body: {response.text[:200]}"
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

    HTTP 418/451 are re-raised because they indicate a
    deployment-level failure, not a per-symbol data gap.
    """
    try:
        data = get_json(
            "/fapi/v1/fundingRate",
            {"symbol": symbol, "limit": 1}
        )
        if not data:
            return None
        return float(data[-1]["fundingRate"]) * 100
    except requests.HTTPError as e:
        if _is_fatal_http_error(e):
            raise
        return None
    except Exception:
        return None


def get_open_interest(symbol):
    """
    Open interest or None if unavailable.

    HTTP 418/451 are re-raised (deployment-level failure).
    """
    try:
        data = get_json("/fapi/v1/openInterest", {"symbol": symbol})
        return float(data["openInterest"])
    except requests.HTTPError as e:
        if _is_fatal_http_error(e):
            raise
        return None
    except Exception:
        return None


def get_open_interest_history(symbol):
    """
    Recent OI % change over the fetched window.

    Endpoint (verified against Binance USDⓈ-M Futures public
    REST API documentation):
        GET /futures/data/openInterestHist
        Params: symbol=<USDT-M perpetual>, period=5m, limit=6
        Response fields: sumOpenInterest, sumOpenInterestValue,
                         timestamp
    Both period=5m and limit=6 are valid for this endpoint.
    The base host for this endpoint is fapi.binance.com
    (USDT-M), not dapi.binance.com (COIN-M).

    WINDOW SPAN:
        period = "5m", limit = 6.
        The first-to-last observation therefore spans
        (6 - 1) * 5 = 25 minutes, NOT 30 minutes.
        The internal key is still "oi_change"; the display
        label makes the ~25m span explicit.

    Returns None (not 0.0) when unavailable so missing data
    contributes 0 score points instead of a neutral value.

    HTTP 418/451 are re-raised (deployment-level failure).
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
    except requests.HTTPError as e:
        if _is_fatal_http_error(e):
            raise
        return None
    except Exception:
        return None


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calculate_rsi(close, period=14):
    """
    SMA-based RSI (NOT Wilder / RMA smoothing).

    Average gain and average loss use a simple rolling mean.
    This is internally consistent and safe, but differs from
    TradingView and Binance chart RSI, which use Wilder's
    smoothing (RMA, i.e. EMA with alpha = 1/period). Values
    produced here are therefore not expected to match those
    references exactly.

    The scoring thresholds in _subscore_rsi() are calibrated
    to THIS implementation and must not be changed without
    recalibrating those thresholds.

    Returns 50.0 when there is not enough data (neutral).
    """
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
    LIVE ACTIVITY score /100 (not a directional prediction):
        momentum (20) + RV (20) + trend (20) + RSI (15)
        + OI (15) + funding (10)

    Missing OI or funding contributes 0 points (NOT a neutral
    value that would trigger the 'low funding' bonus).

    This is DIFFERENT from calculate_backtest_score(), which is
    a technical-only score built from momentum + RV + trend +
    RSI, rescaled to 100. Do not compare the two numerically.
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
    BACKTEST TECHNICAL score on a 0-100 scale, built ONLY from
    features genuinely available historically:
        momentum (20) + RV (20) + trend (20) + RSI (15)
    The 75-point sum is rescaled to 100.

    This is NOT the same as the LIVE ACTIVITY score:
      - Historical OI is not available via the public API.
      - Historical funding is not available via the public API.
    So the backtest intentionally omits those two components
    rather than faking them with 0.0, which would have given
    every sample the 'low funding' bonus.

    Do not compare this number numerically to the live score.
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

    SCOPE (intentional, do not redesign):
        This backtest evaluates the CURRENTLY selected top-N
        symbols returned by the live scanner. It does NOT
        reconstruct historical Binance-wide symbol selection.
        Results reflect the historical behavior of symbols
        that are today's top performers, which is a
        selection bias and must be interpreted accordingly.

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
    research baseline, not a conservative execution simulator features
    (no slippage, no fees, no next-open entry).
    """
    records = []

    log["()
    log("=" * 100)
    log(f"  BACKrelativeTEST - {len(symbols)} symbols, "
        f"horizon {horizon} candles (~{horizon_ * 15volume / 60:.0f}h),"],
 "
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
                    "relative_volume":                    "rsi": features["rsi"],
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

    def _stats
