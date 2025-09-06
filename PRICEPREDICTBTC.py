from __future__ import annotations

"""
btc_daily_pdf_guard_backtest
---------------------------------

This script computes a risk‑neutral probability density function (PDF) for
Bitcoin (BTC) based on current Deribit option prices, then plots the result
as a bell curve each day.  It also implements a simple daily risk guard,
logs each day's predicted median (the 50th percentile of the PDF), and
maintains a journal of trade signals.  When market data for the following
day becomes available, the script grades each previous signal based on
whether the suggested take‑profit (TP) or stop‑loss (SL) was hit first.

Key features:

* **Risk‑neutral PDF**: Uses Breeden–Litzenberger to derive a density from
  call option prices.  Options are pulled from Deribit without any API key.

* **Bell curve visualization**: The PDF is smoothed with a Gaussian
  kernel, plotted with shaded 5‑95% and 25‑75% bands, and annotated with
  the spot price, median, TP and SL levels.

* **Bias, TP and SL**: Chooses a long/short bias based on the PDF.  The
  function ``suggest_trade_levels`` considers both the PDF’s median
  relative to spot and the cumulative mass at spot.  It always chooses a
  side (no neutral signals) by falling back to whichever side the median
  leans.  TP is set at the 25th or 75th percentile (depending on bias);
  SL at the 5th or 95th percentile.

* **Daily risk guard**: Tracks intraday realised and unrealised PnL.  It
  locks trading when a user‑defined loss limit is hit, and resets at the
  start of each Arizona day.  PnL values are currently always zero unless
  integrated with your brokerage.

* **Signal logging and grading**: Each day’s bias/TP/SL is appended to
  ``btc_daily_signals.jsonl``.  The next time you run the script, it
  attempts to grade any ungraded signals by fetching BTC‑USD 30‑minute
  candles for the next day via Yahoo Finance.  When data are missing or
  the next day has not yet happened, the signal remains pending.

Note on data sources
--------------------

Deribit’s public API does not provide historical option chains, so this
script can only compute the PDF for the day it is run.  Historical
performance of the PDF cannot be reconstructed retrospectively.  The
grading of signals relies on Yahoo Finance via yfinance.  In
restricted environments where yfinance cannot be installed or network
access is blocked, signals will remain ``PENDING`` until data are
available.

Usage
-----

Run this script once per trading day (ideally before the New York close)
to generate the PDF, log the signal, and grade any previous signals.  It
stores its state in ``btc_daily_guard_state.json`` and
``btc_daily_signals.jsonl`` within the current working directory.

"""

import os
import json
import time
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import requests
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

try:
    import yfinance as yf  # type: ignore
    import pandas as pd  # type: ignore
except ImportError:
    # In restricted environments yfinance/pandas may be unavailable.
    yf = None  # type: ignore
    pd = None  # type: ignore

DERIBIT = "https://www.deribit.com/api/v2"
SIGNALS_PATH = "btc_daily_signals.jsonl"
STATE_PATH   = "btc_daily_guard_state.json"

# Local CSV file containing Coinbase Bitcoin (CBBTCUSD) closing prices.
# This file should contain two columns: ``observation_date`` (YYYY‑MM‑DD) and
# ``CBBTCUSD`` (closing price).  It can be downloaded from FRED
# (https://fred.stlouisfed.org/series/CBBTCUSD) and placed alongside this
# script.  If present, it will be used to compute actual closing prices
# corresponding to each day’s predicted median when generating the
# predicted‑vs‑actual table.
BTC_PRICE_CSV = "CBBTCUSD.csv"


###############################################################################
# Data model
###############################################################################

# QuoteRow: simple record for an option quote
#
# Deribit returns call and put strikes along with bid and ask prices.  We
# convert these into mid quotes and store them in a QuoteRow object.  This
# dataclass is used throughout the script to type‑hint lists of quotes and
# ensure each quote has numeric strike ``K`` and mid price ``mid``.

@dataclass
class QuoteRow:
    """A container for a single option quote.

    Attributes
    ----------
    K : float
        The option strike price.
    mid : float
        The mid price between bid and ask for the option.
    """
    K: float
    mid: float


def _lower_bool_params(params: Dict) -> Dict:
    """Coerce booleans to 'true'/'false' for the Deribit API."""
    out: Dict[str, str] = {}
    for k, v in params.items():
        if isinstance(v, bool):
            out[k] = "true" if v else "false"
        else:
            out[k] = str(v)
    return out


_session = requests.Session()
_session.headers.update({"User-Agent": "btc-daily-bl/1.3"})


def _deribit_get(path: str, retries: int = 3, backoff: float = 0.6, **params) -> Dict:
    """Perform a GET request against the Deribit API with retry/backoff."""
    q = _lower_bool_params(params)
    url = f"{DERIBIT}/{path}"
    last_err: Optional[Exception] = None
    for a in range(retries):
        try:
            r = _session.get(url, params=q, timeout=15)
            r.raise_for_status()
            data = r.json()
            if "result" not in data:
                raise RuntimeError(f"Unexpected response: {data}")
            return data["result"]  # type: ignore[no-any-return]
        except Exception as e:
            last_err = e
            if a < retries - 1:
                time.sleep(backoff * (2 ** a))
            else:
                raise last_err  # re-raise final exception


# ---------------------------------------------------------------------------
# Daily Risk Guard
# ---------------------------------------------------------------------------

@dataclass
class DailyRiskGuard:
    """Tracks daily PnL and locks trading when a loss limit is exceeded."""
    daily_loss_limit: float
    include_unrealized: bool = True
    tz_offset_hours: int = -7  # Arizona (no DST)
    day: str = ""
    day_start_equity: Optional[float] = None
    day_high_equity: Optional[float] = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    locked: bool = False

    @staticmethod
    def _today_iso(tz_offset_hours: int) -> str:
        return (datetime.utcnow() + timedelta(hours=tz_offset_hours)).date().isoformat()

    def _ensure_day(self) -> None:
        today = self._today_iso(self.tz_offset_hours)
        if self.day != today:
            self.reset_day(self.day_start_equity)

    def reset_day(self, start_equity: Optional[float] = None) -> None:
        """Reset the guard at the start of a new day."""
        self.day = self._today_iso(self.tz_offset_hours)
        self.locked = False
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        if start_equity is not None:
            self.day_start_equity = start_equity
        self.day_high_equity = self.day_start_equity

    def set_start_equity(self, equity: float) -> None:
        self._ensure_day()
        self.day_start_equity = equity
        self.day_high_equity = equity

    def add_realized_pnl(self, pnl: float) -> None:
        self._ensure_day()
        if not self.locked:
            self.realized_pnl += pnl
            self._update_high()

    def set_unrealized_pnl(self, upl: float) -> None:
        self._ensure_day()
        self.unrealized_pnl = upl if self.include_unrealized else 0.0
        self._update_high()

    def _update_high(self) -> None:
        if self.day_start_equity is None:
            return
        eq_now = self.day_start_equity + self.realized_pnl + self.unrealized_pnl
        if self.day_high_equity is None or eq_now > self.day_high_equity:
            self.day_high_equity = eq_now

    def _drawdown_from_start(self) -> float:
        if self.day_start_equity is None:
            # no start equity set; just reflect net losses
            return max(0.0, -(self.realized_pnl + self.unrealized_pnl))
        eq_now = self.day_start_equity + self.realized_pnl + self.unrealized_pnl
        return max(0.0, self.day_start_equity - eq_now)

    def check_lock(self) -> bool:
        self._ensure_day()
        if self._drawdown_from_start() >= self.daily_loss_limit:
            self.locked = True
        return self.locked


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def load_guard(path: str) -> Optional[DailyRiskGuard]:
    """Load a DailyRiskGuard from a JSON file, if it exists.

    The guard's state (day, equity, PnL, locked status) is stored in a
    JSON file between script runs.  If the file does not exist, return
    ``None`` so that a fresh guard can be created.

    Parameters
    ----------
    path : str
        The path to the JSON file to load.

    Returns
    -------
    Optional[DailyRiskGuard]
        A new DailyRiskGuard initialised from the file, or ``None``
        if the file does not exist.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            d = json.load(f)
        return DailyRiskGuard(**d)
    except Exception:
        return None


def save_guard(path: str, guard: DailyRiskGuard) -> None:
    """Save a DailyRiskGuard to a JSON file.

    Parameters
    ----------
    path : str
        The path to the JSON file to write.
    guard : DailyRiskGuard
        The guard to serialize.
    """
    try:
        with open(path, "w") as f:
            json.dump(asdict(guard), f, indent=2)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Option chain helpers
# ---------------------------------------------------------------------------

def fetch_btc_spot() -> float:
    """Return the Deribit BTC index price."""
    res = _deribit_get("public/get_index_price", index_name="btc_usd")
    return float(res["index_price"])


def list_btc_expiries() -> List[int]:
    """Return all upcoming BTC option expiry timestamps (ms)."""
    res = _deribit_get("public/get_instruments", currency="BTC", kind="option", expired=False)
    return sorted({int(inst["expiration_timestamp"]) for inst in res})


def fetch_chain(exp_ms: int) -> Tuple[List[QuoteRow], List[QuoteRow]]:
    """Return (call_list, put_list) of QuoteRow for a given expiry."""
    insts = _deribit_get("public/get_instruments", currency="BTC", kind="option", expired=False)
    calls_meta = [i for i in insts if i.get("option_type") == "call" and int(i["expiration_timestamp"]) == exp_ms]
    puts_meta  = [i for i in insts if i.get("option_type") == "put"  and int(i["expiration_timestamp"]) == exp_ms]

    def _mid(instr: str) -> Optional[float]:
        try:
            book = _deribit_get("public/get_book_summary_by_instrument", instrument_name=instr)
            if not book:
                return None
            bid = book[0].get("bid_price")
            ask = book[0].get("ask_price")
            if bid is None or ask is None:
                return None
            return 0.5 * (bid + ask)
        except Exception:
            return None

    calls: List[QuoteRow] = []
    puts: List[QuoteRow] = []
    for m in calls_meta:
        md = _mid(m.get("instrument_name"))
        if md is not None and math.isfinite(md):
            calls.append(QuoteRow(K=float(m.get("strike")), mid=float(md)))
    for m in puts_meta:
        md = _mid(m.get("instrument_name"))
        if md is not None and math.isfinite(md):
            puts.append(QuoteRow(K=float(m.get("strike")), mid=float(md)))
    calls.sort(key=lambda x: x.K)
    puts.sort(key=lambda x: x.K)
    return calls, puts


def time_to_expiry_years(exp_ms: int) -> float:
    """Time to expiry in years from now, given expiry in ms."""
    now = datetime.now(tz=timezone.utc).timestamp()
    return max(1e-6, (exp_ms / 1000.0 - now) / (365.25 * 24 * 3600))


def calls_from_puts_via_parity(
    puts: List[QuoteRow], S0: float, K: np.ndarray, r: float, T: float, q: float = 0.0
) -> Dict[float, float]:
    """Convert put mids to call mids using put‑call parity: C = P + S*e^{-qT} − K*e^{-rT}."""
    disc_S = math.exp(-q * T)
    disc_K = math.exp(-r * T)
    pm = {row.K: row.mid for row in puts}
    out: Dict[float, float] = {}
    for k in K:
        p = pm.get(float(k))
        if p is not None:
            out[float(k)] = max(0.0, float(p + S0 * disc_S - k * disc_K))
    return out


def build_call_curve(
    calls: List[QuoteRow],
    puts: List[QuoteRow],
    S0: float,
    r: float,
    T: float,
    strike_window: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a sorted strike array and corresponding call mids.

    Real call quotes are merged with parity‑derived call mids from puts.  Any
    tiny negative values (from noise) are clipped at zero.  Only strikes
    within the specified window are returned.
    """
    Ks_calls = np.array([c.K for c in calls], float)
    Cs_calls = np.array([c.mid for c in calls], float)
    all_K = np.unique(np.concatenate([Ks_calls, np.array([p.K for p in puts], float)]))
    parity_calls = calls_from_puts_via_parity(puts, S0, all_K, r=r, T=T)

    merged: Dict[float, float] = {}
    idx_map = {k: i for i, k in enumerate(Ks_calls.tolist())}
    for k in all_K:
        kf = float(k)
        if kf in idx_map:
            merged[kf] = float(Cs_calls[idx_map[kf]])
        elif kf in parity_calls:
            merged[kf] = parity_calls[kf]

    k_lo, k_hi = strike_window
    items = [(k, v) for k, v in merged.items() if k_lo <= k <= k_hi and math.isfinite(v)]
    if not items:
        raise RuntimeError("No strikes after windowing/merge.")
    items.sort(key=lambda t: t[0])
    K_arr = np.array([k for k, _ in items], float)
    C_arr = np.maximum(np.array([v for _, v in items], float), 0.0)
    return K_arr, C_arr


def bl_pdf_from_calls(
    strikes: np.ndarray,
    call_mids: np.ndarray,
    r: float,
    T: float,
    smoothing_scale: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the Breeden–Litzenberger PDF: f(K) = e^{rT} * d²C/dK².

    A cubic spline with gentle smoothing is used to obtain the second
    derivative.  The smoothing scale increases when data are sparse to
    prevent over‑fitting.  The resulting density is normalized to
    integrate to 1.  A minimum of 8 strikes is required.
    """
    n = len(strikes)
    if n < 8:
        raise RuntimeError(f"Too few strikes ({n}) to compute PDF.")
    s = max(1e-6, n * smoothing_scale * (1.5 if n < 18 else 1.0))
    spline = UnivariateSpline(strikes, call_mids, k=3, s=s)
    c2 = spline.derivative(2)(strikes)
    pdf = np.exp(r * T) * np.maximum(c2, 0.0)
    area = np.trapz(pdf, strikes)
    if area > 0:
        pdf /= area
    return strikes, pdf


def gaussian_smooth(y: np.ndarray, sigma_pts: float = 4.0) -> np.ndarray:
    """Apply a Gaussian kernel to smooth the PDF, defined in index space."""
    if sigma_pts <= 0:
        return y
    radius = int(max(3, round(6 * sigma_pts)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kern = np.exp(-(x ** 2) / (2 * sigma_pts ** 2))
    kern /= kern.sum()
    return np.convolve(y, kern, mode="same")


def cdf_from_pdf(x: np.ndarray, pdf: np.ndarray) -> np.ndarray:
    """Compute the cumulative distribution function from the PDF."""
    dx = np.diff(x).mean()
    return np.clip(np.cumsum(pdf) * dx, 0.0, 1.0)


def percentile_level(x: np.ndarray, cdf: np.ndarray, p: float) -> float:
    """Return the value K such that P(X <= K) = p."""
    return float(np.interp(p, cdf, x))


def choose_good_expiry(min_calls: int = 16) -> Tuple[int, List[QuoteRow], List[QuoteRow]]:
    """Pick the nearest expiry with at least ``min_calls`` call quotes.

    If none of the upcoming expiries meet ``min_calls``, fall back to one
    with at least 10 calls, emitting a warning.  Raises if no expiry has
    more than 10 calls.
    """
    expiries = list_btc_expiries()
    for exp in expiries:
        calls, puts = fetch_chain(exp)
        if len(calls) >= min_calls:
            return exp, calls, puts
    for exp in expiries:
        calls, puts = fetch_chain(exp)
        if len(calls) >= 10:
            print(
                f"[warn] sparse chain: {len(calls)} calls at {datetime.fromtimestamp(exp/1000, tz=timezone.utc).date()}"
            )
            return exp, calls, puts
    raise RuntimeError("No upcoming expiry has enough call quotes (>=10).")


# ---------------------------------------------------------------------------
# Bias / TP / SL logic
# ---------------------------------------------------------------------------

def suggest_trade_levels(
    spot: float,
    p05: float,
    p25: float,
    p50: float,
    p75: float,
    p95: float,
    tol: float = 0.002,
    cdf_at_spot: Optional[float] = None,
    edge_cdf: float = 0.01,
) -> Tuple[str, Optional[float], Optional[float]]:
    """Return (bias, take_profit, stop_loss) given percentile levels.

    Bias is "LONG", "SHORT", or "NEUTRAL".  The tolerance ``tol`` is a
    threshold for how far the median can deviate from spot before
    triggering a directional bias.  ``cdf_at_spot`` (the probability mass
    below spot) is used to determine bias when an edge exists; if there is
    ≥ ``edge_cdf`` mass imbalance, the side with more mass becomes the
    bias.  If no directional rule triggers, a fallback chooses the side
    based on whether the median is slightly above or below spot.  TP and
    SL are set at the 25th/75th and 5th/95th percentiles, respectively.
    """
    # Probability mass rule
    if cdf_at_spot is not None:
        if cdf_at_spot >= 0.5 + edge_cdf:
            # More mass below spot -> bearish
            return "SHORT", p25, p95
        if cdf_at_spot <= 0.5 - edge_cdf:
            # More mass above spot -> bullish
            return "LONG", p75, p05
    # Median vs spot rule
    if p50 < spot * (1 - tol):
        return "SHORT", p25, p95
    if p50 > spot * (1 + tol):
        return "LONG", p75, p05
    # Fallback: always pick a side
    return ("SHORT", p25, p95) if (spot - p50) > 0 else ("LONG", p75, p05)


# ---------------------------------------------------------------------------
# Plot and guard integration
# ---------------------------------------------------------------------------

def build_daily_bell_and_guard(
    guard: DailyRiskGuard,
    start_equity_today: Optional[float] = None,
    position: Optional[BTCPosition] = None,
    risk_free_rate: float = 0.0,
    strike_window_pct: float = 0.40,
    min_calls: int = 16,
    bell_sigma_pts: float = 4.0,
    save_png: Optional[str] = None,
) -> Dict[str, float]:
    """Compute and plot the risk‑neutral PDF and update the risk guard.

    Returns a summary dict containing spot price, percentile levels,
    bias, TP/SL, and risk guard state.  The PDF plot is displayed via
    matplotlib and optionally saved.
    """
    if start_equity_today is not None:
        guard.set_start_equity(start_equity_today)

    S0 = fetch_btc_spot()
    exp_ms, calls, puts = choose_good_expiry(min_calls=min_calls)
    T = time_to_expiry_years(exp_ms)
    exp_ts = datetime.fromtimestamp(exp_ms / 1000.0, tz=timezone.utc)

    # Build call curve within ±strike_window_pct around spot
    k_lo, k_hi = (1.0 - strike_window_pct) * S0, (1.0 + strike_window_pct) * S0
    K, C = build_call_curve(calls, puts, S0, r=risk_free_rate, T=T, strike_window=(k_lo, k_hi))

    # Densify strike grid and compute PDF
    grid = np.linspace(K.min(), K.max(), 900)
    pre = UnivariateSpline(K, C, k=3, s=max(1e-6, len(K) * 1e-3))
    Cg = pre(grid)
    x, pdf = bl_pdf_from_calls(grid, Cg, r=risk_free_rate, T=T, smoothing_scale=1e-3)
    pdf = gaussian_smooth(pdf, sigma_pts=bell_sigma_pts)
    area = np.trapz(pdf, x)
    if area > 0:
        pdf /= area
    cdf = cdf_from_pdf(x, pdf)

    # Percentiles and probability mass at spot
    p05 = percentile_level(x, cdf, 0.05)
    p25 = percentile_level(x, cdf, 0.25)
    p50 = percentile_level(x, cdf, 0.50)
    p75 = percentile_level(x, cdf, 0.75)
    p95 = percentile_level(x, cdf, 0.95)
    cdf_at_spot = float(np.interp(S0, x, cdf))

    bias, tp_level, sl_level = suggest_trade_levels(
        S0, p05, p25, p50, p75, p95,
        tol=0.002,
        cdf_at_spot=cdf_at_spot,
        edge_cdf=0.01,
    )

    # Update risk guard with unrealized PnL from open position
    if position is not None:
        upl = position.upl(S0)
        guard.set_unrealized_pnl(upl)
        guard.check_lock()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.fill_between(x, pdf, 0, where=(x >= p05) & (x <= p95), alpha=0.20,
                     label=f"5–95%: ${p05:,.0f}–${p95:,.0f}")
    plt.fill_between(x, pdf, 0, where=(x >= p25) & (x <= p75), alpha=0.35,
                     label=f"25–75%: ${p25:,.0f}–${p75:,.0f}")
    plt.plot(x, pdf, linewidth=1.6)

    def vline(xv: float, ls: str, text: str) -> None:
        plt.axvline(xv, linestyle=ls, linewidth=1.6, alpha=0.9)
        plt.text(xv, plt.ylim()[1] * 0.92, text, rotation=90, va="top", ha="right")

    vline(p50, "--", f"Median ${p50:,.0f}")
    plt.axvline(S0, color="k", linewidth=1.2, alpha=0.6)
    plt.text(S0, plt.ylim()[1] * 0.98, f"Spot ${S0:,.0f}", ha="center", va="top")

    if bias != "NEUTRAL":
        if tp_level is not None:
            vline(tp_level, "-", f"TP ${tp_level:,.0f}")
        if sl_level is not None:
            vline(sl_level, "-", f"SL ${sl_level:,.0f}")

    plt.title(
        f"BTC PDF — Exp {exp_ts.strftime('%Y-%m-%d')} — Day {guard.day or guard._today_iso(guard.tz_offset_hours)}"
    )
    plt.xlabel("BTC price (USD)")
    plt.ylabel("Probability density (relative likelihood)")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper right", frameon=True)
    support_mask = pdf > pdf.max() * 0.002
    if support_mask.any():
        xmin, xmax = x[support_mask][0], x[support_mask][-1]
        span = xmax - xmin
        plt.xlim(xmin - 0.05 * span, xmax + 0.05 * span)
    plt.tight_layout()
    if save_png:
        plt.savefig(save_png, dpi=150)
    plt.show()

    return {
        "spot": S0,
        "p05": p05,
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "p95": p95,
        "bias": bias,
        "tp": tp_level,
        "sl": sl_level,
        "locked": guard.locked,
        "drawdown_from_start": guard._drawdown_from_start(),
        "realized_pnl": guard.realized_pnl,
        "unrealized_pnl": guard.unrealized_pnl,
    }


# ---------------------------------------------------------------------------
# Signal log and grading
# ---------------------------------------------------------------------------

def _append_signal(rec: dict) -> None:
    with open(SIGNALS_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _read_signals() -> List[dict]:
    if not os.path.exists(SIGNALS_PATH):
        return []
    out: List[dict] = []
    with open(SIGNALS_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _write_signals(allrecs: List[dict]) -> None:
    with open(SIGNALS_PATH, "w") as f:
        for rec in allrecs:
            f.write(json.dumps(rec) + "\n")


def az_day_bounds(iso_date: str, tz_offset_hours: int = -7) -> Tuple[datetime, datetime]:
    """Return UTC start/end covering the given Arizona date."""
    local_start = datetime.fromisoformat(iso_date)
    utc_start = datetime(
        local_start.year, local_start.month, local_start.day, tzinfo=timezone.utc
    ) - timedelta(hours=tz_offset_hours)
    utc_end   = utc_start + timedelta(days=1)
    return utc_start, utc_end


def grade_signal_with_next_day(
    rec: dict,
    tz_offset_hours: int = -7,
    within_pct: float = 0.10,
    count_median_touch: bool = True,
) -> dict:
    """Grade a signal by inspecting the next day's BTC price path.

    This function attempts to retrieve 30‑minute bars for BTC‑USD from
    Yahoo Finance via yfinance.  If the next day is in the future or
    data are unavailable, the record is left pending.  Primary rule:
    whichever of TP or SL is hit first sets WIN or LOSS.  Secondary
    rules award a WIN if the median is touched, or if price gets
    within a specified percentage of TP in the favourable direction.
    """
    if rec.get("bias") == "NEUTRAL" or rec.get("tp") is None or rec.get("sl") is None:
        rec["graded"] = True
        rec["result"] = "SKIP"
        return rec

    # Determine next day window in UTC
    d = datetime.fromisoformat(rec["day"])
    next_day_iso = (d + timedelta(days=1)).date().isoformat()
    start_utc, end_utc = az_day_bounds(next_day_iso, tz_offset_hours)

    now_utc = datetime.now(timezone.utc)
    if start_utc > now_utc:
        # Too early to grade
        rec["graded"] = False
        rec["result"] = "PENDING"
        rec["note"] = "Next-day window not available yet"
        return rec

    # Limit the window to available data
    effective_end = min(end_utc, now_utc)

    if yf is None or pd is None:
        # yfinance not installed; cannot grade
        rec["graded"] = False
        rec["result"] = "PENDING"
        rec["note"] = "yfinance unavailable"
        return rec

    # Fetch data
    try:
        df = yf.download(
            "BTC-USD",
            start=start_utc,
            end=effective_end,
            interval="30m",
            progress=False,
            prepost=True,
            auto_adjust=False,
            raise_errors=False,
            threads=False,
        )
    except Exception:
        df = None

    if df is None or df.empty:
        rec["graded"] = False
        rec["result"] = "PENDING"
        rec["note"] = "No data yet"
        return rec

    bias = rec["bias"]
    tp = float(rec["tp"])
    sl = float(rec["sl"])
    p50 = float(rec.get("median") or rec.get("p50") or 0.0)

    result: Optional[str] = None
    for _, row in df.iterrows():
        high = float(row["High"])
        low = float(row["Low"])
        if bias == "LONG":
            hit_tp = high >= tp
            hit_sl = low  <= sl
        else:  # SHORT
            hit_tp = low  <= tp
            hit_sl = high >= sl
        if hit_tp and hit_sl:
            result = "LOSS"  # conservative
            break
        elif hit_tp:
            result = "WIN"
            break
        elif hit_sl:
            result = "LOSS"
            break

    if result is not None:
        rec["graded"] = True
        rec["result"] = result
        return rec

    # Secondary: median touch
    if count_median_touch and p50 > 0:
        touched_median = any(
            (float(row["Low"]) <= p50 <= float(row["High"])) for _, row in df.iterrows()
        )
        if touched_median:
            rec["graded"] = True
            rec["result"] = "WIN"
            rec["note"] = "Median-touch"
            return rec

    # Secondary: within X% of TP
    if within_pct > 0:
        if bias == "LONG":
            thresh = tp * (1.0 - within_pct)
            got_close = any(float(row["High"]) >= thresh for _, row in df.iterrows())
        else:
            thresh = tp * (1.0 + within_pct)
            got_close = any(float(row["Low"]) <= thresh for _, row in df.iterrows())
        if got_close:
            rec["graded"] = True
            rec["result"] = "WIN"
            rec["note"] = f"Within {within_pct:.0%} of TP"
            return rec

    if effective_end < end_utc:
        # day incomplete
        rec["graded"] = False
        rec["result"] = "PENDING"
        rec["note"] = "Next-day still in progress"
        return rec

    # Final: no hits
    rec["graded"] = True
    rec["result"] = "NO_HIT"
    return rec


def compute_win_rate(last_n: int = 30) -> Tuple[int, int, float]:
    """Return (#wins, #trades, win_rate) for last N graded trades."""
    recs = _read_signals()
    graded = [r for r in recs if r.get("graded") and r.get("result") in ("WIN", "LOSS")]
    if not graded:
        return 0, 0, 0.0
    graded = graded[-last_n:]
    wins = sum(1 for r in graded if r["result"] == "WIN")
    total = len(graded)
    return wins, total, (wins / total if total else 0.0)


def summarize_results(last_n: int = 30) -> Dict[str, int]:
    """Return a breakdown of recent results (WIN/LOSS/NO_HIT/SKIP/PENDING)."""
    recs = _read_signals()
    graded = [r for r in recs if r.get("graded")]
    graded = graded[-last_n:]
    counts: Dict[str, int] = {"WIN": 0, "LOSS": 0, "NO_HIT": 0, "SKIP": 0, "PENDING": 0}
    for r in graded:
        res = r.get("result") or "NO_HIT"
        counts[res] = counts.get(res, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Predicted vs Actual Price Table
# ---------------------------------------------------------------------------
def compute_predicted_actual_table(last_n: int = 30) -> Optional[pd.DataFrame]:
    """Return a DataFrame comparing each signal's predicted median to the actual
    closing price from a local CSV file.

    The function looks up actual closing prices in ``BTC_PRICE_CSV`` (a
    CSV downloaded from FRED for the series CBBTCUSD).  It matches the
    ``day`` field from each signal against the ``observation_date`` in the
    CSV, returning a DataFrame with the columns:

    - ``day``: the Arizona trading day (ISO date string)
    - ``predicted_median``: the median (50th percentile) predicted by the PDF
    - ``actual_price``: the BTC close from the CSV on that date, or None
      if not available
    - ``error``: ``actual_price`` minus ``predicted_median`` when both
      values are available

    Parameters
    ----------
    last_n : int, optional
        Number of most recent signals to include (default 30).

    Returns
    -------
    pandas.DataFrame or None
        Returns a DataFrame if pandas and the CSV file are available;
        otherwise returns None and prints a warning.
    """
    # pandas must be available
    if pd is None:
        print("pandas is not available; cannot compute predicted vs actual table.")
        return None
    recs = _read_signals()
    if not recs:
        return pd.DataFrame(columns=["day", "predicted_median", "actual_price", "error"])
    recs = recs[-last_n:]
    # Ensure the CSV file exists
    if not os.path.exists(BTC_PRICE_CSV):
        print(f"Price CSV '{BTC_PRICE_CSV}' not found; cannot compute actual prices.")
        return None
    try:
        price_df = pd.read_csv(BTC_PRICE_CSV)
    except Exception as e:
        print(f"Failed to read {BTC_PRICE_CSV}: {e}")
        return None
    if 'observation_date' not in price_df.columns or 'CBBTCUSD' not in price_df.columns:
        print(f"{BTC_PRICE_CSV} must contain 'observation_date' and 'CBBTCUSD' columns.")
        return None
    # Convert dates
    price_df['observation_date'] = pd.to_datetime(price_df['observation_date'])
    price_df = price_df[['observation_date', 'CBBTCUSD']].dropna(subset=['CBBTCUSD'])
    rows = []
    for rec in recs:
        day = rec.get('day')
        pred = rec.get('median') or rec.get('p50')
        actual_price = None
        if day:
            try:
                date_obj = datetime.fromisoformat(day).date()
                mask = price_df['observation_date'].dt.date == date_obj
                if mask.any():
                    actual_price = float(price_df.loc[mask, 'CBBTCUSD'].iloc[0])
            except Exception:
                actual_price = None
        error = None
        try:
            if pred is not None and actual_price is not None and np.isfinite(float(pred)) and np.isfinite(float(actual_price)):
                error = float(actual_price) - float(pred)
        except Exception:
            error = None
        rows.append({
            'day': day,
            'predicted_median': pred,
            'actual_price': actual_price,
            'error': error
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example configuration: adjust as needed
    DAILY_LOSS_LIMIT = 2500.0
    START_EQUITY_TODAY = 50_000.0

    guard = load_guard(STATE_PATH) or DailyRiskGuard(daily_loss_limit=DAILY_LOSS_LIMIT, tz_offset_hours=-7)
    guard.reset_day(start_equity=START_EQUITY_TODAY)

    summary = build_daily_bell_and_guard(
        guard=guard,
        start_equity_today=START_EQUITY_TODAY,
        position=None,
        risk_free_rate=0.00,
        strike_window_pct=0.40,
        min_calls=16,
        bell_sigma_pts=4.0,
        save_png=f"btc_bl_{guard.day}.png"
    )

    # Display summary
    print(
        f"AZ Day: {guard.day} | Spot ${summary['spot']:,.0f} | "
        f"Median ${summary['p50']:,.0f} | 25–75% ${summary['p25']:,.0f}–${summary['p75']:,.0f} | "
        f"5–95% ${summary['p05']:,.0f}–${summary['p95']:,.0f}"
    )
    print(
        f"Bias: {summary['bias']} | TP: {summary['tp']} | SL: {summary['sl']} | "
        f"PnL R:{summary['realized_pnl']:+.2f} U:{summary['unrealized_pnl']:+.2f} | "
        f"DD: {summary['drawdown_from_start']:.2f} → {'LOCKED' if summary['locked'] else 'OK TO TRADE'}"
    )

    # Persist guard state
    save_guard(STATE_PATH, guard)

    # Log today's signal
    signal_rec = {
        "day": guard.day,
        "spot": summary["spot"],
        "median": summary["p50"],
        "p25": summary["p25"],
        "p75": summary["p75"],
        "p05": summary["p05"],
        "p95": summary["p95"],
        "bias": summary["bias"],
        "tp": summary["tp"],
        "sl": summary["sl"],
        "graded": False,
        "result": None
    }
    _append_signal(signal_rec)

    # Grade any pending signals
    allrecs = _read_signals()
    changed = False
    for idx, r in enumerate(allrecs):
        if not r.get("graded"):
            allrecs[idx] = grade_signal_with_next_day(
                r, tz_offset_hours=-7, within_pct=0.10, count_median_touch=True
            )
            changed = True
    if changed:
        _write_signals(allrecs)

    # Show win rate and breakdown
    wins, total, wr = compute_win_rate(last_n=30)
    breakdown = summarize_results(last_n=30)
    print(f"Rolling 30-signal win rate: {wins}/{total} = {wr:.1%}")
    print(f"Last 30 results breakdown: {breakdown}")

    # ------------------------------------------------------------------
    # Predicted vs Actual Table
    #
    # If pandas is available and a price CSV is present, compute a table
    # that pairs each of the most recent signals (up to 30) with the
    # actual closing price on that day.  This helps assess whether the
    # predicted median was accurate.  If data are missing or the CSV is
    # not present, the table will be omitted.
    try:
        pred_table = compute_predicted_actual_table(last_n=30)
    except Exception as e:
        pred_table = None
        print(f"Error computing predicted vs actual table: {e}")

    if pred_table is not None and not pred_table.empty:
        print("\nPredicted vs Actual (last 30 signals):")
        # Format the DataFrame for display: limit float precision and sort by day
        try:
            display_df = pred_table.copy()
            # Round the error to 2 decimals if present
            if 'error' in display_df.columns:
                display_df['error'] = display_df['error'].map(lambda x: f"{x:,.2f}" if x is not None else "")
            # Format numbers with commas and zero decimals
            if 'predicted_median' in display_df.columns:
                display_df['predicted_median'] = display_df['predicted_median'].map(lambda x: f"{x:,.0f}" if x is not None else "")
            if 'actual_price' in display_df.columns:
                display_df['actual_price'] = display_df['actual_price'].map(lambda x: f"{x:,.0f}" if x is not None else "")
            print(display_df.to_string(index=False))
        except Exception as e:
            # Fallback simple print if formatting fails
            print(pred_table)
