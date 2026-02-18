# app.py — conservative yfinance-only Streamlit IV Z-Score Gauge
# Goals:
# - Minimize Yahoo calls (no .info)
# - Cache results
# - Retries + backoff
# - Session limit: max 5 runs per 30 minutes
# - Cooldown between runs
# - Friendly errors for rate-limits

import time
import random
from collections import deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="IV Gauge", layout="centered")
st.title("IV Gauge (Z-Score)")

# -----------------------
# User-limits (session-based)
# -----------------------
MAX_RUNS = 5
WINDOW_MINUTES = 30
COOLDOWN_SECONDS = 20  # prevent rapid re-clicks

WINDOW = 252
DEFAULT_PERIOD = "2y"

# -----------------------
# Helpers for session tracking
# -----------------------
def _now_ts() -> float:
    return time.time()

def init_limits():
    if "run_times" not in st.session_state:
        st.session_state.run_times = deque()  # timestamps of runs
    if "last_run_ts" not in st.session_state:
        st.session_state.last_run_ts = 0.0

def prune_old_runs():
    """Keep only runs within last WINDOW_MINUTES."""
    cutoff = _now_ts() - WINDOW_MINUTES * 60
    while st.session_state.run_times and st.session_state.run_times[0] < cutoff:
        st.session_state.run_times.popleft()

def remaining_runs_and_reset():
    prune_old_runs()
    used = len(st.session_state.run_times)
    remaining = max(0, MAX_RUNS - used)

    if used == 0:
        reset_in = 0
    else:
        oldest = st.session_state.run_times[0]
        reset_in = int(max(0, (oldest + WINDOW_MINUTES * 60) - _now_ts()))
    return remaining, reset_in, used

def human_time(seconds: int) -> str:
    if seconds <= 0:
        return "now"
    m, s = divmod(seconds, 60)
    if m <= 0:
        return f"{s}s"
    return f"{m}m {s}s"

init_limits()

# -----------------------
# Conservative yfinance fetch
# -----------------------
@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)  # 6 hours cache
def fetch_history_cached(ticker: str, period: str, interval: str = "1d") -> pd.DataFrame:
    # One single Yahoo call (history)
    df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    return df.dropna()

def fetch_with_retries(ticker: str, period: str, max_attempts: int = 5) -> pd.DataFrame:
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fetch_history_cached(ticker, period)
        except Exception as e:
            last_err = e
            # exponential backoff + jitter (kept moderate)
            sleep_s = min(2 ** attempt, 20) + random.uniform(0.2, 1.5)
            time.sleep(sleep_s)
    raise last_err

def is_rate_limit_error(e: Exception) -> bool:
    msg = str(e)
    # yfinance errors vary; this catches the common ones
    keywords = ["Too Many Requests", "YFRateLimitError", "rate limit", "429"]
    return any(k.lower() in msg.lower() for k in keywords)

# -----------------------
# Gauge drawing
# -----------------------
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11

def z_bucket(z: float) -> int:
    z = float(z)
    if z < -2:
        return 1
    elif z < -1:
        return 2
    elif z < 1:
        return 3
    elif z < 2:
        return 4
    else:
        return 5

def draw_arc_text(ax, text, radius, theta_start_deg, theta_end_deg,
                  fontsize=12, color="#F3F4F6", weight="bold"):
    text = str(text)
    n = max(len(text), 1)
    angles = np.linspace(theta_start_deg, theta_end_deg, n)

    for ch, ang_deg in zip(text, angles):
        if ch == " ":
            continue
        ang = np.deg2rad(ang_deg)
        x = radius * np.cos(ang)
        y = radius * np.sin(ang)
        rot = ang_deg - 90  # tangent rotation

        ax.text(
            x, y, ch,
            ha="center", va="center",
            rotation=rot,
            rotation_mode="anchor",
            fontsize=fontsize,
            color=color,
            fontweight=weight
        )

def zscore_speedometer_custom(
    z,
    ticker="",
    date=None,
    window=252,
    bg="#666666",
    arc_label_radius=1.14
):
    z = float(z)
    bucket = z_bucket(z)
    z_display = float(np.clip(z, -3, 3))

    colors = {
        1: "#0B5D1E",  # dark green
        2: "#7CCB7A",  # light green
        3: "#2F6FDB",  # blue
        4: "#FF7F50",  # coral
        5: "#A30234",  # deep red
    }

    sections = [
        (-3, -2, "1", colors[1]),
        (-2, -1, "2", colors[2]),
        (-1,  1, "3", colors[3]),
        ( 1,  2, "4", colors[4]),
        ( 2,  3, "5", colors[5]),
    ]

    def z_to_angle_deg(zv):
        return np.interp(zv, [-3, 3], [180, 0])

    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.set_aspect("equal")
    ax.axis("off")

    outer_r = 1.0
    inner_r = 0.68

    # sections
    for z0, z1, lab, facecol in sections:
        a0 = z_to_angle_deg(z0)
        a1 = z_to_angle_deg(z1)
        a_start, a_end = sorted([a0, a1])

        ax.add_patch(
            Wedge(
                (0, 0), outer_r, a_start, a_end,
                width=outer_r - inner_r,
                facecolor=facecol,
                edgecolor="#111827",
                linewidth=1.2,
                alpha=0.98
            )
        )

        mid_z = (z0 + z1) / 2
        mid_ang = np.deg2rad(z_to_angle_deg(mid_z))
        r_text = (outer_r + inner_r) / 2
        ax.text(
            r_text*np.cos(mid_ang), r_text*np.sin(mid_ang),
            lab,
            ha="center", va="center",
            fontsize=12, fontweight="bold",
            color="white" if lab in {"1", "3", "5"} else "#111827"
        )

    # ticks
    ticks = [-3, -2, -1, 0, 1, 2, 3]
    for tv in ticks:
        ang = np.deg2rad(z_to_angle_deg(tv))
        x0, y0 = inner_r*np.cos(ang), inner_r*np.sin(ang)
        x1, y1 = (inner_r-0.06)*np.cos(ang), (inner_r-0.06)*np.sin(ang)
        ax.plot([x0, x1], [y0, y1], linewidth=1.2, color="#F3F4F6")

        xt, yt = (inner_r-0.14)*np.cos(ang), (inner_r-0.14)*np.sin(ang)
        ax.text(
            xt, yt,
            f"{tv:+d}" if tv != 0 else "0",
            ha="center", va="center",
            fontsize=10, color="#F3F4F6"
        )

    # arc labels
    draw_arc_text(
        ax, "DEBIT",
        radius=arc_label_radius,
        theta_start_deg=z_to_angle_deg(-3),
        theta_end_deg=z_to_angle_deg(-1),
        fontsize=13,
        color="#F3F4F6",
        weight="bold"
    )
    draw_arc_text(
        ax, "CREDIT",
        radius=arc_label_radius,
        theta_start_deg=z_to_angle_deg(1),
        theta_end_deg=z_to_angle_deg(3),
        fontsize=13,
        color="#F3F4F6",
        weight="bold"
    )

    # needle
    needle_ang = np.deg2rad(z_to_angle_deg(z_display))
    needle_len = 0.92
    xN, yN = needle_len*np.cos(needle_ang), needle_len*np.sin(needle_ang)
    ax.plot([0, xN], [0, yN], linewidth=3.2, color="#111827")
    ax.add_patch(Circle((0, 0), 0.045, edgecolor="#111827", facecolor=bg, linewidth=1.6))

    # pointer label
    label_r = 1.04
    xL, yL = label_r*np.cos(needle_ang), label_r*np.sin(needle_ang)
    ax.text(
        xL, yL,
        f"Z = {z:.2f}",
        ha="center", va="center",
        fontsize=11, fontweight="bold",
        color="#111827",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#F3F4F6", edgecolor="#D1D5DB", linewidth=1.0)
    )

    # date formatting
    date_str = ""
    if date is not None:
        try:
            date_str = str(pd.to_datetime(date).date())
        except Exception:
            date_str = str(date)

    # title & subtitle (ticker only; no extra Yahoo calls)
    ax.set_title("IV Z-Score", fontsize=16, pad=22, color="#F3F4F6", fontweight="bold")
    if ticker:
        ax.text(
            0.5, 1.015, f"{ticker}",
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=12, color="#F3F4F6"
        )

    bucket_label = {
        1: "< -2σ",
        2: "[-2σ, -1σ)",
        3: "[-1σ, 1σ)",
        4: "[1σ, 2σ)",
        5: ">= 2σ",
    }[bucket]

    footer_left = f"Latest trading day: {date_str}" if date_str else ""
    footer_right = f"Section: {bucket} ({bucket_label})"

    ax.text(-1.12, -0.22, footer_left, ha="left", va="center", fontsize=10, color="#F3F4F6")
    ax.text( 1.12, -0.22, footer_right, ha="right", va="center", fontsize=10, color="#F3F4F6")

    ax.set_xlim(-1.20, 1.20)
    ax.set_ylim(-0.30, 1.25)

    return fig

# -----------------------
# UI inputs
# -----------------------
ticker = st.text_input("Enter ticker (examples: ^VIX, SPY, AAPL)", value="SPY").strip().upper()

# Display usage status UNDER ticker entry (as requested)
remaining, reset_in, used = remaining_runs_and_reset()
st.caption(
    f"Session limit: **{MAX_RUNS} runs / {WINDOW_MINUTES} minutes**. "
    f"Used: **{used}**, Remaining: **{remaining}**. "
    f"Next reset in: **{human_time(reset_in)}**."
)

col1, col2 = st.columns(2)
with col1:
    window_days = st.number_input("Window (days)", min_value=30, max_value=400, value=WINDOW, step=1)
with col2:
    period = st.text_input("Period (yfinance)", value=DEFAULT_PERIOD).strip()

run = st.button("Run")

# -----------------------
# Run logic
# -----------------------
if run:
    if not ticker:
        st.warning("Please enter a ticker.")
        st.stop()

    # cooldown guard
    now = _now_ts()
    last = st.session_state.last_run_ts
    if now - last < COOLDOWN_SECONDS:
        wait_more = int(COOLDOWN_SECONDS - (now - last))
        st.warning(f"Please wait {wait_more}s before running again (cooldown to avoid rate limits).")
        st.stop()

    # quota guard
    remaining, reset_in, used = remaining_runs_and_reset()
    if remaining <= 0:
        st.error(
            f"You’ve reached the session limit (**{MAX_RUNS} runs per {WINDOW_MINUTES} minutes**).\n\n"
            f"Try again in **{human_time(reset_in)}**."
        )
        st.stop()

    # register this run attempt
    st.session_state.last_run_ts = now
    st.session_state.run_times.append(now)

    with st.spinner("Fetching data from Yahoo Finance…"):
        try:
            df = fetch_with_retries(ticker, period)
        except Exception as e:
            if is_rate_limit_error(e):
                st.error(
                    "Yahoo Finance **rate-limited** this Streamlit Cloud IP.\n\n"
                    "What you can do:\n"
                    "- Wait **3–10 minutes** and try once\n"
                    "- Try a different ticker (SPY is usually safest)\n"
                    "- Avoid repeated rapid clicks (cooldown is active)\n"
                )
            else:
                st.exception(e)
            st.stop()

    if df.empty:
        st.error(f"No data returned for ticker: {ticker}")
        st.stop()

    if len(df) < int(window_days):
        st.error(f"Not enough daily rows for {ticker}. Got {len(df)}, need {int(window_days)}. Increase PERIOD.")
        st.stop()

    hist = df.tail(int(window_days)).copy()

    # IV proxy (your original approach)
    hist["IV"] = hist["Close"]

    mu = float(hist["IV"].mean())
    sigma = float(hist["IV"].std(ddof=1))

    if sigma == 0 or np.isnan(sigma):
        st.error("Std dev is zero/NaN; cannot compute z-score.")
        st.stop()

    current_date = hist.index[-1]
    current_iv = float(hist["IV"].iloc[-1])
    current_z = float((current_iv - mu) / sigma)

    st.success(f"Loaded {len(df)} rows for {ticker}. Latest trading day used: {pd.to_datetime(current_date).date()}")

    st.write(
        f"**IV (proxy)**: {current_iv:.4f}  \n"
        f"**Mean (window)**: {mu:.4f}  \n"
        f"**Std dev (window)**: {sigma:.4f}  \n"
        f"**Z-score**: {current_z:.4f}"
    )

    fig = zscore_speedometer_custom(
        current_z,
        ticker=ticker,
        date=current_date,
        window=int(window_days),
        bg="#666666",
        arc_label_radius=1.14
    )
    st.pyplot(fig, clear_figure=True)

else:
    st.info("Enter a ticker and click Run.")
