import time
import random
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle

# -----------------------------
# Streamlit page setup
# -----------------------------
st.set_page_config(page_title="IV Gauge", layout="centered")
st.title("IV Gauge (Z-Score)")

# -----------------------------
# Settings
# -----------------------------
WINDOW_DEFAULT = 252
PERIOD_DEFAULT = "2y"

# Matplotlib font (safe default)
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11


# -----------------------------
# Helpers
# -----------------------------
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
    """Draw curved text along an arc by placing characters with tangent rotation."""
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


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)  # cache for 6 hours
def fetch_history_cached(ticker: str, period: str, interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV history from Yahoo via yfinance (cached)."""
    df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    return df.dropna()


def fetch_with_retries(ticker: str, period: str, interval: str = "1d", attempts: int = 5) -> pd.DataFrame:
    """Retry wrapper with exponential backoff + jitter (helps with transient Yahoo issues)."""
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            return fetch_history_cached(ticker, period, interval)
        except Exception as e:
            last_err = e
            sleep_s = min(2 ** attempt, 20) + random.uniform(0.2, 1.5)
            time.sleep(sleep_s)
    raise last_err


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)  # cache 24h (optional)
def get_ticker_description_cached(ticker: str) -> str:
    """Optional: Fetch a human-friendly name (this may trigger rate-limit; cached)."""
    try:
        info = yf.Ticker(ticker).info or {}
        for key in ("longName", "shortName", "displayName", "name"):
            val = info.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    except Exception:
        return ""
    return ""


def zscore_speedometer_custom(
    z,
    ticker="",
    date=None,
    window=252,
    bg="#666666",
    arc_label_radius=1.14,
    subtitle=""
):
    """Return a Matplotlib Figure with the Z-score gauge."""
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

    # Draw colored sections
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

        # Section numbers
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

    # Ticks
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

    # Curved arc labels
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

    # Needle
    needle_ang = np.deg2rad(z_to_angle_deg(z_display))
    needle_len = 0.92
    xN, yN = needle_len*np.cos(needle_ang), needle_len*np.sin(needle_ang)
    ax.plot([0, xN], [0, yN], linewidth=3.2, color="#111827")
    ax.add_patch(Circle((0, 0), 0.045, edgecolor="#111827", facecolor=bg, linewidth=1.6))

    # Pointer label
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

    # Date string
    date_str = ""
    if date is not None:
        try:
            date_str = str(pd.to_datetime(date).date())
        except Exception:
            date_str = str(date)

    # Title + subtitle
    ax.set_title("IV Z-Score", fontsize=16, pad=22, color="#F3F4F6", fontweight="bold")
    if subtitle:
        ax.text(
            0.5, 1.015, subtitle,
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=12, color="#F3F4F6"
        )

    # Footer
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


# -----------------------------
# UI (Form)
# -----------------------------
with st.form("run_form"):
    c1, c2 = st.columns([2, 1])
    with c1:
        ticker = st.text_input("Enter ticker (examples: ^VIX, SPY, AAPL)", value="SPY").strip().upper()
    with c2:
        window = st.number_input("Window (days)", min_value=30, max_value=400, value=WINDOW_DEFAULT, step=1)

    period = st.text_input("Period (yfinance)", value=PERIOD_DEFAULT).strip()
    show_desc = st.checkbox("Fetch ticker description (extra Yahoo call)", value=False)

    run = st.form_submit_button("Run")


# -----------------------------
# Run logic
# -----------------------------
if run:
    if not ticker:
        st.warning("Please enter a ticker.")
        st.stop()

    with st.spinner("Fetching data…"):
        try:
            df = fetch_with_retries(ticker, period=period, interval="1d", attempts=5)
        except Exception as e:
            msg = str(e)
            if "Too Many Requests" in msg or "YFRateLimitError" in msg:
                st.error(
                    "Yahoo Finance rate-limited this Streamlit Cloud IP.\n\n"
                    "Try again in 1–5 minutes, or reduce calls, or deploy with a dedicated IP."
                )
            else:
                st.exception(e)
            st.stop()

    if df.empty:
        st.error(f"No data returned for ticker: {ticker}")
        st.stop()

    if len(df) < int(window):
        st.error(f"Not enough daily rows for {ticker}. Got {len(df)}, need {int(window)}. Increase PERIOD.")
        st.stop()

    # Use last WINDOW rows
    hist = df.tail(int(window)).copy()

    # Your current "IV proxy"
    hist["IV"] = hist["Close"]

    mu = float(hist["IV"].mean())
    sigma = float(hist["IV"].std(ddof=1))

    if sigma == 0 or np.isnan(sigma):
        st.error("Std dev is zero/NaN; cannot compute z-score.")
        st.stop()

    current_date = hist.index[-1]
    current_iv = float(hist["IV"].iloc[-1])
    current_z = float((current_iv - mu) / sigma)

    # Show key values on screen (instead of print)
    st.success(f"Loaded {len(df)} rows for {ticker}. Last date: {pd.to_datetime(current_date).date()}")
    st.write(f"**IV (proxy) value:** {current_iv:.4f}")
    st.write(f"**Z-score ({int(window)}D):** {current_z:.4f} → **Section {z_bucket(current_z)}**")

    # Optional description (can cause rate-limit; cached if enabled)
    subtitle = ticker
    if show_desc:
        with st.spinner("Fetching ticker description…"):
            desc = get_ticker_description_cached(ticker)
        if desc and desc.upper() != ticker.upper():
            subtitle = f"{ticker} — {desc}"

    # Build figure and display it
    fig = zscore_speedometer_custom(
        current_z,
        ticker=ticker,
        date=current_date,
        window=int(window),
        bg="#666666",
        arc_label_radius=1.14,
        subtitle=subtitle
    )
    st.pyplot(fig, clear_figure=True)

else:
    st.info("Enter a ticker and click Run.")

