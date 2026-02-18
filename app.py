# Generated from: StreamLit-IV_Gauge.ipynb
# Converted at: 2026-02-18T20:09:56.770Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import numpy as np
import pandas as pd
import yfinance as yf

import streamlit as st

st.set_page_config(page_title="IV Gauge", layout="centered")
st.title("IV Gauge (Z-Score)")

ticker = st.text_input("Enter ticker", value="SPY").strip().upper()
run = st.button("Run")


WINDOW = 252
PERIOD = "2y"

# ✅ workaround: use history() instead of download()
df = yf.Ticker(ticker).history(period=PERIOD, interval="1d", auto_adjust=True).dropna()

if df.empty:
    raise ValueError(f"No data returned for ticker: {ticker}")

if len(df) < WINDOW:
    raise ValueError(f"Not enough data for {ticker}: got {len(df)} rows, need {WINDOW}. Increase PERIOD.")

hist = df.tail(WINDOW).copy()
hist["IV"] = hist["Close"]

mu = hist["IV"].mean()
sigma = hist["IV"].std(ddof=1)

if sigma == 0 or np.isnan(sigma):
    raise ValueError("Std dev is zero/NaN; cannot compute z-score.")

current_date = hist.index[-1]
current_iv = float(hist["IV"].iloc[-1])
current_z  = float((current_iv - mu) / sigma)

print(f"Latest trading day used: {current_date.date()}")
print(f"IV (proxy) value: {current_iv:.4f}")
print(f"Z-score (252D): {current_z:.4f}")


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

bucket = z_bucket(current_z)

labels = {
    1: "< -2σ",
    2: "[-2σ, -1σ)",
    3: "[-1σ, 1σ)",
    4: "[1σ, 2σ)",
    5: ">= 2σ",
}

print(f"Z-score (252D): {current_z:.4f}  →  Section {bucket}")


# ✅ Updated Gauge (your exact tweaks):
# - Section 4 color: #FF7F50
# - Section 5 color: #A30234
# - Background: #666666
# - Curved arc labels: "DEBIT" and "CREDIT" (capitalized)
# - Arc label radius: 1.14
# - Font changed (set globally via Matplotlib rcParams)
# - Title "IV Z-Score" + ticker description on next line with ~1.5 line spacing

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle

# ---- Font (change here if you want a specific installed font) ----
plt.rcParams["font.family"] = "DejaVu Sans"   # safe default (available almost everywhere)
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

def get_ticker_description(ticker: str) -> str:
    """Fetch a human-friendly name from Yahoo via yfinance, with safe fallbacks."""
    try:
        info = yf.Ticker(ticker).info or {}
        for key in ("longName", "shortName", "displayName", "name"):
            val = info.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    except Exception:
        pass
    return ""

def draw_arc_text(ax, text, radius, theta_start_deg, theta_end_deg,
                  fontsize=12, color="#F3F4F6", weight="bold"):
    """
    Draw curved text along an arc. Characters are placed at equal angle steps.
    """
    text = str(text)
    n = max(len(text), 1)
    angles = np.linspace(theta_start_deg, theta_end_deg, n)

    for ch, ang_deg in zip(text, angles):
        if ch == " ":
            continue
        ang = np.deg2rad(ang_deg)
        x = radius * np.cos(ang)
        y = radius * np.sin(ang)

        # Tangent rotation
        rot = ang_deg - 90

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

    # Section colors (your palette)
    colors = {
        1: "#0B5D1E",  # dark green
        2: "#7CCB7A",  # light green
        3: "#2F6FDB",  # blue
        4: "#FF7F50",  # coral (requested)
        5: "#A30234",  # deep red (requested)
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

    # Figure and background
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

    # Curved labels: DEBIT over [-3,-1], CREDIT over [1,3]
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

    # Ticker description
    desc = get_ticker_description(ticker) if ticker else ""
    if ticker and desc and desc.upper() != ticker.upper():
        subtitle = f"{ticker} — {desc}"
    elif ticker:
        subtitle = f"{ticker}"
    else:
        subtitle = ""

    # Title + subtitle with ~1.5 line spacing (placed manually)
    ax.set_title("IV Z-Score", fontsize=16, pad=22, color="#F3F4F6", fontweight="bold")
    if subtitle:
        ax.text(
            0.5, 1.015, subtitle,
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=12, color="#F3F4F6"
        )

    # Footer labels
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

    plt.show()


# ---- Call (uses your existing variables) ----
# Assumes you already have: ticker, current_date, current_z, WINDOW
zscore_speedometer_custom(
    current_z,
    ticker=ticker,
    date=current_date,
    window=WINDOW,
    bg="#666666",
    arc_label_radius=1.14
)
