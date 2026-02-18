https://ivgauge.streamlit.app/

IV Gauge is a simple Streamlit web app that helps options traders understand whether implied volatility is relatively high or low for a ticker.

The app pulls daily market data from Yahoo Finance using yfinance and uses the most recent trading day available.

It calculates a 252-day rolling mean and standard deviation (user-adjustable window) to measure how “normal” today’s IV level is.

Because true options IV is not always available for every ticker, the app uses a practical proxy from the downloaded data to keep it widely usable.

The result is displayed as a Z-score, showing how many standard deviations the current value is from its recent average.

The Z-score is mapped into five easy sections to make interpretation fast (very low → very high).

A speedometer-style gauge chart visualizes the section and highlights where the current value sits on the distribution.

To reduce Yahoo rate-limits, the app uses caching, retries, and conservative request behavior.

It also enforces a session limit of 5 runs per 30 minutes, with a live counter shown under the ticker input.

Overall, this project demonstrates a clean, beginner-friendly way to combine data fetching, statistics, and visualization into a practical trading tool.
