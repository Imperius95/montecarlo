import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Option Monte Carlo",
    page_icon=":chart_with_upwards_trend:",
)

st.title("Monte Carlo Option Pricing")

col1, col2 = st.columns(2)

with col1:
    s0 = st.number_input("Initial stock price", value=100.0, min_value=0.0)
    k = st.number_input("Strike price", value=100.0, min_value=0.0)
    t = st.number_input("Time to maturity (years)", value=1.0, min_value=0.01)

with col2:
    r = st.number_input("Risk-free rate", value=0.05)
    sigma = st.number_input("Volatility", value=0.2, min_value=0.0)
    simulations = st.number_input("Simulations", value=10000, step=1000, min_value=1)
    steps = st.number_input("Steps per simulation", value=252, step=1, min_value=1)

run = st.button("Run simulation")

if run:
    steps = int(steps)
    simulations = int(simulations)

    dt = t / steps
    z = np.random.standard_normal((steps, simulations))
    price_paths = np.zeros_like(z)
    price_paths[0] = s0

    for i in range(1, steps):
        price_paths[i] = price_paths[i - 1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z[i]
        )

    terminal_prices = price_paths[-1]
    call_payoff = np.maximum(terminal_prices - k, 0)
    put_payoff = np.maximum(k - terminal_prices, 0)
    discount = np.exp(-r * t)
    call_price = discount * call_payoff.mean()
    put_price = discount * put_payoff.mean()

    st.subheader("Option prices")
    col_call, col_put = st.columns(2)
    col_call.metric("Call", f"{call_price:.2f}")
    col_put.metric("Put", f"{put_price:.2f}")

    st.subheader("Sample price paths")
    sample_paths = price_paths[:, : min(10, simulations)]
    st.line_chart(pd.DataFrame(sample_paths))

    st.subheader("Distribution of terminal prices")
    counts = pd.Series(terminal_prices).value_counts(bins=50).sort_index()
    hist_df = pd.DataFrame({
        'price': [interval.left for interval in counts.index],
        'count': counts.values,
    }).set_index('price')
    st.bar_chart(hist_df)

