import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from typing import List

def find_breakeven_points(payoff: np.ndarray, prices: np.ndarray) -> List[float]:
    breakeven_points = []
    for i in range(len(payoff) - 1):
        if (payoff[i] <= 0 and payoff[i+1] > 0) or (payoff[i] >= 0 and payoff[i+1] < 0):
            x1, x2 = prices[i], prices[i+1]
            y1, y2 = payoff[i], payoff[i+1]
            if y1 != y2:
                breakeven = x1 - y1 * (x2 - x1) / (y2 - y1)
                breakeven_points.append(round(breakeven, 2))
    return breakeven_points

def identify_strategy_type(options):
    if len(options) == 2:
        if options[0]['is_call'] and options[1]['is_call']:
            if options[0]['is_long'] and not options[1]['is_long']:
                if options[0]['strike'] < options[1]['strike']:
                    return "Bull Call Spread"
                else:
                    return "Bear Call Spread"
        elif not options[0]['is_call'] and not options[1]['is_call']:
            if options[0]['is_long'] and not options[1]['is_long']:
                if options[0]['strike'] > options[1]['strike']:
                    return "Bear Put Spread"
                else:
                    return "Bull Put Spread"
    return "Unknown"

def plot_options_strategy(options, spot_price, net_premium, price_range_percentage=0.5, underlying_name="Asset"):
    strategy_type = identify_strategy_type(options)
    strikes = sorted(set(opt['strike'] for opt in options))
    max_strike = max(strikes)
    min_strike = min(strikes)
    adjusted_max_price = max(spot_price * (1 + price_range_percentage), max_strike * 1.2)
    adjusted_min_price = min(spot_price * (1 - price_range_percentage), min_strike * 0.8)

    base_prices = np.linspace(adjusted_min_price, adjusted_max_price, 1000)
    spot_prices = np.unique(np.concatenate([base_prices, strikes]))

    total_payoff = np.zeros_like(spot_prices, dtype=float)

    for opt in options:
        intrinsic_value = np.maximum(spot_prices - opt['strike'], 0) if opt['is_call'] else np.maximum(opt['strike'] - spot_prices, 0)
        total_payoff += intrinsic_value * opt['quantity'] * (1 if opt['is_long'] else -1)

    total_payoff_before_premium = total_payoff.copy()
    total_payoff -= net_premium

    breakeven_points = find_breakeven_points(total_payoff, spot_prices)
    max_profit = round(np.max(total_payoff), 2)
    max_profit_price = spot_prices[np.argmax(total_payoff)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(spot_prices, total_payoff, label="Net Payoff (incl. premium)", color='blue', linewidth=2.5)
    ax.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax.axvline(spot_price, color='green', linestyle='--', label=f"Spot Price: {spot_price}")
    ax.axvline(x=max_profit_price, color="red", linestyle="-.", alpha=0.8, label=f"Max Profit Level: {max_profit_price:,.2f}")

    for be in breakeven_points:
        ax.axvline(x=be, color="purple", linestyle=":", alpha=0.8, label=f"Breakeven: {be:,.2f}")
        ax.text(be, 0, f"{be:,.2f}", horizontalalignment='right', verticalalignment='bottom', 
                color='purple', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

    ax.set_xlabel(f"{underlying_name} Price at Expiry")
    ax.set_ylabel("Profit / Loss")

    title = f"{underlying_name} {strategy_type} Payoff" if strategy_type != "Unknown" else f"{underlying_name} Options Strategy Payoff"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    max_payoff_pct = (max_profit / net_premium) * 100 if net_premium != 0 else float('inf')
    st.pyplot(fig)

    return {
        "Max Payoff as % of Premium": f"{max_payoff_pct:.2f}%" if max_payoff_pct != float('inf') else "∞",
        "Max Payoff": f"${max_profit:.2f}",
        "Level of Underlying at Max Payoff": f"${max_profit_price:,.2f}",
        "Breakeven Price(s)": ", ".join([f"${be:,.2f}" for be in breakeven_points]) if breakeven_points else "None"
    }

st.title("Options Payoff Calculator 📈")

if st.button("Reset App"):
    st.session_state.clear()
    st.rerun()


spot_price = st.number_input("Current Spot Price", min_value=0.0, value=85000.0)
net_premium = st.number_input("Net Premium Paid (positive) / Received (negative)", value=1000.0)
underlying_name = st.text_input("Underlying Asset Name", value="Asset")
price_range = st.slider("Price Range Percentage", min_value=0.1, max_value=1.0, value=0.5, step=0.1)

st.subheader("Define Your Options Strategy")
num_options = st.number_input("Number of Options", min_value=1, step=1, value=2, key="num_options")

options = []
for i in range(num_options):
    st.markdown(f"### Option {i+1}")
    col1, col2 = st.columns(2)
    with col1:
        strike = st.number_input(f"Strike Price (Option {i+1})", min_value=0.0, value=85000.0 + 2000 * i, key=f"strike_{i}")
        is_long = st.selectbox(f"Position (Option {i+1})", ["Long", "Short"], index=0 if i == 0 else 1, key=f"long_{i}") == "Long"
    with col2:
        is_call = st.selectbox(f"Type (Option {i+1})", ["Call", "Put"], index=0, key=f"type_{i}") == "Call"
        quantity = st.number_input(f"Quantity (Option {i+1})", min_value=1, step=1, value=1, key=f"qty_{i}")

    options.append({
        'strike': strike,
        'is_long': is_long,
        'is_call': is_call,
        'quantity': quantity
    })

if st.button("Calculate Payoff"):
    metrics = plot_options_strategy(options, spot_price, net_premium, price_range_percentage=price_range, underlying_name=underlying_name)

    strategy_type = identify_strategy_type(options)
    if strategy_type != "Unknown":
        st.markdown(f"## {strategy_type} Analysis")
    else:
        st.subheader("Strategy Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Max Payoff", metrics["Max Payoff"])
        st.metric("Max Payoff as % of Premium", metrics["Max Payoff as % of Premium"])
    with col2:
        st.metric("Underlying at Max Payoff", metrics["Level of Underlying at Max Payoff"])
        st.metric("Breakeven Price(s)", metrics["Breakeven Price(s)"])
