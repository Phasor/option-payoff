import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def plot_options_strategy(options, spot_price, net_premium, price_range_percentage=0.5, underlying_name="Asset"):
    """
    Calculates and plots the payoff of an options strategy.
    """
    min_price = spot_price * (1 - price_range_percentage)
    max_price = spot_price * (1 + price_range_percentage)
    spot_prices = np.linspace(min_price, max_price, 500)
    total_payoff = np.zeros_like(spot_prices, dtype=float)

    for opt in options:
        strike = opt['strike']
        is_long = opt['is_long']
        is_call = opt['is_call']
        quantity = opt.get('quantity', 1)

        intrinsic_value = np.maximum(spot_prices - strike, 0) if is_call else np.maximum(strike - spot_prices, 0)
        payoff = intrinsic_value * quantity * (1 if is_long else -1)
        total_payoff += payoff

    total_payoff -= net_premium

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(spot_prices, total_payoff, label="Net Payoff (incl. premium)", color='blue', linewidth=2.5)
    ax.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax.axvline(spot_price, color='green', linestyle='--', label=f"Spot Price: {spot_price}")
    ax.set_xlabel(f"{underlying_name} Price at Expiry")
    ax.set_ylabel("Profit / Loss")
    ax.set_title(f"{underlying_name} Options Strategy Payoff")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

# Streamlit App
st.title("Options Payoff Calculator 📈")

# User Inputs
spot_price = st.number_input("Current Spot Price", min_value=0.0, value=100.0)
net_premium = st.number_input("Net Premium Paid (positive) / Received (negative)", value=0.0)
underlying_name = st.text_input("Underlying Asset Name", value="Asset")

st.subheader("Define Your Options Strategy")
num_options = st.number_input("Number of Options", min_value=1, step=1, value=1)

options = []
for i in range(num_options):
    st.markdown(f"### Option {i+1}")
    strike = st.number_input(f"Strike Price (Option {i+1})", min_value=0.0, value=100.0)
    is_long = st.selectbox(f"Position (Option {i+1})", ["Long", "Short"]) == "Long"
    is_call = st.selectbox(f"Type (Option {i+1})", ["Call", "Put"]) == "Call"
    quantity = st.number_input(f"Quantity (Option {i+1})", min_value=1, step=1, value=1)
    
    options.append({'strike': strike, 'is_long': is_long, 'is_call': is_call, 'quantity': quantity})

# Show Payoff Graph
if st.button("Calculate Payoff"):
    plot_options_strategy(options, spot_price, net_premium, underlying_name=underlying_name)
