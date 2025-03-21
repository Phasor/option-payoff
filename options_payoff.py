import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Any

def find_breakeven_points(payoff: np.ndarray, prices: np.ndarray) -> List[float]:
    """
    Find all breakeven points where payoff crosses zero.
    Uses linear interpolation for more accurate points.
    """
    breakeven_points = []
    
    # Find where the payoff crosses zero
    for i in range(len(payoff) - 1):
        # Check if crossing from negative to positive or positive to negative
        if (payoff[i] <= 0 and payoff[i+1] > 0) or (payoff[i] >= 0 and payoff[i+1] < 0):
            # Linear interpolation to find more accurate breakeven
            x1, x2 = prices[i], prices[i+1]
            y1, y2 = payoff[i], payoff[i+1]
            
            if y1 != y2:  # Avoid division by zero
                breakeven = x1 - y1 * (x2 - x1) / (y2 - y1)
                breakeven_points.append(round(breakeven, 2))
    
    return breakeven_points

def identify_strategy_type(options):
    """Identify common options strategy types"""
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

def get_correct_max_level(options, total_payoff, spot_prices):
    """
    Get the correct underlying level for max profit based on option structure.
    Handles any combination of vanilla puts and calls by analyzing the payoff structure.
    
    Parameters:
    - options: List of option dictionaries with strike, is_long, is_call properties
    - total_payoff: Array of payoff values for each price point
    - spot_prices: Array of underlying price points
    
    Returns:
    - The price level where max profit occurs
    """
    # Find the maximum payoff value
    max_payoff = np.max(total_payoff)
    
    # Find all indices where the payoff equals the maximum value
    max_payoff_indices = np.where(np.isclose(total_payoff, max_payoff))[0]
    
    # If max payoff occurs at a single point, return that price
    if len(max_payoff_indices) == 1:
        return spot_prices[max_payoff_indices[0]]
    
    # If max payoff occurs at multiple points (flat section), we need special handling
    
    # First, extract continuous regions where max payoff occurs
    regions = []
    current_region = []
    
    for i in range(len(max_payoff_indices)):
        if i == 0 or max_payoff_indices[i] != max_payoff_indices[i-1] + 1:
            if current_region:
                regions.append(current_region)
            current_region = [max_payoff_indices[i]]
        else:
            current_region.append(max_payoff_indices[i])
    
    if current_region:
        regions.append(current_region)
    
    # Extract all strike prices
    strikes = sorted(set(opt['strike'] for opt in options))
    
    # Analyze the option structure
    long_calls = [opt['strike'] for opt in options if opt['is_long'] and opt['is_call']]
    short_calls = [opt['strike'] for opt in options if not opt['is_long'] and opt['is_call']]
    long_puts = [opt['strike'] for opt in options if opt['is_long'] and not opt['is_call']]
    short_puts = [opt['strike'] for opt in options if not opt['is_long'] and not opt['is_call']]
    
    # For a standard call spread (long lower strike, short higher strike)
    if long_calls and short_calls and not long_puts and not short_puts:
        if min(long_calls) < min(short_calls):  # Bull call spread
            for strike in short_calls:
                # Find the closest index to this strike
                idx = np.abs(spot_prices - strike).argmin()
                # Check if this index or a nearby one is in one of our max payoff regions
                for region in regions:
                    if any(abs(r - idx) <= 2 for r in region):  # Allow small tolerance
                        return strike
    
    # For a standard put spread (long higher strike, short lower strike)
    elif long_puts and short_puts and not long_calls and not short_calls:
        if max(long_puts) > max(short_puts):  # Bear put spread
            for strike in short_puts:
                # Find the closest index to this strike
                idx = np.abs(spot_prices - strike).argmin()
                # Check if this index or a nearby one is in one of our max payoff regions
                for region in regions:
                    if any(abs(r - idx) <= 2 for r in region):  # Allow small tolerance
                        return strike
    
    # For strategies with both calls and puts
    elif (long_calls or short_calls) and (long_puts or short_puts):
        # For each flat region of maximum payoff
        for region in regions:
            # Get the price range of this region
            start_price = spot_prices[region[0]]
            end_price = spot_prices[region[-1]]
            
            # Check if any strike prices fall within this flat region
            for strike in strikes:
                if start_price <= strike <= end_price:
                    return strike
            
            # If no strike in the region, check if region is bounded by strikes
            for strike in strikes:
                # Find nearest index to this strike
                strike_idx = np.abs(spot_prices - strike).argmin()
                
                # If a strike is just at the boundary of max payoff region
                if abs(strike_idx - region[0]) <= 2:  # Start of region
                    if not any(s for s in strikes if s < strike):  # No lower strikes
                        return strike
                elif abs(strike_idx - region[-1]) <= 2:  # End of region
                    if not any(s for s in strikes if s > strike):  # No higher strikes
                        return strike
    
    # For strategies with just calls
    if (long_calls or short_calls) and not long_puts and not short_puts:
        max_strike = max(strikes)
        # For call-only strategies, max profit often occurs at or above highest strike
        for region in regions:
            region_start_price = spot_prices[region[0]]
            if region_start_price >= max_strike:
                return max_strike
    
    # For strategies with just puts
    if (long_puts or short_puts) and not long_calls and not short_calls:
        min_strike = min(strikes)
        # For put-only strategies, max profit often occurs at or below lowest strike
        for region in regions:
            region_end_price = spot_prices[region[-1]]
            if region_end_price <= min_strike:
                return min_strike
    
    # If no specific pattern is found, use practical heuristics
    # If the flat region extends to infinity, use the appropriate boundary
    if max_payoff_indices[-1] == len(spot_prices) - 1:  # Upper bound
        # For strategies with short calls, use the highest short call strike
        if short_calls:
            return max(short_calls)
        # Otherwise use the edge of our price range
        return spot_prices[max_payoff_indices[0]]
    
    if max_payoff_indices[0] == 0:  # Lower bound
        # For strategies with short puts, use the lowest short put strike
        if short_puts:
            return min(short_puts)
        # Otherwise use the edge of our price range
        return spot_prices[max_payoff_indices[-1]]
    
    # Default: return the midpoint of the first flat region
    region = regions[0]
    midpoint_idx = region[len(region) // 2]
    return spot_prices[midpoint_idx]

def plot_options_strategy(options, spot_price, net_premium, price_range_percentage=0.5, underlying_name="Asset"):
    """
    Calculates and plots the payoff of an options strategy with metrics table.
    """
    # Identify the strategy type
    strategy_type = identify_strategy_type(options)
    
    # For call spreads and similar strategies, we need a wider range to capture max payoff
    # Ensure we go at least 20% beyond the highest strike price
    max_strike = max([opt['strike'] for opt in options])
    min_strike = min([opt['strike'] for opt in options])
    
    adjusted_max_price = max(spot_price * (1 + price_range_percentage), max_strike * 1.2)
    adjusted_min_price = min(spot_price * (1 - price_range_percentage), min_strike * 0.8)
    
    # Create price range with more points for better accuracy
    spot_prices = np.linspace(adjusted_min_price, adjusted_max_price, 1000)
    total_payoff = np.zeros_like(spot_prices, dtype=float)

    for opt in options:
        strike = opt['strike']
        is_long = opt['is_long']
        is_call = opt['is_call']
        quantity = opt.get('quantity', 1)

        intrinsic_value = np.maximum(spot_prices - strike, 0) if is_call else np.maximum(strike - spot_prices, 0)
        payoff = intrinsic_value * quantity * (1 if is_long else -1)
        total_payoff += payoff

    # Save payoff before premium adjustment for calculations
    total_payoff_before_premium = total_payoff.copy()
    
    # Adjust for premium
    total_payoff -= net_premium

    # Finding breakeven points
    breakeven_points = find_breakeven_points(total_payoff, spot_prices)

    # Calculate metrics for the table
    max_profit = np.max(total_payoff)
    max_loss = np.min(total_payoff)
    
    # Get the correct price level for max profit using the enhanced function
    max_profit_price = get_correct_max_level(options, total_payoff, spot_prices)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(spot_prices, total_payoff, label="Net Payoff (incl. premium)", color='blue', linewidth=2.5)
    ax.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax.axvline(spot_price, color='green', linestyle='--', label=f"Spot Price: {spot_price}")
    
    # Add max profit line
    ax.axvline(x=max_profit_price, color="red", linestyle="-.", 
              alpha=0.8, label=f"Max Profit Level: {max_profit_price:,.2f}")
    
    # Add breakeven lines
    for be in breakeven_points:
        ax.axvline(x=be, color="purple", linestyle=":", alpha=0.8, label=f"Breakeven: {be:,.2f}")
        ax.text(be, 0, f"{be:,.2f}", horizontalalignment='right', verticalalignment='bottom', 
                color='purple', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    ax.set_xlabel(f"{underlying_name} Price at Expiry")
    ax.set_ylabel("Profit / Loss")
    
    if strategy_type != "Unknown":
        title = f"{underlying_name} {strategy_type} Payoff"
    else:
        title = f"{underlying_name} Options Strategy Payoff"
        
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Calculate max payoff as percentage of premium invested
    if net_premium > 0:  # Premium paid
        max_payoff_pct = (max_profit / net_premium) * 100 if net_premium != 0 else float('inf')
    else:  # Premium received
        max_payoff_pct = (max_profit / abs(net_premium)) * 100 if net_premium != 0 else float('inf')

    st.pyplot(fig)
    
    # Return metrics for the table
    return {
        "Max Payoff as % of Premium": f"{max_payoff_pct:.2f}%" if max_payoff_pct != float('inf') else "∞",
        "Max Payoff": f"${max_profit:.2f}",
        "Level of Underlying at Max Payoff": f"${max_profit_price:,.2f}",
        "Breakeven Price(s)": ", ".join([f"${be:,.2f}" for be in breakeven_points]) if breakeven_points else "None"
    }

# Streamlit App
st.title("Options Payoff Calculator 📈")

# User Inputs
spot_price = st.number_input("Current Spot Price", min_value=0.0, value=85000.0)
net_premium = st.number_input("Net Premium Paid (positive) / Received (negative)", value=1000.0)
underlying_name = st.text_input("Underlying Asset Name", value="Asset")
price_range = st.slider("Price Range Percentage", min_value=0.1, max_value=1.0, value=0.5, step=0.1,
                       help="Controls how far from the current price to calculate payoffs")

st.subheader("Define Your Options Strategy")
num_options = st.number_input("Number of Options", min_value=1, step=1, value=2)

options = []
for i in range(num_options):
    st.markdown(f"### Option {i+1}")
    col1, col2 = st.columns(2)
    with col1:
        strike_default = 85000.0 if i == 0 else 90000.0  # Default values for the call spread example
        strike = st.number_input(f"Strike Price (Option {i+1})", min_value=0.0, value=strike_default)
        is_long = st.selectbox(f"Position (Option {i+1})", 
                              ["Long", "Short"], 
                              index=0 if i == 0 else 1) == "Long"  # First long, second short for call spread
    with col2:
        is_call = st.selectbox(f"Type (Option {i+1})", ["Call", "Put"], index=0) == "Call"  # Both calls for call spread
        quantity = st.number_input(f"Quantity (Option {i+1})", min_value=1, step=1, value=1)
    
    options.append({'strike': strike, 'is_long': is_long, 'is_call': is_call, 'quantity': quantity})

# Show Payoff Graph
if st.button("Calculate Payoff"):
    # Get metrics from the plot function
    metrics = plot_options_strategy(options, spot_price, net_premium, price_range_percentage=price_range, underlying_name=underlying_name)
    
    # Identify the strategy type
    strategy_type = identify_strategy_type(options)
    
    # Display strategy type if identified
    if strategy_type != "Unknown":
        st.markdown(f"## {strategy_type} Analysis")
    else:
        st.subheader("Strategy Analysis")
    
    # Display metrics in a more organized way
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Max Payoff", metrics["Max Payoff"])
        st.metric("Max Payoff as % of Premium", metrics["Max Payoff as % of Premium"])
    
    with col2:
        st.metric("Underlying at Max Payoff", metrics["Level of Underlying at Max Payoff"])
        st.metric("Breakeven Price(s)", metrics["Breakeven Price(s)"])