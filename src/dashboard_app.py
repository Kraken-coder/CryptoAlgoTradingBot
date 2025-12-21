import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta
import time
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import DatabaseHandler
import config
from binance.client import Client

# Page Config
st.set_page_config(
    page_title="Crypto Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

from feature_engineering import FeatureEngineer
from model_manager import ModelManager

# Initialize
@st.cache_resource
def init_resources():
    db = DatabaseHandler()
    client = Client(config.API_KEY, config.API_SECRET, testnet=True)
    
    # Sync time with Binance server to avoid timestamp errors
    try:
        server_time = client.get_server_time()
        diff = server_time['serverTime'] - int(time.time() * 1000)
        client.timestamp_offset = diff
    except Exception as e:
        st.error(f"Time sync error: {e}")
        
    # Initialize Model
    fe = FeatureEngineer()
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rf_model.pkl')
    mm = ModelManager(model_path)
    try:
        mm.load_model()
    except:
        pass # Handle gracefully if model missing
        
    return db, client, fe, mm

db, client, fe, mm = init_resources()

# --- Sidebar ---
st.sidebar.title("🤖 Trading Bot")
st.sidebar.markdown(f"**Status:** Active 🟢")
st.sidebar.markdown(f"**Strategy:** Random Forest + Volatility")
st.sidebar.markdown(f"**Timeframe:** {config.TIMEFRAME}")

# Auto-refresh
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 300, 60)
st_autorefresh(interval=refresh_interval * 1000, key="dataframerefresh")

if st.sidebar.button("Refresh Now"):
    st.rerun()

# --- Main Functions ---

def get_portfolio_value():
    # 1. Get USDT Balance
    try:
        account = client.get_account()
        balances = {b['asset']: float(b['free']) + float(b['locked']) for b in account['balances']}
        usdt_balance = balances.get('USDT', 0.0)
    except Exception as e:
        st.error(f"Binance API Error: {e}")
        return 0, {}, {}

    # 2. Get Open Positions Value
    positions = db.get_all_open_positions()
    pos_value = 0.0
    
    # Fetch current prices
    prices = {}
    for p in positions:
        sym = p['symbol']
        try:
            ticker = client.get_symbol_ticker(symbol=sym)
            price = float(ticker['price'])
            prices[sym] = price
            pos_value += p['qty'] * price
        except:
            prices[sym] = p['entry_price'] # Fallback
            pos_value += p['qty'] * p['entry_price']

    total_value = usdt_balance + pos_value
    return total_value, balances, prices

def calculate_metrics(closed_trades):
    if not closed_trades:
        return 0, 0, 0, 0
    
    df = pd.DataFrame(closed_trades)
    total_pnl = df['pnl'].sum()
    win_rate = (df['pnl'] > 0).mean()
    n_trades = len(df)
    avg_pnl = df['pnl'].mean()
    
    return total_pnl, win_rate, n_trades, avg_pnl

# --- Dashboard Layout ---

st.title("🚀 Live Trading Dashboard")

# --- Info Note ---
now_utc = datetime.utcnow()
current_block_start_hour = (now_utc.hour // 4) * 4
current_candle_open = now_utc.replace(hour=current_block_start_hour, minute=0, second=0, microsecond=0)
last_completed_candle_open = current_candle_open - timedelta(hours=4)
last_completed_candle_close = current_candle_open

st.info(f"""
**ℹ️ Trading Info:**
- The bot executes trades **only** at the close of 4-hour candles (00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC).
- Predictions displayed here are based on the **current incomplete candle** and may change before the candle closes.
- **Last Actual 4-Hour Candle:** {last_completed_candle_open.strftime('%Y-%m-%d %H:%M')} UTC (Closed at {last_completed_candle_close.strftime('%H:%M')})
""")

# 1. Portfolio Overview
total_value, balances, current_prices = get_portfolio_value()
closed_trades = db.get_closed_trades()
total_pnl, win_rate, n_trades, avg_pnl = calculate_metrics(closed_trades)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Portfolio Value", f"${total_value:,.2f}")

with col2:
    st.metric("Realized P&L", f"${total_pnl:,.2f}", delta_color="normal")

with col3:
    st.metric("Win Rate", f"{win_rate:.1%}")

with col4:
    st.metric("Total Trades", n_trades)

# 2. Active Positions
st.subheader("📊 Active Positions")
positions = db.get_all_open_positions()

if positions:
    pos_data = []
    for p in positions:
        curr_price = current_prices.get(p['symbol'], p['entry_price'])
        unrealized_pnl = (curr_price - p['entry_price']) * p['qty']
        pnl_pct = (curr_price - p['entry_price']) / p['entry_price']
        
        pos_data.append({
            "Symbol": p['symbol'],
            "Tier": p['tier'],
            "Entry Price": p['entry_price'],
            "Current Price": curr_price,
            "Qty": p['qty'],
            "Value": p['qty'] * curr_price,
            "Unrealized P&L ($)": unrealized_pnl,
            "Unrealized P&L (%)": pnl_pct,
            "Planned Exit": p['planned_exit']
        })
    
    df_pos = pd.DataFrame(pos_data)
    
    # Formatting
    st.dataframe(
        df_pos.style.format({
            "Entry Price": "${:.4f}",
            "Current Price": "${:.4f}",
            "Value": "${:.2f}",
            "Unrealized P&L ($)": "${:+.2f}",
            "Unrealized P&L (%)": "{:+.2%}"
        }),
        use_container_width=True
    )
    
    # Portfolio Allocation Chart
    fig_alloc = px.pie(df_pos, values='Value', names='Symbol', title='Current Allocation')
    st.plotly_chart(fig_alloc, use_container_width=True)

else:
    st.info("No active positions currently.")

# 3. Recent Trade History
st.subheader("📜 Recent Trade History")
recent_trades = db.get_recent_trades(20)

if recent_trades:
    df_trades = pd.DataFrame(recent_trades)
    st.dataframe(
        df_trades[['timestamp', 'symbol', 'side', 'qty', 'price', 'pnl', 'strategy_info']],
        use_container_width=True
    )
else:
    st.text("No trades recorded yet.")

# 4. Model & Strategy Insights (Placeholder for now)
st.subheader("🧠 Model Insights")
col_m1, col_m2 = st.columns(2)

with col_m1:
    st.markdown("### Risk Management")
    st.write(f"- **Risk Threshold:** {config.RISK_THRESHOLD:.0%}")
    st.write(f"- **Tier 1 EV:** {config.TIER_1_EV_THRESHOLD}")
    st.write(f"- **Tier 2 EV:** {config.TIER_2_EV_THRESHOLD}")

with col_m2:
    st.markdown("### Asset Universe")
    st.write(", ".join(config.SYMBOLS))

# --- Helper for Predictions ---
def get_prediction(symbol):
    try:
        # Fetch data (need enough for features)
        klines = client.get_klines(symbol=symbol, interval=config.TIMEFRAME, limit=100)
        df = pd.DataFrame(klines, columns=[
            "Open time", "Open", "High", "Low", "Close", "Volume",
            "Close time", "Quote asset volume", "Number of trades",
            "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
        ])
        
        df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")
        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
        
        # Calculate Features
        df_features = fe.calculate_features(df)
        latest_data = df_features.iloc[[-1]].copy()
        X = fe.get_feature_data(latest_data)
        
        if X.isnull().values.any():
            return None, "Insufficient Data"
            
        # Predict
        probs, ev_signal = mm.predict(X)
        ev = ev_signal[0]
        prob_down = probs[0][0] # Prob of Large Down
        
        return ev, prob_down
    except Exception as e:
        return None, str(e)

# 5. Model Predictions & Watchlist
st.subheader("🧠 Model Predictions & Watchlist")

# Session State for Custom Symbols
if 'custom_symbols' not in st.session_state:
    st.session_state.custom_symbols = []

# Input for new symbol
col_add1, col_add2 = st.columns([3, 1])
with col_add1:
    new_symbol = st.text_input("Add Symbol to Watchlist (e.g. DOGEUSDT)")
with col_add2:
    if st.button("Add Symbol"):
        if new_symbol:
            clean_sym = new_symbol.upper().strip()
            # Validate
            try:
                client.get_symbol_ticker(symbol=clean_sym)
                if clean_sym not in config.SYMBOLS and clean_sym not in st.session_state.custom_symbols:
                    st.session_state.custom_symbols.append(clean_sym)
                    st.success(f"Added {clean_sym}")
                else:
                    st.warning("Symbol already in list")
            except:
                st.error("Invalid Symbol on Binance")

# Combine lists
all_symbols = list(config.SYMBOLS) + st.session_state.custom_symbols

# Create Prediction Table
pred_data = []
for sym in all_symbols:
    ev, prob_down = get_prediction(sym)
    
    # Get current price
    try:
        price = float(client.get_symbol_ticker(symbol=sym)['price'])
    except:
        price = 0.0
        
    if ev is not None:
        signal = "HOLD"
        if prob_down > config.RISK_THRESHOLD:
            signal = "RISK STOP 🔴"
        elif ev >= config.TIER_1_EV_THRESHOLD:
            signal = "BUY (Tier 1) 🟢"
        elif ev >= config.TIER_2_EV_THRESHOLD:
            signal = "BUY (Tier 2) 🟡"
            
        pred_data.append({
            "Symbol": sym,
            "Price": price,
            "Expected Value (EV)": ev,
            "Crash Prob": prob_down,
            "Signal": signal
        })
    else:
         pred_data.append({
            "Symbol": sym,
            "Price": price,
            "Expected Value (EV)": 0,
            "Crash Prob": 0,
            "Signal": "Error/No Data"
        })

if pred_data:
    df_pred = pd.DataFrame(pred_data)
    st.dataframe(
        df_pred.style.format({
            "Price": "${:.4f}",
            "Expected Value (EV)": "{:.4f}",
            "Crash Prob": "{:.2%}"
        }).applymap(lambda x: 'color: red' if 'RISK' in str(x) else ('color: green' if 'BUY' in str(x) else ''), subset=['Signal']),
        use_container_width=True
    )

# 6. Market Data (Price Charts)
st.subheader("📈 Market Data")
selected_symbol = st.selectbox("Select Symbol", config.SYMBOLS)

if selected_symbol:
    try:
        # Fetch klines (candlestick data)
        klines = client.get_klines(symbol=selected_symbol, interval=Client.KLINE_INTERVAL_15MINUTE, limit=100)
        df_klines = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df_klines['timestamp'] = pd.to_datetime(df_klines['timestamp'], unit='ms')
        df_klines['close'] = df_klines['close'].astype(float)
        
        fig_price = px.line(df_klines, x='timestamp', y='close', title=f'{selected_symbol} Price (15m)')
        st.plotly_chart(fig_price, use_container_width=True)
    except Exception as e:
        st.error(f"Error fetching market data: {e}")
