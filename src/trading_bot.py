import sys
import os
import time
import pandas as pd
from datetime import datetime, timedelta
from binance.client import Client
from binance.enums import *

# Add current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from database import DatabaseHandler
from feature_engineering import FeatureEngineer
from model_manager import ModelManager

class TradingBot:
    def __init__(self):
        # Initialize components
        self.db = DatabaseHandler()
        # Use testnet=True for Binance Spot Testnet
        self.client = Client(config.API_KEY, config.API_SECRET, testnet=True)
        
        # Sync time with Binance server to avoid timestamp errors
        try:
            server_time = self.client.get_server_time()
            diff = server_time['serverTime'] - int(time.time() * 1000)
            self.client.timestamp_offset = diff
            print(f"Time synced. Offset: {diff}ms")
        except Exception as e:
            print(f"Warning: Could not sync time with Binance: {e}")

        self.fe = FeatureEngineer()
        self.model_manager = ModelManager(config.MODEL_PATH)
        
        # Load model
        try:
            self.model_manager.load_model()
        except FileNotFoundError:
            print("Warning: Model file not found. Please train and save the model first.")

    def startup(self):
        """
        Execute the startup sequence:
        1. Load local state
        2. Query Binance balances
        3. Reconcile
        """
        print("--- Starting Trading Bot ---")
        
        # 1. Load local state
        local_positions = self.db.get_all_open_positions()
        print(f"Local state: {len(local_positions)} open positions.")
        
        # 2. Query Binance balances
        account_info = self.client.get_account()
        balances = {b['asset']: float(b['free']) + float(b['locked']) for b in account_info['balances']}
        
        # 3. Reconcile for each symbol
        for symbol in config.SYMBOLS:
            base_asset = symbol.replace("USDT", "")
            binance_qty = balances.get(base_asset, 0.0)
            print(f"Binance balance for {base_asset}: {binance_qty}")
            self.reconcile(local_positions, binance_qty, base_asset, symbol)
        
        print("Startup complete. System ready.")

    def reconcile(self, local_positions, binance_qty, base_asset, symbol):
        """
        Reconcile local state with Binance state.
        """
        print(f"Reconciling state for {symbol}...")
        
        # Check if we have a local position for the symbol
        local_pos = next((p for p in local_positions if p['symbol'] == symbol), None)
        
        # Threshold for "zero" balance (dust)
        DUST_THRESHOLD = 0.0001 # Adjust based on asset
        
        has_binance_pos = binance_qty > DUST_THRESHOLD
        
        if has_binance_pos and not local_pos:
            print(f"CRITICAL: Binance has {binance_qty} {base_asset} but no local position found.")
            print("Action: Reconstructing position from recent trades.")
            self.reconstruct_position(symbol, binance_qty)
            
        elif local_pos and not has_binance_pos:
            print(f"CRITICAL: Local state shows position but Binance balance is {binance_qty}.")
            print("Action: Marking local position as closed.")
            self.db.close_position(symbol)
            
        elif local_pos and has_binance_pos:
            # Check for major discrepancies
            if abs(local_pos['qty'] - binance_qty) / binance_qty > 0.05: # 5% tolerance
                print(f"WARNING: Quantity mismatch. Local: {local_pos['qty']}, Binance: {binance_qty}")
                # Update local to match Binance? Or alert?
                # For safety, we update local to match reality
                print("Updating local quantity to match Binance.")
                local_pos['qty'] = binance_qty
                self.db.save_position(local_pos)
            else:
                print("State is consistent.")
        else:
            print("No positions open locally or on exchange.")

    def reconstruct_position(self, symbol, qty):
        """
        Attempt to reconstruct a missing position from recent trades.
        """
        trades = self.client.get_my_trades(symbol=symbol, limit=5)
        if not trades:
            print("No recent trades found to reconstruct position. Creating dummy entry.")
            # Create a dummy entry to prevent trading errors, but this requires manual intervention
            entry_price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])
            entry_time = datetime.now().isoformat()
        else:
            # Use the last buy trade
            last_buy = next((t for t in reversed(trades) if t['isBuyer']), None)
            if last_buy:
                entry_price = float(last_buy['price'])
                entry_time = datetime.fromtimestamp(last_buy['time'] / 1000).isoformat()
            else:
                entry_price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])
                entry_time = datetime.now().isoformat()

        # Default to Tier 2 if unknown
        position = {
            'symbol': symbol,
            'qty': qty,
            'entry_price': entry_price,
            'entry_time': entry_time,
            'tier': 2, # Conservative assumption
            'planned_exit': (datetime.now() + timedelta(hours=4)).isoformat(), # Exit next candle
            'ev_at_entry': 0.0, # Unknown
            'status': 'OPEN'
        }
        self.db.save_position(position)
        print(f"Reconstructed position: {position}")

    def get_market_data(self, symbol):
        """Fetch and prepare market data."""
        # Fetch enough data for feature engineering (need at least 40 periods for vol_40)
        klines = self.client.get_historical_klines(symbol, config.TIMEFRAME, "10 days ago UTC")
        
        df = pd.DataFrame(klines, columns=[
            "Open time", "Open", "High", "Low", "Close", "Volume",
            "Close time", "Quote asset volume", "Number of trades",
            "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
        ])
        
        df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")
        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
        
        # Drop the last row (incomplete candle) to match EDA logic
        df = df.iloc[:-1]
        
        return df

    def run_cycle(self):
        """
        Main trading cycle:
        1. Fetch data for ALL symbols
        2. Calculate features & Volatility
        3. Calculate Portfolio Weights (Inverse Volatility)
        4. Predict & Execute for each symbol
        """
        print(f"\n--- Running Cycle: {datetime.now()} ---")
        
        # Store data for portfolio calculation
        symbol_data = {}
        volatilities = {}
        
        # 1. Fetch & Process Data
        for symbol in config.SYMBOLS:
            try:
                df = self.get_market_data(symbol)
                df_features = self.fe.calculate_features(df)
                
                # Get latest volatility (vol_20)
                latest_vol = df_features.iloc[-1]['vol_20']
                volatilities[symbol] = latest_vol
                
                symbol_data[symbol] = df_features
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
        
        # 2. Calculate Weights (Inverse Volatility)
        # Weight = (1/Vol) / Sum(1/Vol)
        inv_vol_sum = sum(1/v for v in volatilities.values() if v > 0)
        weights = {}
        if inv_vol_sum > 0:
            for sym, vol in volatilities.items():
                if vol > 0:
                    weights[sym] = (1/vol) / inv_vol_sum
                else:
                    weights[sym] = 0
        else:
            # Fallback to equal weight if volatility is weird
            print("Warning: Invalid volatilities. Using equal weights.")
            weight = 1.0 / len(config.SYMBOLS)
            weights = {sym: weight for sym in config.SYMBOLS}
            
        print("Portfolio Weights:", {k: f"{v:.2%}" for k, v in weights.items()})

        # 3. Execute Logic for Each Symbol
        for symbol, df_features in symbol_data.items():
            print(f"\nProcessing {symbol}...")
            
            # Get the latest row for prediction
            latest_data = df_features.iloc[[-1]].copy()
            
            # Check if we have enough data
            X = self.fe.get_feature_data(latest_data)
            if X.isnull().values.any():
                print(f"Not enough data for {symbol}. Skipping.")
                continue

            # Predict
            probs, ev_signal = self.model_manager.predict(X)
            
            current_ev = ev_signal[0]
            prob_large_down = probs[0][0] 
            
            print(f"EV: {current_ev:.4f}, Prob(Large Down): {prob_large_down:.4%}")
            
            # Manage Position
            self.manage_positions(symbol, current_ev, prob_large_down, df_features.iloc[-1]['Close'], weights.get(symbol, 0))

    def manage_positions(self, symbol, ev, prob_large_down, current_price, weight):
        """
        Decide to buy, sell, or hold.
        """
        local_pos = self.db.get_open_position(symbol)
        
        # RISK CHECK
        if prob_large_down > config.RISK_THRESHOLD:
            print("Risk Threshold Breached! No new entries.")
            if local_pos:
                print("Closing existing position due to risk.")
                self.close_position(local_pos, current_price, "Risk Stop")
            return

        # ENTRY LOGIC
        if not local_pos:
            if ev >= config.TIER_1_EV_THRESHOLD:
                print("Signal: Tier 1 Entry")
                self.open_position(symbol, tier=1, price=current_price, ev=ev, weight=weight)
            elif ev >= config.TIER_2_EV_THRESHOLD:
                print("Signal: Tier 2 Entry")
                self.open_position(symbol, tier=2, price=current_price, ev=ev, weight=weight)
            else:
                print("No Entry Signal.")
        
        # EXIT / EXTENSION LOGIC
        else:
            planned_exit = datetime.fromisoformat(local_pos['planned_exit'])
            now = datetime.now()
            
            # Check if we are within the exit window (tolerance of 15 mins before planned exit)
            # This handles cases where execution is slightly before the exact second of planned exit
            if now >= (planned_exit - timedelta(minutes=15)):
                # Check for extension (Tier 1 only)
                if local_pos['tier'] == 1 and ev >= config.EXTENSION_EV_THRESHOLD:
                    print("Signal: Extending Tier 1 Position")
                    new_exit = planned_exit + timedelta(hours=4) 
                    local_pos['planned_exit'] = new_exit.isoformat()
                    self.db.save_position(local_pos)
                else:
                    print("Signal: Planned Exit Reached")
                    self.close_position(local_pos, current_price, "Planned Exit")
            else:
                print(f"Holding position. Planned exit: {planned_exit}")

    def open_position(self, symbol, tier, price, ev, weight):
        """Place a buy order and save position."""
        # Calculate quantity based on Portfolio Weight
        # Total Capital = USDT Balance + Value of Open Positions (Simplified: just use USDT for now)
        # Ideally, we should track Total Portfolio Value.
        
        usdt_balance = float(self.client.get_asset_balance(asset='USDT')['free'])
        
        # Target Allocation = Total_Capital * Weight
        # For simplicity in this MVP, let's assume we re-invest available USDT based on weight relative to remaining opportunities?
        # Or just: Trade Size = Current USDT * Weight? 
        # Better: Trade Size = Total Portfolio Value * Weight.
        # But we don't want to over-complicate.
        # Let's use: Trade Size = usdt_balance * weight (This is conservative, assumes full deployment eventually)
        
        trade_value = usdt_balance * weight
        
        if trade_value < 10: # Min trade
            print(f"Insufficient funds for trade (Allocated: ${trade_value:.2f}).")
            return

        qty = trade_value / price
        # Adjust precision (simplified)
        qty = round(qty, 5) 
        
        print(f"Placing BUY order for {qty} {symbol} at ~{price} (Alloc: {weight:.1%})")
        
        try:
            # REAL ORDER (Uncomment when ready)
            # order = self.client.order_market_buy(symbol=symbol, quantity=qty)
            # real_qty = float(order['executedQty'])
            # real_price = float(order['cummulativeQuoteQty']) / real_qty
            
            # SIMULATED
            real_qty = qty
            real_price = price
            
            hold_periods = config.TIER_1_HOLD_PERIOD if tier == 1 else config.TIER_2_HOLD_PERIOD
            planned_exit = datetime.now() + timedelta(hours=4 * hold_periods)
            # Align exit time to the start of the minute/hour (strip seconds/microseconds)
            planned_exit = planned_exit.replace(second=0, microsecond=0)
            
            position = {
                'symbol': symbol,
                'qty': real_qty,
                'entry_price': real_price,
                'entry_time': datetime.now().isoformat(),
                'tier': tier,
                'planned_exit': planned_exit.isoformat(),
                'ev_at_entry': ev,
                'status': 'OPEN'
            }
            self.db.save_position(position)
            print("Position opened and saved.")
            
        except Exception as e:
            print(f"Error placing order: {e}")

    def close_position(self, position, price, reason):
        """Place a sell order and update DB."""
        print(f"Placing SELL order for {position['qty']} {position['symbol']} at ~{price}. Reason: {reason}")
        
        try:
            # REAL ORDER (Uncomment when ready)
            # order = self.client.order_market_sell(symbol=position['symbol'], quantity=position['qty'])
            
            # SIMULATED
            self.db.close_position(position['symbol'])
            
            # Log trade
            trade = {
                'symbol': position['symbol'],
                'side': 'SELL',
                'qty': position['qty'],
                'price': price,
                'timestamp': datetime.now().isoformat(),
                'pnl': (price - position['entry_price']) * position['qty'],
                'strategy_info': {'reason': reason, 'tier': position['tier']}
            }
            self.db.log_trade(trade)
            print("Position closed and logged.")
            
        except Exception as e:
            print(f"Error closing position: {e}")

if __name__ == "__main__":
    bot = TradingBot()
    bot.startup()
    # bot.run_cycle() # Run once or in a loop
