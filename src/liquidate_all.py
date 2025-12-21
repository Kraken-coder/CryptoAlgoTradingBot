import sys
import os
import time
from binance.client import Client
from binance.helpers import round_step_size
from binance.enums import *

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config

def liquidate_all():
    print("⚠️  STARTING LIQUIDATION - SELLING ALL ASSETS FOR USDT ⚠️")
    
    client = Client(config.API_KEY, config.API_SECRET, testnet=True)
    
    # Sync time
    try:
        server_time = client.get_server_time()
        diff = server_time['serverTime'] - int(time.time() * 1000)
        client.timestamp_offset = diff
    except Exception as e:
        print(f"Time sync error: {e}")

    # Get Exchange Info for filters
    print("Fetching exchange info...")
    exchange_info = client.get_exchange_info()
    symbols_info = {s['symbol']: s for s in exchange_info['symbols']}

    # Get Balances
    print("Checking balances...")
    account = client.get_account()
    balances = account['balances']

    for b in balances:
        asset = b['asset']
        free = float(b['free'])
        
        if asset == 'USDT':
            print(f"💰 Keeping USDT Balance: {free}")
            continue
            
        if free > 0:
            symbol = f"{asset}USDT"
            
            if symbol not in symbols_info:
                # print(f"Skipping {asset} (No USDT pair found)")
                continue
                
            # Get filters
            s_info = symbols_info[symbol]
            lot_size_filter = next(f for f in s_info['filters'] if f['filterType'] == 'LOT_SIZE')
            step_size = float(lot_size_filter['stepSize'])
            min_qty = float(lot_size_filter['minQty'])
            
            # Adjust quantity
            qty = round_step_size(free, step_size)
            
            # Ensure we don't sell more than we have due to rounding up
            if qty > free:
                qty = round_step_size(free - step_size, step_size)

            if qty < min_qty:
                print(f"Skipping {asset}: Balance {free} is below min_qty {min_qty}")
                continue

            print(f"Selling {qty} {asset} for USDT...")
            try:
                order = client.create_order(
                    symbol=symbol,
                    side=SIDE_SELL,
                    type=ORDER_TYPE_MARKET,
                    quantity=qty
                )
                print(f"✅ Sold {asset}: {order['status']}")
            except Exception as e:
                print(f"❌ Failed to sell {asset}: {e}")

    # Delete Database to reset bot state
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trading_state.db')
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            print(f"✅ Deleted database: {db_path}")
        except Exception as e:
            print(f"❌ Failed to delete database: {e}")
    else:
        print("ℹ️  No database file found to delete.")

    print("🎉 Liquidation Complete. You are now in USDT.")

if __name__ == "__main__":
    liquidate_all()
