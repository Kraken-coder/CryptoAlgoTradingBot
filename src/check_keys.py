import sys
import os
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import testcred_spot as testcred

import time

def sync_time(client):
    try:
        server_time = client.get_server_time()
        diff = server_time['serverTime'] - int(time.time() * 1000)
        client.timestamp_offset = diff
        print(f"Time synced. Offset: {diff}ms")
    except Exception as e:
        print(f"Warning: Could not sync time: {e}")

def test_spot_keys():
    print("\n--- Testing SPOT Testnet ---")
    try:
        client = Client(testcred.test_key, testcred.test_secret, testnet=True)
        sync_time(client)
        # Try a simple authenticated call
        account = client.get_account()
        print("SUCCESS: Keys are valid for SPOT Testnet.")
        print(f"Balances: {len(account['balances'])} assets found.")
        return True
    except BinanceAPIException as e:
        print(f"FAILED: {e}")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_futures_keys():
    print("\n--- Testing FUTURES Testnet ---")
    try:
        # python-binance uses the same client but different endpoints for futures
        # We don't pass testnet=True here because we want to manually configure if needed, 
        # but actually Client(..., testnet=True) should handle futures calls to the testnet URL automatically 
        # IF the library version is recent. 
        # However, let's be explicit to be sure.
        
        client = Client(testcred.test_key, testcred.test_secret, testnet=True)
        
        # Try a futures authenticated call
        # Note: python-binance might route this to real futures if not careful, 
        # but testnet=True usually sets the config for both.
        
        account = client.futures_account()
        print("SUCCESS: Keys are valid for FUTURES Testnet.")
        print(f"Total Wallet Balance: {account['totalWalletBalance']}")
        return True
    except BinanceAPIException as e:
        print(f"FAILED: {e}")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    print(f"Testing keys from testcred.py...")
    print(f"Key: {testcred.test_key[:4]}...{testcred.test_key[-4:]}")
    
    is_spot = test_spot_keys()
    is_futures = test_futures_keys()
    
    print("\n--- DIAGNOSIS ---")
    if is_spot:
        print("✅ You have SPOT Testnet keys.")
        print("The bot is set up for Spot, so it should work. If it failed before, it might have been a glitch or IP issue.")
    elif is_futures:
        print("⚠️ You have FUTURES Testnet keys.")
        print("The bot is currently configured for SPOT trading.")
        print("You need to either:")
        print("1. Generate SPOT Testnet keys at https://testnet.binance.vision/")
        print("2. Or update the bot to trade Futures (requires code changes).")
    else:
        print("❌ Keys failed for both Spot and Futures Testnets.")
        print("Please regenerate your keys.")
        print("Spot Testnet: https://testnet.binance.vision/")
        print("Futures Testnet: https://testnet.binancefuture.com/")
