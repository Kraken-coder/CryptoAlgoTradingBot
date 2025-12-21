import os
from dotenv import load_dotenv

# Load environment variables from .env file
# Assuming .env is in the parent directory of src (i.e., in crypto/)
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

# API Keys
API_KEY = os.getenv("BINANCE_SPOT_TESTNET_API_KEY")
API_SECRET = os.getenv("BINANCE_SPOT_TESTNET_API_SECRET")

# Trading Configuration

# Symbols (Portfolio)
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT']
TIMEFRAME = "4h"

# Risk Management
RISK_THRESHOLD = 0.20  # Max probability of Large Down
TRANSACTION_COST = 0.001

# Strategy Tiers
TIER_1_EV_THRESHOLD = 0.15
TIER_2_EV_THRESHOLD = 0.10
TIER_1_HOLD_PERIOD = 3
TIER_2_HOLD_PERIOD = 1
EXTENSION_EV_THRESHOLD = 0.05

# Model
MODEL_PATH = "rf_model.pkl"
