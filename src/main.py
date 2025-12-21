import time
import schedule
from datetime import datetime
from trading_bot import TradingBot

def job():
    print("Starting scheduled job...")
    bot.run_cycle()

if __name__ == "__main__":
    bot = TradingBot()
    bot.startup()
    
    # Schedule the job to run every 4 hours (aligned with candles ideally)
    # For now, we just run it immediately and then loop
    # In production, use a proper scheduler or cron
    
    print("Bot started. Waiting for next 4-hour candle close (00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC)...")
    
    # Simple loop
    while True:
        try:
            # Use UTC to align with Binance 4h candles
            now = datetime.utcnow()
            
            # Check if we are in the first 2 minutes of a 4-hour block
            if now.hour % 4 == 0 and 0 <= now.minute < 2:
                print(f"Candle closed at {now}! Running cycle...")
                bot.run_cycle()
                # Sleep to avoid double run in the same window
                time.sleep(120) 
            
            time.sleep(30) # Check every 30 seconds
        except KeyboardInterrupt:
            print("Stopping bot...")
            break
