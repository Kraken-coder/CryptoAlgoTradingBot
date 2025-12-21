import time
import schedule
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
    
    print("Bot started. Press Ctrl+C to exit.")
    
    # Run once immediately
    bot.run_cycle()
    
    # Simple loop
    while True:
        try:
            time.sleep(60) # Check every minute? Or sleep for 4 hours?
            # Ideally, we check if a new candle has closed.
            # For this example, we'll just sleep.
        except KeyboardInterrupt:
            print("Stopping bot...")
            break
