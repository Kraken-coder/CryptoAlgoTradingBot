import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

class DatabaseHandler:
    def __init__(self, db_path: str = "trading_state.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database with necessary tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Positions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                qty REAL NOT NULL,
                entry_price REAL NOT NULL,
                entry_time TEXT NOT NULL,
                tier INTEGER NOT NULL,
                planned_exit TEXT NOT NULL,
                ev_at_entry REAL,
                status TEXT DEFAULT 'OPEN'
            )
        ''')

        # Trades table (history)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                qty REAL NOT NULL,
                price REAL NOT NULL,
                timestamp TEXT NOT NULL,
                pnl REAL,
                commission REAL,
                strategy_info TEXT
            )
        ''')

        # Portfolio state (snapshots)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_state (
                timestamp TEXT PRIMARY KEY,
                total_balance REAL,
                available_balance REAL,
                positions_value REAL,
                risk_exposure REAL
            )
        ''')

        conn.commit()
        conn.close()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def save_position(self, position: Dict[str, Any]):
        """Save or update an open position."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO positions 
            (symbol, qty, entry_price, entry_time, tier, planned_exit, ev_at_entry, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            position['symbol'],
            position['qty'],
            position['entry_price'],
            position['entry_time'],
            position['tier'],
            position['planned_exit'],
            position.get('ev_at_entry'),
            position.get('status', 'OPEN')
        ))
        
        conn.commit()
        conn.close()

    def get_open_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the open position for a symbol."""
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM positions WHERE symbol = ? AND status = "OPEN"', (symbol,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None

    def close_position(self, symbol: str):
        """Mark a position as closed (or delete it if you prefer only keeping active ones in this table)."""
        # Here we delete it from positions and it should be logged in trades
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM positions WHERE symbol = ?', (symbol,))
        conn.commit()
        conn.close()

    def log_trade(self, trade: Dict[str, Any]):
        """Log a completed trade."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades (symbol, side, qty, price, timestamp, pnl, commission, strategy_info)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade['symbol'],
            trade['side'],
            trade['qty'],
            trade['price'],
            trade['timestamp'],
            trade.get('pnl'),
            trade.get('commission'),
            json.dumps(trade.get('strategy_info', {}))
        ))
        
        conn.commit()
        conn.close()

    def get_all_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions."""
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM positions WHERE status = "OPEN"')
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]

    def get_closed_trades(self) -> List[Dict[str, Any]]:
        """Get all closed trades (SELLs) which have PnL."""
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM trades WHERE side = "SELL" ORDER BY timestamp DESC')
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]

    def get_recent_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent trades (BUY and SELL)."""
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?', (limit,))
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
