import sqlite3
import pandas as pd
import numpy as np
import pickle
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database file path
DB_PATH = "trading_platform.db"

# Try to import TensorFlow with fallback
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow not available. LSTM model support will be limited.")
    TENSORFLOW_AVAILABLE = False

class DatabaseManager:
    """Database manager with connection pooling and error handling"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the database connection"""
        self.conn = None
        self._init_database_tables()  # Renamed to avoid circular call
    
    def _init_database_tables(self):
        """Initialize database tables (separate from the module-level init_database)"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Models table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    type TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    train_mse REAL,
                    test_mse REAL,
                    train_mae REAL,
                    test_mae REAL,
                    train_r2 REAL,
                    test_r2 REAL,
                    features TEXT,
                    hyperparameters TEXT,
                    model_data BLOB,
                    scaler_data BLOB,
                    sequence_length INTEGER DEFAULT 60,
                    status TEXT DEFAULT 'active',
                    version INTEGER DEFAULT 1,
                    description TEXT
                )
            ''')
            
            # Market data table with improved indexing
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp)
                )
            ''')
            
            # Create indexes for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp)')
            
            # Trading signals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    current_price REAL NOT NULL,
                    predicted_price REAL NOT NULL,
                    price_change_pct REAL,
                    signal_strength TEXT,
                    position_size REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    status TEXT DEFAULT 'active',
                    result TEXT,
                    actual_return REAL,
                    quality_score REAL,
                    quality_rating TEXT,
                    metadata TEXT
                )
            ''')
            
            # Create indexes for signals
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_model ON trading_signals(model_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_symbol ON trading_signals(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON trading_signals(timestamp)')
            
            # Trading history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trading_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    trade_type TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    amount REAL NOT NULL,
                    commission REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    closed_at TIMESTAMP,
                    order_id TEXT UNIQUE,
                    status TEXT DEFAULT 'open',
                    pnl REAL,
                    pnl_pct REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    signal_id INTEGER,
                    strategy TEXT,
                    metadata TEXT
                )
            ''')
            
            # Model performance tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accuracy_score REAL,
                    precision_score REAL,
                    recall_score REAL,
                    f1_score REAL,
                    mse REAL,
                    mae REAL,
                    r2_score REAL,
                    prediction_count INTEGER,
                    correct_predictions INTEGER,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    win_rate REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    metadata TEXT
                )
            ''')
            
            # Portfolio tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_value REAL NOT NULL,
                    cash_balance REAL NOT NULL,
                    positions TEXT,
                    daily_return REAL,
                    total_return REAL,
                    daily_pnl REAL,
                    unrealized_pnl REAL,
                    realized_pnl REAL,
                    metadata TEXT
                )
            ''')
            
            # Strategy results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    start_date TIMESTAMP NOT NULL,
                    end_date TIMESTAMP NOT NULL,
                    initial_capital REAL NOT NULL,
                    final_value REAL NOT NULL,
                    total_return REAL NOT NULL,
                    total_return_pct REAL NOT NULL,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    win_rate REAL,
                    max_drawdown REAL,
                    max_drawdown_pct REAL,
                    sharpe_ratio REAL,
                    sortino_ratio REAL,
                    calmar_ratio REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            ''')
            
            # Backtest results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    start_date TIMESTAMP NOT NULL,
                    end_date TIMESTAMP NOT NULL,
                    initial_capital REAL NOT NULL,
                    final_value REAL NOT NULL,
                    total_return_pct REAL NOT NULL,
                    annualized_return_pct REAL,
                    volatility_pct REAL,
                    sharpe_ratio REAL,
                    sortino_ratio REAL,
                    max_drawdown_pct REAL,
                    total_trades INTEGER,
                    win_rate REAL,
                    profit_factor REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    parameters TEXT,
                    results TEXT
                )
            ''')
            
            conn.commit()
            logger.info("Database tables initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database tables: {e}")
            raise
    
    def get_connection(self):
        """Get database connection with retry logic"""
        try:
            if self.conn is None:
                self.conn = sqlite3.connect(
                    DB_PATH, 
                    check_same_thread=False,
                    timeout=30
                )
                self.conn.execute("PRAGMA journal_mode=WAL")
                self.conn.execute("PRAGMA synchronous=NORMAL")
                self.conn.execute("PRAGMA cache_size=10000")
            return self.conn
        except Exception as e:
            logger.error(f"Failed to get database connection: {e}")
            raise
    
    def close_connection(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None

# Module-level functions (these use the DatabaseManager singleton)

def init_database():
    """Initialize the database - public interface"""
    db = DatabaseManager()
    # Tables are already initialized in DatabaseManager._init_database_tables()
    logger.info("Database initialization complete")

def store_model(model_name: str, model: Any, scaler: Any, model_info: Dict):
    """Store a trained model in the database with versioning"""
    db = DatabaseManager()
    conn = db.get_connection()
    
    try:
        cursor = conn.cursor()
        
        # Check if model already exists to get version
        cursor.execute('SELECT version FROM models WHERE name = ?', (model_name,))
        existing = cursor.fetchone()
        version = (existing[0] + 1) if existing else 1
        
        # Serialize model and scaler
        model_data = None
        scaler_data = pickle.dumps(scaler) if scaler else None
        
        if model_info.get('type') == 'LSTM' and TENSORFLOW_AVAILABLE:
            # Save Keras model to temporary file and read as binary
            temp_model_path = f"temp_{model_name}_v{version}_model.h5"
            try:
                model.model.save(temp_model_path)
                with open(temp_model_path, 'rb') as f:
                    model_data = f.read()
            finally:
                if os.path.exists(temp_model_path):
                    os.remove(temp_model_path)
        else:
            # For scikit-learn and other models
            model_data = pickle.dumps(model.model if hasattr(model, 'model') else model)
        
        features_json = json.dumps(model_info.get('features', []))
        hyperparams_json = json.dumps(model_info.get('hyperparameters', {}))
        
        # Insert or update model
        cursor.execute('''
            INSERT OR REPLACE INTO models (
                name, type, symbol, interval, train_mse, test_mse,
                train_mae, test_mae, train_r2, test_r2, features,
                hyperparameters, model_data, scaler_data, sequence_length,
                status, version, description, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_name,
            model_info.get('type', 'unknown'),
            model_info.get('symbol', 'UNKNOWN'),
            model_info.get('interval', '1d'),
            model_info.get('train_mse'),
            model_info.get('test_mse'),
            model_info.get('train_mae'),
            model_info.get('test_mae'),
            model_info.get('train_r2'),
            model_info.get('test_r2'),
            features_json,
            hyperparams_json,
            model_data,
            scaler_data,
            model_info.get('sequence_length', 60),
            model_info.get('status', 'active'),
            version,
            model_info.get('description', ''),
            datetime.now()
        ))
        
        conn.commit()
        logger.info(f"Model {model_name} v{version} stored successfully")
        
    except Exception as e:
        logger.error(f"Error storing model {model_name}: {e}")
        conn.rollback()
        raise

def get_models(active_only: bool = True) -> List[Dict]:
    """Get all stored models with optional filtering"""
    db = DatabaseManager()
    conn = db.get_connection()
    
    try:
        cursor = conn.cursor()
        
        query = '''
            SELECT name, type, symbol, interval, created_at, updated_at,
                   train_r2, test_r2, train_mse, test_mse,
                   features, hyperparameters, status, version, description
            FROM models
        '''
        
        if active_only:
            query += " WHERE status = 'active'"
        
        query += ' ORDER BY created_at DESC'
        
        cursor.execute(query)
        
        models = []
        for row in cursor.fetchall():
            features = json.loads(row[10]) if row[10] else []
            hyperparams = json.loads(row[11]) if row[11] else {}
            
            model_info = {
                'name': row[0],
                'type': row[1],
                'symbol': row[2],
                'interval': row[3],
                'created_at': row[4],
                'updated_at': row[5],
                'train_r2': row[6],
                'test_r2': row[7],
                'train_mse': row[8],
                'test_mse': row[9],
                'features': features,
                'hyperparameters': hyperparams,
                'status': row[12],
                'version': row[13],
                'description': row[14]
            }
            models.append(model_info)
        
        return models
        
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return []

def get_models(active_only: bool = True) -> List[Dict]:
    """Get all stored models with optional filtering"""
    db = DatabaseManager()
    conn = db.get_connection()
    
    try:
        cursor = conn.cursor()
        
        query = '''
            SELECT name, type, symbol, interval, created_at, updated_at,
                   train_r2, test_r2, train_mse, test_mse,
                   features, hyperparameters, status, version, description
            FROM models
        '''
        
        if active_only:
            query += " WHERE status = 'active'"
        
        query += ' ORDER BY created_at DESC'
        
        cursor.execute(query)
        
        models = []
        for row in cursor.fetchall():
            features = json.loads(row[10]) if row[10] else []
            hyperparams = json.loads(row[11]) if row[11] else {}
            
            model_info = {
                'name': row[0],
                'type': row[1],
                'symbol': row[2],
                'interval': row[3],
                'created_at': row[4],
                'updated_at': row[5],
                'train_r2': row[6],
                'test_r2': row[7],
                'train_mse': row[8],
                'test_mse': row[9],
                'features': features,
                'hyperparameters': hyperparams,
                'status': row[12],
                'version': row[13],
                'description': row[14]
            }
            models.append(model_info)
        
        return models
        
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return []
    finally:
        pass

def get_model_by_name(model_name: str, version: int = None) -> Optional[Dict]:
    """Get a specific model by name with optional version"""
    db = DatabaseManager()
    conn = db.get_connection()
    
    try:
        cursor = conn.cursor()
        
        query = '''
            SELECT name, type, symbol, interval, created_at, updated_at,
                   train_r2, test_r2, train_mse, test_mse,
                   features, hyperparameters, status, version, description,
                   sequence_length
            FROM models WHERE name = ?
        '''
        
        params = [model_name]
        
        if version:
            query += ' AND version = ?'
            params.append(version)
        else:
            query += ' AND status = "active" ORDER BY version DESC LIMIT 1'
        
        cursor.execute(query, params)
        
        row = cursor.fetchone()
        if row:
            features = json.loads(row[10]) if row[10] else []
            hyperparams = json.loads(row[11]) if row[11] else {}
            
            return {
                'name': row[0],
                'type': row[1],
                'symbol': row[2],
                'interval': row[3],
                'created_at': row[4],
                'updated_at': row[5],
                'train_r2': row[6],
                'test_r2': row[7],
                'train_mse': row[8],
                'test_mse': row[9],
                'features': features,
                'hyperparameters': hyperparams,
                'status': row[12],
                'version': row[13],
                'description': row[14],
                'sequence_length': row[15]
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting model {model_name}: {e}")
        return None

def load_model_from_db(model_name: str, version: int = None) -> Tuple[Any, Any]:
    """Load a model and scaler from database with lazy imports"""
    db = DatabaseManager()
    conn = db.get_connection()
    
    try:
        cursor = conn.cursor()
        
        query = '''
            SELECT type, model_data, scaler_data, sequence_length 
            FROM models WHERE name = ?
        '''
        
        params = [model_name]
        if version:
            query += ' AND version = ?'
            params.append(version)
        else:
            query += ' AND status = "active" ORDER BY version DESC LIMIT 1'
        
        cursor.execute(query, params)
        
        row = cursor.fetchone()
        if not row:
            logger.error(f"Model {model_name} not found in database")
            return None, None
        
        model_type, model_data, scaler_data, sequence_length = row
        
        # Deserialize scaler
        scaler = None
        if scaler_data:
            try:
                scaler = pickle.loads(scaler_data)
            except Exception as e:
                logger.warning(f"Error loading scaler: {e}")
        
        # Deserialize model based on type
        model = None
        
        if model_type == 'LSTM' and TENSORFLOW_AVAILABLE:
            try:
                # Use temporary file for loading
                temp_model_path = f"temp_{model_name}_load.h5"
                with open(temp_model_path, 'wb') as f:
                    f.write(model_data)
                
                # Lazy import to avoid circular imports
                from utils.ml_models import LSTMModel
                
                # Create model instance with proper input shape
                input_shape = (sequence_length or 60, 1)  # Default if not specified
                model_instance = LSTMModel(input_shape=input_shape)
                model_instance.model = tf.keras.models.load_model(temp_model_path)
                model_instance.is_trained = True
                model = model_instance
                
                # Clean up
                if os.path.exists(temp_model_path):
                    os.remove(temp_model_path)
                    
            except Exception as e:
                logger.error(f"Error loading LSTM model: {e}")
        
        else:
            # For scikit-learn models
            try:
                sklearn_model = pickle.loads(model_data)
                
                # Create appropriate wrapper based on model type
                if model_type == 'Random Forest':
                    from utils.ml_models import RandomForestModel
                    wrapper = RandomForestModel()
                    wrapper.model = sklearn_model
                    wrapper.is_trained = True
                    model = wrapper
                    
                elif model_type == 'SVM':
                    from utils.ml_models import SVMModel
                    wrapper = SVMModel()
                    wrapper.model = sklearn_model
                    wrapper.scaler = scaler
                    wrapper.is_trained = True
                    model = wrapper
                    
                else:
                    # Return raw model for unknown types
                    model = sklearn_model
                    
            except Exception as e:
                logger.error(f"Error loading scikit-learn model: {e}")
        
        return model, scaler
        
    except Exception as e:
        logger.error(f"Error loading model from database: {e}")
        return None, None
def store_market_data(symbol: str, df: pd.DataFrame, batch_size: int = 1000):
    """Store market data in database efficiently with batch processing"""
    db = DatabaseManager()
    conn = db.get_connection()
    
    try:
        # Prepare data for insertion
        data_to_insert = []
        current_time = datetime.now()
        
        for idx, row in df.iterrows():
            timestamp = idx if isinstance(idx, datetime) else pd.to_datetime(idx)
            data_to_insert.append((
                symbol,
                timestamp,
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row.get('volume', 0)),
                current_time,
                current_time
            ))
        
        # Insert in batches
        cursor = conn.cursor()
        for i in range(0, len(data_to_insert), batch_size):
            batch = data_to_insert[i:i + batch_size]
            cursor.executemany('''
                INSERT OR REPLACE INTO market_data 
                (symbol, timestamp, open, high, low, close, volume, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', batch)
        
        conn.commit()
        logger.info(f"Stored {len(data_to_insert)} records for {symbol}")
        
    except Exception as e:
        logger.error(f"Error storing market data for {symbol}: {e}")
        conn.rollback()
    finally:
        pass

def get_historical_data(symbol: str, start_date: datetime = None, 
                       end_date: datetime = None, limit: int = None) -> pd.DataFrame:
    """Get historical market data with efficient querying"""
    db = DatabaseManager()
    conn = db.get_connection()
    
    try:
        query = '''
            SELECT timestamp, open, high, low, close, volume
            FROM market_data WHERE symbol = ?
        '''
        params = [symbol]
        
        if start_date:
            query += ' AND timestamp >= ?'
            params.append(start_date)
        
        if end_date:
            query += ' AND timestamp <= ?'
            params.append(end_date)
        
        query += ' ORDER BY timestamp'
        
        if limit:
            query += ' LIMIT ?'
            params.append(limit)
        
        df = pd.read_sql_query(query, conn, params=params, parse_dates=['timestamp'])
        
        if not df.empty:
            df.set_index('timestamp', inplace=True)
            # Ensure numeric types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(inplace=True)
        
        return df
        
    except Exception as e:
        logger.error(f"Error getting historical data for {symbol}: {e}")
        return pd.DataFrame()
    finally:
        pass

def store_signal(signal_data: Dict):
    """Store trading signal in database with comprehensive data"""
    db = DatabaseManager()
    conn = db.get_connection()
    
    try:
        cursor = conn.cursor()
        
        # Calculate expiration time (default 24 hours)
        expires_at = signal_data.get('expires_at')
        if not expires_at and 'timestamp' in signal_data:
            expires_at = pd.to_datetime(signal_data['timestamp']) + timedelta(hours=24)
        
        cursor.execute('''
            INSERT INTO trading_signals (
                model_name, symbol, signal_type, confidence,
                current_price, predicted_price, price_change_pct,
                signal_strength, position_size, stop_loss, take_profit,
                timestamp, expires_at, status, quality_score, quality_rating,
                metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal_data.get('model_name'),
            signal_data.get('symbol'),
            signal_data.get('signal_type', 'HOLD'),
            signal_data.get('confidence', 0),
            signal_data.get('current_price'),
            signal_data.get('predicted_price'),
            signal_data.get('price_change_pct'),
            signal_data.get('signal_strength', 'Weak'),
            signal_data.get('position_size'),
            signal_data.get('stop_loss'),
            signal_data.get('take_profit'),
            signal_data.get('timestamp', datetime.now()),
            expires_at,
            signal_data.get('status', 'active'),
            signal_data.get('quality_score'),
            signal_data.get('quality_rating'),
            json.dumps(signal_data.get('metadata', {}))
        ))
        
        conn.commit()
        signal_id = cursor.lastrowid
        logger.info(f"Stored signal #{signal_id} for {signal_data.get('model_name')}")
        
        return signal_id
        
    except Exception as e:
        logger.error(f"Error storing signal: {e}")
        conn.rollback()
        return None
    finally:
        pass

def get_signals(model_name: str = None, symbol: str = None, 
               start_date: datetime = None, end_date: datetime = None,
               signal_type: str = None, status: str = 'active',
               limit: int = 100) -> List[Dict]:
    """Get trading signals with advanced filtering"""
    db = DatabaseManager()
    conn = db.get_connection()
    
    try:
        query = '''
            SELECT * FROM trading_signals WHERE 1=1
        '''
        params = []
        
        if model_name:
            query += ' AND model_name = ?'
            params.append(model_name)
        
        if symbol:
            query += ' AND symbol = ?'
            params.append(symbol)
        
        if start_date:
            query += ' AND timestamp >= ?'
            params.append(start_date)
        
        if end_date:
            query += ' AND timestamp <= ?'
            params.append(end_date)
        
        if signal_type:
            query += ' AND signal_type = ?'
            params.append(signal_type)
        
        if status:
            query += ' AND status = ?'
            params.append(status)
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        cursor = conn.cursor()
        cursor.execute(query, params)
        
        columns = [description[0] for description in cursor.description]
        signals = []
        
        for row in cursor.fetchall():
            signal_dict = dict(zip(columns, row))
            # Parse metadata JSON
            if signal_dict.get('metadata'):
                try:
                    signal_dict['metadata'] = json.loads(signal_dict['metadata'])
                except:
                    signal_dict['metadata'] = {}
            signals.append(signal_dict)
        
        return signals
        
    except Exception as e:
        logger.error(f"Error getting signals: {e}")
        return []
    finally:
        pass

def update_signal_status(signal_id: int, status: str, result: str = None, 
                        actual_return: float = None):
    """Update signal status and results"""
    db = DatabaseManager()
    conn = db.get_connection()
    
    try:
        cursor = conn.cursor()
        
        query = 'UPDATE trading_signals SET status = ?'
        params = [status]
        
        if result:
            query += ', result = ?'
            params.append(result)
        
        if actual_return is not None:
            query += ', actual_return = ?'
            params.append(actual_return)
        
        query += ' WHERE id = ?'
        params.append(signal_id)
        
        cursor.execute(query, params)
        conn.commit()
        
        logger.info(f"Updated signal #{signal_id} status to {status}")
        
    except Exception as e:
        logger.error(f"Error updating signal #{signal_id}: {e}")
        conn.rollback()
    finally:
        pass

# Additional utility functions for trade management, portfolio tracking, etc.
# would follow the same pattern with comprehensive error handling

def cleanup_old_data(days_to_keep: int = 90, batch_size: int = 10000):
    """Clean up old data from database efficiently"""
    db = DatabaseManager()
    conn = db.get_connection()
    
    try:
        cursor = conn.cursor()
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        tables_to_clean = [
            'market_data',
            'trading_signals', 
            'model_performance',
            'trading_history'
        ]
        
        for table in tables_to_clean:
            logger.info(f"Cleaning up old {table} data...")
            
            # Delete in batches to avoid locking
            while True:
                cursor.execute(f'''
                    DELETE FROM {table} 
                    WHERE created_at < ? 
                    LIMIT {batch_size}
                ''', (cutoff_date,))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count == 0:
                    break
                
                logger.info(f"Deleted {deleted_count} rows from {table}")
        
        logger.info(f"Cleanup completed for data older than {days_to_keep} days")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        conn.rollback()
    finally:
        pass

def get_database_stats() -> Dict[str, Any]:
    """Get database statistics"""
    db = DatabaseManager()
    conn = db.get_connection()
    
    try:
        cursor = conn.cursor()
        stats = {}
        
        # Table row counts
        tables = ['models', 'market_data', 'trading_signals', 'trading_history', 
                 'model_performance', 'portfolio', 'strategy_results']
        
        for table in tables:
            cursor.execute(f'SELECT COUNT(*) FROM {table}')
            stats[f'{table}_count'] = cursor.fetchone()[0]
        
        # Market data stats
        cursor.execute('''
            SELECT symbol, COUNT(*), MIN(timestamp), MAX(timestamp)
            FROM market_data GROUP BY symbol
        ''')
        
        stats['market_data_by_symbol'] = {}
        for row in cursor.fetchall():
            stats['market_data_by_symbol'][row[0]] = {
                'count': row[1],
                'start_date': row[2],
                'end_date': row[3]
            }
        
        # Model stats
        cursor.execute('''
            SELECT type, COUNT(*), AVG(test_r2), MAX(test_r2)
            FROM models WHERE status = 'active' GROUP BY type
        ''')
        
        stats['models_by_type'] = {}
        for row in cursor.fetchall():
            stats['models_by_type'][row[0]] = {
                'count': row[1],
                'avg_r2': row[2],
                'max_r2': row[3]
            }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return {}
    finally:
        pass

# Initialize database when module is imported

# Example usage
if __name__ == "__main__":
    # Test database connection
    print("Database connection test:")
    init_database()
    # Get database statistics
    stats = get_database_stats()
    print(f"Database stats: {stats}")
    
    # Test model storage (would need actual model data)
    print("Database module loaded successfully")