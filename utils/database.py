import sqlite3
import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import joblib
import tensorflow as tf

# Database file path
DB_PATH = "trading_platform.db"

def init_database():
    """Initialize the database with required tables"""
    try:
        conn = sqlite3.connect(DB_PATH)
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
                status TEXT DEFAULT 'active'
            )
        ''')
        
        # Market data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                open_price REAL NOT NULL,
                high_price REAL NOT NULL,
                low_price REAL NOT NULL,
                close_price REAL NOT NULL,
                volume REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )
        ''')
        
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
                position_size REAL,
                stop_loss REAL,
                take_profit REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active',
                result TEXT,
                actual_return REAL
            )
        ''')
        
        # Trading history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                trade_type TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                amount REAL NOT NULL,
                commission REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                order_id TEXT,
                status TEXT DEFAULT 'filled',
                pnl REAL
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
                correct_predictions INTEGER
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
                total_return REAL
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
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                win_rate REAL,
                max_drawdown REAL,
                sharpe_ratio REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        print("Database initialized successfully")
        
    except Exception as e:
        print(f"Error initializing database: {e}")
    finally:
        conn.close()

def store_model(model_name: str, model: Any, scaler: Any, model_info: Dict):
    """Store a trained model in the database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Serialize model and scaler
        if model_info['type'] == 'LSTM':
            # Save Keras model to temporary file and read as binary
            temp_model_path = f"temp_{model_name}_model.h5"
            model.model.save(temp_model_path)
            
            with open(temp_model_path, 'rb') as f:
                model_data = f.read()
            
            # Clean up temporary file
            os.remove(temp_model_path)
            
        else:
            # For scikit-learn models
            model_data = pickle.dumps(model.model)
        
        scaler_data = pickle.dumps(scaler) if scaler else None
        features_json = json.dumps(model_info.get('features', []))
        hyperparams_json = json.dumps(model_info.get('hyperparameters', {}))
        
        # Insert or update model
        cursor.execute('''
            INSERT OR REPLACE INTO models (
                name, type, symbol, interval, train_mse, test_mse,
                train_mae, test_mae, train_r2, test_r2, features,
                hyperparameters, model_data, scaler_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_name,
            model_info['type'],
            model_info['symbol'],
            model_info['interval'],
            model_info.get('train_mse'),
            model_info.get('test_mse'),
            model_info.get('train_mae'),
            model_info.get('test_mae'),
            model_info.get('train_r2'),
            model_info.get('test_r2'),
            features_json,
            hyperparams_json,
            model_data,
            scaler_data
        ))
        
        conn.commit()
        print(f"Model {model_name} stored successfully")
        
    except Exception as e:
        print(f"Error storing model: {e}")
    finally:
        conn.close()

def get_models() -> List[Dict]:
    """Get all stored models"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT name, type, symbol, interval, created_at,
                   train_r2, test_r2, train_mse, test_mse,
                   features, hyperparameters, status
            FROM models WHERE status = 'active'
            ORDER BY created_at DESC
        ''')
        
        models = []
        for row in cursor.fetchall():
            features = json.loads(row[9]) if row[9] else []
            hyperparams = json.loads(row[10]) if row[10] else {}
            
            model_info = {
                'name': row[0],
                'type': row[1],
                'symbol': row[2],
                'interval': row[3],
                'created_at': row[4],
                'train_r2': row[5],
                'test_r2': row[6],
                'train_mse': row[7],
                'test_mse': row[8],
                'features': features,
                'hyperparameters': hyperparams,
                'status': row[11]
            }
            models.append(model_info)
        
        return models
        
    except Exception as e:
        print(f"Error getting models: {e}")
        return []
    finally:
        conn.close()

def get_model_by_name(model_name: str) -> Optional[Dict]:
    """Get a specific model by name"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT name, type, symbol, interval, created_at,
                   train_r2, test_r2, train_mse, test_mse,
                   features, hyperparameters, status
            FROM models WHERE name = ? AND status = 'active'
        ''', (model_name,))
        
        row = cursor.fetchone()
        if row:
            features = json.loads(row[9]) if row[9] else []
            hyperparams = json.loads(row[10]) if row[10] else {}
            
            return {
                'name': row[0],
                'type': row[1],
                'symbol': row[2],
                'interval': row[3],
                'created_at': row[4],
                'train_r2': row[5],
                'test_r2': row[6],
                'train_mse': row[7],
                'test_mse': row[8],
                'features': features,
                'hyperparameters': hyperparams,
                'status': row[11]
            }
        
        return None
        
    except Exception as e:
        print(f"Error getting model by name: {e}")
        return None
    finally:
        conn.close()

def load_model_from_db(model_name: str) -> Tuple[Any, Any]:
    """Load a model and scaler from database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT type, model_data, scaler_data FROM models 
            WHERE name = ? AND status = 'active'
        ''', (model_name,))
        
        row = cursor.fetchone()
        if not row:
            return None, None
        
        model_type, model_data, scaler_data = row
        
        # Deserialize scaler
        scaler = pickle.loads(scaler_data) if scaler_data else None
        
        # Deserialize model
        if model_type == 'LSTM':
            # Save binary data to temporary file and load Keras model
            temp_model_path = f"temp_{model_name}_load.h5"
            
            with open(temp_model_path, 'wb') as f:
                f.write(model_data)
            
            from .ml_models import LSTMModel
            model_instance = LSTMModel(input_shape=(60, 1))  # Placeholder shape
            model_instance.model = tf.keras.models.load_model(temp_model_path)
            model_instance.is_trained = True
            
            # Clean up temporary file
            os.remove(temp_model_path)
            
            model = model_instance
        
        else:
            # For scikit-learn models
            sklearn_model = pickle.loads(model_data)
            
            # Create appropriate model wrapper
            if model_type == 'Random Forest':
                from .ml_models import RandomForestModel
                model = RandomForestModel()
                model.model = sklearn_model
                model.is_trained = True
            elif model_type == 'SVM':
                from .ml_models import SVMModel
                model = SVMModel()
                model.model = sklearn_model
                model.scaler = scaler
                model.is_trained = True
            else:
                model = sklearn_model
        
        return model, scaler
        
    except Exception as e:
        print(f"Error loading model from database: {e}")
        return None, None
    finally:
        conn.close()

def store_market_data(symbol: str, df: pd.DataFrame):
    """Store market data in database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Prepare data for insertion
        data_to_insert = []
        for _, row in df.iterrows():
            data_to_insert.append((
                symbol,
                row.name if hasattr(row, 'name') else row['timestamp'],
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume'])
            ))
        
        # Insert data (ignore duplicates)
        cursor = conn.cursor()
        cursor.executemany('''
            INSERT OR IGNORE INTO market_data 
            (symbol, timestamp, open_price, high_price, low_price, close_price, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', data_to_insert)
        
        conn.commit()
        
    except Exception as e:
        print(f"Error storing market data: {e}")
    finally:
        conn.close()

def get_historical_data(symbol: str, start_date: datetime = None, 
                       end_date: datetime = None) -> pd.DataFrame:
    """Get historical market data"""
    try:
        conn = sqlite3.connect(DB_PATH)
        
        query = '''
            SELECT timestamp, open_price, high_price, low_price, close_price, volume
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
        
        df = pd.read_sql_query(query, conn, params=params)
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.columns = ['open', 'high', 'low', 'close', 'volume']
        
        return df
        
    except Exception as e:
        print(f"Error getting historical data: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def store_signal(signal_data: Dict):
    """Store trading signal in database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trading_signals (
                model_name, symbol, signal_type, confidence,
                current_price, predicted_price, position_size,
                stop_loss, take_profit, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal_data['model_name'],
            signal_data['symbol'],
            signal_data['signal_type'],
            signal_data['confidence'],
            signal_data['current_price'],
            signal_data['predicted_price'],
            signal_data.get('position_size'),
            signal_data.get('stop_loss'),
            signal_data.get('take_profit'),
            signal_data['timestamp']
        ))
        
        conn.commit()
        
    except Exception as e:
        print(f"Error storing signal: {e}")
    finally:
        conn.close()

def get_signals(model_name: str = None, symbol: str = None, 
               start_date: datetime = None, limit: int = 100) -> List[Dict]:
    """Get trading signals"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        query = 'SELECT * FROM trading_signals WHERE 1=1'
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
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        
        columns = [description[0] for description in cursor.description]
        signals = []
        
        for row in cursor.fetchall():
            signal_dict = dict(zip(columns, row))
            signals.append(signal_dict)
        
        return signals
        
    except Exception as e:
        print(f"Error getting signals: {e}")
        return []
    finally:
        conn.close()

def store_trade(trade_data: Dict):
    """Store trade execution in database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trading_history (
                symbol, trade_type, quantity, price, amount,
                commission, timestamp, order_id, status, pnl
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_data['symbol'],
            trade_data['trade_type'],
            trade_data['quantity'],
            trade_data['price'],
            trade_data['amount'],
            trade_data.get('commission', 0),
            trade_data.get('timestamp', datetime.now()),
            trade_data.get('order_id'),
            trade_data.get('status', 'filled'),
            trade_data.get('pnl')
        ))
        
        conn.commit()
        
    except Exception as e:
        print(f"Error storing trade: {e}")
    finally:
        conn.close()

def get_trading_history(symbol: str = None, start_date: datetime = None, 
                       limit: int = 100) -> List[Dict]:
    """Get trading history"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        query = 'SELECT * FROM trading_history WHERE 1=1'
        params = []
        
        if symbol:
            query += ' AND symbol = ?'
            params.append(symbol)
        
        if start_date:
            query += ' AND timestamp >= ?'
            params.append(start_date)
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        
        columns = [description[0] for description in cursor.description]
        trades = []
        
        for row in cursor.fetchall():
            trade_dict = dict(zip(columns, row))
            trades.append(trade_dict)
        
        return trades
        
    except Exception as e:
        print(f"Error getting trading history: {e}")
        return []
    finally:
        conn.close()

def store_model_performance(model_name: str, performance_metrics: Dict):
    """Store model performance metrics"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_performance (
                model_name, accuracy_score, precision_score, recall_score,
                f1_score, mse, mae, r2_score, prediction_count, correct_predictions
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_name,
            performance_metrics.get('accuracy_score'),
            performance_metrics.get('precision_score'),
            performance_metrics.get('recall_score'),
            performance_metrics.get('f1_score'),
            performance_metrics.get('mse'),
            performance_metrics.get('mae'),
            performance_metrics.get('r2_score'),
            performance_metrics.get('prediction_count'),
            performance_metrics.get('correct_predictions')
        ))
        
        conn.commit()
        
    except Exception as e:
        print(f"Error storing model performance: {e}")
    finally:
        conn.close()

def get_model_performance_history(model_name: str = None, days: int = 30) -> List[Dict]:
    """Get model performance history"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        start_date = datetime.now() - timedelta(days=days)
        
        query = '''
            SELECT * FROM model_performance 
            WHERE evaluation_date >= ?
        '''
        params = [start_date]
        
        if model_name:
            query += ' AND model_name = ?'
            params.append(model_name)
        
        query += ' ORDER BY evaluation_date DESC'
        
        cursor.execute(query, params)
        
        columns = [description[0] for description in cursor.description]
        performance_history = []
        
        for row in cursor.fetchall():
            perf_dict = dict(zip(columns, row))
            performance_history.append(perf_dict)
        
        return performance_history
        
    except Exception as e:
        print(f"Error getting model performance history: {e}")
        return []
    finally:
        conn.close()

def store_portfolio_snapshot(portfolio_data: Dict):
    """Store portfolio snapshot"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        positions_json = json.dumps(portfolio_data.get('positions', {}))
        
        cursor.execute('''
            INSERT INTO portfolio (
                total_value, cash_balance, positions, daily_return, total_return
            ) VALUES (?, ?, ?, ?, ?)
        ''', (
            portfolio_data['total_value'],
            portfolio_data['cash_balance'],
            positions_json,
            portfolio_data.get('daily_return'),
            portfolio_data.get('total_return')
        ))
        
        conn.commit()
        
    except Exception as e:
        print(f"Error storing portfolio snapshot: {e}")
    finally:
        conn.close()

def get_portfolio_history(days: int = 30) -> List[Dict]:
    """Get portfolio history"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        start_date = datetime.now() - timedelta(days=days)
        
        cursor.execute('''
            SELECT * FROM portfolio 
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        ''', (start_date,))
        
        columns = [description[0] for description in cursor.description]
        portfolio_history = []
        
        for row in cursor.fetchall():
            portfolio_dict = dict(zip(columns, row))
            if portfolio_dict['positions']:
                portfolio_dict['positions'] = json.loads(portfolio_dict['positions'])
            portfolio_history.append(portfolio_dict)
        
        return portfolio_history
        
    except Exception as e:
        print(f"Error getting portfolio history: {e}")
        return []
    finally:
        conn.close()

def cleanup_old_data(days_to_keep: int = 90):
    """Clean up old data from database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Clean up old market data
        cursor.execute('''
            DELETE FROM market_data 
            WHERE created_at < ?
        ''', (cutoff_date,))
        
        # Clean up old signals
        cursor.execute('''
            DELETE FROM trading_signals 
            WHERE timestamp < ?
        ''', (cutoff_date,))
        
        # Clean up old performance data
        cursor.execute('''
            DELETE FROM model_performance 
            WHERE evaluation_date < ?
        ''', (cutoff_date,))
        
        conn.commit()
        print(f"Cleaned up data older than {days_to_keep} days")
        
    except Exception as e:
        print(f"Error cleaning up old data: {e}")
    finally:
        conn.close()
