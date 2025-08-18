import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import ta
from ta.utils import dropna
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """
    Data processing utility for financial time series data
    """
    
    def __init__(self, scaler_type='standard'):
        """
        Initialize data processor
        
        Args:
            scaler_type: Type of scaler to use ('standard', 'minmax', 'robust')
        """
        self.scaler_type = scaler_type
        self.scaler = self._get_scaler()
        self.feature_columns = []
        self.target_column = None
        
    def _get_scaler(self):
        """Get the appropriate scaler"""
        if self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'minmax':
            return MinMaxScaler()
        elif self.scaler_type == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unsupported scaler type: {self.scaler_type}")
    
    def add_technical_indicators(self, df):
        """
        Add technical indicators to the dataframe
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators
        """
        df_processed = df.copy()
        
        try:
            # Clean data
            df_processed = dropna(df_processed)
            
            # Price-based indicators
            df_processed['sma_10'] = ta.trend.sma_indicator(df_processed['close'], window=10)
            df_processed['sma_20'] = ta.trend.sma_indicator(df_processed['close'], window=20)
            df_processed['sma_50'] = ta.trend.sma_indicator(df_processed['close'], window=50)
            
            df_processed['ema_10'] = ta.trend.ema_indicator(df_processed['close'], window=10)
            df_processed['ema_20'] = ta.trend.ema_indicator(df_processed['close'], window=20)
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df_processed['close'])
            df_processed['bb_high'] = bollinger.bollinger_hband()
            df_processed['bb_low'] = bollinger.bollinger_lband()
            df_processed['bb_mid'] = bollinger.bollinger_mavg()
            df_processed['bb_width'] = (df_processed['bb_high'] - df_processed['bb_low']) / df_processed['bb_mid']
            df_processed['bb_position'] = (df_processed['close'] - df_processed['bb_low']) / (df_processed['bb_high'] - df_processed['bb_low'])
            
            # RSI
            df_processed['rsi'] = ta.momentum.rsi(df_processed['close'], window=14)
            df_processed['rsi_30'] = (df_processed['rsi'] < 30).astype(int)
            df_processed['rsi_70'] = (df_processed['rsi'] > 70).astype(int)
            
            # MACD
            macd = ta.trend.MACD(df_processed['close'])
            df_processed['macd'] = macd.macd()
            df_processed['macd_signal'] = macd.macd_signal()
            df_processed['macd_histogram'] = macd.macd_diff()
            df_processed['macd_bullish'] = (df_processed['macd'] > df_processed['macd_signal']).astype(int)
            
            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(df_processed['high'], df_processed['low'], df_processed['close'])
            df_processed['stoch_k'] = stoch.stoch()
            df_processed['stoch_d'] = stoch.stoch_signal()
            df_processed['stoch_oversold'] = (df_processed['stoch_k'] < 20).astype(int)
            df_processed['stoch_overbought'] = (df_processed['stoch_k'] > 80).astype(int)
            
            # Williams %R
            df_processed['williams_r'] = ta.momentum.williams_r(df_processed['high'], df_processed['low'], df_processed['close'])
            
            # Average True Range (ATR)
            df_processed['atr'] = ta.volatility.average_true_range(df_processed['high'], df_processed['low'], df_processed['close'])
            
            # Commodity Channel Index (CCI)
            df_processed['cci'] = ta.trend.cci(df_processed['high'], df_processed['low'], df_processed['close'])
            
            # Volume indicators (if volume data is available)
            if 'volume' in df_processed.columns:
                df_processed['volume_sma'] = ta.volume.volume_sma(df_processed['close'], df_processed['volume'])
                df_processed['volume_ratio'] = df_processed['volume'] / df_processed['volume_sma']
                
                # On Balance Volume (OBV)
                df_processed['obv'] = ta.volume.on_balance_volume(df_processed['close'], df_processed['volume'])
                
                # Volume Price Trend (VPT)
                df_processed['vpt'] = ta.volume.volume_price_trend(df_processed['close'], df_processed['volume'])
                
                # Money Flow Index (MFI)
                df_processed['mfi'] = ta.volume.money_flow_index(df_processed['high'], df_processed['low'], 
                                                               df_processed['close'], df_processed['volume'])
            
            # Price action features
            df_processed['price_change'] = df_processed['close'].pct_change()
            df_processed['price_change_abs'] = np.abs(df_processed['price_change'])
            
            # High-Low spread
            df_processed['hl_spread'] = (df_processed['high'] - df_processed['low']) / df_processed['close']
            
            # Open-Close spread  
            df_processed['oc_spread'] = (df_processed['close'] - df_processed['open']) / df_processed['open']
            
            # Volatility measures
            df_processed['volatility_10'] = df_processed['price_change'].rolling(window=10).std()
            df_processed['volatility_20'] = df_processed['price_change'].rolling(window=20).std()
            
            # Support and resistance levels
            df_processed['support_20'] = df_processed['low'].rolling(window=20).min()
            df_processed['resistance_20'] = df_processed['high'].rolling(window=20).max()
            df_processed['support_distance'] = (df_processed['close'] - df_processed['support_20']) / df_processed['close']
            df_processed['resistance_distance'] = (df_processed['resistance_20'] - df_processed['close']) / df_processed['close']
            
            # Trend indicators
            df_processed['trend_5'] = np.where(df_processed['close'] > df_processed['close'].shift(5), 1, 0)
            df_processed['trend_10'] = np.where(df_processed['close'] > df_processed['close'].shift(10), 1, 0)
            df_processed['trend_20'] = np.where(df_processed['close'] > df_processed['close'].shift(20), 1, 0)
            
        except Exception as e:
            print(f"Error adding technical indicators: {e}")
        
        # Fill NaN values
        df_processed = df_processed.fillna(method='forward').fillna(method='backward')
        
        return df_processed
    
    def add_lag_features(self, df, columns, lags=[1, 2, 3, 5, 10]):
        """
        Add lagged features
        
        Args:
            df: Input dataframe
            columns: List of columns to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        df_lagged = df.copy()
        
        for col in columns:
            if col in df_lagged.columns:
                for lag in lags:
                    df_lagged[f'{col}_lag_{lag}'] = df_lagged[col].shift(lag)
        
        return df_lagged
    
    def add_rolling_features(self, df, columns, windows=[5, 10, 20]):
        """
        Add rolling statistical features
        
        Args:
            df: Input dataframe
            columns: List of columns to create rolling features for
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        df_rolling = df.copy()
        
        for col in columns:
            if col in df_rolling.columns:
                for window in windows:
                    df_rolling[f'{col}_mean_{window}'] = df_rolling[col].rolling(window=window).mean()
                    df_rolling[f'{col}_std_{window}'] = df_rolling[col].rolling(window=window).std()
                    df_rolling[f'{col}_min_{window}'] = df_rolling[col].rolling(window=window).min()
                    df_rolling[f'{col}_max_{window}'] = df_rolling[col].rolling(window=window).max()
                    
                    # Z-score features
                    rolling_mean = df_rolling[col].rolling(window=window).mean()
                    rolling_std = df_rolling[col].rolling(window=window).std()
                    df_rolling[f'{col}_zscore_{window}'] = (df_rolling[col] - rolling_mean) / rolling_std
        
        return df_rolling
    
    def prepare_features(self, df, include_technical=True, include_volume=True, 
                        include_price_changes=True, include_lags=True, include_rolling=True):
        """
        Prepare comprehensive feature set for ML models
        
        Args:
            df: Input dataframe with OHLCV data
            include_technical: Include technical indicators
            include_volume: Include volume-based features
            include_price_changes: Include price change features
            include_lags: Include lagged features
            include_rolling: Include rolling statistical features
            
        Returns:
            DataFrame with prepared features
        """
        df_features = df.copy()
        
        # Add technical indicators
        if include_technical:
            df_features = self.add_technical_indicators(df_features)
        
        # Add lagged features
        if include_lags:
            price_cols = ['close', 'high', 'low', 'open']
            available_price_cols = [col for col in price_cols if col in df_features.columns]
            if available_price_cols:
                df_features = self.add_lag_features(df_features, available_price_cols, lags=[1, 2, 3, 5])
        
        # Add rolling features
        if include_rolling:
            feature_cols = ['close', 'volume'] if include_volume else ['close']
            available_feature_cols = [col for col in feature_cols if col in df_features.columns]
            if available_feature_cols:
                df_features = self.add_rolling_features(df_features, available_feature_cols, windows=[5, 10])
        
        # Remove columns that shouldn't be features
        columns_to_remove = []
        if not include_volume and 'volume' in df_features.columns:
            volume_cols = [col for col in df_features.columns if 'volume' in col.lower() or col in ['obv', 'vpt', 'mfi']]
            columns_to_remove.extend(volume_cols)
        
        # Remove original OHLC columns except close (target)
        ohlc_to_remove = ['open', 'high', 'low']
        for col in ohlc_to_remove:
            if col in df_features.columns:
                columns_to_remove.append(col)
        
        # Remove timestamp if present
        if 'timestamp' in df_features.columns:
            columns_to_remove.append('timestamp')
        
        df_features = df_features.drop(columns=columns_to_remove, errors='ignore')
        
        # Handle infinite values and NaN
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        df_features = df_features.fillna(method='forward').fillna(method='backward')
        df_features = df_features.fillna(0)
        
        return df_features
    
    def prepare_training_data(self, df, target_col='close', test_size=0.2, 
                            sequence_length=None, future_periods=1):
        """
        Prepare data for training ML models
        
        Args:
            df: Dataframe with features
            target_col: Target column name
            test_size: Proportion of data for testing
            sequence_length: For LSTM models, length of sequences
            future_periods: Number of periods to predict ahead
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, scaler)
        """
        # Create target variable (future price)
        df_processed = df.copy()
        df_processed[f'{target_col}_target'] = df_processed[target_col].shift(-future_periods)
        
        # Remove rows with NaN target
        df_processed = df_processed.dropna()
        
        if len(df_processed) < 50:
            raise ValueError("Insufficient data after preprocessing")
        
        # Separate features and target
        feature_cols = [col for col in df_processed.columns if col != f'{target_col}_target']
        
        # Remove the original target column from features if it exists
        if target_col in feature_cols:
            feature_cols.remove(target_col)
        
        X = df_processed[feature_cols]
        y = df_processed[f'{target_col}_target']
        
        self.feature_columns = feature_cols
        self.target_column = target_col
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if sequence_length is not None:
            # Prepare sequence data for LSTM
            X_sequences = []
            y_sequences = []
            
            for i in range(sequence_length, len(X_scaled)):
                X_sequences.append(X_scaled[i-sequence_length:i])
                y_sequences.append(y.iloc[i])
            
            X_sequences = np.array(X_sequences)
            y_sequences = np.array(y_sequences)
            
            # Split data
            split_idx = int(len(X_sequences) * (1 - test_size))
            
            X_train = X_sequences[:split_idx]
            X_test = X_sequences[split_idx:]
            y_train = y_sequences[:split_idx]
            y_test = y_sequences[split_idx:]
        
        else:
            # Traditional train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y.values, test_size=test_size, shuffle=False
            )
        
        return X_train, X_test, y_train, y_test, self.scaler
    
    def transform_new_data(self, df):
        """
        Transform new data using fitted scaler
        
        Args:
            df: New dataframe to transform
            
        Returns:
            Scaled feature array
        """
        if not self.feature_columns:
            raise ValueError("No feature columns defined. Run prepare_training_data first.")
        
        # Prepare features the same way as training data
        df_features = self.prepare_features(df)
        
        # Select only the features used in training
        available_features = [col for col in self.feature_columns if col in df_features.columns]
        
        if len(available_features) != len(self.feature_columns):
            missing_features = set(self.feature_columns) - set(available_features)
            print(f"Warning: Missing features: {missing_features}")
        
        X = df_features[available_features]
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def create_sequences(self, data, sequence_length):
        """
        Create sequences for LSTM models
        
        Args:
            data: Input data array
            sequence_length: Length of sequences
            
        Returns:
            Array of sequences
        """
        sequences = []
        
        for i in range(sequence_length, len(data)):
            sequences.append(data[i-sequence_length:i])
        
        return np.array(sequences)
    
    def inverse_transform_target(self, y_scaled):
        """
        Inverse transform target values if they were scaled
        
        Args:
            y_scaled: Scaled target values
            
        Returns:
            Original scale target values
        """
        # If target was scaled separately, implement inverse transform here
        # For now, assuming target is not scaled
        return y_scaled
    
    def get_feature_names(self):
        """Get list of feature names"""
        return self.feature_columns.copy()
    
    def calculate_feature_importance_correlation(self, df, target_col='close'):
        """
        Calculate feature importance based on correlation with target
        
        Args:
            df: Dataframe with features
            target_col: Target column name
            
        Returns:
            Dictionary of feature importance scores
        """
        df_features = self.prepare_features(df)
        
        if target_col not in df_features.columns:
            print(f"Target column {target_col} not found")
            return {}
        
        feature_cols = [col for col in df_features.columns if col != target_col]
        correlations = {}
        
        for col in feature_cols:
            try:
                corr = np.corrcoef(df_features[col], df_features[target_col])[0, 1]
                if not np.isnan(corr):
                    correlations[col] = abs(corr)
            except:
                correlations[col] = 0.0
        
        # Sort by importance
        sorted_features = dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_features
