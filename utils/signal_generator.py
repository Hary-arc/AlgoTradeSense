import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Any
import logging
import joblib
import os
import torch
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import internal modules with fallbacks
try:
    from .database import get_model_by_name, load_model_from_db
    DATABASE_AVAILABLE = True
except ImportError:
    logger.warning("Database module not available. Using fallback model loading.")
    DATABASE_AVAILABLE = False
    def get_model_by_name(model_name):
        return None
    def load_model_from_db(model_name):
        return None, None

try:
    from .data_processor import DataProcessor
    DATA_PROCESSOR_AVAILABLE = True
except ImportError:
    logger.warning("DataProcessor module not available. Using fallback feature preparation.")
    DATA_PROCESSOR_AVAILABLE = False

class SignalGenerator:
    """
    AI-powered trading signal generator with comprehensive error handling
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        self.data_processor = DataProcessor() if DATA_PROCESSOR_AVAILABLE else None
        self._model_cache = {}  # Cache loaded models
        self._scaler_cache = {}  # Cache loaded scalers
        
    def generate_signal(self, model_name: str, data: pd.DataFrame, 
                       confidence_threshold: float = None) -> Optional[Dict]:
        """
        Generate trading signal using ML model
        
        Args:
            model_name: Name of the trained model
            data: Recent market data with OHLCV columns
            confidence_threshold: Minimum confidence for signal generation
            
        Returns:
            Dictionary containing signal information or None
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
            
        try:
            # Validate input data
            if not self._validate_input_data(data):
                logger.error("Invalid input data")
                return None
            
            # Load model and metadata
            model, scaler, model_info = self._load_model_and_metadata(model_name)
            if model is None:
                logger.error(f"Could not load model: {model_name}")
                return None
            
            # Prepare data for prediction
            features = self._prepare_prediction_data(data, model_info, scaler)
            if features is None:
                logger.error("Could not prepare features for prediction")
                return None
            
            # Make prediction
            prediction = self._make_prediction(model, features, model_info.get('type', 'unknown'))
            if prediction is None:
                logger.error("Prediction failed")
                return None
            
            # Generate signal based on prediction
            signal_result = self._analyze_prediction(
                prediction, data, model_info, confidence_threshold
            )
            
            # Validate signal quality
            if signal_result:
                signal_result['quality_analysis'] = self.validate_signal_quality(signal_result)
            
            return signal_result
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return self._create_error_signal(data, str(e))
    
    def _validate_input_data(self, data: pd.DataFrame) -> bool:
        """Validate that input data has required columns and sufficient length"""
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            logger.error(f"Missing required columns: {required_cols}")
            return False
        
        if len(data) < 60:
            logger.error(f"Insufficient data: {len(data)} records, need at least 60")
            return False
            
        return True
    
    def _load_model_and_metadata(self, model_name: str) -> Tuple[Any, Any, Dict]:
        """Load model, scaler, and metadata with caching"""
        # Check cache first
        if model_name in self._model_cache:
            return self._model_cache[model_name], self._scaler_cache.get(model_name), {'name': model_name, 'type': 'cached'}
        
        try:
            if DATABASE_AVAILABLE:
                # Try to load from database
                model_info = get_model_by_name(model_name)
                if model_info:
                    model, scaler = load_model_from_db(model_name)
                    if model:
                        # Cache for future use
                        self._model_cache[model_name] = model
                        if scaler:
                            self._scaler_cache[model_name] = scaler
                        return model, scaler, model_info
            
            # Fallback: try to load from file system
            model_path = f"models/{model_name}.pkl"
            if os.path.exists(model_path):
                try:
                    model_data = joblib.load(model_path)
                    model = model_data.get('model')
                    scaler = model_data.get('scaler')
                    model_info = model_data.get('model_info', {'name': model_name, 'type': 'file'})
                    
                    if model:
                        self._model_cache[model_name] = model
                        if scaler:
                            self._scaler_cache[model_name] = scaler
                        return model, scaler, model_info
                except Exception as e:
                    logger.warning(f"Error loading model from file: {e}")
            
            return None, None, {}
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None, None, {}
    
    def _prepare_prediction_data(self, data: pd.DataFrame, model_info: Dict, 
                               scaler: Any) -> Optional[np.ndarray]:
        """Prepare data for model prediction with comprehensive feature engineering"""
        try:
            # Use DataProcessor if available, otherwise use fallback
            if DATA_PROCESSOR_AVAILABLE and self.data_processor:
                df_features = self.data_processor.prepare_features(
                    data, 
                    include_technical=True, 
                    include_volume=True,
                    include_price_changes=True
                )
            else:
                # Fallback feature engineering
                df_features = self._prepare_features_fallback(data)
            
            if df_features.empty:
                logger.error("No features generated")
                return None
            
            # Get model features from metadata or use all available
            model_features = model_info.get('features', [])
            if not model_features:
                model_features = df_features.columns.tolist()
                logger.warning(f"No feature list in model info, using all available: {model_features}")
            
            # Select only the features used in training
            available_features = [col for col in model_features if col in df_features.columns]
            if not available_features:
                logger.error("No matching features found for prediction")
                return None
            
            # Handle sequence models (LSTM)
            model_type = model_info.get('type', '').upper()
            sequence_length = model_info.get('sequence_length', 60)
            
            if model_type == 'LSTM':
                if len(df_features) < sequence_length:
                    logger.error(f"Insufficient data for LSTM: need {sequence_length}, got {len(df_features)}")
                    return None
                
                # Get the sequence
                sequence_data = df_features[available_features].iloc[-sequence_length:].values
                
                # Scale if scaler is available
                if scaler is not None:
                    try:
                        sequence_data = scaler.transform(sequence_data)
                    except Exception as e:
                        logger.warning(f"Scaling failed: {e}")
                
                # Reshape for LSTM: (samples, timesteps, features)
                return sequence_data.reshape(1, sequence_length, len(available_features))
            
            else:
                # Standard models: use latest data point
                feature_data = df_features[available_features].iloc[-1:].values
                
                # Scale if scaler is available
                if scaler is not None:
                    try:
                        feature_data = scaler.transform(feature_data)
                    except Exception as e:
                        logger.warning(f"Scaling failed: {e}")
                
                return feature_data
                
        except Exception as e:
            logger.error(f"Error preparing prediction data: {e}")
            return None
    
    def _prepare_features_fallback(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fallback feature engineering when DataProcessor is not available"""
        features = pd.DataFrame(index=data.index)
        
        # Price features
        features['close'] = data['close']
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Volatility
        features['volatility_20'] = data['close'].pct_change().rolling(20).std()
        features['volatility_50'] = data['close'].pct_change().rolling(50).std()
        
        # Moving averages
        features['sma_20'] = data['close'].rolling(20).mean()
        features['sma_50'] = data['close'].rolling(50).mean()
        features['ema_12'] = data['close'].ewm(span=12).mean()
        features['ema_26'] = data['close'].ewm(span=26).mean()
        
        # Price ratios
        features['price_sma20_ratio'] = data['close'] / features['sma_20']
        features['price_sma50_ratio'] = data['close'] / features['sma_50']
        
        # High-Low features
        features['high_low_ratio'] = data['high'] / data['low']
        features['body_size'] = (data['close'] - data['open']).abs() / data['open']
        
        # Volume features if available
        if 'volume' in data.columns:
            features['volume'] = data['volume']
            features['volume_sma_20'] = data['volume'].rolling(20).mean()
            features['volume_ratio'] = data['volume'] / features['volume_sma_20']
        
        # Technical indicators
        features['rsi_14'] = self._calculate_rsi(data['close'], 14)
        macd, macd_signal = self._calculate_macd(data['close'])
        features['macd'] = macd
        features['macd_signal'] = macd_signal
        features['macd_histogram'] = macd - macd_signal
        
        # Drop NaN values
        features = features.dropna()
        
        return features
    
    def _make_prediction(self, model, features: np.ndarray, model_type: str) -> Optional[float]:
        """Make price prediction using the model"""
        try:
            model_type = model_type.upper()
            
            if 'LSTM' in model_type:
                # Keras/TensorFlow model
                if hasattr(model, 'predict'):
                    prediction = model.predict(features, verbose=0)
                    return float(prediction[0][0])
            
            elif hasattr(model, 'predict'):
                # Scikit-learn model
                prediction = model.predict(features)
                return float(prediction[0])
            
            elif hasattr(model, '__call__'):
                # PyTorch or other callable model
                if isinstance(features, np.ndarray):
                    features = torch.FloatTensor(features)
                prediction = model(features)
                return float(prediction.detach().numpy()[0])
            
            logger.error(f"Unknown model type or prediction method: {model_type}")
            return None
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None
    
    def _analyze_prediction(self, predicted_price: float, data: pd.DataFrame, 
                          model_info: Dict, confidence_threshold: float) -> Dict:
        """Analyze prediction and generate trading signal with comprehensive analysis"""
        try:
            current_price = float(data['close'].iloc[-1])
            price_change = predicted_price - current_price
            price_change_pct = (price_change / current_price) * 100
            
            # Calculate base confidence from model performance
            base_confidence = model_info.get('test_r2', 0.5)
            base_confidence = max(0.1, min(0.9, base_confidence))  # Reasonable bounds
            
            # Adjust for market volatility
            volatility_factor = self._calculate_volatility_factor(data)
            
            # Adjust for prediction magnitude (larger moves = higher confidence)
            magnitude_factor = min(1.0, abs(price_change_pct) / 3.0)
            
            # Combine factors
            confidence = base_confidence * volatility_factor * magnitude_factor
            confidence = max(0.1, min(0.95, confidence))  # Reasonable bounds
            
            # Determine signal type
            signal_type = 'HOLD'
            if confidence >= confidence_threshold:
                if price_change_pct > 1.5:  # 1.5% threshold for BUY
                    signal_type = 'BUY'
                elif price_change_pct < -1.5:  # 1.5% threshold for SELL
                    signal_type = 'SELL'
            
            # Technical confirmation
            signal_strength, technical_factors = self._calculate_signal_strength(
                data, signal_type, confidence
            )
            
            # Prepare comprehensive signal result
            signal_result = {
                'signal_type': signal_type,
                'confidence': round(confidence, 3),
                'predicted_price': round(predicted_price, 4),
                'current_price': round(current_price, 4),
                'price_change': round(price_change, 4),
                'price_change_pct': round(price_change_pct, 2),
                'signal_strength': signal_strength,
                'model_name': model_info.get('name', 'unknown'),
                'model_type': model_info.get('type', 'unknown'),
                'model_accuracy': round(model_info.get('test_r2', 0), 3),
                'timestamp': datetime.now().isoformat(),
                'technical_factors': technical_factors,
                'market_conditions': self._analyze_market_conditions(data)
            }
            
            return signal_result
            
        except Exception as e:
            logger.error(f"Error analyzing prediction: {e}")
            return self._create_error_signal(data, f"Analysis error: {e}")
    
    def _calculate_volatility_factor(self, data: pd.DataFrame) -> float:
        """Calculate volatility adjustment factor"""
        try:
            returns = data['close'].pct_change()
            recent_volatility = returns.rolling(20).std().iloc[-1]
            
            # Normalize volatility (assuming typical daily volatility ~0.01-0.03)
            normalized_vol = min(0.1, max(0.005, recent_volatility))  # Cap extreme values
            volatility_factor = 1.0 - (normalized_vol * 20)  # Reduce confidence in high volatility
            
            return max(0.3, min(1.0, volatility_factor))  # Reasonable bounds
            
        except Exception as e:
            logger.warning(f"Error calculating volatility factor: {e}")
            return 0.7  # Default moderate factor
    
    def _calculate_signal_strength(self, data: pd.DataFrame, signal_type: str, 
                                 confidence: float) -> Tuple[str, Dict]:
        """Calculate signal strength with technical confirmation"""
        strength_score = confidence
        technical_factors = {}
        
        try:
            close_prices = data['close']
            current_price = close_prices.iloc[-1]
            
            # Volume analysis
            if 'volume' in data.columns:
                volumes = data['volume']
                avg_volume = volumes.rolling(20).mean().iloc[-1]
                current_volume = volumes.iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                
                technical_factors['volume_ratio'] = round(volume_ratio, 2)
                
                if volume_ratio > 1.8:
                    strength_score += 0.15
                    technical_factors['volume'] = 'Very High'
                elif volume_ratio > 1.3:
                    strength_score += 0.08
                    technical_factors['volume'] = 'High'
                elif volume_ratio < 0.7:
                    strength_score -= 0.1
                    technical_factors['volume'] = 'Low'
            
            # Trend analysis
            sma_20 = close_prices.rolling(20).mean().iloc[-1]
            sma_50 = close_prices.rolling(50).mean().iloc[-1] if len(close_prices) >= 50 else current_price
            
            technical_factors['trend_20'] = 'Bullish' if current_price > sma_20 else 'Bearish'
            technical_factors['trend_50'] = 'Bullish' if current_price > sma_50 else 'Bearish'
            
            trend_alignment = 0
            if signal_type == 'BUY' and current_price > sma_20:
                trend_alignment += 0.1
            if signal_type == 'BUY' and current_price > sma_50:
                trend_alignment += 0.1
            if signal_type == 'SELL' and current_price < sma_20:
                trend_alignment += 0.1
            if signal_type == 'SELL' and current_price < sma_50:
                trend_alignment += 0.1
                
            strength_score += trend_alignment
            
            # RSI analysis
            rsi = self._calculate_rsi(close_prices, 14)
            if len(rsi) > 0:
                current_rsi = rsi.iloc[-1]
                technical_factors['rsi'] = round(current_rsi, 1)
                
                if signal_type == 'BUY' and current_rsi < 35:
                    strength_score += 0.12
                    technical_factors['rsi_signal'] = 'Oversold'
                elif signal_type == 'SELL' and current_rsi > 65:
                    strength_score += 0.12
                    technical_factors['rsi_signal'] = 'Overbought'
            
            # Determine strength category
            if strength_score >= 0.8:
                return 'Strong', technical_factors
            elif strength_score >= 0.6:
                return 'Medium', technical_factors
            else:
                return 'Weak', technical_factors
                
        except Exception as e:
            logger.warning(f"Error calculating signal strength: {e}")
            return 'Weak', {}
    
    def _analyze_market_conditions(self, data: pd.DataFrame) -> Dict:
        """Analyze current market conditions"""
        try:
            close_prices = data['close']
            returns = close_prices.pct_change()
            
            conditions = {
                'volatility': returns.std(),
                'trend_strength': self._calculate_trend_strength(close_prices),
                'market_regime': self._identify_market_regime(close_prices),
                'volume_trend': self._analyze_volume_trend(data) if 'volume' in data.columns else 'Unknown'
            }
            
            return conditions
            
        except Exception as e:
            logger.warning(f"Error analyzing market conditions: {e}")
            return {'error': str(e)}
    
    def _calculate_trend_strength(self, prices: pd.Series) -> str:
        """Calculate trend strength"""
        try:
            sma_20 = prices.rolling(20).mean()
            sma_50 = prices.rolling(50).mean()
            
            # Check if prices are above both MAs (strong uptrend)
            if prices.iloc[-1] > sma_20.iloc[-1] and prices.iloc[-1] > sma_50.iloc[-1]:
                return 'Strong Uptrend'
            # Check if prices are below both MAs (strong downtrend)
            elif prices.iloc[-1] < sma_20.iloc[-1] and prices.iloc[-1] < sma_50.iloc[-1]:
                return 'Strong Downtrend'
            # Mixed signals (ranging market)
            else:
                return 'Ranging'
                
        except Exception as e:
            return 'Unknown'
    
    def _identify_market_regime(self, prices: pd.Series) -> str:
        """Identify current market regime"""
        try:
            volatility = prices.pct_change().std()
            
            if volatility > 0.03:
                return 'High Volatility'
            elif volatility > 0.015:
                return 'Moderate Volatility'
            else:
                return 'Low Volatility'
                
        except Exception as e:
            return 'Unknown'
    
    def _analyze_volume_trend(self, data: pd.DataFrame) -> str:
        """Analyze volume trend"""
        try:
            volumes = data['volume']
            volume_sma_20 = volumes.rolling(20).mean()
            
            if volumes.iloc[-1] > volume_sma_20.iloc[-1] * 1.5:
                return 'High Volume'
            elif volumes.iloc[-1] < volume_sma_20.iloc[-1] * 0.7:
                return 'Low Volume'
            else:
                return 'Normal Volume'
                
        except Exception as e:
            return 'Unknown'
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator robustly"""
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period, min_periods=1).mean()
            avg_loss = loss.rolling(window=period, min_periods=1).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50)  # Fill NaN with neutral value
            
        except Exception as e:
            logger.warning(f"Error calculating RSI: {e}")
            return pd.Series([50] * len(prices), index=prices.index)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, 
                       signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator robustly"""
        try:
            ema_fast = prices.ewm(span=fast, min_periods=1).mean()
            ema_slow = prices.ewm(span=slow, min_periods=1).mean()
            
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal, min_periods=1).mean()
            
            return macd, macd_signal
            
        except Exception as e:
            logger.warning(f"Error calculating MACD: {e}")
            zero_series = pd.Series([0] * len(prices), index=prices.index)
            return zero_series, zero_series
    
    def _create_error_signal(self, data: pd.DataFrame, error_msg: str) -> Dict:
        """Create error signal response"""
        return {
            'signal_type': 'ERROR',
            'confidence': 0.0,
            'current_price': float(data['close'].iloc[-1]) if len(data) > 0 else 0,
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_batch_signals(self, model_names: List[str], data: pd.DataFrame, 
                             confidence_threshold: float = None) -> Dict:
        """Generate signals for multiple models with enhanced consensus analysis"""
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
            
        signals = {}
        
        for model_name in model_names:
            signal = self.generate_signal(model_name, data, confidence_threshold)
            signals[model_name] = signal
        
        # Consensus analysis
        consensus = self._analyze_signal_consensus(signals)
        
        return {
            'individual_signals': signals,
            'consensus': consensus,
            'timestamp': datetime.now().isoformat()
        }
    
    def _analyze_signal_consensus(self, signals: Dict) -> Dict:
        """Analyze consensus among multiple signals with weighted analysis"""
        valid_signals = {k: v for k, v in signals.items() 
                        if v is not None and v.get('signal_type') not in ['ERROR', 'HOLD'] 
                        and 'error' not in v}
        
        if not valid_signals:
            return {
                'consensus_signal': 'HOLD', 
                'agreement': 0, 
                'weighted_confidence': 0,
                'signal_count': 0
            }
        
        # Weight signals by confidence and model accuracy
        weighted_votes = {'BUY': 0, 'SELL': 0}
        total_weight = 0
        
        for signal_name, signal in valid_signals.items():
            signal_type = signal.get('signal_type')
            confidence = signal.get('confidence', 0.5)
            model_accuracy = signal.get('model_accuracy', 0.5)
            
            # Weight = confidence * model_accuracy
            weight = confidence * model_accuracy
            
            if signal_type in weighted_votes:
                weighted_votes[signal_type] += weight
                total_weight += weight
        
        if total_weight == 0:
            return {
                'consensus_signal': 'HOLD', 
                'agreement': 0, 
                'weighted_confidence': 0,
                'signal_count': len(valid_signals)
            }
        
        # Determine consensus
        if weighted_votes['BUY'] > weighted_votes['SELL']:
            consensus_signal = 'BUY'
            agreement = weighted_votes['BUY'] / total_weight
        else:
            consensus_signal = 'SELL'
            agreement = weighted_votes['SELL'] / total_weight
        
        return {
            'consensus_signal': consensus_signal,
            'agreement': round(agreement, 3),
            'weighted_confidence': round(total_weight / len(valid_signals), 3),
            'signal_count': len(valid_signals),
            'buy_strength': round(weighted_votes['BUY'], 3),
            'sell_strength': round(weighted_votes['SELL'], 3)
        }
    
    def validate_signal_quality(self, signal: Dict) -> Dict:
        """Validate and score signal quality comprehensively"""
        if not signal or signal.get('signal_type') in ['ERROR', 'HOLD']:
            return {
                'quality_score': 0,
                'quality_rating': 'Invalid',
                'recommendation': 'Skip'
            }
        
        quality_score = 0
        quality_factors = []
        
        try:
            # Confidence scoring (0-30 points)
            confidence = signal.get('confidence', 0)
            if confidence >= 0.8:
                quality_score += 30
                quality_factors.append('High confidence (≥0.8)')
            elif confidence >= 0.6:
                quality_score += 20
                quality_factors.append('Medium confidence (0.6-0.8)')
            elif confidence >= 0.4:
                quality_score += 10
                quality_factors.append('Low confidence (0.4-0.6)')
            else:
                quality_factors.append('Very low confidence (<0.4)')
            
            # Signal strength scoring (0-25 points)
            strength = signal.get('signal_strength', 'Weak')
            if strength == 'Strong':
                quality_score += 25
                quality_factors.append('Strong technical confirmation')
            elif strength == 'Medium':
                quality_score += 15
                quality_factors.append('Medium technical confirmation')
            else:
                quality_score += 5
                quality_factors.append('Weak technical confirmation')
            
            # Price movement scoring (0-20 points)
            price_change_pct = abs(signal.get('price_change_pct', 0))
            if price_change_pct >= 4:
                quality_score += 20
                quality_factors.append('Large expected move (≥4%)')
            elif price_change_pct >= 2:
                quality_score += 12
                quality_factors.append('Moderate expected move (2-4%)')
            elif price_change_pct >= 1:
                quality_score += 6
                quality_factors.append('Small expected move (1-2%)')
            else:
                quality_factors.append('Very small expected move (<1%)')
            
            # Market condition scoring (0-15 points)
            market_conditions = signal.get('market_conditions', {})
            volatility = market_conditions.get('volatility', 0)
            if 0.01 <= volatility <= 0.025:
                quality_score += 10
                quality_factors.append('Ideal volatility conditions')
            elif volatility < 0.01:
                quality_score += 5
                quality_factors.append('Low volatility (may lack momentum)')
            else:
                quality_factors.append('High volatility (increased risk)')
            
            # Volume confirmation (0-10 points)
            technical_factors = signal.get('technical_factors', {})
            volume_ratio = technical_factors.get('volume_ratio', 1)
            if volume_ratio > 1.5:
                quality_score += 8
                quality_factors.append('High volume confirmation')
            elif volume_ratio > 1.2:
                quality_score += 4
                quality_factors.append('Moderate volume confirmation')
            
            # Final quality assessment
            if quality_score >= 80:
                quality_rating = 'Excellent'
                recommendation = 'Strong Trade'
            elif quality_score >= 60:
                quality_rating = 'Good'
                recommendation = 'Trade'
            elif quality_score >= 40:
                quality_rating = 'Fair'
                recommendation = 'Consider'
            else:
                quality_rating = 'Poor'
                recommendation = 'Skip'
            
            return {
                'quality_score': quality_score,
                'quality_rating': quality_rating,
                'quality_factors': quality_factors,
                'recommendation': recommendation
            }
            
        except Exception as e:
            logger.error(f"Error validating signal quality: {e}")
            return {
                'quality_score': 0,
                'quality_rating': 'Error',
                'recommendation': 'Skip',
                'error': str(e)
            }
    
    def clear_cache(self):
        """Clear cached models and scalers"""
        self._model_cache.clear()
        self._scaler_cache.clear()
        logger.info("Model cache cleared")

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    sample_data = pd.DataFrame({
        'open': prices,
        'high': prices + np.abs(np.random.randn(100) * 0.3),
        'low': prices - np.abs(np.random.randn(100) * 0.3),
        'close': prices,
        'volume': np.random.randint(1000, 5000, 100)
    }, index=dates)
    
    # Test signal generation
    generator = SignalGenerator(confidence_threshold=0.6)
    
    # Test with a mock model (in real usage, you'd have actual trained models)
    print("Testing signal generation...")
    
    # This would normally come from your model database
    mock_model_info = {
        'name': 'test_model',
        'type': 'RandomForest',
        'test_r2': 0.75,
        'features': ['close', 'returns', 'volatility_20', 'sma_20']
    }
    
    # For demonstration, we'll create a simple signal
    signal = generator._analyze_prediction(
        predicted_price=105.0,
        data=sample_data,
        model_info=mock_model_info,
        confidence_threshold=0.6
    )
    
    if signal:
        print("Generated Signal:")
        for key, value in signal.items():
            print(f"  {key}: {value}")
        
        # Validate quality
        quality = generator.validate_signal_quality(signal)
        print("\nSignal Quality:")
        for key, value in quality.items():
            print(f"  {key}: {value}")
    else:
        print("Failed to generate signal")