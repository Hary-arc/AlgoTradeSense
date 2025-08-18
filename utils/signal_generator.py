import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List
import joblib
import os
from .database import get_model_by_name, load_model_from_db
from .data_processor import DataProcessor

class SignalGenerator:
    """
    AI-powered trading signal generator
    """
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.confidence_threshold = 0.7
        
    def generate_signal(self, model_name: str, data: pd.DataFrame, 
                       confidence_threshold: float = 0.7) -> Optional[Dict]:
        """
        Generate trading signal using ML model
        
        Args:
            model_name: Name of the trained model
            data: Recent market data
            confidence_threshold: Minimum confidence for signal generation
            
        Returns:
            Dictionary containing signal information or None
        """
        try:
            # Load model information
            model_info = get_model_by_name(model_name)
            if not model_info:
                raise ValueError(f"Model {model_name} not found")
            
            # Load the actual model
            model, scaler = load_model_from_db(model_name)
            if model is None:
                raise ValueError(f"Could not load model {model_name}")
            
            # Prepare data for prediction
            features = self._prepare_prediction_data(data, model_info, scaler)
            
            if features is None:
                return None
            
            # Make prediction
            prediction = self._make_prediction(model, features, model_info['type'])
            
            if prediction is None:
                return None
            
            # Generate signal based on prediction
            signal_result = self._analyze_prediction(
                prediction, data, model_info, confidence_threshold
            )
            
            return signal_result
            
        except Exception as e:
            print(f"Error generating signal: {e}")
            return None
    
    def _prepare_prediction_data(self, data: pd.DataFrame, model_info: Dict, 
                               scaler) -> Optional[np.ndarray]:
        """Prepare data for model prediction"""
        try:
            # Get the latest data point for prediction
            if len(data) < 60:  # Need sufficient history for features
                return None
            
            # Prepare features using the same process as training
            df_features = self.data_processor.prepare_features(
                data, 
                include_technical=True, 
                include_volume=True,
                include_price_changes=True
            )
            
            if len(df_features) == 0:
                return None
            
            # Get model features
            model_features = model_info.get('features', [])
            
            # Select only the features used in training
            available_features = [col for col in model_features if col in df_features.columns]
            
            if len(available_features) == 0:
                print("No matching features found for prediction")
                return None
            
            # Get the latest data point
            feature_data = df_features[available_features].iloc[-1:].values
            
            # Scale features
            if scaler is not None:
                feature_data = scaler.transform(feature_data)
            
            # Handle LSTM sequence requirements
            if model_info['type'] == 'LSTM':
                sequence_length = 60  # Default sequence length
                
                if len(df_features) >= sequence_length:
                    # Create sequence
                    sequence_data = df_features[available_features].iloc[-sequence_length:].values
                    if scaler is not None:
                        sequence_data = scaler.transform(sequence_data)
                    
                    # Reshape for LSTM
                    feature_data = sequence_data.reshape(1, sequence_length, len(available_features))
                else:
                    return None
            
            return feature_data
            
        except Exception as e:
            print(f"Error preparing prediction data: {e}")
            return None
    
    def _make_prediction(self, model, features: np.ndarray, model_type: str) -> Optional[float]:
        """Make price prediction using the model"""
        try:
            if model_type == 'LSTM':
                # Keras/TensorFlow model
                prediction = model.predict(features, verbose=0)
                return float(prediction[0][0])
            
            else:
                # Scikit-learn model
                prediction = model.predict(features)
                return float(prediction[0])
                
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def _analyze_prediction(self, predicted_price: float, data: pd.DataFrame, 
                          model_info: Dict, confidence_threshold: float) -> Dict:
        """Analyze prediction and generate trading signal"""
        try:
            current_price = float(data['close'].iloc[-1])
            
            # Calculate price change
            price_change = predicted_price - current_price
            price_change_pct = (price_change / current_price) * 100
            
            # Calculate prediction confidence based on model accuracy and market conditions
            base_confidence = model_info.get('test_r2', 0.5)
            
            # Adjust confidence based on market volatility
            recent_volatility = data['close'].pct_change().rolling(20).std().iloc[-1]
            volatility_factor = max(0.5, 1 - (recent_volatility * 10))  # Reduce confidence in high volatility
            
            # Adjust confidence based on prediction magnitude
            magnitude_factor = min(1.0, abs(price_change_pct) / 5)  # Higher confidence for larger moves
            
            confidence = base_confidence * volatility_factor * magnitude_factor
            confidence = max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
            
            # Generate signal based on prediction and confidence
            signal_type = 'HOLD'
            
            if confidence >= confidence_threshold:
                if price_change_pct > 2:  # Minimum 2% move for BUY signal
                    signal_type = 'BUY'
                elif price_change_pct < -2:  # Minimum 2% drop for SELL signal
                    signal_type = 'SELL'
            
            # Additional technical analysis for signal confirmation
            signal_strength = self._calculate_signal_strength(data, signal_type, confidence)
            
            # Prepare signal result
            signal_result = {
                'signal_type': signal_type,
                'confidence': confidence,
                'predicted_price': predicted_price,
                'current_price': current_price,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'signal_strength': signal_strength,
                'model_name': model_info['name'],
                'model_type': model_info['type'],
                'timestamp': datetime.now(),
                'technical_factors': self._get_technical_factors(data)
            }
            
            return signal_result
            
        except Exception as e:
            print(f"Error analyzing prediction: {e}")
            return {
                'signal_type': 'HOLD',
                'confidence': 0.0,
                'predicted_price': predicted_price,
                'current_price': data['close'].iloc[-1],
                'error': str(e)
            }
    
    def _calculate_signal_strength(self, data: pd.DataFrame, signal_type: str, confidence: float) -> str:
        """Calculate signal strength based on technical analysis"""
        try:
            # Get recent price data
            close_prices = data['close']
            volumes = data.get('volume', pd.Series([1] * len(close_prices)))
            
            strength_score = confidence
            
            # Volume confirmation
            avg_volume = volumes.rolling(20).mean().iloc[-1]
            current_volume = volumes.iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            if volume_ratio > 1.5:
                strength_score += 0.1
            elif volume_ratio < 0.5:
                strength_score -= 0.1
            
            # Trend confirmation
            sma_20 = close_prices.rolling(20).mean().iloc[-1]
            current_price = close_prices.iloc[-1]
            
            if signal_type == 'BUY' and current_price > sma_20:
                strength_score += 0.1
            elif signal_type == 'SELL' and current_price < sma_20:
                strength_score += 0.1
            
            # RSI confirmation
            rsi = self._calculate_rsi(close_prices)
            if len(rsi) > 0:
                current_rsi = rsi.iloc[-1]
                
                if signal_type == 'BUY' and current_rsi < 30:  # Oversold
                    strength_score += 0.15
                elif signal_type == 'SELL' and current_rsi > 70:  # Overbought
                    strength_score += 0.15
            
            # Classify strength
            if strength_score >= 0.8:
                return 'Strong'
            elif strength_score >= 0.6:
                return 'Medium'
            else:
                return 'Weak'
                
        except Exception as e:
            print(f"Error calculating signal strength: {e}")
            return 'Weak'
    
    def _get_technical_factors(self, data: pd.DataFrame) -> Dict:
        """Get technical analysis factors"""
        try:
            close_prices = data['close']
            
            # Moving averages
            sma_20 = close_prices.rolling(20).mean().iloc[-1]
            sma_50 = close_prices.rolling(50).mean().iloc[-1] if len(close_prices) >= 50 else sma_20
            current_price = close_prices.iloc[-1]
            
            # RSI
            rsi = self._calculate_rsi(close_prices)
            current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
            
            # MACD
            macd, macd_signal = self._calculate_macd(close_prices)
            macd_value = macd.iloc[-1] if len(macd) > 0 else 0
            macd_signal_value = macd_signal.iloc[-1] if len(macd_signal) > 0 else 0
            
            # Support and resistance
            recent_highs = close_prices.rolling(20).max().iloc[-1]
            recent_lows = close_prices.rolling(20).min().iloc[-1]
            
            technical_factors = {
                'price_vs_sma20': 'Above' if current_price > sma_20 else 'Below',
                'price_vs_sma50': 'Above' if current_price > sma_50 else 'Below',
                'rsi': current_rsi,
                'rsi_condition': 'Overbought' if current_rsi > 70 else 'Oversold' if current_rsi < 30 else 'Neutral',
                'macd_bullish': macd_value > macd_signal_value,
                'near_resistance': abs(current_price - recent_highs) / current_price < 0.02,
                'near_support': abs(current_price - recent_lows) / current_price < 0.02
            }
            
            return technical_factors
            
        except Exception as e:
            print(f"Error getting technical factors: {e}")
            return {}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            print(f"Error calculating RSI: {e}")
            return pd.Series([50] * len(prices))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD indicator"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            
            return macd, macd_signal
            
        except Exception as e:
            print(f"Error calculating MACD: {e}")
            return pd.Series([0] * len(prices)), pd.Series([0] * len(prices))
    
    def generate_batch_signals(self, model_names: List[str], data: pd.DataFrame, 
                             confidence_threshold: float = 0.7) -> Dict:
        """Generate signals for multiple models"""
        signals = {}
        
        for model_name in model_names:
            signal = self.generate_signal(model_name, data, confidence_threshold)
            signals[model_name] = signal
        
        # Consensus analysis
        consensus = self._analyze_signal_consensus(signals)
        
        return {
            'individual_signals': signals,
            'consensus': consensus
        }
    
    def _analyze_signal_consensus(self, signals: Dict) -> Dict:
        """Analyze consensus among multiple signals"""
        valid_signals = {k: v for k, v in signals.items() if v is not None and 'error' not in v}
        
        if not valid_signals:
            return {'consensus_signal': 'HOLD', 'agreement': 0, 'confidence': 0}
        
        # Count signal types
        signal_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        total_confidence = 0
        
        for signal in valid_signals.values():
            signal_type = signal.get('signal_type', 'HOLD')
            confidence = signal.get('confidence', 0)
            
            signal_counts[signal_type] += 1
            total_confidence += confidence
        
        # Determine consensus
        total_signals = len(valid_signals)
        consensus_signal = max(signal_counts, key=signal_counts.get)
        agreement = signal_counts[consensus_signal] / total_signals
        avg_confidence = total_confidence / total_signals
        
        return {
            'consensus_signal': consensus_signal,
            'agreement': agreement,
            'confidence': avg_confidence,
            'signal_distribution': signal_counts,
            'total_models': total_signals
        }
    
    def validate_signal_quality(self, signal: Dict) -> Dict:
        """Validate and score signal quality"""
        quality_score = 0
        quality_factors = []
        
        # Confidence check
        confidence = signal.get('confidence', 0)
        if confidence >= 0.8:
            quality_score += 30
            quality_factors.append('High confidence')
        elif confidence >= 0.6:
            quality_score += 20
            quality_factors.append('Medium confidence')
        else:
            quality_score += 5
            quality_factors.append('Low confidence')
        
        # Signal strength check
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
        
        # Price change magnitude
        price_change_pct = abs(signal.get('price_change_pct', 0))
        if price_change_pct >= 5:
            quality_score += 20
            quality_factors.append('Significant price target')
        elif price_change_pct >= 2:
            quality_score += 10
            quality_factors.append('Moderate price target')
        else:
            quality_score += 2
            quality_factors.append('Small price target')
        
        # Technical factors check
        technical_factors = signal.get('technical_factors', {})
        if technical_factors:
            quality_score += 15
            quality_factors.append('Technical analysis available')
        
        # Final quality assessment
        if quality_score >= 80:
            quality_rating = 'Excellent'
        elif quality_score >= 60:
            quality_rating = 'Good'
        elif quality_score >= 40:
            quality_rating = 'Fair'
        else:
            quality_rating = 'Poor'
        
        return {
            'quality_score': quality_score,
            'quality_rating': quality_rating,
            'quality_factors': quality_factors,
            'recommendation': 'Trade' if quality_score >= 60 else 'Monitor' if quality_score >= 40 else 'Skip'
        }
