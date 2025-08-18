import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import ta

class PatternRecognizer:
    """
    Advanced candlestick pattern recognition for trading signals.
    Implements detection for channels, triangles, wedges, and other technical patterns.
    """
    
    def __init__(self, min_pattern_length: int = 20, confidence_threshold: float = 0.7):
        self.min_pattern_length = min_pattern_length
        self.confidence_threshold = confidence_threshold
        
    def detect_all_patterns(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Detect all supported patterns in the given dataframe."""
        patterns = {}
        
        if len(df) < self.min_pattern_length:
            return patterns
            
        # Channel patterns
        patterns.update(self._detect_channels(df))
        
        # Triangle patterns
        patterns.update(self._detect_triangles(df))
        
        # Wedge patterns  
        patterns.update(self._detect_wedges(df))
        
        # Support/Resistance levels
        patterns['support_resistance'] = self._detect_support_resistance(df)
        
        return patterns
    
    def _detect_channels(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Detect ranging, uptrend, and downtrend channels."""
        patterns = {}
        
        # Get recent data for pattern analysis
        recent_data = df.tail(50)
        highs = recent_data['high'].values
        lows = recent_data['low'].values
        closes = recent_data['close'].values
        
        # Find swing highs and lows
        swing_highs = self._find_swing_points(highs, type='high')
        swing_lows = self._find_swing_points(lows, type='low')
        
        if len(swing_highs) >= 3 and len(swing_lows) >= 3:
            # Calculate trend lines for highs and lows
            high_trend = self._calculate_trend_line(swing_highs, highs)
            low_trend = self._calculate_trend_line(swing_lows, lows)
            
            if high_trend and low_trend:
                high_slope = high_trend['slope']
                low_slope = low_trend['slope']
                
                # Classify channel type based on slopes
                if abs(high_slope) < 0.1 and abs(low_slope) < 0.1:
                    patterns['ranging_channel'] = {
                        'type': 'Ranging Channel',
                        'confidence': 0.8,
                        'upper_line': high_trend,
                        'lower_line': low_trend,
                        'signal': 'NEUTRAL',
                        'description': 'Price moving sideways between support and resistance'
                    }
                elif high_slope > 0.1 and low_slope > 0.1:
                    patterns['uptrend_channel'] = {
                        'type': 'Uptrend Channel',
                        'confidence': 0.85,
                        'upper_line': high_trend,
                        'lower_line': low_trend,
                        'signal': 'BULLISH',
                        'description': 'Rising channel - buy on lower line, sell on upper line'
                    }
                elif high_slope < -0.1 and low_slope < -0.1:
                    patterns['downtrend_channel'] = {
                        'type': 'Downtrend Channel',
                        'confidence': 0.85,
                        'upper_line': high_trend,
                        'lower_line': low_trend,
                        'signal': 'BEARISH',
                        'description': 'Falling channel - sell on upper line, cover on lower line'
                    }
        
        return patterns
    
    def _detect_triangles(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Detect ascending, descending, and expanding triangles."""
        patterns = {}
        
        recent_data = df.tail(40)
        highs = recent_data['high'].values
        lows = recent_data['low'].values
        
        swing_highs = self._find_swing_points(highs, type='high')
        swing_lows = self._find_swing_points(lows, type='low')
        
        if len(swing_highs) >= 3 and len(swing_lows) >= 3:
            high_trend = self._calculate_trend_line(swing_highs, highs)
            low_trend = self._calculate_trend_line(swing_lows, lows)
            
            if high_trend and low_trend:
                high_slope = high_trend['slope']
                low_slope = low_trend['slope']
                
                # Ascending Triangle: flat resistance, rising support
                if abs(high_slope) < 0.05 and low_slope > 0.1:
                    patterns['ascending_triangle'] = {
                        'type': 'Ascending Triangle',
                        'confidence': 0.8,
                        'upper_line': high_trend,
                        'lower_line': low_trend,
                        'signal': 'BULLISH',
                        'description': 'Bullish pattern - expect upward breakout'
                    }
                
                # Descending Triangle: falling resistance, flat support
                elif high_slope < -0.1 and abs(low_slope) < 0.05:
                    patterns['descending_triangle'] = {
                        'type': 'Descending Triangle',
                        'confidence': 0.8,
                        'upper_line': high_trend,
                        'lower_line': low_trend,
                        'signal': 'BEARISH',
                        'description': 'Bearish pattern - expect downward breakout'
                    }
                
                # Expanding Triangle: diverging lines
                elif high_slope > 0.05 and low_slope < -0.05:
                    patterns['expanding_triangle'] = {
                        'type': 'Expanding Triangle',
                        'confidence': 0.75,
                        'upper_line': high_trend,
                        'lower_line': low_trend,
                        'signal': 'VOLATILE',
                        'description': 'High volatility pattern - trade breakouts carefully'
                    }
        
        return patterns
    
    def _detect_wedges(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Detect rising and falling wedge patterns."""
        patterns = {}
        
        recent_data = df.tail(35)
        highs = recent_data['high'].values
        lows = recent_data['low'].values
        
        swing_highs = self._find_swing_points(highs, type='high')
        swing_lows = self._find_swing_points(lows, type='low')
        
        if len(swing_highs) >= 3 and len(swing_lows) >= 3:
            high_trend = self._calculate_trend_line(swing_highs, highs)
            low_trend = self._calculate_trend_line(swing_lows, lows)
            
            if high_trend and low_trend:
                high_slope = high_trend['slope']
                low_slope = low_trend['slope']
                
                # Rising Wedge: both lines rising, but upper line rises faster
                if high_slope > 0.1 and low_slope > 0.05 and high_slope > low_slope * 1.5:
                    patterns['rising_wedge'] = {
                        'type': 'Rising Wedge',
                        'confidence': 0.75,
                        'upper_line': high_trend,
                        'lower_line': low_trend,
                        'signal': 'BEARISH',
                        'description': 'Bearish reversal pattern - expect downward breakout'
                    }
                
                # Falling Wedge: both lines falling, but lower line falls faster  
                elif high_slope < -0.05 and low_slope < -0.1 and abs(low_slope) > abs(high_slope) * 1.5:
                    patterns['falling_wedge'] = {
                        'type': 'Falling Wedge',
                        'confidence': 0.75,
                        'upper_line': high_trend,
                        'lower_line': low_trend,
                        'signal': 'BULLISH',
                        'description': 'Bullish reversal pattern - expect upward breakout'
                    }
        
        return patterns
    
    def _find_swing_points(self, data: np.ndarray, type: str = 'high', window: int = 5) -> List[Tuple[int, float]]:
        """Find swing highs or lows in price data."""
        swing_points = []
        
        for i in range(window, len(data) - window):
            if type == 'high':
                is_swing = all(data[i] >= data[i-j] for j in range(1, window+1)) and \
                          all(data[i] >= data[i+j] for j in range(1, window+1))
            else:  # low
                is_swing = all(data[i] <= data[i-j] for j in range(1, window+1)) and \
                          all(data[i] <= data[i+j] for j in range(1, window+1))
            
            if is_swing:
                swing_points.append((i, data[i]))
        
        return swing_points
    
    def _calculate_trend_line(self, swing_points: List[Tuple[int, float]], data: np.ndarray) -> Optional[Dict]:
        """Calculate trend line from swing points."""
        if len(swing_points) < 2:
            return None
            
        x_coords = [point[0] for point in swing_points]
        y_coords = [point[1] for point in swing_points]
        
        # Linear regression for trend line
        slope, intercept, r_value, _, _ = stats.linregress(x_coords, y_coords)
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'start_point': (x_coords[0], y_coords[0]),
            'end_point': (x_coords[-1], y_coords[-1]),
            'points': swing_points
        }
    
    def _detect_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Dict:
        """Detect key support and resistance levels."""
        recent_data = df.tail(100)
        highs = recent_data['high']
        lows = recent_data['low']
        closes = recent_data['close']
        
        # Calculate support and resistance levels
        resistance_levels = []
        support_levels = []
        
        # Rolling max/min for dynamic levels
        for i in range(window, len(recent_data)):
            local_high = highs.iloc[i-window:i].max()
            local_low = lows.iloc[i-window:i].min()
            
            # Check if current price is near these levels
            current_price = closes.iloc[i]
            
            if abs(current_price - local_high) / current_price < 0.01:
                resistance_levels.append(local_high)
            if abs(current_price - local_low) / current_price < 0.01:
                support_levels.append(local_low)
        
        return {
            'resistance_levels': list(set(resistance_levels))[-3:],  # Top 3 recent
            'support_levels': list(set(support_levels))[-3:],       # Top 3 recent
            'current_price': float(closes.iloc[-1])
        }
    
    def get_trading_signals(self, patterns: Dict) -> List[Dict]:
        """Generate trading signals based on detected patterns."""
        signals = []
        
        for pattern_name, pattern_data in patterns.items():
            if pattern_name == 'support_resistance':
                continue
                
            signal_strength = pattern_data.get('confidence', 0.5)
            signal_type = pattern_data.get('signal', 'NEUTRAL')
            
            if signal_strength >= self.confidence_threshold:
                signals.append({
                    'pattern': pattern_data['type'],
                    'signal': signal_type,
                    'strength': signal_strength,
                    'description': pattern_data['description'],
                    'timestamp': pd.Timestamp.now()
                })
        
        return signals