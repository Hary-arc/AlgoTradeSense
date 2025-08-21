import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from scipy.signal import argrelextrema
import logging
from dataclasses import dataclass
from .support_resistance import detect_support_resistance
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrendLine:
    """Data class for trend line information"""
    slope: float
    intercept: float
    r_squared: float
    start_point: Tuple[int, float]
    end_point: Tuple[int, float]
    points: List[Tuple[int, float]]
    length: int

@dataclass
class Pattern:
    """Data class for pattern information"""
    type: str
    confidence: float
    upper_line: Optional[TrendLine] = None
    lower_line: Optional[TrendLine] = None
    signal: str = "NEUTRAL"
    description: str = ""
    breakout_level: Optional[float] = None
    volume_confirmation: bool = False

class PatternRecognizer:
    """
    Advanced candlestick pattern recognition for trading signals.
    Implements detection for channels, triangles, wedges, and other technical patterns.
    """
    
    def __init__(self, min_pattern_length: int = 20, confidence_threshold: float = 0.7,
                 swing_window: int = 5, min_swing_points: int = 3):
        self.min_pattern_length = min_pattern_length
        self.confidence_threshold = confidence_threshold
        self.swing_window = swing_window
        self.min_swing_points = min_swing_points
        self._cached_swing_points = {}
        
    def detect_all_patterns(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Detect all supported patterns in the given dataframe."""
        patterns = {}
        
        if len(df) < self.min_pattern_length:
            logger.warning(f"Insufficient data: {len(df)} records, need at least {self.min_pattern_length}")
            return patterns
        
        try:
            # Ensure we have required columns
            required_cols = ['high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"DataFrame must contain columns: {required_cols}")
            
            # Channel patterns
            patterns.update(self._detect_channels(df))
            
            # Triangle patterns
            patterns.update(self._detect_triangles(df))
            
            # Wedge patterns  
            patterns.update(self._detect_wedges(df))
            
            # Support/Resistance levels
            patterns['support_resistance'] = self._detect_support_resistance(df)
            
            # Candlestick patterns (simple implementation)
            patterns.update(self._detect_candlestick_patterns(df))
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
        
        return patterns
    
    def _detect_channels(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Detect ranging, uptrend, and downtrend channels."""
        patterns = {}
        
        try:
            # Get recent data for pattern analysis
            recent_data = df.tail(60)  # Use more data for better trend detection
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            
            # Find swing highs and lows
            swing_highs = self._find_swing_points(highs, type='high')
            swing_lows = self._find_swing_points(lows, type='low')
            
            if len(swing_highs) >= self.min_swing_points and len(swing_lows) >= self.min_swing_points:
                # Calculate trend lines for highs and lows
                high_trend = self._calculate_trend_line(swing_highs, highs)
                low_trend = self._calculate_trend_line(swing_lows, lows)
                
                if high_trend and low_trend and high_trend.r_squared > 0.6 and low_trend.r_squared > 0.6:
                    high_slope = high_trend.slope
                    low_slope = low_trend.slope
                    
                    # Calculate channel properties
                    channel_width = abs(high_trend.intercept - low_trend.intercept)
                    avg_price = np.mean(highs[-20:])  # Average of recent prices
                    relative_width = channel_width / avg_price
                    
                    # Only consider valid channels with sufficient width
                    if relative_width > 0.02:  # At least 2% channel width
                        # Classify channel type based on slopes and parallelism
                        slope_diff = abs(high_slope - low_slope)
                        are_parallel = slope_diff / max(abs(high_slope), abs(low_slope)) < 0.3 if max(abs(high_slope), abs(low_slope)) > 0 else True
                        
                        if are_parallel:
                            if abs(high_slope) < 0.001 and abs(low_slope) < 0.001:
                                patterns['ranging_channel'] = self._create_pattern_dict(
                                    'Ranging Channel', 0.8, 'NEUTRAL', 
                                    'Price moving sideways between support and resistance',
                                    high_trend, low_trend
                                )
                            elif high_slope > 0.001 and low_slope > 0.001:
                                patterns['uptrend_channel'] = self._create_pattern_dict(
                                    'Uptrend Channel', 0.85, 'BULLISH',
                                    'Rising channel - buy on lower line, sell on upper line',
                                    high_trend, low_trend
                                )
                            elif high_slope < -0.001 and low_slope < -0.001:
                                patterns['downtrend_channel'] = self._create_pattern_dict(
                                    'Downtrend Channel', 0.85, 'BEARISH',
                                    'Falling channel - sell on upper line, cover on lower line',
                                    high_trend, low_trend
                                )
        
        except Exception as e:
            logger.error(f"Error detecting channels: {e}")
        
        return patterns
    
    def _detect_triangles(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Detect ascending, descending, and symmetrical triangles."""
        patterns = {}
        
        try:
            recent_data = df.tail(50)
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            
            swing_highs = self._find_swing_points(highs, type='high')
            swing_lows = self._find_swing_points(lows, type='low')
            
            if len(swing_highs) >= self.min_swing_points and len(swing_lows) >= self.min_swing_points:
                high_trend = self._calculate_trend_line(swing_highs, highs)
                low_trend = self._calculate_trend_line(swing_lows, lows)
                
                if high_trend and low_trend and high_trend.r_squared > 0.5 and low_trend.r_squared > 0.5:
                    high_slope = high_trend.slope
                    low_slope = low_trend.slope
                    
                    # Calculate convergence (triangles have converging lines)
                    convergence = abs(high_slope - low_slope) > 0.002
                    
                    if convergence:
                        # Ascending Triangle: flat resistance, rising support
                        if abs(high_slope) < 0.001 and low_slope > 0.002:
                            patterns['ascending_triangle'] = self._create_pattern_dict(
                                'Ascending Triangle', 0.8, 'BULLISH',
                                'Bullish pattern - expect upward breakout',
                                high_trend, low_trend
                            )
                        
                        # Descending Triangle: falling resistance, flat support
                        elif high_slope < -0.002 and abs(low_slope) < 0.001:
                            patterns['descending_triangle'] = self._create_pattern_dict(
                                'Descending Triangle', 0.8, 'BEARISH',
                                'Bearish pattern - expect downward breakout',
                                high_trend, low_trend
                            )
                        
                        # Symmetrical Triangle: both lines converging
                        elif (high_slope < -0.001 and low_slope > 0.001) or (high_slope > 0.001 and low_slope < -0.001):
                            patterns['symmetrical_triangle'] = self._create_pattern_dict(
                                'Symmetrical Triangle', 0.75, 'NEUTRAL',
                                'Consolidation pattern - trade breakout direction',
                                high_trend, low_trend
                            )
        
        except Exception as e:
            logger.error(f"Error detecting triangles: {e}")
        
        return patterns
    
    def _detect_wedges(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Detect rising and falling wedge patterns."""
        patterns = {}
        
        try:
            recent_data = df.tail(45)
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            
            swing_highs = self._find_swing_points(highs, type='high')
            swing_lows = self._find_swing_points(lows, type='low')
            
            if len(swing_highs) >= self.min_swing_points and len(swing_lows) >= self.min_swing_points:
                high_trend = self._calculate_trend_line(swing_highs, highs)
                low_trend = self._calculate_trend_line(swing_lows, lows)
                
                if high_trend and low_trend and high_trend.r_squared > 0.5 and low_trend.r_squared > 0.5:
                    high_slope = high_trend.slope
                    low_slope = low_trend.slope
                    
                    # Wedges have slopes in the same direction but different angles
                    same_direction = (high_slope > 0 and low_slope > 0) or (high_slope < 0 and low_slope < 0)
                    
                    if same_direction:
                        slope_ratio = abs(high_slope / low_slope) if abs(low_slope) > 1e-5 else float('inf')
                        
                        # Rising Wedge: both lines rising, but upper line rises faster
                        if high_slope > 0 and low_slope > 0 and slope_ratio > 1.3:
                            patterns['rising_wedge'] = self._create_pattern_dict(
                                'Rising Wedge', 0.75, 'BEARISH',
                                'Bearish reversal pattern - expect downward breakout',
                                high_trend, low_trend
                            )
                        
                        # Falling Wedge: both lines falling, but lower line falls faster  
                        elif high_slope < 0 and low_slope < 0 and slope_ratio < 0.7:
                            patterns['falling_wedge'] = self._create_pattern_dict(
                                'Falling Wedge', 0.75, 'BULLISH',
                                'Bullish reversal pattern - expect upward breakout',
                                high_trend, low_trend
                            )
        
        except Exception as e:
            logger.error(f"Error detecting wedges: {e}")
        
        return patterns
    
    def _find_swing_points(self, data: np.ndarray, type: str = 'high', window: int = None) -> List[Tuple[int, float]]:
        """Find swing highs or lows in price data using scipy's argrelextrema."""
        if window is None:
            window = self.swing_window
        
        cache_key = (tuple(data), type, window)
        if cache_key in self._cached_swing_points:
            return self._cached_swing_points[cache_key]
        
        try:
            if type == 'high':
                indices = argrelextrema(data, np.greater, order=window)[0]
            else:  # low
                indices = argrelextrema(data, np.less, order=window)[0]
            
            swing_points = [(int(idx), float(data[idx])) for idx in indices if idx < len(data)]
            
            # Filter out weak swing points (optional)
            if len(swing_points) > 5:
                # Keep only the most significant swings
                values = [point[1] for point in swing_points]
                if type == 'high':
                    significant_indices = np.argsort(values)[-5:]  # Top 5 highs
                else:
                    significant_indices = np.argsort(values)[:5]   # Bottom 5 lows
                
                swing_points = [swing_points[i] for i in significant_indices]
                swing_points.sort(key=lambda x: x[0])  # Sort by index
            
            self._cached_swing_points[cache_key] = swing_points
            return swing_points
            
        except Exception as e:
            logger.error(f"Error finding swing points: {e}")
            return []
    
    def _calculate_trend_line(self, swing_points: List[Tuple[int, float]], data: np.ndarray) -> Optional[TrendLine]:
        """Calculate trend line from swing points using linear regression."""
        if len(swing_points) < 2:
            return None
        
        try:
            x_coords = np.array([point[0] for point in swing_points])
            y_coords = np.array([point[1] for point in swing_points])
            
            # Linear regression for trend line
            slope, intercept, r_value, _, _ = stats.linregress(x_coords, y_coords)
            
            # Calculate start and end points
            start_idx = min(x_coords)
            end_idx = max(x_coords)
            start_price = slope * start_idx + intercept
            end_price = slope * end_idx + intercept
            
            return TrendLine(
                slope=float(slope),
                intercept=float(intercept),
                r_squared=float(r_value ** 2),
                start_point=(int(start_idx), float(start_price)),
                end_point=(int(end_idx), float(end_price)),
                points=swing_points,
                length=int(end_idx - start_idx)
            )
            
        except Exception as e:
            logger.error(f"Error calculating trend line: {e}")
            return None
    
    def _detect_support_resistance(self, df: pd.DataFrame, window: int = 20, tolerance: float = 0.01) -> Dict:
        """Detect support and resistance levels using the professional detector."""
        return detect_support_resistance(df, window=window, tolerance=tolerance)
    
    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Detect basic candlestick patterns."""
        patterns = {}
        
        try:
            if len(df) < 3:
                return patterns
            
            # Get last few candles
            recent = df.tail(3)
            opens = recent['open'].values
            highs = recent['high'].values
            lows = recent['low'].values
            closes = recent['close'].values
            volumes = recent['volume'].values if 'volume' in recent.columns else np.ones(3)
            
            # Hammer pattern (single candle)
            if self._is_hammer(opens[-1], highs[-1], lows[-1], closes[-1]):
                patterns['hammer'] = {
                    'type': 'Hammer',
                    'confidence': 0.7,
                    'signal': 'BULLISH' if closes[-1] > opens[-1] else 'BEARISH',
                    'description': 'Potential reversal pattern',
                    'timestamp': recent.index[-1]
                }
            
            # Engulfing pattern (two candles)
            if len(recent) >= 2:
                if self._is_engulfing(opens[-2], closes[-2], opens[-1], closes[-1]):
                    patterns['engulfing'] = {
                        'type': 'Engulfing',
                        'confidence': 0.75,
                        'signal': 'BULLISH' if closes[-1] > opens[-1] else 'BEARISH',
                        'description': 'Strong reversal pattern',
                        'timestamp': recent.index[-1]
                    }
            
            # Doji pattern (single candle)
            if self._is_doji(opens[-1], highs[-1], lows[-1], closes[-1]):
                patterns['doji'] = {
                    'type': 'Doji',
                    'confidence': 0.65,
                    'signal': 'NEUTRAL',
                    'description': 'Indecision pattern',
                    'timestamp': recent.index[-1]
                }
                
        except Exception as e:
            logger.error(f"Error detecting candlestick patterns: {e}")
        
        return patterns
    
    def _is_hammer(self, open, high, low, close, ratio=0.7):
        """Check if candle is a hammer pattern."""
        body_size = abs(close - open)
        total_range = high - low
        if total_range == 0:
            return False
        lower_wick = min(open, close) - low
        upper_wick = high - max(open, close)
        return (lower_wick >= 2 * body_size and upper_wick <= body_size * 0.5)
    
    def _is_engulfing(self, open1, close1, open2, close2):
        """Check if two candles form an engulfing pattern."""
        body1 = abs(close1 - open1)
        body2 = abs(close2 - open2)
        if body2 > body1 * 1.5:
            if (close1 > open1 and close2 < open2) or (close1 < open1 and close2 > open2):
                return True
        return False
    
    def _is_doji(self, open, high, low, close, doji_threshold=0.05):
        """Check if candle is a doji pattern."""
        body_size = abs(close - open)
        total_range = high - low
        if total_range == 0:
            return False
        return body_size / total_range < doji_threshold
    
    def _create_pattern_dict(self, pattern_type: str, confidence: float, signal: str,
                           description: str, upper_line: TrendLine, lower_line: TrendLine) -> Dict:
        """Helper method to create pattern dictionary."""
        return {
            'type': pattern_type,
            'confidence': confidence,
            'upper_line': self._trendline_to_dict(upper_line),
            'lower_line': self._trendline_to_dict(lower_line),
            'signal': signal,
            'description': description,
            'timestamp': pd.Timestamp.now()
        }
    
    def _trendline_to_dict(self, trendline: TrendLine) -> Dict:
        """Convert TrendLine object to dictionary."""
        return {
            'slope': trendline.slope,
            'intercept': trendline.intercept,
            'r_squared': trendline.r_squared,
            'start_point': trendline.start_point,
            'end_point': trendline.end_point,
            'length': trendline.length
        }
    
    def get_trading_signals(self, patterns: Dict) -> List[Dict]:
        """Generate trading signals based on detected patterns."""
        signals = []
        
        try:
            for pattern_name, pattern_data in patterns.items():
                if pattern_name == 'support_resistance':
                    # Generate signals from support/resistance
                    current_price = pattern_data.get('current_price', 0)
                    support_levels = pattern_data.get('support_levels', [])
                    resistance_levels = pattern_data.get('resistance_levels', [])
                    
                    # Check if price is near support
                    for support in support_levels:
                        if abs(current_price - support) / current_price < 0.01:
                            signals.append({
                                'pattern': 'Support Level',
                                'signal': 'BULLISH',
                                'strength': 0.7,
                                'description': f'Price near support at {support:.4f}',
                                'timestamp': pd.Timestamp.now()
                            })
                    
                    # Check if price is near resistance
                    for resistance in resistance_levels:
                        if abs(current_price - resistance) / current_price < 0.01:
                            signals.append({
                                'pattern': 'Resistance Level',
                                'signal': 'BEARISH',
                                'strength': 0.7,
                                'description': f'Price near resistance at {resistance:.4f}',
                                'timestamp': pd.Timestamp.now()
                            })
                    
                    continue
                
                signal_strength = pattern_data.get('confidence', 0.5)
                signal_type = pattern_data.get('signal', 'NEUTRAL')
                
                if signal_strength >= self.confidence_threshold:
                    signals.append({
                        'pattern': pattern_data['type'],
                        'signal': signal_type,
                        'strength': signal_strength,
                        'description': pattern_data['description'],
                        'timestamp': pattern_data.get('timestamp', pd.Timestamp.now())
                    })
        
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
        
        return signals
    
    def clear_cache(self):
        """Clear cached swing points."""
        self._cached_swing_points.clear()

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    prices = np.cumsum(np.random.randn(100) * 0.01) + 100
    highs = prices + np.abs(np.random.randn(100) * 0.5)
    lows = prices - np.abs(np.random.randn(100) * 0.5)
    
    df = pd.DataFrame({
        'open': prices,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Test pattern recognition
    recognizer = PatternRecognizer()
    patterns = recognizer.detect_all_patterns(df)
    signals = recognizer.get_trading_signals(patterns)
    
    print("Detected Patterns:")
    for name, pattern in patterns.items():
        if name != 'support_resistance':
            print(f"  {name}: {pattern['type']} ({pattern['signal']})")
    
    print("\nTrading Signals:")
    for signal in signals:
        print(f"  {signal['pattern']}: {signal['signal']} (Strength: {signal['strength']})")