import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import technical analysis library
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    logger.warning("TA library not available. Technical indicators will be disabled.")
    TA_AVAILABLE = False

# Color scheme for consistent styling
COLOR_SCHEME = {
    'bullish': '#00D4AA',
    'bearish': '#FF4B4B',
    'neutral': '#808080',
    'sma_20': '#FFA500',
    'sma_50': '#800080',
    'ema_12': '#FFFF00',
    'ema_26': '#00FFFF',
    'bollinger': '#ADCFFF',
    'rsi': '#FFD700',
    'macd': '#00D4AA',
    'signal': '#FF4B4B',
    'background': 'rgba(0,0,0,0)',
    'grid': 'rgba(128, 128, 128, 0.3)'
}

def _validate_ohlcv_data(df: pd.DataFrame) -> bool:
    """Validate that DataFrame contains required OHLCV columns."""
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"Missing required columns. Needed: {required_cols}, Got: {list(df.columns)}")
        return False
    if len(df) < 5:
        logger.error(f"Insufficient data: {len(df)} records")
        return False
    return True

def _calculate_sma(series: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average with validation."""
    if len(series) < window:
        logger.warning(f"Not enough data for SMA{window}. Need {window}, got {len(series)}")
        return pd.Series(np.nan, index=series.index)
    return series.rolling(window=window).mean()

def _calculate_ema(series: pd.Series, window: int) -> pd.Series:
    """Calculate Exponential Moving Average with validation."""
    if len(series) < window:
        logger.warning(f"Not enough data for EMA{window}. Need {window}, got {len(series)}")
        return pd.Series(np.nan, index=series.index)
    return series.ewm(span=window, adjust=False).mean()

def create_candlestick_chart(df: pd.DataFrame, symbol: str, height: int = 600, 
                           title: str = None, show_volume: bool = True) -> go.Figure:
    """
    Create a professional TradingView-style candlestick chart
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Trading pair symbol
        height: Chart height in pixels
        title: Custom chart title
        show_volume: Whether to show volume subplot
        
    Returns:
        Plotly figure object
    """
    if not _validate_ohlcv_data(df):
        return go.Figure()
    
    try:
        if show_volume:
            # Create subplots with secondary y-axis for volume
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=[f'{symbol} Price Chart', 'Volume'],
                row_width=[0.2, 0.7]
            )
            
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name="Price",
                    increasing_line_color=COLOR_SCHEME['bullish'],
                    decreasing_line_color=COLOR_SCHEME['bearish'],
                    increasing_fillcolor=COLOR_SCHEME['bullish'],
                    decreasing_fillcolor=COLOR_SCHEME['bearish']
                ),
                row=1, col=1
            )
            
            # Volume bars
            if 'volume' in df.columns:
                colors = [COLOR_SCHEME['bullish'] if close >= open else COLOR_SCHEME['bearish'] 
                         for close, open in zip(df['close'], df['open'])]
                
                fig.add_trace(
                    go.Bar(
                        x=df.index,
                        y=df['volume'],
                        name="Volume",
                        marker_color=colors,
                        opacity=0.7
                    ),
                    row=2, col=1
                )
            else:
                logger.warning("Volume data not available")
        else:
            # Create single chart without volume
            fig = go.Figure()
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name="Price",
                    increasing_line_color=COLOR_SCHEME['bullish'],
                    decreasing_line_color=COLOR_SCHEME['bearish'],
                    increasing_fillcolor=COLOR_SCHEME['bullish'],
                    decreasing_fillcolor=COLOR_SCHEME['bearish']
                )
            )
        
        # Update layout for professional appearance
        chart_title = title or f"{symbol} Trading Chart"
        fig.update_layout(
            title=chart_title,
            height=height,
            template='plotly_dark',
            showlegend=False,
            xaxis_rangeslider_visible=False,
            plot_bgcolor=COLOR_SCHEME['background'],
            paper_bgcolor=COLOR_SCHEME['background'],
            font=dict(color='white', size=12),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Update axes
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=COLOR_SCHEME['grid'],
            showspikes=True,
            spikecolor="white",
            spikesnap="cursor",
            spikemode="across"
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=COLOR_SCHEME['grid'],
            showspikes=True,
            spikecolor="white",
            spikesnap="cursor",
            spikemode="across"
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating candlestick chart: {e}")
        return go.Figure()

def add_technical_indicators(fig: go.Figure, df: pd.DataFrame, 
                           indicators: list = None, row: int = 1, col: int = 1) -> go.Figure:
    """
    Add technical indicators to the chart
    
    Args:
        fig: Existing plotly figure
        df: DataFrame with OHLCV data
        indicators: List of indicators to add
        row: Subplot row number
        col: Subplot column number
        
    Returns:
        Updated plotly figure
    """
    
    if not _validate_ohlcv_data(df):
        return fig
    
    if indicators is None:
        indicators = ['sma_20', 'sma_50']
    
    try:
        # Moving Averages
        if 'sma_20' in indicators:
            sma_20 = _calculate_sma(df['close'], 20)
            if not sma_20.isna().all():
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=sma_20,
                        mode='lines',
                        name='SMA 20',
                        line=dict(color=COLOR_SCHEME['sma_20'], width=1.5),
                        opacity=0.8
                    ),
                    row=row, col=col
                )
        
        if 'sma_50' in indicators:
            sma_50 = _calculate_sma(df['close'], 50)
            if not sma_50.isna().all():
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=sma_50,
                        mode='lines',
                        name='SMA 50',
                        line=dict(color=COLOR_SCHEME['sma_50'], width=1.5),
                        opacity=0.8
                    ),
                    row=row, col=col
                )
        
        # EMA (Exponential Moving Average)
        if 'ema_12' in indicators:
            ema_12 = _calculate_ema(df['close'], 12)
            if not ema_12.isna().all():
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=ema_12,
                        mode='lines',
                        name='EMA 12',
                        line=dict(color=COLOR_SCHEME['ema_12'], width=1.5),
                        opacity=0.8
                    ),
                    row=row, col=col
                )
        
        if 'ema_26' in indicators:
            ema_26 = _calculate_ema(df['close'], 26)
            if not ema_26.isna().all():
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=ema_26,
                        mode='lines',
                        name='EMA 26',
                        line=dict(color=COLOR_SCHEME['ema_26'], width=1.5),
                        opacity=0.8
                    ),
                    row=row, col=col
                )
        
        # Bollinger Bands (using TA library if available)
        if 'bollinger' in indicators and TA_AVAILABLE:
            try:
                bollinger = ta.volatility.BollingerBands(df['close'])
                bb_high = bollinger.bollinger_hband()
                bb_low = bollinger.bollinger_lband()
                bb_mid = bollinger.bollinger_mavg()
                
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=bb_high,
                        mode='lines',
                        name='BB Upper',
                        line=dict(color=COLOR_SCHEME['bollinger'], width=1),
                        opacity=0.7
                    ),
                    row=row, col=col
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=bb_low,
                        mode='lines',
                        name='BB Lower',
                        line=dict(color=COLOR_SCHEME['bollinger'], width=1),
                        opacity=0.7,
                        fill='tonexty',
                        fillcolor='rgba(173, 204, 255, 0.1)'
                    ),
                    row=row, col=col
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=bb_mid,
                        mode='lines',
                        name='BB Mid',
                        line=dict(color=COLOR_SCHEME['bollinger'], width=1, dash='dash'),
                        opacity=0.6
                    ),
                    row=row, col=col
                )
            except Exception as e:
                logger.warning(f"Could not add Bollinger Bands: {e}")
        
    except Exception as e:
        logger.error(f"Error adding technical indicators: {e}")
    
    return fig

def create_rsi_chart(df: pd.DataFrame, height: int = 200, window: int = 14) -> go.Figure:
    """Create RSI indicator chart"""
    if not _validate_ohlcv_data(df):
        return go.Figure()
    
    try:
        # Calculate RSI manually if TA not available
        if TA_AVAILABLE:
            rsi = ta.momentum.rsi(df['close'], window=window)
        else:
            # Manual RSI calculation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=rsi,
                mode='lines',
                name='RSI',
                line=dict(color=COLOR_SCHEME['rsi'], width=2)
            )
        )
        
        # Add overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3)
        
        fig.update_layout(
            title=f"RSI ({window})",
            height=height,
            template='plotly_dark',
            showlegend=False,
            yaxis=dict(range=[0, 100]),
            plot_bgcolor=COLOR_SCHEME['background'],
            paper_bgcolor=COLOR_SCHEME['background'],
            font=dict(color='white', size=10)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating RSI chart: {e}")
        return go.Figure()

def create_macd_chart(df: pd.DataFrame, height: int = 200) -> go.Figure:
    """Create MACD indicator chart"""
    if not _validate_ohlcv_data(df):
        return go.Figure()
    
    try:
        if TA_AVAILABLE:
            macd_indicator = ta.trend.MACD(df['close'])
            macd = macd_indicator.macd()
            macd_signal = macd_indicator.macd_signal()
            macd_histogram = macd_indicator.macd_diff()
        else:
            # Manual MACD calculation
            ema_12 = _calculate_ema(df['close'], 12)
            ema_26 = _calculate_ema(df['close'], 26)
            macd = ema_12 - ema_26
            macd_signal = _calculate_ema(macd, 9)
            macd_histogram = macd - macd_signal
        
        fig = go.Figure()
        
        # MACD line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=macd,
                mode='lines',
                name='MACD',
                line=dict(color=COLOR_SCHEME['macd'], width=2)
            )
        )
        
        # Signal line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=macd_signal,
                mode='lines',
                name='Signal',
                line=dict(color=COLOR_SCHEME['signal'], width=2)
            )
        )
        
        # Histogram
        colors = ['green' if val >= 0 else 'red' for val in macd_histogram]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=macd_histogram,
                name='Histogram',
                marker_color=colors,
                opacity=0.6
            )
        )
        
        # Add zero line
        fig.add_hline(y=0, line_width=1, line_color='white', opacity=0.3)
        
        fig.update_layout(
            title="MACD",
            height=height,
            template='plotly_dark',
            showlegend=True,
            plot_bgcolor=COLOR_SCHEME['background'],
            paper_bgcolor=COLOR_SCHEME['background'],
            font=dict(color='white', size=10)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating MACD chart: {e}")
        return go.Figure()

def create_volume_profile_chart(df: pd.DataFrame, height: int = 400, bins: int = 50) -> go.Figure:
    """Create volume profile chart using efficient calculation"""
    if not _validate_ohlcv_data(df) or 'volume' not in df.columns:
        return go.Figure()
    
    try:
        # Use numpy for efficient volume profile calculation
        low, high = df['low'].min(), df['high'].max()
        price_range = high - low
        
        if price_range <= 0:
            logger.warning("Invalid price range for volume profile")
            return go.Figure()
        
        # Create price bins
        bin_edges = np.linspace(low, high, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate volume per price bin
        volume_profile = np.zeros(bins)
        
        for idx, row in df.iterrows():
            # Find which bins this candle contributes to
            candle_low, candle_high = row['low'], row['high']
            candle_volume = row['volume']
            
            # Find overlapping bins
            start_bin = np.searchsorted(bin_edges, candle_low, side='right') - 1
            end_bin = np.searchsorted(bin_edges, candle_high, side='left')
            
            # Distribute volume proportionally to overlapping bins
            for bin_idx in range(max(0, start_bin), min(bins, end_bin)):
                bin_low = bin_edges[bin_idx]
                bin_high = bin_edges[bin_idx + 1]
                
                # Calculate overlap
                overlap_low = max(candle_low, bin_low)
                overlap_high = min(candle_high, bin_high)
                overlap_pct = max(0, overlap_high - overlap_low) / (candle_high - candle_low)
                
                volume_profile[bin_idx] += candle_volume * overlap_pct
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=volume_profile,
                y=bin_centers,
                orientation='h',
                name='Volume Profile',
                marker_color='rgba(0, 212, 170, 0.6)',
                marker_line=dict(color='rgba(0, 212, 170, 1)', width=1)
            )
        )
        
        fig.update_layout(
            title="Volume Profile",
            height=height,
            template='plotly_dark',
            showlegend=False,
            xaxis_title="Volume",
            yaxis_title="Price",
            plot_bgcolor=COLOR_SCHEME['background'],
            paper_bgcolor=COLOR_SCHEME['background'],
            font=dict(color='white', size=10)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating volume profile chart: {e}")
        return go.Figure()

def add_support_resistance_levels(fig: go.Figure, df: pd.DataFrame, 
                                 window: int = 20, num_levels: int = 5) -> go.Figure:
    """Add support and resistance levels to chart using improved detection"""
    if not _validate_ohlcv_data(df):
        return fig
    
    try:
        from scipy.signal import argrelextrema
        
        # Find local minima (support) and maxima (resistance)
        local_minima = argrelextrema(df['low'].values, np.less, order=window)[0]
        local_maxima = argrelextrema(df['high'].values, np.greater, order=window)[0]
        
        # Get the most recent levels
        recent_minima = local_minima[-num_levels:] if len(local_minima) > 0 else []
        recent_maxima = local_maxima[-num_levels:] if len(local_maxima) > 0 else []
        
        # Add support levels
        for idx in recent_minima:
            if idx < len(df):
                level = df['low'].iloc[idx]
                fig.add_hline(
                    y=level,
                    line_dash="dot",
                    line_color="green",
                    opacity=0.5,
                    annotation_text=f"S: {level:.4f}",
                    annotation_position="right",
                    annotation_font_size=10
                )
        
        # Add resistance levels
        for idx in recent_maxima:
            if idx < len(df):
                level = df['high'].iloc[idx]
                fig.add_hline(
                    y=level,
                    line_dash="dot",
                    line_color="red",
                    opacity=0.5,
                    annotation_text=f"R: {level:.4f}",
                    annotation_position="right",
                    annotation_font_size=10
                )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error adding support/resistance levels: {e}")
        return fig

def add_trade_markers(fig: go.Figure, trades: list, row: int = 1, col: int = 1) -> go.Figure:
    """Add buy/sell trade markers to chart with improved formatting"""
    try:
        if not trades:
            return fig
        
        buy_trades = [t for t in trades if str(t.get('type', '')).upper() in ['BUY', 'LONG']]
        sell_trades = [t for t in trades if str(t.get('type', '')).upper() in ['SELL', 'SHORT']]
        
        if buy_trades:
            buy_x = [t.get('timestamp') for t in buy_trades if 'timestamp' in t]
            buy_y = [t.get('price') for t in buy_trades if 'price' in t]
            
            if buy_x and buy_y:
                fig.add_trace(
                    go.Scatter(
                        x=buy_x,
                        y=buy_y,
                        mode='markers+text',
                        name='Buy Orders',
                        marker=dict(
                            symbol='triangle-up',
                            size=15,
                            color='green',
                            line=dict(color='white', width=2)
                        ),
                        text=["BUY"] * len(buy_x),
                        textposition="top center",
                        textfont=dict(color='green', size=10)
                    ),
                    row=row, col=col
                )
        
        if sell_trades:
            sell_x = [t.get('timestamp') for t in sell_trades if 'timestamp' in t]
            sell_y = [t.get('price') for t in sell_trades if 'price' in t]
            
            if sell_x and sell_y:
                fig.add_trace(
                    go.Scatter(
                        x=sell_x,
                        y=sell_y,
                        mode='markers+text',
                        name='Sell Orders',
                        marker=dict(
                            symbol='triangle-down',
                            size=15,
                            color='red',
                            line=dict(color='white', width=2)
                        ),
                        text=["SELL"] * len(sell_x),
                        textposition="bottom center",
                        textfont=dict(color='red', size=10)
                    ),
                    row=row, col=col
                )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error adding trade markers: {e}")
        return fig

def create_performance_dashboard(portfolio_history: list, benchmark_data: pd.DataFrame = None, 
                               trades: list = None, height: int = 800) -> go.Figure:
    """Create comprehensive performance dashboard"""
    try:
        if not portfolio_history:
            return go.Figure()
        
        # Create subplots: portfolio value, drawdown, daily returns
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Portfolio Value', 'Drawdown', 'Daily Returns'],
            vertical_spacing=0.1
        )
        
        # Extract portfolio data
        dates = [p['timestamp'] for p in portfolio_history]
        values = [p['total_value'] for p in portfolio_history]
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=values,
                mode='lines',
                name='Portfolio',
                line=dict(color=COLOR_SCHEME['bullish'], width=3)
            ),
            row=1, col=1
        )
        
        # Benchmark comparison
        if benchmark_data is not None and not benchmark_data.empty:
            start_value = values[0] if values else 10000
            benchmark_normalized = (benchmark_data['close'] / benchmark_data['close'].iloc[0]) * start_value
            
            fig.add_trace(
                go.Scatter(
                    x=benchmark_data.index,
                    y=benchmark_normalized,
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='orange', width=2, dash='dash')
                ),
                row=1, col=1
            )
        
        # Drawdown calculation
        portfolio_series = pd.Series(values, index=dates)
        rolling_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - rolling_max) / rolling_max * 100
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=drawdown,
                mode='lines',
                name='Drawdown',
                line=dict(color='red', width=2),
                fill='tozeroy'
            ),
            row=2, col=1
        )
        
        # Daily returns
        daily_returns = portfolio_series.pct_change() * 100
        
        colors = [COLOR_SCHEME['bullish'] if ret >= 0 else COLOR_SCHEME['bearish'] 
                 for ret in daily_returns]
        
        fig.add_trace(
            go.Bar(
                x=dates,
                y=daily_returns,
                name='Daily Returns',
                marker_color=colors,
                opacity=0.7
            ),
            row=3, col=1
        )
        
        # Add zero lines
        fig.add_hline(y=0, row=2, col=1, line_width=1, line_color='white', opacity=0.5)
        fig.add_hline(y=0, row=3, col=1, line_width=1, line_color='white', opacity=0.5)
        
        # Update layout
        fig.update_layout(
            title="Portfolio Performance Dashboard",
            height=height,
            template='plotly_dark',
            showlegend=True,
            plot_bgcolor=COLOR_SCHEME['background'],
            paper_bgcolor=COLOR_SCHEME['background'],
            font=dict(color='white')
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating performance dashboard: {e}")
        return go.Figure()

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.abs(np.random.randn(100) * 0.5),
        'low': prices - np.abs(np.random.randn(100) * 0.5),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Test chart creation
    fig = create_candlestick_chart(df, 'TEST')
    fig = add_technical_indicators(fig, df, ['sma_20', 'sma_50'])
    fig = add_support_resistance_levels(fig, df)
    
    # Create sample trades
    sample_trades = [
        {'timestamp': dates[30], 'price': df['close'].iloc[30], 'type': 'BUY'},
        {'timestamp': dates[70], 'price': df['close'].iloc[70], 'type': 'SELL'}
    ]
    fig = add_trade_markers(fig, sample_trades)
    
    fig.show()
    
    # Create indicator charts
    rsi_fig = create_rsi_chart(df)
    macd_fig = create_macd_chart(df)
    volume_fig = create_volume_profile_chart(df)
    
    print("Charts created successfully!")