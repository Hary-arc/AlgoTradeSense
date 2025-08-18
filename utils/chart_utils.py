import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta

def create_candlestick_chart(df: pd.DataFrame, symbol: str, height: int = 600) -> go.Figure:
    """
    Create a professional TradingView-style candlestick chart
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Trading pair symbol
        height: Chart height in pixels
        
    Returns:
        Plotly figure object
    """
    
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
            increasing_line_color='#00D4AA',
            decreasing_line_color='#FF4B4B',
            increasing_fillcolor='#00D4AA',
            decreasing_fillcolor='#FF4B4B'
        ),
        row=1, col=1
    )
    
    # Volume bars
    colors = ['#00D4AA' if close >= open else '#FF4B4B' 
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
    
    # Update layout for professional appearance
    fig.update_layout(
        title=f"{symbol} Trading Chart",
        height=height,
        template='plotly_dark',
        showlegend=False,
        xaxis_rangeslider_visible=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Update x-axis
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128, 128, 128, 0.3)',
        showspikes=True,
        spikecolor="white",
        spikesnap="cursor",
        spikemode="across"
    )
    
    # Update y-axis
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128, 128, 128, 0.3)',
        showspikes=True,
        spikecolor="white",
        spikesnap="cursor",
        spikemode="across"
    )
    
    return fig

def add_technical_indicators(fig: go.Figure, df: pd.DataFrame, 
                           indicators: list = None) -> go.Figure:
    """
    Add technical indicators to the chart
    
    Args:
        fig: Existing plotly figure
        df: DataFrame with OHLCV data
        indicators: List of indicators to add
        
    Returns:
        Updated plotly figure
    """
    
    if indicators is None:
        indicators = ['sma_20', 'sma_50', 'bollinger', 'rsi']
    
    try:
        # Moving Averages
        if 'sma_20' in indicators:
            sma_20 = ta.trend.sma_indicator(df['close'], window=20)
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=sma_20,
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='orange', width=1),
                    opacity=0.8
                ),
                row=1, col=1
            )
        
        if 'sma_50' in indicators:
            sma_50 = ta.trend.sma_indicator(df['close'], window=50)
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=sma_50,
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='purple', width=1),
                    opacity=0.8
                ),
                row=1, col=1
            )
        
        # Bollinger Bands
        if 'bollinger' in indicators:
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
                    line=dict(color='rgba(173, 204, 255, 0.8)', width=1),
                    fill=None
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=bb_low,
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='rgba(173, 204, 255, 0.8)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(173, 204, 255, 0.1)'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=bb_mid,
                    mode='lines',
                    name='BB Mid',
                    line=dict(color='rgba(173, 204, 255, 0.6)', width=1, dash='dash'),
                ),
                row=1, col=1
            )
        
        # EMA
        if 'ema_12' in indicators:
            ema_12 = ta.trend.ema_indicator(df['close'], window=12)
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=ema_12,
                    mode='lines',
                    name='EMA 12',
                    line=dict(color='yellow', width=1),
                    opacity=0.8
                ),
                row=1, col=1
            )
        
        if 'ema_26' in indicators:
            ema_26 = ta.trend.ema_indicator(df['close'], window=26)
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=ema_26,
                    mode='lines',
                    name='EMA 26',
                    line=dict(color='cyan', width=1),
                    opacity=0.8
                ),
                row=1, col=1
            )
        
    except Exception as e:
        print(f"Error adding technical indicators: {e}")
    
    return fig

def create_rsi_chart(df: pd.DataFrame, height: int = 200) -> go.Figure:
    """Create RSI indicator chart"""
    try:
        rsi = ta.momentum.rsi(df['close'], window=14)
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=rsi,
                mode='lines',
                name='RSI',
                line=dict(color='#FFD700', width=2)
            )
        )
        
        # Add overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3)
        
        fig.update_layout(
            title="RSI (14)",
            height=height,
            template='plotly_dark',
            showlegend=False,
            yaxis=dict(range=[0, 100]),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=10)
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating RSI chart: {e}")
        return go.Figure()

def create_macd_chart(df: pd.DataFrame, height: int = 200) -> go.Figure:
    """Create MACD indicator chart"""
    try:
        macd_indicator = ta.trend.MACD(df['close'])
        macd = macd_indicator.macd()
        macd_signal = macd_indicator.macd_signal()
        macd_histogram = macd_indicator.macd_diff()
        
        fig = go.Figure()
        
        # MACD line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=macd,
                mode='lines',
                name='MACD',
                line=dict(color='#00D4AA', width=2)
            )
        )
        
        # Signal line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=macd_signal,
                mode='lines',
                name='Signal',
                line=dict(color='#FF4B4B', width=2)
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
        
        fig.update_layout(
            title="MACD",
            height=height,
            template='plotly_dark',
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=10)
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating MACD chart: {e}")
        return go.Figure()

def create_volume_profile_chart(df: pd.DataFrame, height: int = 400) -> go.Figure:
    """Create volume profile chart"""
    try:
        # Calculate volume profile
        price_bins = 50
        price_range = df['high'].max() - df['low'].min()
        bin_size = price_range / price_bins
        
        volume_profile = []
        price_levels = []
        
        for i in range(price_bins):
            price_level = df['low'].min() + (i * bin_size)
            price_levels.append(price_level)
            
            # Calculate volume at this price level
            mask = (
                (df['low'] <= price_level) & 
                (df['high'] >= price_level)
            )
            volume_at_level = df.loc[mask, 'volume'].sum()
            volume_profile.append(volume_at_level)
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=volume_profile,
                y=price_levels,
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
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=10)
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating volume profile chart: {e}")
        return go.Figure()

def add_support_resistance_levels(fig: go.Figure, df: pd.DataFrame, 
                                 window: int = 20) -> go.Figure:
    """Add support and resistance levels to chart"""
    try:
        # Calculate support and resistance
        support_levels = df['low'].rolling(window=window, center=True).min()
        resistance_levels = df['high'].rolling(window=window, center=True).max()
        
        # Get significant levels (local minima/maxima)
        support_peaks = []
        resistance_peaks = []
        
        for i in range(window, len(df) - window):
            if support_levels.iloc[i] == df['low'].iloc[i]:
                support_peaks.append((df.index[i], df['low'].iloc[i]))
            
            if resistance_levels.iloc[i] == df['high'].iloc[i]:
                resistance_peaks.append((df.index[i], df['high'].iloc[i]))
        
        # Add support levels
        for timestamp, level in support_peaks[-5:]:  # Last 5 support levels
            fig.add_hline(
                y=level,
                line_dash="dot",
                line_color="green",
                opacity=0.5,
                annotation_text=f"Support: {level:.4f}",
                annotation_position="right"
            )
        
        # Add resistance levels
        for timestamp, level in resistance_peaks[-5:]:  # Last 5 resistance levels
            fig.add_hline(
                y=level,
                line_dash="dot",
                line_color="red",
                opacity=0.5,
                annotation_text=f"Resistance: {level:.4f}",
                annotation_position="right"
            )
        
        return fig
        
    except Exception as e:
        print(f"Error adding support/resistance levels: {e}")
        return fig

def create_multi_timeframe_chart(symbol: str, data_dict: dict) -> go.Figure:
    """Create multi-timeframe analysis chart"""
    try:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f'{symbol} - 1H', f'{symbol} - 4H', 
                           f'{symbol} - 1D', f'{symbol} - Volume Analysis'],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        timeframes = ['1h', '4h', '1d']
        positions = [(1, 1), (1, 2), (2, 1)]
        
        for tf, pos in zip(timeframes, positions):
            if tf in data_dict:
                df = data_dict[tf]
                
                # Add candlestick
                fig.add_trace(
                    go.Candlestick(
                        x=df.index,
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name=f"{tf.upper()}",
                        increasing_line_color='#00D4AA',
                        decreasing_line_color='#FF4B4B'
                    ),
                    row=pos[0], col=pos[1]
                )
                
                # Add SMA
                sma_20 = ta.trend.sma_indicator(df['close'], window=20)
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=sma_20,
                        mode='lines',
                        name=f'SMA 20 {tf.upper()}',
                        line=dict(color='orange', width=1),
                        opacity=0.8
                    ),
                    row=pos[0], col=pos[1]
                )
        
        # Add volume analysis in bottom right
        if '1h' in data_dict:
            df = data_dict['1h']
            colors = ['#00D4AA' if close >= open else '#FF4B4B' 
                     for close, open in zip(df['close'], df['open'])]
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['volume'],
                    name="Volume",
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            template='plotly_dark',
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=10)
        )
        
        # Remove range slider for all subplots
        fig.update_xaxes(rangeslider_visible=False)
        
        return fig
        
    except Exception as e:
        print(f"Error creating multi-timeframe chart: {e}")
        return go.Figure()

def create_correlation_matrix(symbols: list, data_dict: dict) -> go.Figure:
    """Create correlation matrix heatmap"""
    try:
        # Prepare price data
        price_data = {}
        for symbol in symbols:
            if symbol in data_dict:
                price_data[symbol] = data_dict[symbol]['close']
        
        if not price_data:
            return go.Figure()
        
        # Create correlation matrix
        correlation_df = pd.DataFrame(price_data).corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_df.values,
            x=correlation_df.columns,
            y=correlation_df.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_df.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 12},
            showscale=True
        ))
        
        fig.update_layout(
            title="Asset Correlation Matrix",
            template='plotly_dark',
            height=400,
            font=dict(color='white')
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating correlation matrix: {e}")
        return go.Figure()

def add_trade_markers(fig: go.Figure, trades: list) -> go.Figure:
    """Add buy/sell trade markers to chart"""
    try:
        buy_trades = [t for t in trades if t['type'] in ['BUY', 'buy']]
        sell_trades = [t for t in trades if t['type'] in ['SELL', 'sell']]
        
        if buy_trades:
            buy_x = [t['timestamp'] for t in buy_trades]
            buy_y = [t['price'] for t in buy_trades]
            
            fig.add_trace(
                go.Scatter(
                    x=buy_x,
                    y=buy_y,
                    mode='markers',
                    name='Buy Orders',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='green',
                        line=dict(color='white', width=2)
                    )
                )
            )
        
        if sell_trades:
            sell_x = [t['timestamp'] for t in sell_trades]
            sell_y = [t['price'] for t in sell_trades]
            
            fig.add_trace(
                go.Scatter(
                    x=sell_x,
                    y=sell_y,
                    mode='markers',
                    name='Sell Orders',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='red',
                        line=dict(color='white', width=2)
                    )
                )
            )
        
        return fig
        
    except Exception as e:
        print(f"Error adding trade markers: {e}")
        return fig

def create_portfolio_performance_chart(portfolio_history: list, 
                                     benchmark_data: pd.DataFrame = None) -> go.Figure:
    """Create portfolio performance comparison chart"""
    try:
        if not portfolio_history:
            return go.Figure()
        
        # Extract portfolio data
        dates = [p['timestamp'] for p in portfolio_history]
        values = [p['total_value'] for p in portfolio_history]
        
        fig = go.Figure()
        
        # Portfolio performance
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=values,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#00D4AA', width=3)
            )
        )
        
        # Add benchmark if provided
        if benchmark_data is not None and not benchmark_data.empty:
            # Normalize benchmark to same starting value
            start_value = values[0] if values else 10000
            benchmark_normalized = (benchmark_data['close'] / benchmark_data['close'].iloc[0]) * start_value
            
            fig.add_trace(
                go.Scatter(
                    x=benchmark_data.index,
                    y=benchmark_normalized,
                    mode='lines',
                    name='Benchmark (Buy & Hold)',
                    line=dict(color='orange', width=2, dash='dash')
                )
            )
        
        fig.update_layout(
            title="Portfolio Performance",
            height=400,
            template='plotly_dark',
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating portfolio performance chart: {e}")
        return go.Figure()
