import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
from utils.binance_client import BinanceClient
from utils.pattern_recognition import PatternRecognizer
import os

st.set_page_config(page_title="Trading Dashboard", page_icon="📈", layout="wide")

# Initialize Binance client
@st.cache_resource
def get_binance_client():
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
    
    if not api_key or not api_secret:
        return None
    
    try:
        client = BinanceClient(api_key, api_secret, testnet=testnet)
        return client
    except Exception as e:
        st.error(f"Failed to initialize Binance client: {str(e)}")
        return None

def create_tradingview_chart(df, symbol, patterns=None):
    """Create professional TradingView-style candlestick chart with pattern overlays."""
    
    # Create subplot with secondary y-axis for volume
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{symbol} Price Action', 'Volume'),
        row_heights=[0.7, 0.3]
    )
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            increasing_fillcolor='rgba(38, 166, 154, 0.5)',
            decreasing_fillcolor='rgba(239, 83, 80, 0.5)'
        ),
        row=1, col=1
    )
    
    # Add volume bars
    colors = ['#26a69a' if close >= open else '#ef5350' 
              for close, open in zip(df['close'], df['open'])]
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            marker_color=colors,
            name='Volume',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Add pattern overlays if available
    if patterns:
        for pattern_name, pattern_data in patterns.items():
            if pattern_name == 'support_resistance':
                # Add support/resistance lines
                sr_data = pattern_data
                current_time = df.index[-1]
                start_time = df.index[0]
                
                for level in sr_data.get('resistance_levels', []):
                    fig.add_shape(
                        type="line",
                        x0=start_time, y0=level, x1=current_time, y1=level,
                        line=dict(color="red", width=2, dash="dash"),
                        row=1, col=1
                    )
                    fig.add_annotation(
                        x=current_time, y=level,
                        text=f"R: {level:.2f}",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor="red",
                        bgcolor="red",
                        bordercolor="red",
                        font=dict(color="white", size=10),
                        row=1, col=1
                    )
                
                for level in sr_data.get('support_levels', []):
                    fig.add_shape(
                        type="line",
                        x0=start_time, y0=level, x1=current_time, y1=level,
                        line=dict(color="green", width=2, dash="dash"),
                        row=1, col=1
                    )
                    fig.add_annotation(
                        x=current_time, y=level,
                        text=f"S: {level:.2f}",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor="green",
                        bgcolor="green",
                        bordercolor="green",
                        font=dict(color="white", size=10),
                        row=1, col=1
                    )
            
            else:
                # Add trend lines for patterns
                if 'upper_line' in pattern_data and 'lower_line' in pattern_data:
                    upper_line = pattern_data['upper_line']
                    lower_line = pattern_data['lower_line']
                    
                    # Calculate line points
                    x_start = df.index[upper_line['start_point'][0]]
                    x_end = df.index[min(upper_line['end_point'][0], len(df)-1)]
                    
                    # Upper trend line
                    y_start_upper = upper_line['start_point'][1]
                    y_end_upper = upper_line['end_point'][1]
                    
                    fig.add_shape(
                        type="line",
                        x0=x_start, y0=y_start_upper,
                        x1=x_end, y1=y_end_upper,
                        line=dict(color="#2196F3", width=2, dash="dot"),
                        row=1, col=1
                    )
                    
                    # Lower trend line
                    y_start_lower = lower_line['start_point'][1]
                    y_end_lower = lower_line['end_point'][1]
                    
                    fig.add_shape(
                        type="line",
                        x0=x_start, y0=y_start_lower,
                        x1=x_end, y1=y_end_lower,
                        line=dict(color="#2196F3", width=2, dash="dot"),
                        row=1, col=1
                    )
                    
                    # Add pattern label
                    mid_x = x_start + (x_end - x_start) / 2
                    mid_y = y_start_upper + (y_start_lower - y_start_upper) / 2
                    
                    fig.add_annotation(
                        x=mid_x, y=mid_y,
                        text=pattern_data['type'],
                        showarrow=False,
                        bgcolor="rgba(33, 150, 243, 0.8)",
                        bordercolor="#2196F3",
                        font=dict(color="white", size=12, family="Arial Black"),
                        row=1, col=1
                    )
    
    # Update layout for professional appearance
    fig.update_layout(
        title=dict(
            text=f"{symbol} Live Chart with Pattern Recognition",
            x=0.5,
            font=dict(size=20, family="Arial Black")
        ),
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Update axes
    fig.update_xaxes(
        title_text="Time",
        showgrid=True,
        gridcolor='rgba(128, 128, 128, 0.2)',
        row=2, col=1
    )
    fig.update_yaxes(
        title_text="Price (USDT)",
        showgrid=True,
        gridcolor='rgba(128, 128, 128, 0.2)',
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Volume",
        showgrid=True,
        gridcolor='rgba(128, 128, 128, 0.2)',
        row=2, col=1
    )
    
    return fig

def main():
    st.title("📈 Live Trading Dashboard with Pattern Recognition")
    st.markdown("---")
    
    client = get_binance_client()
    if not client:
        st.error("⚠️ Unable to connect to Binance API. Please check your credentials.")
        return
    
    # Initialize pattern recognizer
    pattern_recognizer = PatternRecognizer(min_pattern_length=20, confidence_threshold=0.7)
    
    # Symbol selection
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        symbol = st.selectbox(
            "Select Trading Pair",
            ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT", "BNBUSDT"],
            index=0
        )
    
    with col2:
        interval = st.selectbox(
            "Timeframe",
            ["1m", "5m", "15m", "1h", "4h", "1d"],
            index=3
        )
    
    with col3:
        limit = st.selectbox(
            "Data Points",
            [100, 200, 500, 1000],
            index=1
        )
    
    with col4:
        auto_refresh = st.checkbox("Auto Refresh", value=True)
    
    # Fetch and display data
    try:
        # Get market data
        klines = client.get_klines(symbol, interval, limit)
        if not klines:
            st.error("Unable to fetch market data")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Process data
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        
        # Detect patterns
        patterns = pattern_recognizer.detect_all_patterns(df)
        trading_signals = pattern_recognizer.get_trading_signals(patterns)
        
        # Display current market info
        current_price = float(df['close'].iloc[-1])
        price_change = float(df['close'].iloc[-1]) - float(df['close'].iloc[-2])
        price_change_pct = (price_change / float(df['close'].iloc[-2])) * 100
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Current Price", f"${current_price:,.2f}", f"{price_change:+.2f} ({price_change_pct:+.1f}%)")
        
        with col2:
            st.metric("24h High", f"${df['high'].max():,.2f}")
        
        with col3:
            st.metric("24h Low", f"${df['low'].min():,.2f}")
        
        with col4:
            st.metric("Volume", f"{df['volume'].iloc[-1]:,.0f}")
        
        with col5:
            st.metric("Patterns Found", len([p for p in patterns.keys() if p != 'support_resistance']))
        
        st.markdown("---")
        
        # Create and display chart
        chart_fig = create_tradingview_chart(df, symbol, patterns)
        st.plotly_chart(chart_fig, use_container_width=True)
        
        # Display pattern analysis
        if patterns:
            st.subheader("🔍 Pattern Analysis")
            
            # Create pattern summary cards
            pattern_keys = [p for p in patterns.keys() if p != 'support_resistance']
            if pattern_keys:
                pattern_cols = st.columns(min(3, len(pattern_keys)))
                
                for pattern_idx, pattern_name in enumerate(pattern_keys):
                    if pattern_idx < len(pattern_cols):
                        pattern_data = patterns[pattern_name]
                        with pattern_cols[pattern_idx]:
                            signal_color = {
                                'BULLISH': '🟢',
                                'BEARISH': '🔴', 
                                'NEUTRAL': '🟡',
                                'VOLATILE': '🟠'
                            }.get(pattern_data['signal'], '⚪')
                            
                            st.markdown(f"""
                            <div style='padding: 1rem; border: 1px solid #ddd; border-radius: 0.5rem; margin: 0.5rem 0;'>
                                <h4>{signal_color} {pattern_data['type']}</h4>
                                <p><strong>Signal:</strong> {pattern_data['signal']}</p>
                                <p><strong>Confidence:</strong> {pattern_data['confidence']:.1%}</p>
                                <p style='font-size: 0.9em;'>{pattern_data['description']}</p>
                            </div>
                            """, unsafe_allow_html=True)
        
        # Display trading signals
        if trading_signals:
            st.subheader("🎯 Trading Signals")
            
            signal_df = pd.DataFrame(trading_signals)
            st.dataframe(
                signal_df[['pattern', 'signal', 'strength', 'description']],
                use_container_width=True,
                hide_index=True
            )
        
        # Support and Resistance levels
        if 'support_resistance' in patterns:
            st.subheader("📊 Key Levels")
            
            sr_data = patterns['support_resistance']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Resistance Levels:**")
                for level in sr_data.get('resistance_levels', []):
                    distance = ((level - current_price) / current_price) * 100
                    st.write(f"🔴 ${level:.2f} ({distance:+.1f}%)")
            
            with col2:
                st.write("**Support Levels:**")
                for level in sr_data.get('support_levels', []):
                    distance = ((level - current_price) / current_price) * 100
                    st.write(f"🟢 ${level:.2f} ({distance:+.1f}%)")
        
        # Auto-refresh mechanism
        if auto_refresh:
            time.sleep(30)
            st.rerun()
    
    except Exception as e:
        st.error(f"Error loading market data: {str(e)}")

if __name__ == "__main__":
    main()