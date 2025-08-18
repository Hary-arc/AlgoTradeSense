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

def create_tradingview_chart(df, symbol, patterns=None, current_price=None, price_change_pct=None):
    """Create professional TradingView-style candlestick chart matching the reference design."""
    
    # Create subplot with volume at bottom (matching TradingView layout)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.75, 0.25],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Add candlestick chart with TradingView colors
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=symbol,
            increasing_line_color='#089981',  # TradingView green
            decreasing_line_color='#F23645',  # TradingView red
            increasing_fillcolor='#089981',
            decreasing_fillcolor='#F23645',
            line=dict(width=1)
        ),
        row=1, col=1
    )
    
    # Add volume bars with TradingView styling
    volume_colors = []
    for i in range(len(df)):
        if i == 0:
            # First candle - use close vs open
            color = '#089981' if df['close'].iloc[i] >= df['open'].iloc[i] else '#F23645'
        else:
            # Compare close to previous close (TradingView style)
            color = '#089981' if df['close'].iloc[i] >= df['close'].iloc[i-1] else '#F23645'
        volume_colors.append(color)
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            marker_color=volume_colors,
            name='Volume',
            opacity=0.6,
            showlegend=False
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
    
    # Update layout to match TradingView exactly
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#131722',  # TradingView background
        plot_bgcolor='#131722',
        font=dict(color='#D1D4DC', family='Arial', size=12),
        xaxis_rangeslider_visible=False,
        height=600,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        hovermode='x unified'
    )
    
    # Update axes to match TradingView
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(54, 58, 79, 0.5)',
        showline=True,
        linewidth=1,
        linecolor='rgba(54, 58, 79, 0.8)',
        tickfont=dict(color='#868B93', size=10),
        showticklabels=True,
        row=2, col=1
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(54, 58, 79, 0.5)',
        showline=False,
        tickfont=dict(color='#868B93', size=10),
        side='right',
        showticklabels=True,
        row=1, col=1
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(54, 58, 79, 0.5)',
        showline=False,
        tickfont=dict(color='#868B93', size=10),
        side='right',
        showticklabels=True,
        row=2, col=1
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(54, 58, 79, 0.5)',
        showline=False,
        showticklabels=False,
        row=1, col=1
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
        # Note: Binance doesn't support sub-minute intervals like 5s, 10s
        # Available intervals: 1s, 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
        interval = st.selectbox(
            "Timeframe",
            ["1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"],
            index=6,  # Default to 1h
            help="Note: Ultra-short intervals (1s) may have limited historical data"
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
        
        # Calculate market metrics
        current_price = float(df['close'].iloc[-1])
        price_change = float(df['close'].iloc[-1]) - float(df['close'].iloc[-2])
        price_change_pct = (price_change / float(df['close'].iloc[-2])) * 100
        high_24h = float(df['high'].max())
        low_24h = float(df['low'].min())
        volume = float(df['volume'].iloc[-1])
        
        # Create TradingView-style metrics bar
        st.markdown(f"""
        <div style='background-color: #131722; padding: 12px; border-radius: 6px; margin-bottom: 15px; border: 1px solid #363A4E;'>
            <div style='display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 20px;'>
                <div style='display: flex; align-items: center; gap: 15px;'>
                    <span style='color: #868B93; font-size: 12px;'>O</span>
                    <span style='color: #D1D4DC; font-size: 12px; font-weight: 500;'>{df["open"].iloc[-1]:.4f}</span>
                    <span style='color: #868B93; font-size: 12px;'>H</span>
                    <span style='color: #089981; font-size: 12px; font-weight: 500;'>{high_24h:.4f}</span>
                    <span style='color: #868B93; font-size: 12px;'>L</span>
                    <span style='color: #F23645; font-size: 12px; font-weight: 500;'>{low_24h:.4f}</span>
                    <span style='color: #868B93; font-size: 12px;'>C</span>
                    <span style='color: #D1D4DC; font-size: 12px; font-weight: 500;'>{current_price:.4f}</span>
                    <span style='color: #868B93; font-size: 12px;'>Volume</span>
                    <span style='color: #D1D4DC; font-size: 12px; font-weight: 500;'>{volume:,.0f}</span>
                </div>
                <div style='display: flex; align-items: center; gap: 10px;'>
                    <span style='color: #868B93; font-size: 12px;'>Patterns</span>
                    <span style='color: #2962FF; font-size: 12px; font-weight: 500;'>{len([p for p in patterns.keys() if p != "support_resistance"])}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create TradingView-style header
        st.markdown(f"""
        <div style='background-color: #131722; padding: 16px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #363A4E;'>
            <div style='display: flex; align-items: center; justify-content: space-between;'>
                <div style='display: flex; align-items: center; gap: 15px;'>
                    <h2 style='color: #D1D4DC; margin: 0; font-size: 22px; font-weight: 600;'>{symbol}</h2>
                    <span style='background-color: #363A4E; color: #868B93; font-size: 11px; padding: 3px 6px; border-radius: 3px;'>BINANCE</span>
                    <span style='color: {"#089981" if price_change_pct >= 0 else "#F23645"}; font-size: 24px; font-weight: 600; font-family: monospace;'>
                        {current_price:.4f}
                    </span>
                    <span style='color: {"#089981" if price_change_pct >= 0 else "#F23645"}; font-size: 14px; font-weight: 500;'>
                        {"+" if price_change >= 0 else ""}{price_change:.4f} ({price_change_pct:+.2f}%)
                    </span>
                </div>
                <div style='display: flex; align-items: center; gap: 15px;'>
                    <span style='background-color: #2962FF; color: white; font-size: 11px; padding: 2px 8px; border-radius: 12px; font-weight: 500;'>{interval}</span>
                    <span style='color: #089981; font-size: 12px; font-weight: 500;'>● LIVE</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create and display chart
        chart_fig = create_tradingview_chart(df, symbol, patterns, current_price, price_change_pct)
        st.plotly_chart(chart_fig, use_container_width=True, config={'displayModeBar': False})
        
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