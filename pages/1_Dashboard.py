import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
from utils.binance_client import BinanceClient
from utils.chart_utils import create_candlestick_chart, add_technical_indicators
from utils.database import get_historical_data, store_market_data
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

def main():
    st.title("📈 Real-Time Trading Dashboard")
    st.markdown("---")
    
    client = get_binance_client()
    if not client:
        st.error("⚠️ Unable to connect to Binance API. Please check your credentials.")
        return
    
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
    
    # Create placeholder for real-time updates
    chart_placeholder = st.empty()
    metrics_placeholder = st.empty()
    
    # Auto-refresh mechanism
    if auto_refresh:
        refresh_rate = 30  # seconds
        while True:
            try:
                # Fetch real-time data
                klines = client.get_klines(symbol, interval, limit)
                
                if klines:
                    # Convert to DataFrame
                    df = pd.DataFrame(klines, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    
                    # Convert types
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col])
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # Store data in database
                    store_market_data(symbol, df)
                    
                    # Update chart
                    with chart_placeholder.container():
                        fig = create_candlestick_chart(df, symbol)
                        fig = add_technical_indicators(fig, df)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Update metrics
                    with metrics_placeholder.container():
                        current_price = df['close'].iloc[-1]
                        price_change = current_price - df['close'].iloc[-2]
                        price_change_pct = (price_change / df['close'].iloc[-2]) * 100
                        volume_24h = df['volume'].sum()
                        high_24h = df['high'].max()
                        low_24h = df['low'].min()
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            st.metric(
                                label="Current Price",
                                value=f"${current_price:.4f}",
                                delta=f"{price_change_pct:.2f}%"
                            )
                        
                        with col2:
                            st.metric(
                                label="24h High",
                                value=f"${high_24h:.4f}"
                            )
                        
                        with col3:
                            st.metric(
                                label="24h Low",
                                value=f"${low_24h:.4f}"
                            )
                        
                        with col4:
                            st.metric(
                                label="24h Volume",
                                value=f"{volume_24h:.0f}"
                            )
                        
                        with col5:
                            st.metric(
                                label="Last Update",
                                value=datetime.now().strftime("%H:%M:%S")
                            )
                
                # Wait before next update
                if auto_refresh:
                    time.sleep(refresh_rate)
                else:
                    break
                    
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                break
    
    else:
        # Static data fetch
        try:
            klines = client.get_klines(symbol, interval, limit)
            
            if klines:
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Display chart
                fig = create_candlestick_chart(df, symbol)
                fig = add_technical_indicators(fig, df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display metrics
                current_price = df['close'].iloc[-1]
                price_change = current_price - df['close'].iloc[-2]
                price_change_pct = (price_change / df['close'].iloc[-2]) * 100
                volume_24h = df['volume'].sum()
                high_24h = df['high'].max()
                low_24h = df['low'].min()
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric(
                        label="Current Price",
                        value=f"${current_price:.4f}",
                        delta=f"{price_change_pct:.2f}%"
                    )
                
                with col2:
                    st.metric(
                        label="24h High",
                        value=f"${high_24h:.4f}"
                    )
                
                with col3:
                    st.metric(
                        label="24h Low",
                        value=f"${low_24h:.4f}"
                    )
                
                with col4:
                    st.metric(
                        label="24h Volume",
                        value=f"{volume_24h:.0f}"
                    )
                
                with col5:
                    st.metric(
                        label="Last Update",
                        value=datetime.now().strftime("%H:%M:%S")
                    )
                
                # Additional market information
                st.markdown("---")
                st.subheader("📊 Market Information")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 Price Analysis")
                    
                    # Support and resistance levels
                    recent_highs = df['high'].rolling(window=20).max()
                    recent_lows = df['low'].rolling(window=20).min()
                    
                    st.write(f"**Resistance Level**: ${recent_highs.iloc[-1]:.4f}")
                    st.write(f"**Support Level**: ${recent_lows.iloc[-1]:.4f}")
                    
                    # Moving averages
                    ma_20 = df['close'].rolling(window=20).mean().iloc[-1]
                    ma_50 = df['close'].rolling(window=min(50, len(df))).mean().iloc[-1]
                    
                    st.write(f"**20-period MA**: ${ma_20:.4f}")
                    st.write(f"**50-period MA**: ${ma_50:.4f}")
                
                with col2:
                    st.subheader("💹 Technical Indicators")
                    
                    # RSI calculation
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    
                    current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
                    
                    st.write(f"**RSI (14)**: {current_rsi:.2f}")
                    
                    if current_rsi > 70:
                        st.write("🔴 Overbought condition")
                    elif current_rsi < 30:
                        st.write("🟢 Oversold condition")
                    else:
                        st.write("🟡 Neutral condition")
                    
                    # Volume analysis
                    avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
                    current_volume = df['volume'].iloc[-1]
                    volume_ratio = current_volume / avg_volume
                    
                    st.write(f"**Volume Ratio**: {volume_ratio:.2f}x")
                    
                    if volume_ratio > 1.5:
                        st.write("📈 High volume activity")
                    elif volume_ratio < 0.5:
                        st.write("📉 Low volume activity")
                    else:
                        st.write("➡️ Normal volume activity")
                
        except Exception as e:
            st.error(f"Error fetching market data: {str(e)}")

if __name__ == "__main__":
    main()
