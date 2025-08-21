import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, environment variables should be set manually
    pass

# Set TensorFlow environment variables to reduce warnings
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

from utils.database import init_database
from utils.binance_client import BinanceClient

# Page configuration
st.set_page_config(
    page_title="AI Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database on startup
init_database()

# Initialize Binance client
@st.cache_resource
def get_binance_client():
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
    
    if not api_key or not api_secret:
        st.error("⚠️ Binance API credentials not found in environment variables.")
        st.info("Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables.")
        return None
    
    try:
        client = BinanceClient(api_key, api_secret, testnet=testnet)
        return client
    except Exception as e:
        st.error(f"Failed to initialize Binance client: {str(e)}")
        return None

# Main page
def main():
    st.title("🤖 AI-Powered Trading Platform")
    st.markdown("---")
    
    # Sidebar for global settings
    with st.sidebar:
        st.header("🔧 Global Settings")
        
        # Trading mode selection
        trading_mode = st.selectbox(
            "Trading Mode",
            ["Paper Trading", "Live Trading"],
            help="Paper trading uses simulated funds for testing"
        )
        
        # Risk management settings
        st.subheader("Risk Management")
        max_position_size = st.slider("Max Position Size (%)", 1, 50, 10)
        stop_loss_pct = st.slider("Stop Loss (%)", 1, 20, 5)
        take_profit_pct = st.slider("Take Profit (%)", 1, 50, 15)
        
        # Store settings in session state
        st.session_state.trading_mode = trading_mode
        st.session_state.max_position_size = max_position_size
        st.session_state.stop_loss_pct = stop_loss_pct
        st.session_state.take_profit_pct = take_profit_pct
        
        st.markdown("---")
        
        # Connection status
        st.subheader("🔗 Connection Status")
        client = get_binance_client()
        if client and client.test_connection():
            st.success("✅ Binance API Connected")
            account_info = client.get_account_info()
            if account_info:
                st.info(f"Account Type: {account_info.get('accountType', 'Unknown')}")
        else:
            st.error("❌ Binance API Disconnected")
    
    # Main content area
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Active Models",
            value="3",
            delta="1 new today"
        )
    
    with col2:
        st.metric(
            label="Portfolio Value",
            value="$10,000",
            delta="2.5%"
        )
    
    with col3:
        st.metric(
            label="Win Rate",
            value="68.5%",
            delta="5.2%"
        )
    
    # Quick overview
    st.subheader("📊 Platform Overview")
    
    tab1, tab2, tab3 = st.tabs(["📈 Recent Signals", "🎯 Model Performance", "📰 Market Status"])
    
    with tab1:
        st.info("Recent trading signals will appear here based on ML model predictions.")
        st.write("Navigate to the Signal Generation page to view and generate new signals.")
    
    with tab2:
        st.info("Model performance metrics and accuracy tracking will be displayed here.")
        st.write("Navigate to the Analytics page for detailed performance analysis.")
    
    with tab3:
        st.info("Current market status and trending assets will be shown here.")
        st.write("Navigate to the Dashboard page for real-time market data and charts.")
    
    # Getting started guide
    st.subheader("🚀 Getting Started")
    
    steps = [
        "1. **Dashboard**: View real-time market data and professional trading charts",
        "2. **Model Training**: Train ML models on historical data with various algorithms",
        "3. **Signal Generation**: Generate trading signals based on trained models",
        "4. **Backtesting**: Test your strategies on historical data",
        "5. **Analytics**: Monitor performance and model accuracy",
        "6. **Model Comparison**: Compare different ML models and strategies"
    ]
    
    for step in steps:
        st.markdown(step)
    
    st.markdown("---")
    st.info("💡 **Tip**: Start with paper trading mode to test your strategies before using real funds.")

if __name__ == "__main__":
    main()
