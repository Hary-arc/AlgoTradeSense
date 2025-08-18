import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from utils.signal_generator import SignalGenerator
from utils.database import get_models, get_historical_data, store_signal
from utils.binance_client import BinanceClient
from utils.data_processor import DataProcessor
import os

st.set_page_config(page_title="Signal Generation", page_icon="🎯", layout="wide")

def main():
    st.title("🎯 AI Signal Generation")
    st.markdown("---")
    
    # Sidebar for signal configuration
    with st.sidebar:
        st.header("⚙️ Signal Configuration")
        
        # Model selection
        try:
            models = get_models()
            if models:
                model_names = [model['name'] for model in models]
                selected_model = st.selectbox(
                    "Select Model",
                    model_names,
                    help="Choose a trained model for signal generation"
                )
                
                # Find selected model info
                model_info = next((m for m in models if m['name'] == selected_model), None)
                
                if model_info:
                    st.info(f"**Type**: {model_info['type']}")
                    st.info(f"**Symbol**: {model_info['symbol']}")
                    st.info(f"**Test R²**: {model_info.get('test_r2', 'N/A'):.4f}")
            else:
                st.error("No trained models found. Please train a model first.")
                return
                
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return
        
        # Signal parameters
        st.subheader("📊 Signal Parameters")
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            0.5, 0.95, 0.7,
            help="Minimum confidence required for signal generation"
        )
        
        prediction_horizon = st.selectbox(
            "Prediction Horizon",
            ["1 hour", "4 hours", "1 day", "3 days"],
            index=1
        )
        
        signal_strength = st.selectbox(
            "Signal Strength Filter",
            ["All Signals", "Strong Only", "Medium+", "Weak Only"],
            index=1
        )
        
        # Risk management
        st.subheader("⚠️ Risk Management")
        
        position_size = st.slider(
            "Position Size (%)",
            1, 50, 10,
            help="Percentage of portfolio to allocate"
        )
        
        stop_loss = st.slider(
            "Stop Loss (%)",
            1, 20, 5,
            help="Maximum loss tolerance"
        )
        
        take_profit = st.slider(
            "Take Profit (%)",
            1, 50, 15,
            help="Target profit level"
        )
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🎯 Live Signals", "📈 Signal History", "⚙️ Signal Analysis"])
    
    with tab1:
        st.subheader("🎯 Real-Time Signal Generation")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Generate signal button
            if st.button("🚀 Generate Signal", type="primary"):
                try:
                    # Initialize components
                    api_key = os.getenv("BINANCE_API_KEY", "")
                    api_secret = os.getenv("BINANCE_API_SECRET", "")
                    testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
                    
                    if not api_key or not api_secret:
                        st.error("Binance API credentials not found")
                        return
                    
                    client = BinanceClient(api_key, api_secret, testnet=testnet)
                    signal_gen = SignalGenerator()
                    processor = DataProcessor()
                    
                    # Get recent data for the model's symbol
                    symbol = model_info['symbol']
                    interval = model_info['interval']
                    
                    # Fetch recent data
                    klines = client.get_klines(symbol, interval, 100)
                    
                    if klines:
                        df = pd.DataFrame(klines, columns=[
                            'timestamp', 'open', 'high', 'low', 'close', 'volume',
                            'close_time', 'quote_asset_volume', 'number_of_trades',
                            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                        ])
                        
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            df[col] = pd.to_numeric(df[col])
                        
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df.set_index('timestamp', inplace=True)
                        
                        # Generate signal
                        with st.spinner("Generating signal..."):
                            signal = signal_gen.generate_signal(
                                model_name=selected_model,
                                data=df,
                                confidence_threshold=confidence_threshold
                            )
                        
                        if signal:
                            # Display signal
                            signal_type = signal['signal_type']
                            confidence = signal['confidence']
                            predicted_price = signal['predicted_price']
                            current_price = signal['current_price']
                            
                            # Color coding for signal type
                            if signal_type == 'BUY':
                                signal_color = '🟢'
                                signal_emoji = '📈'
                            elif signal_type == 'SELL':
                                signal_color = '🔴'
                                signal_emoji = '📉'
                            else:
                                signal_color = '🟡'
                                signal_emoji = '➡️'
                            
                            st.success(f"✅ Signal Generated Successfully!")
                            
                            # Signal display box
                            signal_container = st.container()
                            with signal_container:
                                st.markdown(f"""
                                <div style="border: 2px solid {'#00ff00' if signal_type == 'BUY' else '#ff0000' if signal_type == 'SELL' else '#ffff00'}; 
                                           border-radius: 10px; padding: 20px; margin: 10px 0;">
                                    <h3>{signal_color} {signal_type} SIGNAL {signal_emoji}</h3>
                                    <p><strong>Symbol:</strong> {symbol}</p>
                                    <p><strong>Confidence:</strong> {confidence:.2%}</p>
                                    <p><strong>Current Price:</strong> ${current_price:.6f}</p>
                                    <p><strong>Predicted Price:</strong> ${predicted_price:.6f}</p>
                                    <p><strong>Expected Change:</strong> {((predicted_price - current_price) / current_price * 100):+.2f}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Risk management recommendations
                            if signal_type in ['BUY', 'SELL']:
                                st.subheader("💼 Position Recommendations")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric(
                                        "Recommended Position",
                                        f"{position_size}% of portfolio"
                                    )
                                
                                with col2:
                                    if signal_type == 'BUY':
                                        stop_price = current_price * (1 - stop_loss/100)
                                        take_profit_price = current_price * (1 + take_profit/100)
                                    else:
                                        stop_price = current_price * (1 + stop_loss/100)
                                        take_profit_price = current_price * (1 - take_profit/100)
                                    
                                    st.metric(
                                        "Stop Loss",
                                        f"${stop_price:.6f}"
                                    )
                                
                                with col3:
                                    st.metric(
                                        "Take Profit",
                                        f"${take_profit_price:.6f}"
                                    )
                                
                                # Store signal in database
                                signal_data = {
                                    'model_name': selected_model,
                                    'symbol': symbol,
                                    'signal_type': signal_type,
                                    'confidence': confidence,
                                    'current_price': current_price,
                                    'predicted_price': predicted_price,
                                    'position_size': position_size,
                                    'stop_loss': stop_price,
                                    'take_profit': take_profit_price,
                                    'timestamp': datetime.now()
                                }
                                
                                try:
                                    store_signal(signal_data)
                                    st.info("📝 Signal stored in history")
                                except Exception as e:
                                    st.warning(f"Could not store signal: {str(e)}")
                        else:
                            st.warning("⚠️ No signal generated. Market conditions may not meet criteria.")
                    
                    else:
                        st.error("Could not fetch market data")
                
                except Exception as e:
                    st.error(f"Error generating signal: {str(e)}")
        
        with col2:
            st.subheader("📊 Model Info")
            
            if model_info:
                st.write(f"**Model Type**: {model_info['type']}")
                st.write(f"**Training Symbol**: {model_info['symbol']}")
                st.write(f"**Timeframe**: {model_info['interval']}")
                st.write(f"**Test Accuracy**: {model_info.get('test_r2', 0):.2%}")
                st.write(f"**Created**: {model_info['created_at']}")
                
                # Model performance indicator
                accuracy = model_info.get('test_r2', 0)
                if accuracy > 0.8:
                    st.success("🟢 High Accuracy Model")
                elif accuracy > 0.6:
                    st.warning("🟡 Medium Accuracy Model")
                else:
                    st.error("🔴 Low Accuracy Model")
                
                # Feature importance (if available)
                if 'features' in model_info:
                    st.subheader("🎯 Model Features")
                    for feature in model_info['features'][:5]:  # Show top 5 features
                        st.write(f"• {feature}")
    
    with tab2:
        st.subheader("📈 Signal History")
        
        # Display recent signals
        try:
            # This would get signals from database
            st.info("Signal history will be displayed here once signals are generated.")
            
            # Placeholder for signal history table
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write("Recent signals will appear in this table:")
                
                # Example structure for when signals exist
                sample_data = pd.DataFrame({
                    'Timestamp': ['2024-01-01 10:00:00'],
                    'Symbol': ['BTCUSDT'],
                    'Signal': ['BUY'],
                    'Confidence': ['75%'],
                    'Entry Price': ['$45,000'],
                    'Status': ['Pending']
                })
                
                st.dataframe(sample_data, use_container_width=True)
            
            with col2:
                st.subheader("📊 Signal Stats")
                st.metric("Total Signals", "0")
                st.metric("Win Rate", "0%")
                st.metric("Avg Confidence", "0%")
                
        except Exception as e:
            st.error(f"Error loading signal history: {str(e)}")
    
    with tab3:
        st.subheader("⚙️ Signal Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Signal Distribution")
            st.info("Signal type distribution charts will be displayed here.")
            
            # Placeholder for signal distribution
            sample_distribution = {
                'BUY': 45,
                'SELL': 35,
                'HOLD': 20
            }
            
            fig = go.Figure(data=[go.Pie(
                labels=list(sample_distribution.keys()),
                values=list(sample_distribution.values()),
                hole=0.3
            )])
            
            fig.update_layout(
                title="Signal Type Distribution (Sample)",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Confidence Trends")
            st.info("Signal confidence trends over time will be shown here.")
            
            # Placeholder for confidence trends
            dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
            confidence_trend = np.random.uniform(0.6, 0.9, 30)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=confidence_trend,
                mode='lines+markers',
                name='Average Confidence',
                line=dict(color='#00D4AA')
            ))
            
            fig.update_layout(
                title="Signal Confidence Trend (Sample)",
                xaxis_title="Date",
                yaxis_title="Confidence",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Signal performance metrics
        st.subheader("📊 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Signal Accuracy", "0%", "N/A")
        
        with col2:
            st.metric("Avg Hold Time", "0 hours", "N/A")
        
        with col3:
            st.metric("Best Signal", "+0%", "N/A")
        
        with col4:
            st.metric("Worst Signal", "0%", "N/A")

if __name__ == "__main__":
    main()
