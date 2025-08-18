import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from utils.ml_models import LSTMModel, RandomForestModel, SVMModel
from utils.data_processor import DataProcessor
from utils.database import get_historical_data, store_model, get_models
from utils.binance_client import BinanceClient
import os
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

st.set_page_config(page_title="Model Training", page_icon="🧠", layout="wide")

def main():
    st.title("🧠 ML Model Training")
    st.markdown("---")
    
    # Sidebar for model configuration
    with st.sidebar:
        st.header("🔧 Training Configuration")
        
        # Model selection
        model_type = st.selectbox(
            "Select Model Type",
            ["LSTM", "Random Forest", "SVM"],
            help="Choose the machine learning algorithm to train"
        )
        
        # Data configuration
        st.subheader("📊 Data Configuration")
        symbol = st.selectbox(
            "Trading Pair",
            ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT", "BNBUSDT"]
        )
        
        interval = st.selectbox(
            "Timeframe",
            ["1h", "4h", "1d"],
            index=1
        )
        
        lookback_days = st.slider("Training Data (days)", 30, 365, 90)
        
        # Feature selection
        st.subheader("🎯 Features")
        include_technical = st.checkbox("Technical Indicators", value=True)
        include_volume = st.checkbox("Volume Data", value=True)
        include_price_changes = st.checkbox("Price Changes", value=True)
        
        # Model hyperparameters
        st.subheader("⚙️ Hyperparameters")
        
        # Initialize default values
        lstm_units, dropout_rate, epochs, batch_size = 128, 0.2, 50, 32
        n_estimators, max_depth, min_samples_split = 100, 15, 5
        C, kernel, gamma = 1.0, "rbf", "scale"
        
        if model_type == "LSTM":
            lstm_units = st.slider("LSTM Units", 32, 256, 128)
            dropout_rate = st.slider("Dropout Rate", 0.1, 0.5, 0.2)
            epochs = st.slider("Epochs", 10, 200, 50)
            batch_size = st.slider("Batch Size", 16, 128, 32)
            
        elif model_type == "Random Forest":
            n_estimators = st.slider("Number of Trees", 50, 500, 100)
            max_depth = st.slider("Max Depth", 5, 50, 15)
            min_samples_split = st.slider("Min Samples Split", 2, 20, 5)
            
        elif model_type == "SVM":
            C = st.slider("C Parameter", 0.1, 10.0, 1.0)
            kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"])
            gamma = st.selectbox("Gamma", ["scale", "auto"])
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Training Data Preview")
        
        # Data loading and preprocessing
        try:
            # Initialize Binance client
            api_key = os.getenv("BINANCE_API_KEY", "")
            api_secret = os.getenv("BINANCE_API_SECRET", "")
            testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
            
            if api_key and api_secret:
                client = BinanceClient(api_key, api_secret, testnet=testnet)
                
                # Fetch historical data
                end_time = datetime.now()
                start_time = end_time - timedelta(days=lookback_days)
                
                # Calculate limit based on interval
                interval_minutes = {
                    "1h": 60,
                    "4h": 240,
                    "1d": 1440
                }
                
                limit = min(1000, (lookback_days * 1440) // interval_minutes[interval])
                
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
                    df.set_index('timestamp', inplace=True)
                    
                    # Display data info
                    st.info(f"📊 Loaded {len(df)} data points from {start_time.date()} to {end_time.date()}")
                    
                    # Show data preview
                    st.dataframe(df[['open', 'high', 'low', 'close', 'volume']].tail(10))
                    
                    # Data visualization
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='#00D4AA')
                    ))
                    
                    fig.update_layout(
                        title=f"{symbol} Price Data",
                        xaxis_title="Time",
                        yaxis_title="Price",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error("No data available for the selected parameters")
                    df = None
            else:
                st.error("Binance API credentials not found")
                df = None
                
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            df = None
    
    with col2:
        st.subheader("🎯 Training Status")
        
        # Training button
        if st.button("🚀 Start Training", type="primary"):
            if df is not None and len(df) > 50:
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Initialize data processor
                    processor = DataProcessor()
                    
                    # Prepare features
                    status_text.text("Preparing features...")
                    progress_bar.progress(10)
                    
                    features = processor.prepare_features(
                        df,
                        include_technical=include_technical,
                        include_volume=include_volume,
                        include_price_changes=include_price_changes
                    )
                    
                    status_text.text("Splitting data...")
                    progress_bar.progress(20)
                    
                    # Prepare training data
                    X_train, X_test, y_train, y_test, scaler = processor.prepare_training_data(
                        features, 
                        target_col='close',
                        test_size=0.2,
                        sequence_length=60 if model_type == "LSTM" else None
                    )
                    
                    status_text.text(f"Training {model_type} model...")
                    progress_bar.progress(30)
                    
                    # Initialize and train model
                    if model_type == "LSTM":
                        model = LSTMModel(
                            input_shape=(X_train.shape[1], X_train.shape[2]),
                            units=lstm_units,
                            dropout_rate=dropout_rate
                        )
                        
                        # Train with progress updates
                        history = model.train(
                            X_train, y_train,
                            validation_data=(X_test, y_test),
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=0
                        )
                        
                        # Update progress during training
                        for i in range(30, 80, 10):
                            progress_bar.progress(i)
                            time.sleep(0.1)
                    
                    elif model_type == "Random Forest":
                        model = RandomForestModel(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split
                        )
                        model.train(X_train, y_train)
                    
                    elif model_type == "SVM":
                        model = SVMModel(
                            C=C,
                            kernel=kernel,
                            gamma=gamma
                        )
                        model.train(X_train, y_train)
                    
                    progress_bar.progress(80)
                    status_text.text("Evaluating model...")
                    
                    # Make predictions
                    train_pred = model.predict(X_train)
                    test_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    train_mse = mean_squared_error(y_train, train_pred)
                    test_mse = mean_squared_error(y_test, test_pred)
                    train_mae = mean_absolute_error(y_train, train_pred)
                    test_mae = mean_absolute_error(y_test, test_pred)
                    train_r2 = r2_score(y_train, train_pred)
                    test_r2 = r2_score(y_test, test_pred)
                    
                    progress_bar.progress(90)
                    status_text.text("Saving model...")
                    
                    # Save model
                    model_name = f"{model_type}_{symbol}_{interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    model_data = {
                        'name': model_name,
                        'type': model_type,
                        'symbol': symbol,
                        'interval': interval,
                        'created_at': datetime.now(),
                        'train_mse': train_mse,
                        'test_mse': test_mse,
                        'train_mae': train_mae,
                        'test_mae': test_mae,
                        'train_r2': train_r2,
                        'test_r2': test_r2,
                        'features': list(features.columns),
                        'hyperparameters': {
                            'LSTM': {'units': lstm_units, 'dropout': dropout_rate, 'epochs': epochs, 'batch_size': batch_size},
                            'Random Forest': {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_split': min_samples_split},
                            'SVM': {'C': C, 'kernel': kernel, 'gamma': gamma}
                        }[model_type]
                    }
                    
                    # Store model in database
                    store_model(model_name, model, scaler, model_data)
                    
                    progress_bar.progress(100)
                    status_text.text("Training completed!")
                    
                    # Display results
                    st.success("✅ Model training completed!")
                    
                    st.metric("Test R² Score", f"{test_r2:.4f}")
                    st.metric("Test MSE", f"{test_mse:.6f}")
                    st.metric("Test MAE", f"{test_mae:.6f}")
                    
                    # Plot predictions vs actual
                    st.subheader("📊 Model Performance")
                    
                    fig = go.Figure()
                    
                    # Plot actual vs predicted for test set
                    test_indices = range(len(y_test))
                    
                    fig.add_trace(go.Scatter(
                        x=test_indices,
                        y=y_test.flatten() if hasattr(y_test, 'flatten') else y_test,
                        mode='lines',
                        name='Actual',
                        line=dict(color='blue')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=test_indices,
                        y=test_pred.flatten() if hasattr(test_pred, 'flatten') else test_pred,
                        mode='lines',
                        name='Predicted',
                        line=dict(color='red')
                    ))
                    
                    fig.update_layout(
                        title="Actual vs Predicted Prices",
                        xaxis_title="Time Steps",
                        yaxis_title="Price",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
                    progress_bar.progress(0)
                    status_text.text("Training failed")
            
            else:
                st.error("Insufficient data for training. Please load more data.")
        
        # Display existing models
        st.subheader("📚 Trained Models")
        
        try:
            models = get_models()
            if models:
                for model_info in models[-5:]:  # Show last 5 models
                    with st.expander(f"{model_info['name']}"):
                        st.write(f"**Type**: {model_info['type']}")
                        st.write(f"**Symbol**: {model_info['symbol']}")
                        st.write(f"**Test R²**: {model_info.get('test_r2', 'N/A'):.4f}")
                        st.write(f"**Created**: {model_info['created_at']}")
            else:
                st.info("No trained models found. Train your first model!")
                
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")

if __name__ == "__main__":
    main()
