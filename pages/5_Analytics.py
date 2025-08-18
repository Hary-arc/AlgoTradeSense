import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from utils.database import get_models, get_model_performance_history, get_trading_history
import sqlite3

st.set_page_config(page_title="Analytics Dashboard", page_icon="📊", layout="wide")

def main():
    st.title("📊 Analytics Dashboard")
    st.markdown("---")
    
    # Sidebar for analytics filters
    with st.sidebar:
        st.header("🔍 Analytics Filters")
        
        # Time period selection
        time_period = st.selectbox(
            "Time Period",
            ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Last Year", "All Time"],
            index=2
        )
        
        # Model filter
        try:
            models = get_models()
            if models:
                model_names = ["All Models"] + [model['name'] for model in models]
                selected_models = st.multiselect(
                    "Select Models",
                    model_names,
                    default=["All Models"]
                )
            else:
                st.warning("No models found")
                selected_models = []
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            selected_models = []
        
        # Metric selection
        st.subheader("📈 Metrics to Display")
        show_accuracy = st.checkbox("Model Accuracy", value=True)
        show_trading = st.checkbox("Trading Performance", value=True)
        show_risk = st.checkbox("Risk Metrics", value=True)
        show_signals = st.checkbox("Signal Analysis", value=True)
    
    # Main dashboard
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Model Performance", "💹 Trading Analytics", "⚠️ Risk Analysis"])
    
    with tab1:
        st.subheader("📊 Platform Overview")
        
        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            try:
                total_models = len(get_models()) if get_models() else 0
                st.metric("Total Models", total_models, "Active")
            except:
                st.metric("Total Models", "0", "N/A")
        
        with col2:
            st.metric("Portfolio Value", "$10,000", "+2.5%")
        
        with col3:
            st.metric("Total Trades", "45", "+5 today")
        
        with col4:
            st.metric("Win Rate", "68.5%", "+3.2%")
        
        with col5:
            st.metric("Active Signals", "3", "2 pending")
        
        st.markdown("---")
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Portfolio Performance")
            
            # Sample portfolio performance data
            dates = pd.date_range(start='2024-01-01', periods=90, freq='D')
            portfolio_values = 10000 * (1 + np.cumsum(np.random.normal(0.001, 0.02, 90)))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=portfolio_values,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#00D4AA', width=2),
                fill='tonexty'
            ))
            
            fig.update_layout(
                title="Portfolio Value Over Time",
                xaxis_title="Date",
                yaxis_title="Value ($)",
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Model Accuracy Trends")
            
            # Sample model accuracy data
            model_names = ["LSTM_BTC", "RF_ETH", "SVM_ADA"]
            accuracies = [0.75, 0.68, 0.72]
            
            fig = go.Figure(data=[go.Bar(
                x=model_names,
                y=accuracies,
                marker_color=['#00D4AA', '#FF6B6B', '#4ECDC4']
            )])
            
            fig.update_layout(
                title="Current Model Accuracies",
                xaxis_title="Models",
                yaxis_title="Accuracy",
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent activity
        st.subheader("🕒 Recent Activity")
        
        # Sample recent activity data
        activity_data = {
            'Time': ['10:30 AM', '09:45 AM', '09:15 AM', '08:30 AM'],
            'Activity': [
                'New BUY signal generated for BTCUSDT',
                'Model LSTM_BTC_001 completed training',
                'Trade executed: SELL ETHUSDT at $3,245.67',
                'Backtest completed for RF_ETH_002'
            ],
            'Type': ['Signal', 'Training', 'Trade', 'Backtest'],
            'Status': ['Active', 'Completed', 'Filled', 'Completed']
        }
        
        activity_df = pd.DataFrame(activity_data)
        st.dataframe(activity_df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("🎯 Model Performance Analysis")
        
        if show_accuracy:
            # Model comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Model Comparison")
                
                # Sample model performance data
                model_data = {
                    'Model': ['LSTM_BTC_001', 'RF_ETH_002', 'SVM_ADA_003', 'LSTM_DOT_004'],
                    'Type': ['LSTM', 'Random Forest', 'SVM', 'LSTM'],
                    'Symbol': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT'],
                    'Accuracy': [0.847, 0.723, 0.689, 0.756],
                    'Precision': [0.821, 0.698, 0.654, 0.732],
                    'Recall': [0.789, 0.756, 0.723, 0.778],
                    'F1-Score': [0.805, 0.726, 0.687, 0.755]
                }
                
                model_df = pd.DataFrame(model_data)
                st.dataframe(model_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.subheader("📈 Accuracy Over Time")
                
                # Sample accuracy trend data
                dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
                lstm_acc = 0.7 + 0.1 * np.sin(np.linspace(0, 4*np.pi, 30)) + np.random.normal(0, 0.02, 30)
                rf_acc = 0.65 + 0.08 * np.sin(np.linspace(0, 3*np.pi, 30)) + np.random.normal(0, 0.015, 30)
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=lstm_acc,
                    mode='lines+markers',
                    name='LSTM Models',
                    line=dict(color='#00D4AA')
                ))
                
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=rf_acc,
                    mode='lines+markers',
                    name='Random Forest',
                    line=dict(color='#FF6B6B')
                ))
                
                fig.update_layout(
                    title="Model Accuracy Trends",
                    xaxis_title="Date",
                    yaxis_title="Accuracy",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance analysis
            st.subheader("🎯 Feature Importance Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Feature importance for selected model
                features = ['RSI', 'MACD', 'SMA_20', 'Volume', 'Price_Change', 'Bollinger_Bands', 'ATR', 'OBV']
                importance = [0.18, 0.15, 0.14, 0.12, 0.11, 0.10, 0.08, 0.07]
                
                fig = go.Figure(data=[go.Bar(
                    x=importance,
                    y=features,
                    orientation='h',
                    marker_color='#00D4AA'
                )])
                
                fig.update_layout(
                    title="Feature Importance (LSTM_BTC_001)",
                    xaxis_title="Importance Score",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Model prediction accuracy by market condition
                conditions = ['Trending Up', 'Trending Down', 'Sideways', 'High Volatility']
                accuracies = [0.82, 0.75, 0.68, 0.71]
                
                fig = go.Figure(data=[go.Bar(
                    x=conditions,
                    y=accuracies,
                    marker_color=['#00D4AA', '#FF6B6B', '#FFE66D', '#4ECDC4']
                )])
                
                fig.update_layout(
                    title="Accuracy by Market Condition",
                    xaxis_title="Market Condition",
                    yaxis_title="Accuracy",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("💹 Trading Performance Analytics")
        
        if show_trading:
            # Trading metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total P&L", "$1,247.32", "+12.47%")
            
            with col2:
                st.metric("Avg Trade Return", "2.8%", "+0.3%")
            
            with col3:
                st.metric("Best Trade", "$234.56", "BTCUSDT")
            
            with col4:
                st.metric("Worst Trade", "-$89.23", "ADAUSDT")
            
            # Trading performance charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 P&L Distribution")
                
                # Sample P&L data
                pnl_data = np.random.normal(25, 50, 100)  # Average $25 profit, $50 std dev
                
                fig = go.Figure(data=[go.Histogram(
                    x=pnl_data,
                    nbinsx=20,
                    marker_color='#00D4AA'
                )])
                
                fig.update_layout(
                    title="Trade P&L Distribution",
                    xaxis_title="P&L ($)",
                    yaxis_title="Frequency",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📈 Win Rate by Symbol")
                
                symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
                win_rates = [0.72, 0.68, 0.65, 0.71, 0.63]
                
                fig = go.Figure(data=[go.Bar(
                    x=symbols,
                    y=win_rates,
                    marker_color='#00D4AA'
                )])
                
                fig.update_layout(
                    title="Win Rate by Trading Pair",
                    xaxis_title="Symbol",
                    yaxis_title="Win Rate",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Trading timeline
            st.subheader("⏰ Trading Timeline")
            
            # Sample trading data
            trade_times = pd.date_range(start='2024-01-01', periods=20, freq='2D')
            trade_pnl = np.random.normal(25, 40, 20)
            trade_symbols = np.random.choice(['BTCUSDT', 'ETHUSDT', 'ADAUSDT'], 20)
            
            fig = go.Figure()
            
            colors = ['green' if pnl > 0 else 'red' for pnl in trade_pnl]
            
            fig.add_trace(go.Scatter(
                x=trade_times,
                y=trade_pnl,
                mode='markers',
                marker=dict(
                    color=colors,
                    size=10,
                    line=dict(width=1, color='white')
                ),
                text=trade_symbols,
                name='Trades'
            ))
            
            fig.update_layout(
                title="Trade P&L Timeline",
                xaxis_title="Date",
                yaxis_title="P&L ($)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("⚠️ Risk Analysis")
        
        if show_risk:
            # Risk metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Portfolio VaR (95%)", "$892.15", "Daily")
            
            with col2:
                st.metric("Max Drawdown", "8.7%", "-1.2%")
            
            with col3:
                st.metric("Sharpe Ratio", "1.85", "+0.15")
            
            with col4:
                st.metric("Beta", "0.72", "vs BTC")
            
            # Risk analysis charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📉 Drawdown Analysis")
                
                # Sample drawdown data
                dates = pd.date_range(start='2024-01-01', periods=90, freq='D')
                cumulative_returns = np.cumsum(np.random.normal(0.001, 0.02, 90))
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdown = (cumulative_returns - running_max) * 100
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=drawdown,
                    fill='tonexty',
                    mode='lines',
                    name='Drawdown',
                    line=dict(color='red'),
                    fillcolor='rgba(255, 0, 0, 0.3)'
                ))
                
                fig.update_layout(
                    title="Portfolio Drawdown",
                    xaxis_title="Date",
                    yaxis_title="Drawdown (%)",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📊 Risk Distribution")
                
                # Risk by asset
                assets = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'Others']
                risk_contribution = [35, 28, 15, 12, 10]
                
                fig = go.Figure(data=[go.Pie(
                    labels=assets,
                    values=risk_contribution,
                    hole=0.3
                )])
                
                fig.update_layout(
                    title="Risk Contribution by Asset",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Risk-Return Analysis
            st.subheader("📈 Risk-Return Analysis")
            
            # Sample data for different strategies/models
            strategies = ['LSTM_BTC', 'RF_ETH', 'SVM_ADA', 'LSTM_DOT', 'Portfolio']
            returns = [0.15, 0.12, 0.08, 0.10, 0.11]
            volatilities = [0.25, 0.20, 0.15, 0.18, 0.19]
            colors = ['#00D4AA', '#FF6B6B', '#4ECDC4', '#FFE66D', '#FF8C94']
            
            fig = go.Figure()
            
            for i, strategy in enumerate(strategies):
                fig.add_trace(go.Scatter(
                    x=[volatilities[i]],
                    y=[returns[i]],
                    mode='markers',
                    marker=dict(
                        color=colors[i],
                        size=15,
                        line=dict(width=2, color='white')
                    ),
                    name=strategy,
                    text=strategy
                ))
            
            fig.update_layout(
                title="Risk-Return Profile",
                xaxis_title="Volatility (Risk)",
                yaxis_title="Return",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk alerts
        st.subheader("🚨 Risk Alerts")
        
        # Sample risk alerts
        alerts = [
            {"type": "Warning", "message": "Portfolio concentration above 30% in BTCUSDT", "severity": "Medium"},
            {"type": "Info", "message": "Volatility increased 15% from last week", "severity": "Low"},
            {"type": "Critical", "message": "Model accuracy dropped below 60% for SVM_ADA_003", "severity": "High"}
        ]
        
        for alert in alerts:
            severity_color = {
                "Low": "info",
                "Medium": "warning", 
                "High": "error"
            }
            
            getattr(st, severity_color[alert["severity"]])(f"**{alert['type']}**: {alert['message']}")

if __name__ == "__main__":
    main()
