import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from utils.backtesting_engine import BacktestingEngine
from utils.database import get_models, get_historical_data
from utils.binance_client import BinanceClient
from utils.data_processor import DataProcessor
import os

st.set_page_config(page_title="Backtesting", page_icon="🔄", layout="wide")

def main():
    st.title("🔄 Strategy Backtesting")
    st.markdown("---")
    
    # Sidebar for backtesting configuration
    with st.sidebar:
        st.header("⚙️ Backtest Configuration")
        
        # Model selection
        try:
            models = get_models()
            if models:
                model_names = [model['name'] for model in models]
                selected_model = st.selectbox(
                    "Select Model",
                    model_names,
                    help="Choose a trained model for backtesting"
                )
                
                model_info = next((m for m in models if m['name'] == selected_model), None)
                
                if model_info:
                    st.info(f"**Type**: {model_info['type']}")
                    st.info(f"**Symbol**: {model_info['symbol']}")
                    st.info(f"**Accuracy**: {model_info.get('test_r2', 0):.2%}")
            else:
                st.error("No trained models found. Please train a model first.")
                return
                
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return
        
        # Backtesting period
        st.subheader("📅 Backtesting Period")
        
        end_date = st.date_input(
            "End Date",
            value=datetime.now().date(),
            max_value=datetime.now().date()
        )
        
        backtest_days = st.slider(
            "Backtest Period (days)",
            7, 365, 90,
            help="Number of days to backtest"
        )
        
        start_date = end_date - timedelta(days=backtest_days)
        st.write(f"Start Date: {start_date}")
        
        # Trading parameters
        st.subheader("💰 Trading Parameters")
        
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=100,
            max_value=1000000,
            value=10000,
            step=100
        )
        
        position_size = st.slider(
            "Position Size (%)",
            1, 100, 10,
            help="Percentage of capital per trade"
        )
        
        commission = st.slider(
            "Commission (%)",
            0.0, 1.0, 0.1,
            step=0.01,
            help="Trading commission percentage"
        )
        
        # Risk management
        st.subheader("⚠️ Risk Management")
        
        stop_loss = st.slider(
            "Stop Loss (%)",
            1, 20, 5,
            help="Maximum loss per trade"
        )
        
        take_profit = st.slider(
            "Take Profit (%)",
            1, 50, 15,
            help="Target profit per trade"
        )
        
        confidence_threshold = st.slider(
            "Min Confidence",
            0.5, 0.95, 0.7,
            help="Minimum signal confidence"
        )
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🚀 Run Backtest", "📊 Results Analysis", "📈 Performance Metrics"])
    
    with tab1:
        st.subheader("🚀 Backtest Execution")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Backtest parameters summary
            st.subheader("📋 Backtest Summary")
            
            if models and model_info:
                summary_data = {
                    'Parameter': [
                        'Model', 'Symbol', 'Period', 'Initial Capital',
                        'Position Size', 'Commission', 'Stop Loss', 'Take Profit'
                    ],
                    'Value': [
                        selected_model,
                        model_info.get('symbol', 'N/A'),
                        f"{start_date} to {end_date}",
                        f"${initial_capital:,}",
                        f"{position_size}%",
                        f"{commission}%",
                        f"{stop_loss}%",
                        f"{take_profit}%"
                    ]
                }
            
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
            else:
                st.error("No model selected for backtesting")
            
            # Run backtest button
            if st.button("🔄 Run Backtest", type="primary"):
                try:
                    # Initialize components
                    api_key = os.getenv("BINANCE_API_KEY", "")
                    api_secret = os.getenv("BINANCE_API_SECRET", "")
                    testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
                    
                    if not api_key or not api_secret:
                        st.error("Binance API credentials not found")
                        return
                    
                    client = BinanceClient(api_key, api_secret, testnet=testnet)
                    backtest_engine = BacktestingEngine()
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Fetch historical data
                    status_text.text("Fetching historical data...")
                    progress_bar.progress(10)
                    
                    symbol = model_info.get('symbol', 'BTCUSDT')
                    interval = model_info.get('interval', '1h')
                    
                    # Calculate data points needed
                    interval_minutes = {
                        "1h": 60,
                        "4h": 240,
                        "1d": 1440
                    }
                    
                    limit = min(1000, (backtest_days * 1440) // interval_minutes[interval])
                    klines = client.get_klines(symbol, interval, limit)
                    
                    if not klines:
                        st.error("Could not fetch historical data")
                        return
                    
                    # Convert klines to DataFrame
                    df_data = []
                    for kline in klines:
                        df_data.append({
                            'timestamp': kline[0],
                            'open': float(kline[1]),
                            'high': float(kline[2]),
                            'low': float(kline[3]),
                            'close': float(kline[4]),
                            'volume': float(kline[5])
                        })
                    
                    df = pd.DataFrame(df_data)
                    
                    # Data is already numeric from above conversion
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    # Filter data to backtest period
                    start_datetime = pd.to_datetime(start_date)
                    end_datetime = pd.to_datetime(end_date)
                    df = df[(df.index >= start_datetime) & (df.index <= end_datetime)]
                    
                    if len(df) < 10:
                        st.error("Insufficient data for the selected period")
                        return
                    
                    status_text.text("Running backtest...")
                    progress_bar.progress(30)
                    
                    # Configure backtest parameters
                    backtest_config = {
                        'initial_capital': initial_capital,
                        'position_size_pct': position_size / 100,
                        'commission_pct': commission / 100,
                        'stop_loss_pct': stop_loss / 100,
                        'take_profit_pct': take_profit / 100,
                        'confidence_threshold': confidence_threshold
                    }
                    
                    # Run backtest
                    if models and model_info and selected_model:
                        results = backtest_engine.run_backtest(
                            model_name=selected_model,
                            data=df,
                            config=backtest_config
                        )
                    else:
                        st.error("Missing model information for backtesting")
                        return
                    
                    progress_bar.progress(80)
                    status_text.text("Analyzing results...")
                    
                    if results:
                        # Store results in session state
                        st.session_state.backtest_results = results
                        
                        progress_bar.progress(100)
                        status_text.text("Backtest completed!")
                        
                        st.success("✅ Backtest completed successfully!")
                        
                        # Display key metrics
                        st.subheader("📊 Key Results")
                        
                        final_value = results['portfolio_values'][-1]
                        total_return = (final_value - initial_capital) / initial_capital * 100
                        total_trades = len(results['trades'])
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Final Portfolio Value",
                                f"${final_value:,.2f}",
                                f"{total_return:+.2f}%"
                            )
                        
                        with col2:
                            st.metric(
                                "Total Trades",
                                total_trades
                            )
                        
                        with col3:
                            winning_trades = sum(1 for trade in results['trades'] if trade['pnl'] > 0)
                            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                            st.metric(
                                "Win Rate",
                                f"{win_rate:.1f}%"
                            )
                        
                        with col4:
                            max_drawdown = results.get('max_drawdown', 0) * 100
                            st.metric(
                                "Max Drawdown",
                                f"{max_drawdown:.2f}%"
                            )
                        
                        # Portfolio value chart
                        st.subheader("📈 Portfolio Performance")
                        
                        fig = go.Figure()
                        
                        portfolio_dates = pd.date_range(
                            start=start_datetime,
                            periods=len(results['portfolio_values']),
                            freq='H' if interval == '1h' else 'D'
                        )
                        
                        fig.add_trace(go.Scatter(
                            x=portfolio_dates,
                            y=results['portfolio_values'],
                            mode='lines',
                            name='Portfolio Value',
                            line=dict(color='#00D4AA', width=2)
                        ))
                        
                        # Add buy/sell markers
                        for trade in results['trades']:
                            color = 'green' if trade['type'] == 'BUY' else 'red'
                            portfolio_value_at_trade = trade.get('portfolio_value_before', initial_capital)
                            fig.add_trace(go.Scatter(
                                x=[trade['timestamp']],
                                y=[portfolio_value_at_trade],
                                mode='markers',
                                marker=dict(
                                    color=color,
                                    size=10,
                                    symbol='triangle-up' if trade['type'] == 'BUY' else 'triangle-down'
                                ),
                                name=f"{trade['type']} Signal",
                                showlegend=False
                            ))
                        
                        fig.update_layout(
                            title="Portfolio Value Over Time",
                            xaxis_title="Date",
                            yaxis_title="Portfolio Value ($)",
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    else:
                        st.error("Backtest failed to produce results")
                
                except Exception as e:
                    st.error(f"Backtest failed: {str(e)}")
                    if 'progress_bar' in locals():
                        progress_bar.progress(0)
                    if 'status_text' in locals():
                        status_text.text("Backtest failed")
                    st.exception(e)
        
        with col2:
            st.subheader("📊 Expected Performance")
            
            # Display model statistics
            if model_info:
                accuracy = model_info.get('test_r2', 0)
                
                st.write(f"**Model Accuracy**: {accuracy:.2%}")
                
                if accuracy > 0.8:
                    st.success("🟢 High accuracy model - Strong backtest expected")
                elif accuracy > 0.6:
                    st.warning("🟡 Medium accuracy model - Moderate performance expected")
                else:
                    st.error("🔴 Low accuracy model - Use with caution")
                
                # Risk assessment
                st.subheader("⚠️ Risk Assessment")
                
                risk_score = (position_size + stop_loss) / 2
                
                if risk_score > 15:
                    st.error("🔴 High Risk Strategy")
                elif risk_score > 10:
                    st.warning("🟡 Medium Risk Strategy")
                else:
                    st.success("🟢 Low Risk Strategy")
                
                st.write(f"Risk Score: {risk_score:.1f}/25")
                
                # Recommendations
                st.subheader("💡 Recommendations")
                
                recommendations = []
                
                if position_size > 20:
                    recommendations.append("Consider reducing position size")
                
                if stop_loss < 3:
                    recommendations.append("Stop loss might be too tight")
                
                if take_profit < stop_loss * 2:
                    recommendations.append("Consider better risk/reward ratio")
                
                if confidence_threshold < 0.7:
                    recommendations.append("Higher confidence threshold may improve results")
                
                if not recommendations:
                    recommendations.append("Configuration looks balanced")
                
                for rec in recommendations:
                    st.write(f"• {rec}")
    
    with tab2:
        st.subheader("📊 Detailed Results Analysis")
        
        if 'backtest_results' in st.session_state:
            results = st.session_state.backtest_results
            
            # Trade analysis
            if results['trades']:
                trades_df = pd.DataFrame(results['trades'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📋 Trade Log")
                    
                    # Format trades for display
                    display_trades = trades_df.copy()
                    display_trades['PnL %'] = (display_trades['pnl'] / initial_capital * 100).round(2)
                    display_trades['Timestamp'] = pd.to_datetime(display_trades['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
                    
                    st.dataframe(
                        display_trades[['Timestamp', 'type', 'price', 'confidence', 'PnL %']],
                        use_container_width=True
                    )
                
                with col2:
                    st.subheader("📈 Trade Distribution")
                    
                    # PnL distribution
                    pnl_values = trades_df['pnl'].values
                    
                    fig = go.Figure(data=[go.Histogram(
                        x=pnl_values,
                        nbinsx=20,
                        marker_color='#00D4AA'
                    )])
                    
                    fig.update_layout(
                        title="P&L Distribution",
                        xaxis_title="P&L ($)",
                        yaxis_title="Frequency",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info("No trades were executed during the backtest period.")
        
        else:
            st.info("Run a backtest to see detailed results analysis.")
    
    with tab3:
        st.subheader("📈 Performance Metrics")
        
        if 'backtest_results' in st.session_state:
            results = st.session_state.backtest_results
            
            # Calculate comprehensive metrics
            portfolio_values = np.array(results['portfolio_values'])
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            # Risk metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("📊 Return Metrics")
                
                total_return = (portfolio_values[-1] - initial_capital) / initial_capital
                annualized_return = (1 + total_return) ** (365 / backtest_days) - 1
                
                st.metric("Total Return", f"{total_return:.2%}")
                st.metric("Annualized Return", f"{annualized_return:.2%}")
                
                if len(returns) > 0:
                    avg_daily_return = np.mean(returns)
                    st.metric("Avg Daily Return", f"{avg_daily_return:.4%}")
            
            with col2:
                st.subheader("⚠️ Risk Metrics")
                
                max_drawdown = results.get('max_drawdown', 0)
                st.metric("Max Drawdown", f"{max_drawdown:.2%}")
                
                if len(returns) > 1:
                    volatility = np.std(returns) * np.sqrt(252)  # Annualized
                    st.metric("Volatility (Annual)", f"{volatility:.2%}")
                    
                    if volatility > 0:
                        sharpe_ratio = annualized_return / volatility
                        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            
            with col3:
                st.subheader("🎯 Trade Metrics")
                
                trades = results['trades']
                if trades:
                    total_trades = len(trades)
                    winning_trades = sum(1 for trade in trades if trade['pnl'] > 0)
                    win_rate = winning_trades / total_trades
                    
                    st.metric("Win Rate", f"{win_rate:.2%}")
                    st.metric("Total Trades", total_trades)
                    
                    avg_win = np.mean([trade['pnl'] for trade in trades if trade['pnl'] > 0]) if winning_trades > 0 else 0
                    avg_loss = np.mean([trade['pnl'] for trade in trades if trade['pnl'] < 0]) if (total_trades - winning_trades) > 0 else 0
                    
                    if avg_loss != 0:
                        profit_factor = abs(avg_win / avg_loss)
                        st.metric("Profit Factor", f"{profit_factor:.2f}")
            
            # Performance comparison
            st.subheader("📊 Performance Comparison")
            
            # Compare with buy and hold
            buy_hold_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
            
            comparison_data = {
                'Strategy': ['ML Trading', 'Buy & Hold'],
                'Total Return': [f"{total_return:.2%}", f"{buy_hold_return:.2%}"],
                'Annualized Return': [f"{annualized_return:.2%}", f"{(1 + buy_hold_return) ** (365 / backtest_days) - 1:.2%}"],
                'Max Drawdown': [f"{max_drawdown:.2%}", "N/A"],
                'Number of Trades': [len(trades), "1"]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Strategy vs benchmark chart
            fig = go.Figure()
            
            # Normalize prices to start from same value
            normalized_prices = df['close'].values / df['close'].values[0] * initial_capital
            
            portfolio_dates = pd.date_range(
                start=pd.to_datetime(start_date),
                periods=len(results['portfolio_values']),
                freq='H' if model_info['interval'] == '1h' else 'D'
            )
            
            benchmark_dates = df.index[:len(portfolio_dates)]
            
            fig.add_trace(go.Scatter(
                x=portfolio_dates,
                y=results['portfolio_values'],
                mode='lines',
                name='ML Strategy',
                line=dict(color='#00D4AA', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=benchmark_dates,
                y=normalized_prices[:len(portfolio_dates)],
                mode='lines',
                name='Buy & Hold',
                line=dict(color='orange', width=2)
            ))
            
            fig.update_layout(
                title="Strategy vs Buy & Hold Performance",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Run a backtest to see performance metrics.")

if __name__ == "__main__":
    main()
