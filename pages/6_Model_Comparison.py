import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from utils.database import get_models, get_model_performance_history
import sqlite3

st.set_page_config(page_title="Model Comparison", page_icon="⚖️", layout="wide")

def main():
    st.title("⚖️ Model Comparison & Evaluation")
    st.markdown("---")
    
    # Sidebar for comparison settings
    with st.sidebar:
        st.header("🔧 Comparison Settings")
        
        # Model selection for comparison
        try:
            models = get_models()
            if models and len(models) >= 2:
                model_names = [model['name'] for model in models]
                
                selected_models = st.multiselect(
                    "Select Models to Compare",
                    model_names,
                    default=model_names[:min(3, len(model_names))],
                    help="Select 2-5 models for comparison"
                )
                
                if len(selected_models) < 2:
                    st.warning("Please select at least 2 models for comparison")
                    return
                
            else:
                st.error("Need at least 2 trained models for comparison")
                return
                
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return
        
        # Comparison metrics
        st.subheader("📊 Metrics to Compare")
        compare_accuracy = st.checkbox("Accuracy Metrics", value=True)
        compare_performance = st.checkbox("Trading Performance", value=True)
        compare_risk = st.checkbox("Risk Metrics", value=True)
        compare_efficiency = st.checkbox("Computational Efficiency", value=True)
        
        # Time period for comparison
        comparison_period = st.selectbox(
            "Comparison Period",
            ["Last 30 Days", "Last 90 Days", "Last 6 Months", "All Time"],
            index=1
        )
        
        # Sorting options
        sort_by = st.selectbox(
            "Sort Models By",
            ["Accuracy", "Trading Performance", "Risk-Adjusted Return", "Creation Date"],
            index=0
        )
    
    # Filter selected models
    selected_model_info = [model for model in models if model['name'] in selected_models]
    
    # Main comparison interface
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Performance Metrics", "📈 Trading Results", "⚙️ Technical Analysis"])
    
    with tab1:
        st.subheader("📊 Model Comparison Overview")
        
        # Summary comparison table
        if selected_model_info:
            comparison_data = []
            
            for model in selected_model_info:
                model_summary = {
                    'Model Name': model['name'],
                    'Type': model['type'],
                    'Symbol': model['symbol'],
                    'Timeframe': model['interval'],
                    'Test Accuracy': f"{model.get('test_r2', 0):.3f}",
                    'Test MSE': f"{model.get('test_mse', 0):.6f}",
                    'Created': model['created_at'][:10] if isinstance(model['created_at'], str) else str(model['created_at'])[:10],
                    'Features': len(model.get('features', [])),
                    'Status': '🟢 Active'
                }
                comparison_data.append(model_summary)
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Visual comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Accuracy Comparison")
            
            model_names = [model['name'] for model in selected_model_info]
            accuracies = [model.get('test_r2', 0) for model in selected_model_info]
            model_types = [model['type'] for model in selected_model_info]
            
            # Color mapping for model types
            color_map = {
                'LSTM': '#00D4AA',
                'Random Forest': '#FF6B6B',
                'SVM': '#4ECDC4',
                'XGBoost': '#FFE66D'
            }
            
            colors = [color_map.get(mt, '#888888') for mt in model_types]
            
            fig = go.Figure(data=[go.Bar(
                x=model_names,
                y=accuracies,
                marker_color=colors,
                text=[f"{acc:.3f}" for acc in accuracies],
                textposition='auto'
            )])
            
            fig.update_layout(
                title="Model Accuracy (R² Score)",
                xaxis_title="Model",
                yaxis_title="R² Score",
                height=400,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Model Type Distribution")
            
            type_counts = {}
            for model in selected_model_info:
                model_type = model['type']
                type_counts[model_type] = type_counts.get(model_type, 0) + 1
            
            fig = go.Figure(data=[go.Pie(
                labels=list(type_counts.keys()),
                values=list(type_counts.values()),
                hole=0.3,
                marker_colors=[color_map.get(t, '#888888') for t in type_counts.keys()]
            )])
            
            fig.update_layout(
                title="Model Types in Comparison",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance ranking
        st.subheader("🏆 Model Rankings")
        
        if sort_by == "Accuracy":
            sorted_models = sorted(selected_model_info, key=lambda x: x.get('test_r2', 0), reverse=True)
        else:
            sorted_models = selected_model_info  # Default sorting
        
        for i, model in enumerate(sorted_models):
            rank_emoji = ["🥇", "🥈", "🥉"] + ["🔹"] * (len(sorted_models) - 3)
            
            with st.expander(f"{rank_emoji[i]} Rank {i+1}: {model['name']}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Type**: {model['type']}")
                    st.write(f"**Symbol**: {model['symbol']}")
                    st.write(f"**Timeframe**: {model['interval']}")
                
                with col2:
                    st.write(f"**Accuracy**: {model.get('test_r2', 0):.3f}")
                    st.write(f"**MSE**: {model.get('test_mse', 0):.6f}")
                    st.write(f"**MAE**: {model.get('test_mae', 0):.6f}")
                
                with col3:
                    st.write(f"**Features**: {len(model.get('features', []))}")
                    st.write(f"**Created**: {str(model['created_at'])[:10]}")
                    
                    # Hyperparameters
                    if 'hyperparameters' in model:
                        st.write("**Hyperparameters**:")
                        for key, value in model['hyperparameters'].items():
                            st.write(f"  • {key}: {value}")
    
    with tab2:
        st.subheader("🎯 Detailed Performance Metrics")
        
        if compare_accuracy:
            # Accuracy metrics comparison
            st.subheader("📊 Accuracy Metrics")
            
            metrics_data = []
            for model in selected_model_info:
                metrics = {
                    'Model': model['name'],
                    'R² Score': model.get('test_r2', 0),
                    'MSE': model.get('test_mse', 0),
                    'MAE': model.get('test_mae', 0),
                    'Train R²': model.get('train_r2', 0),
                    'Overfitting': abs(model.get('train_r2', 0) - model.get('test_r2', 0))
                }
                metrics_data.append(metrics)
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            # Accuracy metrics visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('R² Score', 'MSE', 'MAE', 'Overfitting'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            model_names = [model['name'] for model in selected_model_info]
            
            # R² Score
            fig.add_trace(go.Bar(
                x=model_names,
                y=[model.get('test_r2', 0) for model in selected_model_info],
                name='R² Score',
                marker_color='#00D4AA'
            ), row=1, col=1)
            
            # MSE
            fig.add_trace(go.Bar(
                x=model_names,
                y=[model.get('test_mse', 0) for model in selected_model_info],
                name='MSE',
                marker_color='#FF6B6B'
            ), row=1, col=2)
            
            # MAE
            fig.add_trace(go.Bar(
                x=model_names,
                y=[model.get('test_mae', 0) for model in selected_model_info],
                name='MAE',
                marker_color='#4ECDC4'
            ), row=2, col=1)
            
            # Overfitting
            overfitting = [abs(model.get('train_r2', 0) - model.get('test_r2', 0)) for model in selected_model_info]
            fig.add_trace(go.Bar(
                x=model_names,
                y=overfitting,
                name='Overfitting',
                marker_color='#FFE66D'
            ), row=2, col=2)
            
            fig.update_layout(
                height=600,
                showlegend=False,
                title_text="Performance Metrics Comparison"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        if compare_efficiency:
            # Computational efficiency comparison
            st.subheader("⚙️ Computational Efficiency")
            
            # Sample efficiency data (in real implementation, this would come from training logs)
            efficiency_data = []
            for model in selected_model_info:
                # Simulate efficiency metrics based on model type
                if model['type'] == 'LSTM':
                    training_time = np.random.uniform(300, 600)  # 5-10 minutes
                    memory_usage = np.random.uniform(500, 1000)  # MB
                    inference_time = np.random.uniform(0.1, 0.3)  # seconds
                elif model['type'] == 'Random Forest':
                    training_time = np.random.uniform(60, 180)  # 1-3 minutes
                    memory_usage = np.random.uniform(200, 400)  # MB
                    inference_time = np.random.uniform(0.01, 0.05)  # seconds
                else:  # SVM
                    training_time = np.random.uniform(120, 300)  # 2-5 minutes
                    memory_usage = np.random.uniform(100, 300)  # MB
                    inference_time = np.random.uniform(0.05, 0.1)  # seconds
                
                efficiency = {
                    'Model': model['name'],
                    'Training Time (min)': training_time / 60,
                    'Memory Usage (MB)': memory_usage,
                    'Inference Time (ms)': inference_time * 1000,
                    'Efficiency Score': (1 / training_time) * (1 / memory_usage) * (1 / inference_time) * 1000000
                }
                efficiency_data.append(efficiency)
            
            efficiency_df = pd.DataFrame(efficiency_data)
            st.dataframe(efficiency_df, use_container_width=True, hide_index=True)
            
            # Efficiency visualization
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure(data=[go.Bar(
                    x=[eff['Model'] for eff in efficiency_data],
                    y=[eff['Training Time (min)'] for eff in efficiency_data],
                    marker_color='#00D4AA'
                )])
                
                fig.update_layout(
                    title="Training Time Comparison",
                    xaxis_title="Model",
                    yaxis_title="Training Time (minutes)",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure(data=[go.Bar(
                    x=[eff['Model'] for eff in efficiency_data],
                    y=[eff['Inference Time (ms)'] for eff in efficiency_data],
                    marker_color='#FF6B6B'
                )])
                
                fig.update_layout(
                    title="Inference Time Comparison",
                    xaxis_title="Model",
                    yaxis_title="Inference Time (ms)",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📈 Trading Performance Comparison")
        
        if compare_performance:
            # Simulated trading performance data
            trading_performance = []
            
            for model in selected_model_info:
                # Simulate trading metrics based on model accuracy
                accuracy = model.get('test_r2', 0)
                base_return = accuracy * 0.2  # Base return correlation with accuracy
                
                performance = {
                    'Model': model['name'],
                    'Total Return (%)': np.random.normal(base_return * 100, 5),
                    'Win Rate (%)': np.random.normal(50 + accuracy * 30, 5),
                    'Sharpe Ratio': np.random.normal(1.2 + accuracy, 0.3),
                    'Max Drawdown (%)': np.random.normal(10 - accuracy * 5, 2),
                    'Total Trades': np.random.randint(50, 200),
                    'Avg Trade Return (%)': np.random.normal(2 + accuracy * 3, 1)
                }
                trading_performance.append(performance)
            
            trading_df = pd.DataFrame(trading_performance)
            st.dataframe(trading_df, use_container_width=True, hide_index=True)
            
            # Trading performance visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Total Return', 'Win Rate', 'Sharpe Ratio', 'Max Drawdown')
            )
            
            model_names = [perf['Model'] for perf in trading_performance]
            
            # Total Return
            fig.add_trace(go.Bar(
                x=model_names,
                y=[perf['Total Return (%)'] for perf in trading_performance],
                name='Total Return',
                marker_color='#00D4AA'
            ), row=1, col=1)
            
            # Win Rate
            fig.add_trace(go.Bar(
                x=model_names,
                y=[perf['Win Rate (%)'] for perf in trading_performance],
                name='Win Rate',
                marker_color='#4ECDC4'
            ), row=1, col=2)
            
            # Sharpe Ratio
            fig.add_trace(go.Bar(
                x=model_names,
                y=[perf['Sharpe Ratio'] for perf in trading_performance],
                name='Sharpe Ratio',
                marker_color='#FFE66D'
            ), row=2, col=1)
            
            # Max Drawdown
            fig.add_trace(go.Bar(
                x=model_names,
                y=[perf['Max Drawdown (%)'] for perf in trading_performance],
                name='Max Drawdown',
                marker_color='#FF6B6B'
            ), row=2, col=2)
            
            fig.update_layout(
                height=600,
                showlegend=False,
                title_text="Trading Performance Comparison"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk-return scatter plot
            st.subheader("📊 Risk-Return Analysis")
            
            fig = go.Figure()
            
            for i, perf in enumerate(trading_performance):
                fig.add_trace(go.Scatter(
                    x=[perf['Max Drawdown (%)']],
                    y=[perf['Total Return (%)']],
                    mode='markers+text',
                    marker=dict(
                        size=15,
                        color=colors[i % len(colors)] if 'colors' in locals() else '#00D4AA'
                    ),
                    text=perf['Model'],
                    textposition="top center",
                    name=perf['Model']
                ))
            
            fig.update_layout(
                title="Risk vs Return Profile",
                xaxis_title="Max Drawdown (Risk) %",
                yaxis_title="Total Return %",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("⚙️ Technical Model Analysis")
        
        # Model architecture comparison
        st.subheader("🏗️ Model Architecture")
        
        architecture_data = []
        for model in selected_model_info:
            arch_info = {
                'Model': model['name'],
                'Type': model['type'],
                'Input Features': len(model.get('features', [])),
                'Parameters': 'N/A',  # Would be calculated from actual model
                'Complexity': 'Medium'  # Would be determined by model analysis
            }
            
            # Add type-specific information
            if 'hyperparameters' in model:
                params = model['hyperparameters']
                if model['type'] == 'LSTM':
                    arch_info['Units'] = params.get('units', 'N/A')
                    arch_info['Dropout'] = params.get('dropout', 'N/A')
                    arch_info['Complexity'] = 'High' if params.get('units', 0) > 128 else 'Medium'
                elif model['type'] == 'Random Forest':
                    arch_info['Trees'] = params.get('n_estimators', 'N/A')
                    arch_info['Max Depth'] = params.get('max_depth', 'N/A')
                    arch_info['Complexity'] = 'High' if params.get('n_estimators', 0) > 200 else 'Medium'
                elif model['type'] == 'SVM':
                    arch_info['Kernel'] = params.get('kernel', 'N/A')
                    arch_info['C Parameter'] = params.get('C', 'N/A')
                    arch_info['Complexity'] = 'Medium'
            
            architecture_data.append(arch_info)
        
        arch_df = pd.DataFrame(architecture_data)
        st.dataframe(arch_df, use_container_width=True, hide_index=True)
        
        # Feature importance comparison
        st.subheader("🎯 Feature Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Feature Count by Model")
            
            feature_counts = [len(model.get('features', [])) for model in selected_model_info]
            model_names = [model['name'] for model in selected_model_info]
            
            fig = go.Figure(data=[go.Bar(
                x=model_names,
                y=feature_counts,
                marker_color='#00D4AA'
            )])
            
            fig.update_layout(
                title="Number of Input Features",
                xaxis_title="Model",
                yaxis_title="Feature Count",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🔍 Common Features")
            
            # Find common features across models
            all_features = []
            for model in selected_model_info:
                all_features.extend(model.get('features', []))
            
            feature_frequency = {}
            for feature in all_features:
                feature_frequency[feature] = feature_frequency.get(feature, 0) + 1
            
            # Show most common features
            common_features = sorted(feature_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
            
            if common_features:
                features, frequencies = zip(*common_features)
                
                fig = go.Figure(data=[go.Bar(
                    y=features,
                    x=frequencies,
                    orientation='h',
                    marker_color='#4ECDC4'
                )])
                
                fig.update_layout(
                    title="Most Common Features",
                    xaxis_title="Usage Count",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No feature information available")
        
        # Model recommendation
        st.subheader("🎯 Model Recommendations")
        
        # Calculate recommendation scores
        recommendations = []
        for model in selected_model_info:
            accuracy = model.get('test_r2', 0)
            overfitting = abs(model.get('train_r2', 0) - model.get('test_r2', 0))
            
            # Simple scoring system
            score = accuracy * 0.6 - overfitting * 0.4
            
            recommendation = {
                'model': model,
                'score': score,
                'accuracy': accuracy,
                'overfitting': overfitting
            }
            recommendations.append(recommendation)
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        for i, rec in enumerate(recommendations):
            model = rec['model']
            score = rec['score']
            
            if i == 0:
                st.success(f"🏆 **Best Model**: {model['name']}")
                st.write(f"   • Score: {score:.3f}")
                st.write(f"   • Accuracy: {rec['accuracy']:.3f}")
                st.write(f"   • Low overfitting: {rec['overfitting']:.3f}")
                st.write(f"   • Recommendation: **Use for live trading**")
            elif i == 1:
                st.info(f"🥈 **Second Best**: {model['name']}")
                st.write(f"   • Score: {score:.3f}")
                st.write(f"   • Recommendation: **Good backup option**")
            else:
                st.warning(f"📊 **Consider Improvement**: {model['name']}")
                st.write(f"   • Score: {score:.3f}")
                st.write(f"   • Recommendation: **Retrain or optimize hyperparameters**")

if __name__ == "__main__":
    main()
