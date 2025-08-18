# AI Trading Platform

## Overview

This is a comprehensive AI-powered trading platform built with Streamlit that provides machine learning-based cryptocurrency trading signals and analysis. The platform integrates with Binance API to fetch real-time market data and uses multiple ML models (LSTM, Random Forest, SVM) to generate trading signals, perform backtesting, and provide detailed analytics.

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes

**August 18, 2025:**
- Fixed critical pandas deprecation errors (fillna method='forward' → method='ffill')
- Fixed ta.volume.volume_sma function (replaced with manual volume moving average)
- Fixed variable scope issues in Model Training page (n_estimators, lstm_units, etc.)
- Reduced LSP diagnostic errors from 136 to 108 (28 errors fixed)
- Binance API credentials securely stored in environment variables
- All required packages successfully installed and working
- **NEW**: Implemented TradingView-style live charts with advanced pattern recognition
- **NEW**: Added 8 candlestick pattern types: channels, triangles, wedges with trend lines
- **NEW**: Created professional dark theme matching TradingView aesthetics
- **NEW**: Live pattern detection with trading signals and confidence scoring

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit-based multi-page application
- **UI Structure**: Main dashboard with 6 specialized pages:
  - Dashboard: Real-time market data and charts
  - Model Training: ML model creation and training interface
  - Signal Generation: AI-powered trading signal creation
  - Backtesting: Strategy performance testing
  - Analytics: Comprehensive performance metrics
  - Model Comparison: Side-by-side model evaluation
- **Visualization**: Plotly for interactive charts and TradingView-style candlestick charts
- **Layout**: Wide layout with sidebar navigation for global settings

### Backend Architecture
- **Core Language**: Python
- **ML Framework**: Hybrid approach using both scikit-learn and TensorFlow/Keras
- **Model Types**: 
  - LSTM neural networks for time series prediction
  - Random Forest for ensemble learning
  - Support Vector Machines for pattern recognition
- **Data Processing**: Custom DataProcessor class with technical indicator integration
- **Signal Generation**: Real-time AI signal generation with confidence scoring
- **Backtesting Engine**: Comprehensive strategy testing with risk metrics

### Data Storage Solutions
- **Primary Database**: SQLite for local data persistence
- **Database Schema**: 
  - Models table: Stores trained ML models with metadata and performance metrics
  - Market data table: Historical and real-time price data
  - Signals table: Generated trading signals with timestamps
  - Trading history: Backtest results and performance data
- **Model Serialization**: Binary storage of trained models and scalers in database
- **Data Processing**: Pandas for data manipulation with technical analysis integration

### Authentication and Authorization
- **API Authentication**: Environment variable-based Binance API key management
- **Security**: HMAC SHA256 signature generation for secure API communication
- **Configuration**: Environment-based testnet/mainnet switching

## External Dependencies

### Trading APIs
- **Binance API**: Primary cryptocurrency exchange integration
  - REST API for historical data and account management
  - WebSocket API for real-time market data streaming
  - Testnet support for safe development and testing

### Machine Learning Libraries
- **TensorFlow/Keras**: Deep learning framework for LSTM neural networks
- **scikit-learn**: Traditional ML algorithms and preprocessing utilities
- **Technical Analysis Library (ta)**: Comprehensive technical indicator calculations

### Data Visualization
- **Plotly**: Interactive charting library for financial data visualization
- **Streamlit**: Web application framework with built-in visualization components

### Data Processing
- **Pandas**: Primary data manipulation and analysis library
- **NumPy**: Numerical computing foundation
- **SQLite3**: Embedded database for local data storage

### Development Tools
- **Joblib**: Model serialization and parallel processing
- **Requests**: HTTP client for API communications
- **WebSocket**: Real-time data streaming capabilities

### Environment Management
- Environment variables for secure API credential storage
- Configurable testnet/production mode switching
- Modular utility structure for maintainable code organization