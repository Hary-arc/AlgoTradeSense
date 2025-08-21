import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import sqlite3
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import internal modules with fallbacks
try:
    from .database import get_model_by_name
    DATABASE_AVAILABLE = True
except ImportError:
    logger.warning("Database module not available. Using fallback.")
    DATABASE_AVAILABLE = False
    def get_model_by_name(model_name):
        return None

try:
    from .signal_generator import SignalGenerator
    SIGNAL_GENERATOR_AVAILABLE = True
except ImportError:
    logger.warning("SignalGenerator module not available. Using fallback.")
    SIGNAL_GENERATOR_AVAILABLE = False

try:
    from .data_processor import DataProcessor
    DATA_PROCESSOR_AVAILABLE = True
except ImportError:
    logger.warning("DataProcessor module not available. Using fallback.")
    DATA_PROCESSOR_AVAILABLE = False


class BacktestingEngine:
    """
    Comprehensive backtesting engine for ML trading strategies
    """
    
    def __init__(self):
        self.signal_generator = SignalGenerator() if SIGNAL_GENERATOR_AVAILABLE else None
        self.data_processor = DataProcessor() if DATA_PROCESSOR_AVAILABLE else None
        self.results = {}
        
    def run_backtest(self, model_name: str, data: pd.DataFrame, config: Dict) -> Dict:
        """
        Run backtesting for a specific model and configuration
        
        Args:
            model_name: Name of the ML model to use
            data: Historical price data with datetime index
            config: Backtesting configuration
            
        Returns:
            Dictionary containing backtest results
        """
        try:
            # Validate input data
            if data.empty or len(data) < 100:
                return {
                    'error': 'Insufficient data for backtesting (min 100 records required)',
                    'final_value': config.get('initial_capital', 10000),
                    'portfolio_values': [config.get('initial_capital', 10000)],
                    'trades': []
                }
            
            # Initialize backtesting parameters
            initial_capital = config.get('initial_capital', 10000)
            position_size_pct = config.get('position_size_pct', 0.1)
            commission_pct = config.get('commission_pct', 0.001)
            stop_loss_pct = config.get('stop_loss_pct', 0.05)
            take_profit_pct = config.get('take_profit_pct', 0.15)
            confidence_threshold = config.get('confidence_threshold', 0.7)
            risk_free_rate = config.get('risk_free_rate', 0.02)  # 2% annual
            
            # Initialize tracking variables
            cash = initial_capital
            position = 0.0  # Number of shares/coins held
            entry_price = 0.0
            
            trades = []
            portfolio_values = [initial_capital]
            daily_returns = []
            timestamps = [data.index[0]]
            
            # Track open positions for stop loss/take profit
            open_positions = []
            
            # Process each data point (start from 60 to have enough history)
            for i in range(60, len(data)):
                current_time = data.index[i]
                current_data = data.iloc[:i+1]
                current_price = current_data['close'].iloc[-1]
                
                # Generate signal
                signal_result = self._generate_signal_for_backtest(
                    model_name, current_data, confidence_threshold
                )
                
                # Calculate current portfolio value
                position_value = position * current_price
                current_portfolio_value = cash + position_value
                
                # Check stop loss and take profit for existing positions
                if position > 0 and open_positions:
                    for pos in open_positions:
                        stop_loss_price = pos['stop_loss']
                        take_profit_price = pos['take_profit']
                        
                        # Check stop loss
                        if current_price <= stop_loss_price:
                            # Execute stop loss
                            sell_amount = position * current_price
                            commission = sell_amount * commission_pct
                            cash += (sell_amount - commission)
                            
                            # Calculate P&L
                            pnl = (sell_amount - commission) - pos['cost_basis']
                            
                            # Record trade
                            trade = {
                                'timestamp': current_time,
                                'type': 'SELL_STOP_LOSS',
                                'price': current_price,
                                'quantity': position,
                                'amount': sell_amount,
                                'commission': commission,
                                'confidence': 0,
                                'portfolio_value_before': current_portfolio_value,
                                'pnl': pnl,
                                'entry_price': pos['entry_price']
                            }
                            trades.append(trade)
                            
                            position = 0.0
                            open_positions = []
                            break
                        
                        # Check take profit
                        elif current_price >= take_profit_price:
                            # Execute take profit
                            sell_amount = position * current_price
                            commission = sell_amount * commission_pct
                            cash += (sell_amount - commission)
                            
                            # Calculate P&L
                            pnl = (sell_amount - commission) - pos['cost_basis']
                            
                            # Record trade
                            trade = {
                                'timestamp': current_time,
                                'type': 'SELL_TAKE_PROFIT',
                                'price': current_price,
                                'quantity': position,
                                'amount': sell_amount,
                                'commission': commission,
                                'confidence': 0,
                                'portfolio_value_before': current_portfolio_value,
                                'pnl': pnl,
                                'entry_price': pos['entry_price']
                            }
                            trades.append(trade)
                            
                            position = 0.0
                            open_positions = []
                            break
                
                # Execute new trades based on signals
                if signal_result and signal_result.get('signal_type') in ['BUY', 'SELL']:
                    signal_type = signal_result.get('signal_type')
                    confidence = signal_result.get('confidence', 0)
                    
                    if signal_type == 'BUY' and position == 0 and cash > 0:
                        # Buy signal
                        trade_amount = cash * position_size_pct
                        commission = trade_amount * commission_pct
                        
                        if trade_amount > commission:
                            shares_to_buy = (trade_amount - commission) / current_price
                            
                            # Execute buy
                            position = shares_to_buy
                            cash -= trade_amount
                            cost_basis = trade_amount
                            
                            # Record open position
                            open_positions.append({
                                'entry_time': current_time,
                                'entry_price': current_price,
                                'quantity': shares_to_buy,
                                'cost_basis': cost_basis,
                                'stop_loss': current_price * (1 - stop_loss_pct),
                                'take_profit': current_price * (1 + take_profit_pct)
                            })
                            
                            # Record trade
                            trade = {
                                'timestamp': current_time,
                                'type': 'BUY',
                                'price': current_price,
                                'quantity': shares_to_buy,
                                'amount': trade_amount,
                                'commission': commission,
                                'confidence': confidence,
                                'portfolio_value_before': current_portfolio_value,
                                'pnl': 0,  # Will be calculated when closed
                                'entry_price': current_price
                            }
                            trades.append(trade)
                    
                    elif signal_type == 'SELL' and position > 0:
                        # Sell signal
                        sell_amount = position * current_price
                        commission = sell_amount * commission_pct
                        
                        # Execute sell
                        cash += (sell_amount - commission)
                        
                        # Calculate P&L
                        pnl = (sell_amount - commission) - open_positions[0]['cost_basis']
                        
                        # Record trade
                        trade = {
                            'timestamp': current_time,
                            'type': 'SELL',
                            'price': current_price,
                            'quantity': position,
                            'amount': sell_amount,
                            'commission': commission,
                            'confidence': confidence,
                            'portfolio_value_before': current_portfolio_value,
                            'pnl': pnl,
                            'entry_price': open_positions[0]['entry_price']
                        }
                        trades.append(trade)
                        
                        position = 0.0
                        open_positions = []
                
                # Update portfolio tracking
                position_value = position * current_price
                current_portfolio_value = cash + position_value
                portfolio_values.append(current_portfolio_value)
                timestamps.append(current_time)
                
                # Calculate daily return (if we have a previous value)
                if len(portfolio_values) > 1:
                    daily_return = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
                    daily_returns.append(daily_return)
            
            # Close any remaining position at final price
            if position > 0:
                final_price = data['close'].iloc[-1]
                sell_amount = position * final_price
                commission = sell_amount * commission_pct
                cash += (sell_amount - commission)
                
                # Calculate final P&L
                pnl = (sell_amount - commission) - open_positions[0]['cost_basis']
                
                trade = {
                    'timestamp': data.index[-1],
                    'type': 'SELL_FINAL',
                    'price': final_price,
                    'quantity': position,
                    'amount': sell_amount,
                    'commission': commission,
                    'confidence': 0,
                    'portfolio_value_before': portfolio_values[-1],
                    'pnl': pnl,
                    'entry_price': open_positions[0]['entry_price']
                }
                trades.append(trade)
                
                portfolio_values[-1] = cash  # Update final portfolio value
            
            # Calculate performance metrics
            results = self._calculate_performance_metrics(
                initial_capital, portfolio_values, trades, daily_returns, data, risk_free_rate
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Backtesting error: {e}")
            return {
                'error': str(e),
                'portfolio_values': [initial_capital],
                'trades': [],
                'final_value': initial_capital
            }
    
    def _generate_signal_for_backtest(self, model_name: str, data: pd.DataFrame, 
                                    confidence_threshold: float) -> Optional[Dict]:
        """Generate signal for backtesting"""
        try:
            # Try to use ML model if available
            if DATABASE_AVAILABLE:
                model_info = get_model_by_name(model_name)
                if model_info and self.signal_generator:
                    # Use actual ML model signal generation
                    signal = self.signal_generator.generate_signal(model_name, data)
                    if signal and signal.get('confidence', 0) >= confidence_threshold:
                        return signal
            
            # Fallback to technical analysis signals
            return self._generate_technical_signal(data)
            
        except Exception as e:
            logger.warning(f"Error generating signal: {e}. Using technical fallback.")
            return self._generate_technical_signal(data)
    
    def _generate_technical_signal(self, data: pd.DataFrame) -> Optional[Dict]:
        """Generate technical analysis signal as fallback"""
        if len(data) < 20:
            return None
            
        try:
            # Calculate technical indicators
            close_prices = data['close']
            current_price = close_prices.iloc[-1]
            
            # Moving averages
            ma_short = close_prices.rolling(window=10).mean().iloc[-1]
            ma_long = close_prices.rolling(window=20).mean().iloc[-1]
            
            # RSI
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            
            # Generate signal
            signal_type = 'HOLD'
            confidence = 0.5
            reason = 'No_Clear_Signal'
            
            # Bullish signals
            if ma_short > ma_long and current_price > ma_short and current_rsi < 70:
                signal_type = 'BUY'
                confidence = 0.7
                reason = 'MA_Crossover_Bullish_RSI_Ok'
            # Bearish signals
            elif ma_short < ma_long and current_price < ma_short and current_rsi > 30:
                signal_type = 'SELL'
                confidence = 0.7
                reason = 'MA_Crossover_Bearish_RSI_Ok'
            # Overbought/oversold
            elif current_rsi > 80:
                signal_type = 'SELL'
                confidence = 0.6
                reason = 'RSI_Overbought'
            elif current_rsi < 20:
                signal_type = 'BUY'
                confidence = 0.6
                reason = 'RSI_Oversold'
            
            return {
                'signal_type': signal_type,
                'confidence': confidence,
                'price': current_price,
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"Error generating technical signal: {e}")
            return None
    
    def _calculate_performance_metrics(self, initial_capital: float, portfolio_values: List[float],
                                     trades: List[Dict], daily_returns: List[float], 
                                     price_data: pd.DataFrame, risk_free_rate: float = 0.02) -> Dict:
        """Calculate comprehensive performance metrics"""
        try:
            final_value = portfolio_values[-1] if portfolio_values else initial_capital
            total_return = (final_value - initial_capital) / initial_capital
            
            # Basic metrics
            metrics = {
                'initial_capital': initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'portfolio_values': portfolio_values,
                'trades': trades,
                'total_trades': len(trades),
                'daily_returns': daily_returns
            }
            
            # Trade analysis
            closed_trades = [t for t in trades if t.get('pnl') is not None and t['type'].startswith('SELL')]
            
            if closed_trades:
                winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
                losing_trades = [t for t in closed_trades if t.get('pnl', 0) < 0]
                
                metrics['winning_trades'] = len(winning_trades)
                metrics['losing_trades'] = len(losing_trades)
                metrics['win_rate'] = len(winning_trades) / len(closed_trades) if closed_trades else 0
                
                # P&L analysis
                total_pnl = sum(t.get('pnl', 0) for t in closed_trades)
                metrics['total_pnl'] = total_pnl
                
                if winning_trades:
                    avg_win = np.mean([t['pnl'] for t in winning_trades])
                    max_win = max([t['pnl'] for t in winning_trades])
                    metrics['average_win'] = avg_win
                    metrics['max_win'] = max_win
                else:
                    metrics['average_win'] = 0
                    metrics['max_win'] = 0
                
                if losing_trades:
                    avg_loss = np.mean([t['pnl'] for t in losing_trades])
                    max_loss = min([t['pnl'] for t in losing_trades])
                    metrics['average_loss'] = avg_loss
                    metrics['max_loss'] = max_loss
                else:
                    metrics['average_loss'] = 0
                    metrics['max_loss'] = 0
                
                # Profit factor
                total_wins = sum([t['pnl'] for t in winning_trades])
                total_losses = abs(sum([t['pnl'] for t in losing_trades]))
                
                if total_losses > 0:
                    metrics['profit_factor'] = total_wins / total_losses
                else:
                    metrics['profit_factor'] = float('inf') if total_wins > 0 else 0
                
                # Average holding period
                buy_trades = {t['timestamp']: t for t in trades if t['type'] == 'BUY'}
                holding_periods = []
                
                for trade in closed_trades:
                    if trade['entry_price']:
                        # Find corresponding buy trade
                        for buy_time, buy_trade in buy_trades.items():
                            if abs(buy_trade['price'] - trade['entry_price']) < 0.0001:  # Float comparison tolerance
                                if hasattr(buy_time, 'timestamp') and hasattr(trade['timestamp'], 'timestamp'):
                                    holding_days = (trade['timestamp'].timestamp() - buy_time.timestamp()) / (24 * 3600)
                                    holding_periods.append(holding_days)
                                break
                
                if holding_periods:
                    metrics['avg_holding_period_days'] = np.mean(holding_periods)
                else:
                    metrics['avg_holding_period_days'] = 0
            else:
                # No trades executed
                metrics.update({
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'total_pnl': 0,
                    'average_win': 0,
                    'max_win': 0,
                    'average_loss': 0,
                    'max_loss': 0,
                    'profit_factor': 0,
                    'avg_holding_period_days': 0
                })
            
            # Risk metrics
            if len(daily_returns) > 1:
                returns_array = np.array(daily_returns)
                
                # Volatility (annualized)
                volatility = np.std(returns_array) * np.sqrt(252)
                metrics['volatility'] = volatility
                
                # Sharpe ratio (with risk-free rate)
                if volatility > 0:
                    annualized_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
                    excess_return = annualized_return - risk_free_rate
                    sharpe_ratio = excess_return / volatility
                    metrics['sharpe_ratio'] = sharpe_ratio
                else:
                    metrics['sharpe_ratio'] = 0
                
                # Sortino ratio (only downside volatility)
                downside_returns = returns_array[returns_array < 0]
                if len(downside_returns) > 0:
                    downside_volatility = np.std(downside_returns) * np.sqrt(252)
                    if downside_volatility > 0:
                        sortino_ratio = (annualized_return - risk_free_rate) / downside_volatility
                        metrics['sortino_ratio'] = sortino_ratio
                    else:
                        metrics['sortino_ratio'] = float('inf') if annualized_return > risk_free_rate else 0
                else:
                    metrics['sortino_ratio'] = float('inf') if annualized_return > risk_free_rate else 0
                
                # Maximum drawdown
                portfolio_array = np.array(portfolio_values)
                peak = np.maximum.accumulate(portfolio_array)
                drawdown = (peak - portfolio_array) / peak
                max_drawdown = np.max(drawdown)
                
                metrics['max_drawdown'] = max_drawdown
                metrics['max_drawdown_pct'] = max_drawdown * 100
                
                # Calmar ratio
                if max_drawdown > 0:
                    metrics['calmar_ratio'] = annualized_return / max_drawdown
                else:
                    metrics['calmar_ratio'] = float('inf') if annualized_return > 0 else 0
                
                # Recovery factor
                if max_drawdown > 0:
                    metrics['recovery_factor'] = total_return / max_drawdown
                else:
                    metrics['recovery_factor'] = float('inf') if total_return > 0 else 0
            else:
                metrics.update({
                    'volatility': 0,
                    'sharpe_ratio': 0,
                    'sortino_ratio': 0,
                    'max_drawdown': 0,
                    'max_drawdown_pct': 0,
                    'calmar_ratio': 0,
                    'recovery_factor': 0
                })
            
            # Benchmark comparison (Buy and Hold)
            if len(price_data) > 0:
                start_price = price_data['close'].iloc[0]
                end_price = price_data['close'].iloc[-1]
                buy_hold_return = (end_price - start_price) / start_price
                
                metrics['benchmark_return'] = buy_hold_return
                metrics['benchmark_return_pct'] = buy_hold_return * 100
                metrics['alpha'] = total_return - buy_hold_return
                metrics['alpha_pct'] = metrics['alpha'] * 100
                
                # Information ratio
                benchmark_returns = price_data['close'].pct_change().dropna()
                if len(benchmark_returns) > 0 and len(daily_returns) > 0:
                    min_len = min(len(benchmark_returns), len(daily_returns))
                    active_returns = np.array(daily_returns[:min_len]) - np.array(benchmark_returns[-min_len:])
                    tracking_error = np.std(active_returns) * np.sqrt(252)
                    
                    if tracking_error > 0:
                        information_ratio = (annualized_return - (buy_hold_return * 252 / len(price_data))) / tracking_error
                        metrics['information_ratio'] = information_ratio
                    else:
                        metrics['information_ratio'] = 0
                else:
                    metrics['information_ratio'] = 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {
                'error': str(e),
                'portfolio_values': portfolio_values,
                'trades': trades,
                'final_value': final_value
            }
    
    def run_walk_forward_analysis(self, model_name: str, data: pd.DataFrame, 
                                config: Dict, window_size: int = 252, step_size: int = 63) -> Dict:
        """
        Run walk-forward analysis for more robust backtesting
        
        Args:
            model_name: Name of the ML model
            data: Historical data
            config: Backtesting configuration
            window_size: Size of the rolling window in days
            step_size: Step size between windows in days
            
        Returns:
            Walk-forward analysis results
        """
        results = []
        
        # Convert step size to number of records
        if step_size <= 0:
            step_size = window_size // 4
        
        for start_idx in range(0, len(data) - window_size, step_size):
            end_idx = start_idx + window_size
            test_end_idx = min(len(data), end_idx + step_size)
            
            # Training window
            train_data = data.iloc[start_idx:end_idx]
            
            # Test window
            test_data = data.iloc[end_idx:test_end_idx]
            
            if len(test_data) < 10:
                break
            
            # Run backtest on test window
            window_result = self.run_backtest(model_name, test_data, config)
            window_result['start_date'] = test_data.index[0]
            window_result['end_date'] = test_data.index[-1]
            window_result['window_size'] = len(test_data)
            
            results.append(window_result)
        
        # Aggregate results
        aggregated_results = self._aggregate_walk_forward_results(results)
        
        return {
            'individual_windows': results,
            'aggregated_metrics': aggregated_results
        }
    
    def _aggregate_walk_forward_results(self, results: List[Dict]) -> Dict:
        """Aggregate walk-forward analysis results"""
        if not results:
            return {}
        
        # Filter out failed windows
        successful_results = [r for r in results if 'error' not in r]
        
        if not successful_results:
            return {
                'num_windows': len(results),
                'successful_windows': 0,
                'consistency_score': 0
            }
        
        # Calculate average metrics
        total_returns = [r.get('total_return', 0) for r in successful_results]
        win_rates = [r.get('win_rate', 0) for r in successful_results]
        sharpe_ratios = [r.get('sharpe_ratio', 0) for r in successful_results]
        max_drawdowns = [r.get('max_drawdown', 0) for r in successful_results]
        
        aggregated = {
            'num_windows': len(results),
            'successful_windows': len(successful_results),
            'consistency_score': len(successful_results) / len(results),
            'avg_return': np.mean(total_returns),
            'median_return': np.median(total_returns),
            'std_return': np.std(total_returns),
            'min_return': np.min(total_returns),
            'max_return': np.max(total_returns),
            'avg_win_rate': np.mean(win_rates),
            'avg_sharpe_ratio': np.mean(sharpe_ratios),
            'avg_max_drawdown': np.mean(max_drawdowns),
            'positive_windows': sum(1 for r in total_returns if r > 0),
            'negative_windows': sum(1 for r in total_returns if r < 0)
        }
        
        return aggregated
    
    def compare_strategies(self, models: List[str], data: pd.DataFrame, config: Dict) -> Dict:
        """
        Compare multiple models/strategies
        
        Args:
            models: List of model names to compare
            data: Historical data
            config: Backtesting configuration
            
        Returns:
            Comparison results
        """
        results = {}
        
        for model_name in models:
            logger.info(f"Backtesting {model_name}...")
            model_result = self.run_backtest(model_name, data, config)
            results[model_name] = model_result
        
        # Create comparison summary
        comparison = self._create_strategy_comparison(results)
        
        return {
            'individual_results': results,
            'comparison_summary': comparison
        }
    
    def _create_strategy_comparison(self, results: Dict) -> Dict:
        """Create strategy comparison summary"""
        comparison_metrics = []
        
        for strategy_name, result in results.items():
            if 'error' not in result:
                metrics = {
                    'strategy': strategy_name,
                    'total_return': result.get('total_return', 0),
                    'total_return_pct': result.get('total_return_pct', 0),
                    'win_rate': result.get('win_rate', 0),
                    'sharpe_ratio': result.get('sharpe_ratio', 0),
                    'sortino_ratio': result.get('sortino_ratio', 0),
                    'max_drawdown': result.get('max_drawdown', 0),
                    'max_drawdown_pct': result.get('max_drawdown_pct', 0),
                    'total_trades': result.get('total_trades', 0),
                    'profit_factor': result.get('profit_factor', 0),
                    'calmar_ratio': result.get('calmar_ratio', 0)
                }
                comparison_metrics.append(metrics)
        
        # Rank strategies by multiple criteria
        if comparison_metrics:
            # Sort by multiple metrics
            ranked_by_sharpe = sorted(comparison_metrics, key=lambda x: x['sharpe_ratio'], reverse=True)
            ranked_by_return = sorted(comparison_metrics, key=lambda x: x['total_return'], reverse=True)
            ranked_by_drawdown = sorted(comparison_metrics, key=lambda x: x['max_drawdown'])
            
            return {
                'ranked_by_sharpe': ranked_by_sharpe,
                'ranked_by_return': ranked_by_return,
                'ranked_by_drawdown': ranked_by_drawdown,
                'best_overall': ranked_by_sharpe[0] if ranked_by_sharpe else None,
                'num_strategies': len(comparison_metrics)
            }
        
        return {}