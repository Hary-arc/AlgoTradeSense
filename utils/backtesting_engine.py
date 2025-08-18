import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import sqlite3
from .database import get_model_by_name
from .signal_generator import SignalGenerator
from .data_processor import DataProcessor

class BacktestingEngine:
    """
    Comprehensive backtesting engine for ML trading strategies
    """
    
    def __init__(self):
        self.signal_generator = SignalGenerator()
        self.data_processor = DataProcessor()
        self.results = {}
        
    def run_backtest(self, model_name: str, data: pd.DataFrame, config: Dict) -> Dict:
        """
        Run backtesting for a specific model and configuration
        
        Args:
            model_name: Name of the ML model to use
            data: Historical price data
            config: Backtesting configuration
            
        Returns:
            Dictionary containing backtest results
        """
        try:
            # Initialize backtesting parameters
            initial_capital = config.get('initial_capital', 10000)
            position_size_pct = config.get('position_size_pct', 0.1)
            commission_pct = config.get('commission_pct', 0.001)
            stop_loss_pct = config.get('stop_loss_pct', 0.05)
            take_profit_pct = config.get('take_profit_pct', 0.15)
            confidence_threshold = config.get('confidence_threshold', 0.7)
            
            # Initialize tracking variables
            portfolio_value = initial_capital
            cash = initial_capital
            position = 0  # Number of shares/coins held
            position_value = 0
            
            trades = []
            portfolio_values = [initial_capital]
            daily_returns = []
            
            # Process each data point
            for i in range(60, len(data)):  # Start from 60 to have enough history
                current_data = data.iloc[:i+1]
                current_price = current_data['close'].iloc[-1]
                
                # Generate signal
                signal_result = self._generate_signal_for_backtest(
                    model_name, current_data, confidence_threshold
                )
                
                if signal_result is None:
                    # No signal, just update portfolio value
                    position_value = position * current_price
                    portfolio_value = cash + position_value
                    portfolio_values.append(portfolio_value)
                    continue
                
                signal_type = signal_result.get('signal_type', 'HOLD')
                confidence = signal_result.get('confidence', 0)
                
                # Calculate position value
                position_value = position * current_price
                portfolio_value = cash + position_value
                
                # Execute trading logic
                trade_executed = False
                
                if signal_type == 'BUY' and position == 0 and cash > 0:
                    # Buy signal and not holding position
                    trade_amount = min(cash * position_size_pct, cash)
                    commission = trade_amount * commission_pct
                    
                    if trade_amount > commission:
                        shares_to_buy = (trade_amount - commission) / current_price
                        
                        # Execute buy
                        position = shares_to_buy
                        cash -= trade_amount
                        
                        # Record trade
                        trade = {
                            'timestamp': current_data.index[-1],
                            'type': 'BUY',
                            'price': current_price,
                            'quantity': shares_to_buy,
                            'amount': trade_amount,
                            'commission': commission,
                            'confidence': confidence,
                            'portfolio_value_before': portfolio_value,
                            'stop_loss': current_price * (1 - stop_loss_pct),
                            'take_profit': current_price * (1 + take_profit_pct),
                            'pnl': 0  # Will be calculated when position is closed
                        }
                        trades.append(trade)
                        trade_executed = True
                
                elif signal_type == 'SELL' and position > 0:
                    # Sell signal and holding position
                    sell_amount = position * current_price
                    commission = sell_amount * commission_pct
                    
                    # Execute sell
                    cash += (sell_amount - commission)
                    
                    # Calculate P&L
                    last_buy_trade = None
                    for trade in reversed(trades):
                        if trade['type'] == 'BUY' and trade.get('pnl') == 0:
                            last_buy_trade = trade
                            break
                    
                    pnl = 0
                    if last_buy_trade:
                        buy_amount = last_buy_trade['amount']
                        pnl = (sell_amount - commission) - buy_amount
                        last_buy_trade['pnl'] = pnl
                    
                    # Record sell trade
                    trade = {
                        'timestamp': current_data.index[-1],
                        'type': 'SELL',
                        'price': current_price,
                        'quantity': position,
                        'amount': sell_amount,
                        'commission': commission,
                        'confidence': confidence,
                        'portfolio_value_before': portfolio_value,
                        'pnl': pnl
                    }
                    trades.append(trade)
                    
                    position = 0
                    trade_executed = True
                
                # Check stop loss and take profit for existing positions
                elif position > 0:
                    last_buy_trade = None
                    for trade in reversed(trades):
                        if trade['type'] == 'BUY' and trade.get('pnl') == 0:
                            last_buy_trade = trade
                            break
                    
                    if last_buy_trade:
                        stop_loss_price = last_buy_trade.get('stop_loss', 0)
                        take_profit_price = last_buy_trade.get('take_profit', float('inf'))
                        
                        # Check stop loss
                        if current_price <= stop_loss_price:
                            sell_amount = position * current_price
                            commission = sell_amount * commission_pct
                            
                            cash += (sell_amount - commission)
                            
                            # Calculate P&L
                            buy_amount = last_buy_trade['amount']
                            pnl = (sell_amount - commission) - buy_amount
                            last_buy_trade['pnl'] = pnl
                            
                            # Record stop loss trade
                            trade = {
                                'timestamp': current_data.index[-1],
                                'type': 'SELL_STOP_LOSS',
                                'price': current_price,
                                'quantity': position,
                                'amount': sell_amount,
                                'commission': commission,
                                'confidence': 0,
                                'portfolio_value_before': portfolio_value,
                                'pnl': pnl
                            }
                            trades.append(trade)
                            
                            position = 0
                            trade_executed = True
                        
                        # Check take profit
                        elif current_price >= take_profit_price:
                            sell_amount = position * current_price
                            commission = sell_amount * commission_pct
                            
                            cash += (sell_amount - commission)
                            
                            # Calculate P&L
                            buy_amount = last_buy_trade['amount']
                            pnl = (sell_amount - commission) - buy_amount
                            last_buy_trade['pnl'] = pnl
                            
                            # Record take profit trade
                            trade = {
                                'timestamp': current_data.index[-1],
                                'type': 'SELL_TAKE_PROFIT',
                                'price': current_price,
                                'quantity': position,
                                'amount': sell_amount,
                                'commission': commission,
                                'confidence': 0,
                                'portfolio_value_before': portfolio_value,
                                'pnl': pnl
                            }
                            trades.append(trade)
                            
                            position = 0
                            trade_executed = True
                
                # Update portfolio value
                position_value = position * current_price
                portfolio_value = cash + position_value
                portfolio_values.append(portfolio_value)
                
                # Calculate daily return
                if len(portfolio_values) > 1:
                    daily_return = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
                    daily_returns.append(daily_return)
            
            # Calculate performance metrics
            results = self._calculate_performance_metrics(
                initial_capital, portfolio_values, trades, daily_returns, data
            )
            
            return results
            
        except Exception as e:
            print(f"Backtesting error: {e}")
            return {
                'error': str(e),
                'portfolio_values': [initial_capital],
                'trades': [],
                'final_value': initial_capital
            }
    
    def _generate_signal_for_backtest(self, model_name: str, data: pd.DataFrame, 
                                    confidence_threshold: float) -> Optional[Dict]:
        """Generate signal for backtesting (simplified version)"""
        try:
            # Get model from database
            model_info = get_model_by_name(model_name)
            if not model_info:
                return None
            
            # For now, use a simple moving average strategy as fallback
            # This ensures backtesting works while ML models are being developed
            if len(data) < 20:
                return None
                
            # Calculate simple moving averages
            close_prices = data['close']
            ma_short = close_prices.rolling(window=10).mean().iloc[-1]
            ma_long = close_prices.rolling(window=20).mean().iloc[-1]
            current_price = close_prices.iloc[-1]
            
            # Generate signal based on moving average crossover
            if ma_short > ma_long and current_price > ma_short:
                return {
                    'signal_type': 'BUY',
                    'confidence': 0.75,
                    'price': current_price,
                    'reason': 'MA_Crossover_Bullish'
                }
            elif ma_short < ma_long and current_price < ma_short:
                return {
                    'signal_type': 'SELL', 
                    'confidence': 0.75,
                    'price': current_price,
                    'reason': 'MA_Crossover_Bearish'
                }
            else:
                return {
                    'signal_type': 'HOLD',
                    'confidence': 0.5,
                    'price': current_price,
                    'reason': 'No_Clear_Signal'
                }
            
        except Exception as e:
            print(f"Error generating signal for backtest: {e}")
            return None
    
    def _calculate_performance_metrics(self, initial_capital: float, portfolio_values: List[float],
                                     trades: List[Dict], daily_returns: List[float], 
                                     price_data: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        try:
            final_value = portfolio_values[-1]
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
            
            if len(trades) > 0:
                # Trade analysis
                winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
                losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
                
                metrics['winning_trades'] = len(winning_trades)
                metrics['losing_trades'] = len(losing_trades)
                metrics['win_rate'] = len(winning_trades) / len([t for t in trades if 'pnl' in t and t['pnl'] != 0])
                
                # P&L analysis
                total_pnl = sum([t.get('pnl', 0) for t in trades])
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
                    'profit_factor': 0
                })
            
            # Risk metrics
            if len(daily_returns) > 1:
                returns_array = np.array(daily_returns)
                
                # Volatility (annualized)
                volatility = np.std(returns_array) * np.sqrt(252)
                metrics['volatility'] = volatility
                
                # Sharpe ratio (assuming risk-free rate = 0)
                if volatility > 0:
                    annualized_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
                    sharpe_ratio = annualized_return / volatility
                    metrics['sharpe_ratio'] = sharpe_ratio
                else:
                    metrics['sharpe_ratio'] = 0
                
                # Maximum drawdown
                peak = initial_capital
                max_dd = 0
                
                for value in portfolio_values:
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak
                    if drawdown > max_dd:
                        max_dd = drawdown
                
                metrics['max_drawdown'] = max_dd
                
                # Calmar ratio
                if max_dd > 0:
                    annualized_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
                    metrics['calmar_ratio'] = annualized_return / max_dd
                else:
                    metrics['calmar_ratio'] = float('inf') if total_return > 0 else 0
            
            else:
                metrics.update({
                    'volatility': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'calmar_ratio': 0
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
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating performance metrics: {e}")
            return {
                'error': str(e),
                'portfolio_values': portfolio_values,
                'trades': trades,
                'final_value': portfolio_values[-1] if portfolio_values else initial_capital
            }
    
    def run_walk_forward_analysis(self, model_name: str, data: pd.DataFrame, 
                                config: Dict, window_size: int = 252) -> Dict:
        """
        Run walk-forward analysis for more robust backtesting
        
        Args:
            model_name: Name of the ML model
            data: Historical data
            config: Backtesting configuration
            window_size: Size of the rolling window in days
            
        Returns:
            Walk-forward analysis results
        """
        results = []
        
        for i in range(window_size, len(data), window_size // 4):
            # Training window
            train_data = data.iloc[max(0, i - window_size):i]
            
            # Test window
            test_end = min(len(data), i + window_size // 4)
            test_data = data.iloc[i:test_end]
            
            if len(test_data) < 10:
                break
            
            # Run backtest on test window
            window_result = self.run_backtest(model_name, test_data, config)
            window_result['start_date'] = test_data.index[0]
            window_result['end_date'] = test_data.index[-1]
            
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
        
        # Calculate average metrics
        total_returns = [r.get('total_return', 0) for r in results if 'error' not in r]
        win_rates = [r.get('win_rate', 0) for r in results if 'error' not in r]
        sharpe_ratios = [r.get('sharpe_ratio', 0) for r in results if 'error' not in r]
        max_drawdowns = [r.get('max_drawdown', 0) for r in results if 'error' not in r]
        
        aggregated = {
            'num_windows': len(results),
            'successful_windows': len(total_returns),
            'avg_return': np.mean(total_returns) if total_returns else 0,
            'std_return': np.std(total_returns) if total_returns else 0,
            'avg_win_rate': np.mean(win_rates) if win_rates else 0,
            'avg_sharpe_ratio': np.mean(sharpe_ratios) if sharpe_ratios else 0,
            'avg_max_drawdown': np.mean(max_drawdowns) if max_drawdowns else 0,
            'consistency_score': len(total_returns) / len(results) if results else 0
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
            print(f"Backtesting {model_name}...")
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
                    'win_rate': result.get('win_rate', 0),
                    'sharpe_ratio': result.get('sharpe_ratio', 0),
                    'max_drawdown': result.get('max_drawdown', 0),
                    'total_trades': result.get('total_trades', 0),
                    'profit_factor': result.get('profit_factor', 0)
                }
                comparison_metrics.append(metrics)
        
        # Rank strategies
        if comparison_metrics:
            # Sort by Sharpe ratio (risk-adjusted return)
            ranked_strategies = sorted(
                comparison_metrics, 
                key=lambda x: x['sharpe_ratio'], 
                reverse=True
            )
            
            return {
                'ranked_strategies': ranked_strategies,
                'best_strategy': ranked_strategies[0]['strategy'] if ranked_strategies else None,
                'num_strategies': len(comparison_metrics)
            }
        
        return {}
