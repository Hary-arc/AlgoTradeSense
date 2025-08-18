import os
import time
import hmac
import hashlib
import requests
from datetime import datetime
import websocket
import json
import threading

class BinanceClient:
    """
    Binance API client for market data and trading operations
    """
    
    def __init__(self, api_key, api_secret, testnet=True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        if testnet:
            self.base_url = "https://testnet.binance.vision"
            self.ws_base_url = "wss://testnet.binance.vision/ws"
        else:
            self.base_url = "https://api.binance.com"
            self.ws_base_url = "wss://stream.binance.com:9443/ws"
        
        self.session = requests.Session()
        self.session.headers.update({
            'X-MBX-APIKEY': self.api_key
        })
    
    def _generate_signature(self, data):
        """Generate HMAC SHA256 signature"""
        return hmac.new(
            self.api_secret.encode('utf-8'),
            data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _make_request(self, method, endpoint, params=None, signed=False):
        """Make API request to Binance"""
        url = f"{self.base_url}{endpoint}"
        
        if params is None:
            params = {}
        
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            params['signature'] = self._generate_signature(query_string)
        
        try:
            if method == 'GET':
                response = self.session.get(url, params=params, timeout=10)
            elif method == 'POST':
                response = self.session.post(url, params=params, timeout=10)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return None
        except Exception as e:
            print(f"Error in API request: {e}")
            return None
    
    def test_connection(self):
        """Test connection to Binance API"""
        try:
            result = self._make_request('GET', '/api/v3/ping')
            return result is not None
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
    
    def get_server_time(self):
        """Get server time"""
        return self._make_request('GET', '/api/v3/time')
    
    def get_exchange_info(self):
        """Get exchange information"""
        return self._make_request('GET', '/api/v3/exchangeInfo')
    
    def get_symbol_price(self, symbol):
        """Get current price for a symbol"""
        params = {'symbol': symbol}
        return self._make_request('GET', '/api/v3/ticker/price', params)
    
    def get_24hr_ticker(self, symbol=None):
        """Get 24hr ticker statistics"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        return self._make_request('GET', '/api/v3/ticker/24hr', params)
    
    def get_klines(self, symbol, interval, limit=500, start_time=None, end_time=None):
        """
        Get kline/candlestick data
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            limit: Number of klines to return (max 1000)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': min(limit, 1000)
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        
        return self._make_request('GET', '/api/v3/klines', params)
    
    def get_orderbook(self, symbol, limit=100):
        """Get order book depth"""
        params = {
            'symbol': symbol,
            'limit': limit
        }
        return self._make_request('GET', '/api/v3/depth', params)
    
    def get_recent_trades(self, symbol, limit=500):
        """Get recent trades"""
        params = {
            'symbol': symbol,
            'limit': limit
        }
        return self._make_request('GET', '/api/v3/trades', params)
    
    def get_account_info(self):
        """Get account information"""
        return self._make_request('GET', '/api/v3/account', signed=True)
    
    def get_open_orders(self, symbol=None):
        """Get open orders"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        return self._make_request('GET', '/api/v3/openOrders', params, signed=True)
    
    def get_all_orders(self, symbol, limit=500):
        """Get all orders for a symbol"""
        params = {
            'symbol': symbol,
            'limit': limit
        }
        return self._make_request('GET', '/api/v3/allOrders', params, signed=True)
    
    def place_order(self, symbol, side, order_type, quantity, price=None, time_in_force='GTC'):
        """
        Place a new order
        
        Args:
            symbol: Trading pair
            side: 'BUY' or 'SELL'
            order_type: 'LIMIT', 'MARKET', 'STOP_LOSS', etc.
            quantity: Order quantity
            price: Order price (required for LIMIT orders)
            time_in_force: Time in force ('GTC', 'IOC', 'FOK')
        """
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': quantity
        }
        
        if order_type == 'LIMIT':
            if price is None:
                raise ValueError("Price is required for LIMIT orders")
            params['price'] = price
            params['timeInForce'] = time_in_force
        
        return self._make_request('POST', '/api/v3/order', params, signed=True)
    
    def cancel_order(self, symbol, order_id):
        """Cancel an active order"""
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
        return self._make_request('DELETE', '/api/v3/order', params, signed=True)
    
    def get_trade_history(self, symbol, limit=500):
        """Get trade history"""
        params = {
            'symbol': symbol,
            'limit': limit
        }
        return self._make_request('GET', '/api/v3/myTrades', params, signed=True)

class BinanceWebSocket:
    """
    Binance WebSocket client for real-time data
    """
    
    def __init__(self, testnet=True):
        self.testnet = testnet
        self.ws = None
        self.callbacks = {}
        
        if testnet:
            self.ws_base_url = "wss://testnet.binance.vision/ws"
        else:
            self.ws_base_url = "wss://stream.binance.com:9443/ws"
    
    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            if 'stream' in data:
                stream = data['stream']
                if stream in self.callbacks:
                    self.callbacks[stream](data['data'])
            
        except Exception as e:
            print(f"Error processing WebSocket message: {e}")
    
    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        print(f"WebSocket error: {error}")
    
    def on_close(self, ws):
        """Handle WebSocket close"""
        print("WebSocket connection closed")
    
    def subscribe_kline(self, symbol, interval, callback):
        """Subscribe to kline/candlestick streams"""
        stream = f"{symbol.lower()}@kline_{interval}"
        self.callbacks[stream] = callback
        
        ws_url = f"{self.ws_base_url}/{stream}"
        
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        # Run in a separate thread
        ws_thread = threading.Thread(target=self.ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
        
        return ws_thread
    
    def subscribe_ticker(self, symbol, callback):
        """Subscribe to 24hr ticker streams"""
        stream = f"{symbol.lower()}@ticker"
        self.callbacks[stream] = callback
        
        ws_url = f"{self.ws_base_url}/{stream}"
        
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        ws_thread = threading.Thread(target=self.ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
        
        return ws_thread
    
    def close(self):
        """Close WebSocket connection"""
        if self.ws:
            self.ws.close()
