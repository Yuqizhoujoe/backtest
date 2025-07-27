# Phase 2: Data Layer & IBKR Integration (Week 3)

## Overview

Implement robust data fetching, validation, and storage systems with comprehensive Interactive Brokers API integration. This phase establishes the data foundation that will feed all backtesting operations.

## Objectives

- Implement Interactive Brokers API integration with ib_insync
- Create comprehensive data validation and quality assurance systems
- Build efficient data storage and retrieval mechanisms
- Develop data simulation capabilities for testing
- Establish rate limiting and error handling for API operations

## Technical Deliverables

### 1. Enhanced Project Structure

```
src/
├── data/
│   ├── __init__.py
│   ├── base_fetcher.py         # Abstract data fetcher interface
│   ├── ibkr_fetcher.py         # IBKR data implementation
│   ├── data_manager.py         # Enhanced database operations
│   ├── data_validator.py       # Data quality checks
│   ├── cache_manager.py        # Data caching system
│   ├── simulators.py           # Data simulation for testing
│   └── rate_limiter.py         # API rate limiting
├── utils/
│   ├── connection_manager.py   # IBKR connection handling
│   └── retry_handler.py        # Retry logic for API calls
└── config/
    └── ibkr_config.py          # IBKR-specific configuration

tests/
├── integration/
│   ├── test_ibkr_integration.py
│   ├── test_data_validation.py
│   └── test_cache_performance.py
└── fixtures/
    ├── mock_ibkr_data.py
    └── sample_market_data.py
```

### 2. Abstract Data Fetcher Interface (`src/data/base_fetcher.py`)

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union
from datetime import datetime, date
import pandas as pd

class BaseDataFetcher(ABC):
    """Abstract base class for data fetchers"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.is_connected = False
        
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to data source"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to data source"""
        pass
    
    @abstractmethod
    async def get_stock_data(self, symbol: str, start_date: date, 
                           end_date: date, timeframe: str = '1d') -> pd.DataFrame:
        """Fetch historical stock data"""
        pass
    
    @abstractmethod
    async def get_option_chain(self, symbol: str, expiration: date) -> List[Dict]:
        """Fetch option chain for given symbol and expiration"""
        pass
    
    @abstractmethod
    async def get_option_data(self, contract_id: str, start_date: date,
                            end_date: date) -> pd.DataFrame:
        """Fetch historical option data"""
        pass
    
    @abstractmethod
    async def get_real_time_quote(self, contract_id: str) -> Dict:
        """Get real-time quote for contract"""
        pass
    
    @abstractmethod
    async def get_contract_details(self, symbol: str) -> Dict:
        """Get contract specifications"""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_connected:
            asyncio.run(self.disconnect())
```

### 3. IBKR Data Fetcher (`src/data/ibkr_fetcher.py`)

```python
import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Union
import pandas as pd
from ib_insync import IB, Stock, Option, Contract, util
from decimal import Decimal

from .base_fetcher import BaseDataFetcher
from .rate_limiter import RateLimiter
from .data_validator import DataValidator
from ..utils.connection_manager import IBKRConnectionManager
from ..core.exceptions import DataFetchError, ConnectionError

class IBKRDataFetcher(BaseDataFetcher):
    """Interactive Brokers data fetcher implementation"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.ib = IB()
        self.connection_manager = IBKRConnectionManager(config)
        self.rate_limiter = RateLimiter(
            requests_per_second=config.get('rate_limit', 50),
            burst_limit=config.get('burst_limit', 100)
        )
        self.validator = DataValidator()
        self.logger = logging.getLogger(__name__)
        
        # IBKR Configuration
        self.host = config.get('host', '127.0.0.1')
        self.port = config.get('port', 7497)  # TWS paper trading port
        self.client_id = config.get('client_id', 1)
        self.timeout = config.get('timeout', 30)
        
        # Data settings
        self.max_days_per_request = 365  # IBKR limit
        self.retry_attempts = 3
        self.retry_delay = 1.0
        
    async def connect(self) -> bool:
        """Establish connection to IBKR TWS/Gateway"""
        try:
            await self.rate_limiter.acquire()
            
            self.ib.connect(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                timeout=self.timeout
            )
            
            # Verify connection
            if self.ib.isConnected():
                self.is_connected = True
                self.logger.info(f"Connected to IBKR at {self.host}:{self.port}")
                return True
            else:
                raise ConnectionError("Failed to establish IBKR connection")
                
        except Exception as e:
            self.logger.error(f"IBKR connection failed: {e}")
            raise ConnectionError(f"IBKR connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Close IBKR connection"""
        if self.is_connected:
            self.ib.disconnect()
            self.is_connected = False
            self.logger.info("Disconnected from IBKR")
    
    async def get_stock_data(self, symbol: str, start_date: date, 
                           end_date: date, timeframe: str = '1 day') -> pd.DataFrame:
        """Fetch historical stock data from IBKR"""
        await self.rate_limiter.acquire()
        
        try:
            # Create stock contract
            stock = Stock(symbol, 'SMART', 'USD')
            
            # Qualify contract
            contracts = self.ib.qualifyContracts(stock)
            if not contracts:
                raise DataFetchError(f"Could not qualify contract for {symbol}")
            
            qualified_contract = contracts[0]
            
            # Calculate duration string for IBKR API
            duration = self._calculate_duration(start_date, end_date)
            
            # Request historical data
            bars = self.ib.reqHistoricalData(
                contract=qualified_contract,
                endDateTime=end_date,
                durationStr=duration,
                barSizeSetting=timeframe,
                whatToShow='TRADES',
                useRTH=True,  # Regular trading hours
                formatDate=1,
                keepUpToDate=False
            )
            
            # Convert to DataFrame
            df = util.df(bars)
            
            if df.empty:
                self.logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Validate and clean data
            df = self.validator.validate_stock_data(df, symbol)
            
            self.logger.info(f"Fetched {len(df)} bars for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching stock data for {symbol}: {e}")
            raise DataFetchError(f"Failed to fetch stock data: {e}")
    
    async def get_option_chain(self, symbol: str, expiration: date) -> List[Dict]:
        """Fetch complete option chain for given expiration"""
        await self.rate_limiter.acquire()
        
        try:
            # Create stock contract for underlying
            stock = Stock(symbol, 'SMART', 'USD')
            
            # Get contract details
            stock_contracts = self.ib.qualifyContracts(stock)
            if not stock_contracts:
                raise DataFetchError(f"Could not qualify stock contract for {symbol}")
            
            # Request option chain
            chains = self.ib.reqSecDefOptParams(
                underlyingSymbol=symbol,
                futFopExchange='',
                underlyingSecType='STK',
                underlyingConId=stock_contracts[0].conId
            )
            
            if not chains:
                raise DataFetchError(f"No option chains found for {symbol}")
            
            # Find chain for specific expiration
            target_chain = None
            for chain in chains:
                if expiration in chain.expirations:
                    target_chain = chain
                    break
            
            if not target_chain:
                raise DataFetchError(f"No option chain found for {symbol} expiring {expiration}")
            
            # Build option contracts for all strikes
            option_contracts = []
            
            for strike in target_chain.strikes:
                # Create CALL option
                call_option = Option(symbol, expiration.strftime('%Y%m%d'), 
                                   strike, 'C', 'SMART')
                option_contracts.append(('CALL', call_option))
                
                # Create PUT option
                put_option = Option(symbol, expiration.strftime('%Y%m%d'), 
                                  strike, 'P', 'SMART')
                option_contracts.append(('PUT', put_option))
            
            # Qualify all contracts in batch
            all_contracts = [contract for _, contract in option_contracts]
            qualified_contracts = self.ib.qualifyContracts(*all_contracts)
            
            # Request market data for all qualified contracts
            option_data = []
            
            for i, contract in enumerate(qualified_contracts):
                if contract:
                    # Get market data
                    ticker = self.ib.reqMktData(contract, '', False, False)
                    self.ib.sleep(0.1)  # Small delay to avoid rate limiting
                    
                    option_type = option_contracts[i][0]
                    
                    option_info = {
                        'symbol': symbol,
                        'expiration': expiration,
                        'strike': contract.strike,
                        'option_type': option_type,
                        'contract': contract,
                        'bid': ticker.bid if ticker.bid != -1 else None,
                        'ask': ticker.ask if ticker.ask != -1 else None,
                        'last': ticker.last if ticker.last != -1 else None,
                        'volume': ticker.volume,
                        'open_interest': ticker.putOpenInterest if option_type == 'PUT' else ticker.callOpenInterest,
                        'implied_volatility': ticker.impliedVolatility,
                        'delta': ticker.delta,
                        'gamma': ticker.gamma,
                        'theta': ticker.theta,
                        'vega': ticker.vega
                    }
                    
                    option_data.append(option_info)
            
            # Cancel market data subscriptions
            for ticker in self.ib.tickers():
                self.ib.cancelMktData(ticker.contract)
            
            # Validate option chain data
            validated_data = self.validator.validate_option_chain(option_data, symbol, expiration)
            
            self.logger.info(f"Fetched option chain for {symbol} expiring {expiration}: {len(validated_data)} contracts")
            return validated_data
            
        except Exception as e:
            self.logger.error(f"Error fetching option chain for {symbol}: {e}")
            raise DataFetchError(f"Failed to fetch option chain: {e}")
    
    async def get_option_data(self, option_contract: Dict, start_date: date,
                            end_date: date) -> pd.DataFrame:
        """Fetch historical option data"""
        await self.rate_limiter.acquire()
        
        try:
            # Create option contract
            option = Option(
                symbol=option_contract['symbol'],
                lastTradeDateOrContractMonth=option_contract['expiration'].strftime('%Y%m%d'),
                strike=option_contract['strike'],
                right=option_contract['option_type'][0],  # C or P
                exchange='SMART'
            )
            
            # Qualify contract
            contracts = self.ib.qualifyContracts(option)
            if not contracts:
                raise DataFetchError(f"Could not qualify option contract")
            
            qualified_contract = contracts[0]
            
            # Calculate duration
            duration = self._calculate_duration(start_date, end_date)
            
            # Request historical data
            bars = self.ib.reqHistoricalData(
                contract=qualified_contract,
                endDateTime=end_date,
                durationStr=duration,
                barSizeSetting='1 day',
                whatToShow='OPTION_IMPLIED_VOLATILITY',
                useRTH=True,
                formatDate=1,
                keepUpToDate=False
            )
            
            # Convert to DataFrame
            df = util.df(bars)
            
            if df.empty:
                self.logger.warning(f"No option data returned")
                return pd.DataFrame()
            
            # Validate option data
            df = self.validator.validate_option_data(df, option_contract)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching option data: {e}")
            raise DataFetchError(f"Failed to fetch option data: {e}")
    
    async def get_real_time_quote(self, contract: Union[Stock, Option]) -> Dict:
        """Get real-time market data"""
        await self.rate_limiter.acquire()
        
        try:
            # Request market data
            ticker = self.ib.reqMktData(contract, '', False, False)
            
            # Wait for data
            timeout = 5  # seconds
            start_time = datetime.now()
            
            while (datetime.now() - start_time).seconds < timeout:
                self.ib.sleep(0.1)
                if ticker.bid != -1 or ticker.ask != -1:
                    break
            
            # Extract quote data
            quote = {
                'timestamp': datetime.now(),
                'bid': ticker.bid if ticker.bid != -1 else None,
                'ask': ticker.ask if ticker.ask != -1 else None,
                'last': ticker.last if ticker.last != -1 else None,
                'volume': ticker.volume,
                'high': ticker.high if ticker.high != -1 else None,
                'low': ticker.low if ticker.low != -1 else None
            }
            
            # Add Greeks for options
            if isinstance(contract, Option):
                quote.update({
                    'delta': ticker.delta,
                    'gamma': ticker.gamma,
                    'theta': ticker.theta,
                    'vega': ticker.vega,
                    'implied_volatility': ticker.impliedVolatility
                })
            
            # Cancel market data subscription
            self.ib.cancelMktData(contract)
            
            return quote
            
        except Exception as e:
            self.logger.error(f"Error getting real-time quote: {e}")
            raise DataFetchError(f"Failed to get real-time quote: {e}")
    
    async def get_contract_details(self, symbol: str, contract_type: str = 'STK') -> Dict:
        """Get detailed contract information"""
        await self.rate_limiter.acquire()
        
        try:
            if contract_type == 'STK':
                contract = Stock(symbol, 'SMART', 'USD')
            else:
                raise ValueError(f"Unsupported contract type: {contract_type}")
            
            # Get contract details
            details = self.ib.reqContractDetails(contract)
            
            if not details:
                raise DataFetchError(f"No contract details found for {symbol}")
            
            detail = details[0]
            
            contract_info = {
                'symbol': symbol,
                'exchange': detail.contract.primaryExchange,
                'currency': detail.contract.currency,
                'multiplier': detail.contract.multiplier,
                'trading_hours': detail.tradingHours,
                'time_zone': detail.timeZoneId,
                'min_tick': detail.minTick,
                'market_cap': detail.marketCap if hasattr(detail, 'marketCap') else None,
                'industry': detail.industry if hasattr(detail, 'industry') else None,
                'category': detail.category if hasattr(detail, 'category') else None
            }
            
            return contract_info
            
        except Exception as e:
            self.logger.error(f"Error getting contract details for {symbol}: {e}")
            raise DataFetchError(f"Failed to get contract details: {e}")
    
    def _calculate_duration(self, start_date: date, end_date: date) -> str:
        """Calculate IBKR duration string"""
        days = (end_date - start_date).days
        
        if days <= 30:
            return f"{days} D"
        elif days <= 365:
            weeks = days // 7
            return f"{weeks} W"
        else:
            years = days // 365
            return f"{years} Y"
    
    async def health_check(self) -> bool:
        """Check if IBKR connection is healthy"""
        try:
            if not self.is_connected:
                return False
            
            # Try a simple request
            contract = Stock('AAPL', 'SMART', 'USD')
            contracts = self.ib.qualifyContracts(contract)
            return len(contracts) > 0
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
```

### 4. Data Validator (`src/data/data_validator.py`)

```python
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, date
import logging
from decimal import Decimal

from ..core.exceptions import DataValidationError

class DataValidator:
    """Validates and cleans market data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Validation thresholds
        self.max_price_change = 0.5  # 50% max daily change
        self.min_volume = 0
        self.max_bid_ask_spread = 0.2  # 20% max spread
        
    def validate_stock_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate and clean stock data"""
        if df.empty:
            return df
        
        original_count = len(df)
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise DataValidationError(f"Missing required columns: {missing_columns}")
        
        # Remove rows with null prices
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Validate price relationships
        invalid_prices = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close']) |
            (df['open'] <= 0) |
            (df['close'] <= 0)
        )
        
        if invalid_prices.any():
            self.logger.warning(f"Removing {invalid_prices.sum()} rows with invalid prices for {symbol}")
            df = df[~invalid_prices]
        
        # Check for extreme price changes
        if len(df) > 1:
            df = df.sort_index()
            price_changes = df['close'].pct_change().abs()
            extreme_changes = price_changes > self.max_price_change
            
            if extreme_changes.any():
                self.logger.warning(f"Found {extreme_changes.sum()} extreme price changes for {symbol}")
                # Log but don't remove - might be valid (earnings, splits, etc.)
        
        # Validate volume
        negative_volume = df['volume'] < 0
        if negative_volume.any():
            self.logger.warning(f"Removing {negative_volume.sum()} rows with negative volume for {symbol}")
            df = df[~negative_volume]
        
        # Add derived fields
        df['mid_price'] = (df['open'] + df['close']) / 2
        df['price_range'] = df['high'] - df['low']
        df['price_range_pct'] = df['price_range'] / df['close']
        
        removed_count = original_count - len(df)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} invalid rows from {symbol} stock data")
        
        return df
    
    def validate_option_chain(self, option_data: List[Dict], symbol: str, expiration: date) -> List[Dict]:
        """Validate option chain data"""
        validated_options = []
        
        for option in option_data:
            try:
                # Validate required fields
                required_fields = ['symbol', 'expiration', 'strike', 'option_type']
                if not all(field in option for field in required_fields):
                    self.logger.warning(f"Skipping option with missing required fields: {option}")
                    continue
                
                # Validate strike price
                if option['strike'] <= 0:
                    self.logger.warning(f"Skipping option with invalid strike: {option['strike']}")
                    continue
                
                # Validate option type
                if option['option_type'] not in ['CALL', 'PUT']:
                    self.logger.warning(f"Skipping option with invalid type: {option['option_type']}")
                    continue
                
                # Validate bid/ask spread
                bid = option.get('bid')
                ask = option.get('ask')
                
                if bid is not None and ask is not None and bid > 0 and ask > 0:
                    spread_pct = (ask - bid) / ((ask + bid) / 2)
                    if spread_pct > self.max_bid_ask_spread:
                        self.logger.warning(f"Wide bid-ask spread for {symbol} {option['strike']} {option['option_type']}: {spread_pct:.2%}")
                        # Don't skip - just log warning
                
                # Validate Greeks (if present)
                if 'delta' in option and option['delta'] is not None:
                    delta = option['delta']
                    if option['option_type'] == 'CALL' and not (0 <= delta <= 1):
                        self.logger.warning(f"Invalid delta for call option: {delta}")
                    elif option['option_type'] == 'PUT' and not (-1 <= delta <= 0):
                        self.logger.warning(f"Invalid delta for put option: {delta}")
                
                validated_options.append(option)
                
            except Exception as e:
                self.logger.error(f"Error validating option {option}: {e}")
                continue
        
        self.logger.info(f"Validated {len(validated_options)} options for {symbol} expiring {expiration}")
        return validated_options
    
    def validate_option_data(self, df: pd.DataFrame, option_contract: Dict) -> pd.DataFrame:
        """Validate historical option data"""
        if df.empty:
            return df
        
        original_count = len(df)
        
        # Remove null price data
        df = df.dropna(subset=['close'])
        
        # Validate positive prices
        invalid_prices = df['close'] <= 0
        if invalid_prices.any():
            self.logger.warning(f"Removing {invalid_prices.sum()} rows with non-positive prices")
            df = df[~invalid_prices]
        
        # For options, validate that prices don't exceed intrinsic value bounds
        # This requires underlying price, which we'll implement later
        
        removed_count = original_count - len(df)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} invalid rows from option data")
        
        return df
    
    def validate_real_time_data(self, quote: Dict) -> Dict:
        """Validate real-time quote data"""
        validated_quote = {}
        
        # Copy valid fields
        for field, value in quote.items():
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                validated_quote[field] = value
        
        # Validate bid/ask relationship
        bid = validated_quote.get('bid')
        ask = validated_quote.get('ask')
        
        if bid is not None and ask is not None:
            if bid > ask:
                self.logger.warning(f"Bid ({bid}) > Ask ({ask}) - possible data error")
            elif bid <= 0 or ask <= 0:
                self.logger.warning(f"Non-positive bid ({bid}) or ask ({ask})")
        
        return validated_quote
    
    def interpolate_missing_data(self, df: pd.DataFrame, method: str = 'linear') -> pd.DataFrame:
        """Interpolate missing data points"""
        if df.empty:
            return df
        
        # Identify missing data
        missing_count = df.isnull().sum().sum()
        if missing_count == 0:
            return df
        
        self.logger.info(f"Interpolating {missing_count} missing data points using {method} method")
        
        # Interpolate numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if method == 'linear':
            df[numeric_columns] = df[numeric_columns].interpolate(method='linear')
        elif method == 'forward_fill':
            df[numeric_columns] = df[numeric_columns].fillna(method='ffill')
        elif method == 'backward_fill':
            df[numeric_columns] = df[numeric_columns].fillna(method='bfill')
        else:
            raise ValueError(f"Unsupported interpolation method: {method}")
        
        return df
    
    def detect_outliers(self, df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.Series:
        """Detect outliers in data"""
        if column not in df.columns:
            raise ValueError(f"Column {column} not found in DataFrame")
        
        series = df[column].dropna()
        
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (series < lower_bound) | (series > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((series - series.mean()) / series.std())
            outliers = z_scores > 3
            
        else:
            raise ValueError(f"Unsupported outlier detection method: {method}")
        
        return outliers
```

### 5. Rate Limiter (`src/data/rate_limiter.py`)

```python
import asyncio
import time
from typing import Optional
from collections import deque

class RateLimiter:
    """Rate limiter for API requests"""
    
    def __init__(self, requests_per_second: float = 50, burst_limit: int = 100):
        self.requests_per_second = requests_per_second
        self.burst_limit = burst_limit
        self.min_interval = 1.0 / requests_per_second
        
        # Token bucket implementation
        self.tokens = burst_limit
        self.last_update = time.time()
        
        # Request tracking
        self.request_times = deque()
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire permission to make a request"""
        async with self.lock:
            now = time.time()
            
            # Add tokens based on elapsed time
            elapsed = now - self.last_update
            self.tokens = min(self.burst_limit, 
                            self.tokens + elapsed * self.requests_per_second)
            self.last_update = now
            
            # Wait if no tokens available
            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.requests_per_second
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1
            
            # Track request time
            self.request_times.append(now)
            
            # Clean old request times (older than 1 second)
            while self.request_times and self.request_times[0] < now - 1:
                self.request_times.popleft()
    
    def get_stats(self) -> dict:
        """Get rate limiting statistics"""
        now = time.time()
        recent_requests = len([t for t in self.request_times if t > now - 1])
        
        return {
            'requests_last_second': recent_requests,
            'tokens_available': self.tokens,
            'requests_per_second_limit': self.requests_per_second,
            'burst_limit': self.burst_limit
        }
```

### 6. Cache Manager (`src/data/cache_manager.py`)

```python
import pickle
import hashlib
import os
from datetime import datetime, timedelta
from typing import Any, Optional, Dict
import pandas as pd
from pathlib import Path

class CacheManager:
    """Manages data caching for improved performance"""
    
    def __init__(self, cache_dir: str = "data/cache", default_ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl  # seconds
        
    def _get_cache_key(self, key_data: Dict) -> str:
        """Generate cache key from data"""
        key_string = str(sorted(key_data.items()))
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path"""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def get(self, key_data: Dict, ttl: Optional[int] = None) -> Optional[Any]:
        """Get data from cache"""
        cache_key = self._get_cache_key(key_data)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        # Check if cache is expired
        ttl = ttl or self.default_ttl
        cache_age = time.time() - cache_path.stat().st_mtime
        
        if cache_age > ttl:
            # Remove expired cache
            cache_path.unlink()
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            # Remove corrupted cache
            cache_path.unlink()
            return None
    
    def set(self, key_data: Dict, data: Any) -> None:
        """Store data in cache"""
        cache_key = self._get_cache_key(key_data)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Failed to cache data: {e}")
    
    def clear_expired(self) -> int:
        """Clear expired cache files"""
        removed_count = 0
        current_time = time.time()
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_age = current_time - cache_file.stat().st_mtime
            if cache_age > self.default_ttl:
                cache_file.unlink()
                removed_count += 1
        
        return removed_count
    
    def clear_all(self) -> int:
        """Clear all cache files"""
        removed_count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
            removed_count += 1
        return removed_count
```

## Testing Requirements

### Integration Tests

1. **IBKR Connection Tests** (`tests/integration/test_ibkr_integration.py`)
   - Connection establishment and cleanup
   - Error handling for connection failures
   - Rate limiting compliance
   - Data fetching end-to-end

2. **Data Validation Tests** (`tests/integration/test_data_validation.py`)
   - Real data validation scenarios
   - Edge case handling
   - Performance with large datasets

3. **Cache Performance Tests** (`tests/integration/test_cache_performance.py`)
   - Cache hit/miss scenarios
   - Memory usage optimization
   - Cache expiration handling

## Configuration

### IBKR Configuration (`config/ibkr_config.py`)

```python
IBKR_CONFIG = {
    'paper_trading': {
        'host': '127.0.0.1',
        'port': 7497,
        'client_id': 1
    },
    'live_trading': {
        'host': '127.0.0.1',
        'port': 7496,
        'client_id': 1
    },
    'rate_limits': {
        'requests_per_second': 50,
        'burst_limit': 100,
        'daily_limit': 10000
    },
    'data_settings': {
        'max_days_per_request': 365,
        'retry_attempts': 3,
        'retry_delay': 1.0,
        'timeout': 30
    }
}
```

## Performance Requirements

- **API Response Time**: <500ms for single contract requests
- **Option Chain Fetch**: <5 seconds for 100 strike prices
- **Data Validation**: <100ms for 1000 data points
- **Cache Performance**: <10ms for cache hits
- **Memory Usage**: <500MB for typical daily operations

## Error Handling

- Comprehensive retry logic with exponential backoff
- Circuit breaker pattern for persistent failures  
- Graceful degradation when IBKR is unavailable
- Data validation with configurable tolerance levels
- Detailed logging for debugging and monitoring

## Success Criteria

- [ ] Stable IBKR connection with automatic reconnection
- [ ] Option chains fetched with >95% data completeness
- [ ] Data validation removes <5% of records
- [ ] Cache system achieves >80% hit rate
- [ ] Rate limiting prevents API quota violations
- [ ] All integration tests pass consistently
- [ ] Performance benchmarks met under load