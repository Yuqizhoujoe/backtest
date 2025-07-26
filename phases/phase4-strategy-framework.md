# Phase 4: Strategy Framework (Week 5)

## Overview

Build a flexible, extensible strategy framework that supports multiple options trading strategies with configurable parameters, signal generation, and position management. This phase creates the foundation for implementing various options strategies with proper risk controls.

## Objectives

- Create abstract strategy framework with pluggable implementations
- Implement sell put and covered call strategies with realistic entry/exit logic
- Build signal generation system with market condition analysis
- Develop position sizing algorithms with risk management
- Create strategy factory for configuration-driven strategy creation
- Implement comprehensive strategy backtesting interface

## Technical Deliverables

### 1. Enhanced Project Structure

```
src/
├── strategies/
│   ├── __init__.py
│   ├── base_strategy.py          # Abstract strategy interface
│   ├── sell_put.py               # Cash-secured put strategy
│   ├── covered_call.py           # Covered call strategy
│   ├── iron_condor.py            # Iron condor strategy (future)
│   ├── strategy_factory.py       # Strategy creation factory
│   ├── signals/
│   │   ├── __init__.py
│   │   ├── technical_signals.py  # Technical analysis signals
│   │   ├── fundamental_signals.py # Fundamental analysis signals
│   │   ├── volatility_signals.py # Volatility-based signals
│   │   └── market_regime.py      # Market regime detection
│   ├── position_sizing/
│   │   ├── __init__.py
│   │   ├── kelly_criterion.py    # Kelly criterion sizing
│   │   ├── fixed_sizing.py       # Fixed position sizing
│   │   ├── volatility_sizing.py  # Volatility-based sizing
│   │   └── risk_parity.py        # Risk parity approach
│   └── utils/
│       ├── __init__.py
│       ├── strategy_utils.py     # Common strategy utilities
│       └── validation.py         # Strategy parameter validation

tests/
├── unit/
│   ├── test_strategies.py
│   ├── test_signals.py
│   ├── test_position_sizing.py
│   └── test_strategy_factory.py
└── integration/
    ├── test_strategy_execution.py
    └── test_strategy_performance.py

config/
└── strategies/
    ├── sell_put_config.yaml
    ├── covered_call_config.yaml
    └── strategy_defaults.yaml
```

### 2. Abstract Strategy Base (`src/strategies/base_strategy.py`)

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date
import logging
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum

from ..core.contracts import OptionContract, StockContract
from ..core.position import Position
from ..core.portfolio import Portfolio
from ..core.exceptions import StrategyError

class SignalType(Enum):
    """Signal types for strategy decisions"""
    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    HOLD = "hold"
    CLOSE_ALL = "close_all"

@dataclass
class StrategySignal:
    """Represents a strategy signal"""
    signal_type: SignalType
    contract: Optional[Any] = None
    quantity: Optional[int] = None
    target_price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict] = None
    timestamp: Optional[datetime] = None

@dataclass
class StrategyConfig:
    """Base configuration for strategies"""
    name: str
    max_position_size: Decimal = Decimal('0.1')  # 10% of portfolio
    max_positions: int = 10
    risk_per_trade: Decimal = Decimal('0.02')  # 2% risk per trade
    min_dte: int = 30  # Minimum days to expiration
    max_dte: int = 60  # Maximum days to expiration
    profit_target: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    commission_per_contract: Decimal = Decimal('1.0')
    
class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, config: StrategyConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Strategy state
        self.is_active = True
        self.current_positions: List[Position] = []
        self.signals_history: List[StrategySignal] = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = Decimal('0')
        self.max_drawdown = Decimal('0')
        
        # Validate configuration
        self._validate_config()
    
    @abstractmethod
    def generate_signals(self, market_data: Dict, portfolio: Portfolio, 
                        current_time: datetime) -> List[StrategySignal]:
        """
        Generate trading signals based on market conditions
        
        Args:
            market_data: Current market data including prices, Greeks, etc.
            portfolio: Current portfolio state
            current_time: Current timestamp
            
        Returns:
            List of strategy signals
        """
        pass
    
    @abstractmethod
    def validate_signal(self, signal: StrategySignal, portfolio: Portfolio,
                       market_data: Dict) -> bool:
        """
        Validate if a signal can be executed given current conditions
        
        Args:
            signal: Strategy signal to validate
            portfolio: Current portfolio state
            market_data: Current market data
            
        Returns:
            True if signal is valid and can be executed
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: StrategySignal, portfolio: Portfolio,
                               market_data: Dict) -> int:
        """
        Calculate appropriate position size for a signal
        
        Args:
            signal: Strategy signal
            portfolio: Current portfolio state
            market_data: Current market data
            
        Returns:
            Position size (number of contracts)
        """
        pass
    
    @abstractmethod
    def should_exit_position(self, position: Position, market_data: Dict,
                           current_time: datetime) -> bool:
        """
        Determine if an existing position should be closed
        
        Args:
            position: Current position
            market_data: Current market data
            current_time: Current timestamp
            
        Returns:
            True if position should be closed
        """
        pass
    
    def update_positions(self, portfolio: Portfolio) -> None:
        """Update internal position tracking from portfolio"""
        # Filter positions for this strategy's contracts
        strategy_positions = []
        for position in portfolio.positions.values():
            if self._is_strategy_position(position):
                strategy_positions.append(position)
        
        self.current_positions = strategy_positions
    
    def get_strategy_metrics(self) -> Dict:
        """Calculate and return strategy performance metrics"""
        win_rate = (self.winning_trades / self.total_trades * 100 
                   if self.total_trades > 0 else 0)
        
        avg_pnl_per_trade = (self.total_pnl / self.total_trades 
                           if self.total_trades > 0 else Decimal('0'))
        
        return {
            'name': self.config.name,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'avg_pnl_per_trade': avg_pnl_per_trade,
            'max_drawdown': self.max_drawdown,
            'current_positions': len(self.current_positions),
            'is_active': self.is_active
        }
    
    def log_signal(self, signal: StrategySignal) -> None:
        """Log a strategy signal for analysis"""
        signal.timestamp = datetime.now()
        self.signals_history.append(signal)
        
        self.logger.info(f"Generated signal: {signal.signal_type.value} "
                        f"for {signal.contract} at {signal.timestamp}")
    
    def _validate_config(self) -> None:
        """Validate strategy configuration"""
        if self.config.max_position_size <= 0 or self.config.max_position_size > 1:
            raise StrategyError("max_position_size must be between 0 and 1")
        
        if self.config.max_positions <= 0:
            raise StrategyError("max_positions must be positive")
        
        if self.config.risk_per_trade <= 0 or self.config.risk_per_trade > 0.1:
            raise StrategyError("risk_per_trade must be between 0 and 0.1")
        
        if self.config.min_dte < 0 or self.config.max_dte < self.config.min_dte:
            raise StrategyError("Invalid DTE configuration")
    
    def _is_strategy_position(self, position: Position) -> bool:
        """Check if a position belongs to this strategy"""
        # This would be implemented based on position metadata
        # For now, assume all option positions belong to strategy
        return isinstance(position.contract, OptionContract)
    
    def get_required_margin(self, signal: StrategySignal, market_data: Dict) -> Decimal:
        """Calculate required margin for a signal"""
        # Basic implementation - should be overridden by specific strategies
        if isinstance(signal.contract, OptionContract):
            underlying_price = market_data.get('underlying_price', Decimal('100'))
            return underlying_price * Decimal('0.2')  # 20% of underlying value
        return Decimal('0')
    
    def check_risk_limits(self, signal: StrategySignal, portfolio: Portfolio) -> bool:
        """Check if signal violates risk limits"""
        # Check position count limit
        if len(self.current_positions) >= self.config.max_positions:
            return False
        
        # Check portfolio allocation limit
        signal_value = self.get_required_margin(signal, {})
        portfolio_value = portfolio.total_value
        
        if signal_value / portfolio_value > self.config.max_position_size:
            return False
        
        return True
```

### 3. Cash-Secured Put Strategy (`src/strategies/sell_put.py`)

```python
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from decimal import Decimal
import numpy as np

from .base_strategy import BaseStrategy, StrategySignal, SignalType, StrategyConfig
from ..core.contracts import OptionContract, StockContract
from ..core.position import Position
from ..core.portfolio import Portfolio
from ..pricing.black_scholes import BlackScholesCalculator
from ..strategies.signals.volatility_signals import VolatilitySignalGenerator
from ..strategies.signals.technical_signals import TechnicalSignalGenerator

class SellPutConfig(StrategyConfig):
    """Configuration for sell put strategy"""
    target_delta: Decimal = Decimal('0.3')  # Target delta for puts
    delta_tolerance: Decimal = Decimal('0.05')  # Allowable delta range
    min_premium: Decimal = Decimal('0.5')  # Minimum premium to collect
    max_strikes_otm: int = 5  # Maximum strikes out of the money
    profit_target_pct: Decimal = Decimal('0.5')  # 50% profit target
    dte_exit: int = 7  # Exit when DTE reaches this value
    avoid_earnings: bool = True  # Avoid positions over earnings
    min_volume: int = 100  # Minimum option volume
    min_open_interest: int = 50  # Minimum open interest
    
class SellPutStrategy(BaseStrategy):
    """Cash-secured put selling strategy"""
    
    def __init__(self, config: SellPutConfig):
        super().__init__(config)
        self.config: SellPutConfig = config
        self.bs_calculator = BlackScholesCalculator()
        self.vol_signals = VolatilitySignalGenerator()
        self.tech_signals = TechnicalSignalGenerator()
        
    def generate_signals(self, market_data: Dict, portfolio: Portfolio, 
                        current_time: datetime) -> List[StrategySignal]:
        """Generate sell put signals"""
        signals = []
        
        # Get available symbols from market data
        symbols = market_data.get('symbols', [])
        
        for symbol in symbols:
            symbol_data = market_data.get(symbol, {})
            
            # Check if we should generate signals for this symbol
            if not self._should_trade_symbol(symbol, symbol_data, current_time):
                continue
            
            # Get option chain
            option_chain = symbol_data.get('option_chain', {})
            underlying_price = symbol_data.get('underlying_price')
            
            if not option_chain or not underlying_price:
                continue
            
            # Find suitable put options
            put_signals = self._find_put_opportunities(
                symbol, option_chain, underlying_price, symbol_data
            )
            
            signals.extend(put_signals)
        
        return signals
    
    def _should_trade_symbol(self, symbol: str, symbol_data: Dict, 
                           current_time: datetime) -> bool:
        """Determine if we should consider trading this symbol"""
        # Check if we already have a position in this symbol
        for position in self.current_positions:
            if (isinstance(position.contract, OptionContract) and 
                position.contract.symbol == symbol):
                return False  # Don't add new positions in same symbol
        
        # Check volatility conditions
        vol_signal = self.vol_signals.analyze_volatility(symbol_data)
        if vol_signal.get('regime') == 'high_vol':
            return True  # Good time to sell puts in high vol
        
        # Check technical conditions
        tech_signal = self.tech_signals.analyze_trend(symbol_data)
        if tech_signal.get('trend') in ['bullish', 'neutral']:
            return True  # Sell puts in bullish/neutral markets
        
        return False
    
    def _find_put_opportunities(self, symbol: str, option_chain: Dict,
                              underlying_price: Decimal, symbol_data: Dict) -> List[StrategySignal]:
        """Find suitable put options to sell"""
        signals = []
        
        # Look through expirations
        for expiration_date, chain_data in option_chain.items():
            # Check DTE
            dte = (expiration_date - datetime.now().date()).days
            if dte < self.config.min_dte or dte > self.config.max_dte:
                continue
            
            # Avoid earnings if configured
            if self.config.avoid_earnings:
                earnings_date = symbol_data.get('next_earnings')
                if earnings_date and earnings_date <= expiration_date:
                    continue
            
            # Find puts with target delta
            puts = [opt for opt in chain_data if opt['option_type'] == 'PUT']
            
            for put_option in puts:
                if self._is_suitable_put(put_option, underlying_price, symbol_data):
                    signal = self._create_put_signal(put_option, symbol_data)
                    signals.append(signal)
        
        # Sort by premium collected (descending)
        signals.sort(key=lambda s: s.metadata.get('premium', 0), reverse=True)
        
        # Return top signal
        return signals[:1] if signals else []
    
    def _is_suitable_put(self, put_option: Dict, underlying_price: Decimal,
                        symbol_data: Dict) -> bool:
        """Check if put option meets strategy criteria"""
        # Check delta
        delta = abs(put_option.get('delta', 0))
        target_delta = float(self.config.target_delta)
        delta_tolerance = float(self.config.delta_tolerance)
        
        if not (target_delta - delta_tolerance <= delta <= target_delta + delta_tolerance):
            return False
        
        # Check premium
        bid = put_option.get('bid', 0)
        if bid < float(self.config.min_premium):
            return False
        
        # Check volume and open interest
        volume = put_option.get('volume', 0)
        open_interest = put_option.get('open_interest', 0)
        
        if volume < self.config.min_volume or open_interest < self.config.min_open_interest:
            return False
        
        # Check if strike is reasonable distance from current price
        strike = put_option['strike']
        otm_amount = float(underlying_price - strike) / float(underlying_price)
        
        if otm_amount < 0 or otm_amount > 0.2:  # More than 20% OTM
            return False
        
        return True
    
    def _create_put_signal(self, put_option: Dict, symbol_data: Dict) -> StrategySignal:
        """Create a sell put signal"""
        # Create option contract
        contract = OptionContract(
            symbol=put_option['symbol'],
            strike=put_option['strike'],
            expiration=put_option['expiration'],
            option_type='PUT',
            exchange='SMART',
            bid=put_option.get('bid'),
            ask=put_option.get('ask'),
            delta=put_option.get('delta'),
            gamma=put_option.get('gamma'),
            theta=put_option.get('theta'),
            vega=put_option.get('vega'),
            implied_volatility=put_option.get('implied_volatility')
        )
        
        # Calculate target prices
        premium = put_option.get('bid', 0)  # Sell at bid
        profit_target = premium * (1 - float(self.config.profit_target_pct))
        
        return StrategySignal(
            signal_type=SignalType.ENTER_SHORT,
            contract=contract,
            target_price=Decimal(str(premium)),
            take_profit=Decimal(str(profit_target)),
            confidence=0.8,  # Base confidence
            metadata={
                'strategy': 'sell_put',
                'premium': premium,
                'delta': put_option.get('delta'),
                'dte': (put_option['expiration'] - datetime.now().date()).days,
                'underlying_price': symbol_data.get('underlying_price')
            }
        )
    
    def validate_signal(self, signal: StrategySignal, portfolio: Portfolio,
                       market_data: Dict) -> bool:
        """Validate sell put signal"""
        if signal.signal_type != SignalType.ENTER_SHORT:
            return False
        
        if not isinstance(signal.contract, OptionContract):
            return False
        
        # Check if we have enough cash for cash-secured put
        required_cash = signal.contract.strike * signal.contract.multiplier
        
        # Account for quantity if calculated
        if hasattr(signal, 'quantity') and signal.quantity:
            required_cash *= signal.quantity
        
        if portfolio.cash < required_cash:
            return False
        
        # Check risk limits
        if not self.check_risk_limits(signal, portfolio):
            return False
        
        return True
    
    def calculate_position_size(self, signal: StrategySignal, portfolio: Portfolio,
                               market_data: Dict) -> int:
        """Calculate position size for sell put"""
        # For cash-secured puts, we're limited by available cash
        contract_value = signal.contract.strike * signal.contract.multiplier
        max_contracts_by_cash = int(portfolio.cash // contract_value)
        
        # Apply position size limit
        max_position_value = portfolio.total_value * self.config.max_position_size
        max_contracts_by_size = int(max_position_value // contract_value)
        
        # Apply risk per trade limit
        risk_amount = portfolio.total_value * self.config.risk_per_trade
        # For puts, maximum risk is the strike price minus premium
        premium = signal.metadata.get('premium', 0)
        max_risk_per_contract = float(signal.contract.strike) - premium
        max_contracts_by_risk = int(risk_amount / max_risk_per_contract) if max_risk_per_contract > 0 else 0
        
        # Take minimum of all constraints
        position_size = min(max_contracts_by_cash, max_contracts_by_size, max_contracts_by_risk)
        
        return max(1, position_size)  # At least 1 contract
    
    def should_exit_position(self, position: Position, market_data: Dict,
                           current_time: datetime) -> bool:
        """Determine if put position should be closed"""
        if not isinstance(position.contract, OptionContract):
            return False
        
        contract = position.contract
        
        # Check DTE
        dte = (contract.expiration - current_time.date()).days
        if dte <= self.config.dte_exit:
            return True
        
        # Check profit target
        current_price = market_data.get(contract.contract_id, {}).get('bid', 0)
        if current_price <= 0:
            return False
        
        entry_price = position.avg_price
        profit_pct = (entry_price - current_price) / entry_price
        
        if profit_pct >= float(self.config.profit_target_pct):
            return True
        
        # Check if assignment risk is high
        underlying_price = market_data.get(contract.symbol, {}).get('underlying_price')
        if underlying_price and underlying_price <= contract.strike * Decimal('1.05'):
            # Underlying within 5% of strike - consider closing
            return True
        
        return False
    
    def get_required_margin(self, signal: StrategySignal, market_data: Dict) -> Decimal:
        """Calculate required margin for cash-secured put"""
        # Cash-secured puts require 100% cash collateral
        return signal.contract.strike * signal.contract.multiplier
```

### 4. Covered Call Strategy (`src/strategies/covered_call.py`)

```python
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from decimal import Decimal

from .base_strategy import BaseStrategy, StrategySignal, SignalType, StrategyConfig
from ..core.contracts import OptionContract, StockContract
from ..core.position import Position
from ..core.portfolio import Portfolio

class CoveredCallConfig(StrategyConfig):
    """Configuration for covered call strategy"""
    target_delta: Decimal = Decimal('0.3')  # Target delta for calls
    delta_tolerance: Decimal = Decimal('0.05')
    min_premium: Decimal = Decimal('1.0')  # Minimum premium to collect
    profit_target_pct: Decimal = Decimal('0.5')  # 50% profit target
    roll_dte: int = 21  # Roll calls when DTE reaches this value
    roll_delta_threshold: Decimal = Decimal('0.5')  # Roll when delta exceeds this
    max_roll_attempts: int = 2  # Maximum times to roll same call
    
class CoveredCallStrategy(BaseStrategy):
    """Covered call strategy for stock positions"""
    
    def __init__(self, config: CoveredCallConfig):
        super().__init__(config)
        self.config: CoveredCallConfig = config
        
    def generate_signals(self, market_data: Dict, portfolio: Portfolio, 
                        current_time: datetime) -> List[StrategySignal]:
        """Generate covered call signals"""
        signals = []
        
        # Find stock positions without covered calls
        stock_positions = self._get_uncovered_stock_positions(portfolio)
        
        for position in stock_positions:
            symbol = position.contract.symbol
            symbol_data = market_data.get(symbol, {})
            
            if not symbol_data:
                continue
            
            # Check if we should sell a call on this stock
            call_signal = self._generate_call_signal(position, symbol_data, current_time)
            if call_signal:
                signals.append(call_signal)
        
        return signals
    
    def _get_uncovered_stock_positions(self, portfolio: Portfolio) -> List[Position]:
        """Find stock positions that don't have covered calls"""
        stock_positions = []
        covered_symbols = set()
        
        # First, identify symbols with existing call positions
        for position in portfolio.positions.values():
            if (isinstance(position.contract, OptionContract) and 
                position.contract.option_type.value == 'CALL' and
                position.is_short):
                covered_symbols.add(position.contract.symbol)
        
        # Find stock positions not in covered set
        for position in portfolio.positions.values():
            if (isinstance(position.contract, StockContract) and
                position.is_long and
                position.contract.symbol not in covered_symbols):
                
                # Check if we have enough shares (minimum 100 for 1 call)
                if position.quantity >= 100:
                    stock_positions.append(position)
        
        return stock_positions
    
    def _generate_call_signal(self, stock_position: Position, symbol_data: Dict,
                            current_time: datetime) -> Optional[StrategySignal]:
        """Generate call signal for stock position"""
        option_chain = symbol_data.get('option_chain', {})
        underlying_price = symbol_data.get('underlying_price')
        
        if not option_chain or not underlying_price:
            return None
        
        # Find suitable call options
        best_call = None
        best_premium = Decimal('0')
        
        for expiration_date, chain_data in option_chain.items():
            # Check DTE
            dte = (expiration_date - current_time.date()).days
            if dte < self.config.min_dte or dte > self.config.max_dte:
                continue
            
            # Find calls with target delta
            calls = [opt for opt in chain_data if opt['option_type'] == 'CALL']
            
            for call_option in calls:
                if self._is_suitable_call(call_option, underlying_price):
                    premium = call_option.get('bid', 0)
                    if premium > best_premium:
                        best_premium = Decimal(str(premium))
                        best_call = call_option
        
        if best_call:
            return self._create_call_signal(best_call, stock_position, symbol_data)
        
        return None
    
    def _is_suitable_call(self, call_option: Dict, underlying_price: Decimal) -> bool:
        """Check if call option meets strategy criteria"""
        # Check delta
        delta = call_option.get('delta', 0)
        target_delta = float(self.config.target_delta)
        delta_tolerance = float(self.config.delta_tolerance)
        
        if not (target_delta - delta_tolerance <= delta <= target_delta + delta_tolerance):
            return False
        
        # Check premium
        bid = call_option.get('bid', 0)
        if bid < float(self.config.min_premium):
            return False
        
        # Check if strike is above current price (OTM)
        strike = call_option['strike']
        if strike <= underlying_price:
            return False
        
        return True
    
    def _create_call_signal(self, call_option: Dict, stock_position: Position,
                          symbol_data: Dict) -> StrategySignal:
        """Create a sell call signal"""
        # Create option contract
        contract = OptionContract(
            symbol=call_option['symbol'],
            strike=call_option['strike'],
            expiration=call_option['expiration'],
            option_type='CALL',
            exchange='SMART',
            bid=call_option.get('bid'),
            ask=call_option.get('ask'),
            delta=call_option.get('delta'),
            gamma=call_option.get('gamma'),
            theta=call_option.get('theta'),
            vega=call_option.get('vega'),
            implied_volatility=call_option.get('implied_volatility')
        )
        
        # Calculate quantity based on stock position
        max_calls = stock_position.quantity // 100
        
        # Calculate target prices
        premium = call_option.get('bid', 0)
        profit_target = premium * (1 - float(self.config.profit_target_pct))
        
        return StrategySignal(
            signal_type=SignalType.ENTER_SHORT,
            contract=contract,
            quantity=max_calls,
            target_price=Decimal(str(premium)),
            take_profit=Decimal(str(profit_target)),
            confidence=0.7,
            metadata={
                'strategy': 'covered_call',
                'premium': premium,
                'delta': call_option.get('delta'),
                'dte': (call_option['expiration'] - datetime.now().date()).days,
                'underlying_price': symbol_data.get('underlying_price'),
                'stock_position_id': id(stock_position)
            }
        )
    
    def validate_signal(self, signal: StrategySignal, portfolio: Portfolio,
                       market_data: Dict) -> bool:
        """Validate covered call signal"""
        if signal.signal_type != SignalType.ENTER_SHORT:
            return False
        
        if not isinstance(signal.contract, OptionContract):
            return False
        
        # Check if we have enough stock shares
        symbol = signal.contract.symbol
        required_shares = signal.quantity * 100
        
        stock_position = None
        for position in portfolio.positions.values():
            if (isinstance(position.contract, StockContract) and
                position.contract.symbol == symbol and
                position.is_long):
                stock_position = position
                break
        
        if not stock_position or stock_position.quantity < required_shares:
            return False
        
        return True
    
    def calculate_position_size(self, signal: StrategySignal, portfolio: Portfolio,
                               market_data: Dict) -> int:
        """Calculate position size for covered call"""
        # Position size already calculated in signal generation
        return signal.quantity or 1
    
    def should_exit_position(self, position: Position, market_data: Dict,
                           current_time: datetime) -> bool:
        """Determine if call position should be closed or rolled"""
        if not isinstance(position.contract, OptionContract):
            return False
        
        contract = position.contract
        
        # Check DTE for rolling
        dte = (contract.expiration - current_time.date()).days
        if dte <= self.config.roll_dte:
            return True
        
        # Check if call is deep ITM (high assignment risk)
        current_delta = market_data.get(contract.contract_id, {}).get('delta', 0)
        if current_delta >= float(self.config.roll_delta_threshold):
            return True
        
        # Check profit target
        current_price = market_data.get(contract.contract_id, {}).get('bid', 0)
        if current_price <= 0:
            return False
        
        entry_price = position.avg_price
        profit_pct = (entry_price - current_price) / entry_price
        
        if profit_pct >= float(self.config.profit_target_pct):
            return True
        
        return False
    
    def get_required_margin(self, signal: StrategySignal, market_data: Dict) -> Decimal:
        """Calculate required margin for covered call"""
        # Covered calls don't require additional margin if we own the stock
        return Decimal('0')
```

### 5. Strategy Factory (`src/strategies/strategy_factory.py`)

```python
from typing import Dict, Type, Any, Optional
import yaml
from pathlib import Path

from .base_strategy import BaseStrategy, StrategyConfig
from .sell_put import SellPutStrategy, SellPutConfig
from .covered_call import CoveredCallStrategy, CoveredCallConfig
from ..core.exceptions import StrategyError

class StrategyFactory:
    """Factory for creating strategy instances from configuration"""
    
    # Registry of available strategies
    STRATEGY_REGISTRY: Dict[str, Dict[str, Type]] = {
        'sell_put': {
            'strategy_class': SellPutStrategy,
            'config_class': SellPutConfig
        },
        'covered_call': {
            'strategy_class': CoveredCallStrategy,
            'config_class': CoveredCallConfig
        }
    }
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else Path("config/strategies")
    
    def create_strategy(self, strategy_name: str, 
                       config_override: Optional[Dict] = None) -> BaseStrategy:
        """
        Create strategy instance from name and optional config override
        
        Args:
            strategy_name: Name of strategy to create
            config_override: Optional config parameters to override defaults
            
        Returns:
            Configured strategy instance
        """
        if strategy_name not in self.STRATEGY_REGISTRY:
            available_strategies = list(self.STRATEGY_REGISTRY.keys())
            raise StrategyError(f"Unknown strategy: {strategy_name}. "
                              f"Available strategies: {available_strategies}")
        
        strategy_info = self.STRATEGY_REGISTRY[strategy_name]
        strategy_class = strategy_info['strategy_class']
        config_class = strategy_info['config_class']
        
        # Load default configuration
        config_data = self._load_strategy_config(strategy_name)
        
        # Apply overrides
        if config_override:
            config_data.update(config_override)
        
        # Create config instance
        try:
            config = config_class(**config_data)
        except TypeError as e:
            raise StrategyError(f"Invalid configuration for {strategy_name}: {e}")
        
        # Create and return strategy instance
        return strategy_class(config)
    
    def create_strategy_from_config_file(self, config_file_path: str) -> BaseStrategy:
        """Create strategy from configuration file"""
        config_path = Path(config_file_path)
        
        if not config_path.exists():
            raise StrategyError(f"Config file not found: {config_file_path}")
        
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
        
        strategy_name = full_config.get('strategy_type')
        if not strategy_name:
            raise StrategyError("Configuration must specify 'strategy_type'")
        
        strategy_config = full_config.get('config', {})
        
        return self.create_strategy(strategy_name, strategy_config)
    
    def _load_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """Load default configuration for strategy"""
        config_file = self.config_dir / f"{strategy_name}_config.yaml"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f) or {}
        else:
            # Return minimal default config
            return {
                'name': strategy_name,
                'max_position_size': 0.1,
                'max_positions': 10,
                'risk_per_trade': 0.02
            }
    
    def list_available_strategies(self) -> List[str]:
        """Get list of available strategy names"""
        return list(self.STRATEGY_REGISTRY.keys())
    
    def get_strategy_config_template(self, strategy_name: str) -> Dict[str, Any]:
        """Get configuration template for strategy"""
        if strategy_name not in self.STRATEGY_REGISTRY:
            raise StrategyError(f"Unknown strategy: {strategy_name}")
        
        config_class = self.STRATEGY_REGISTRY[strategy_name]['config_class']
        
        # Create instance with defaults to get template
        default_config = config_class(name=strategy_name)
        
        # Convert to dictionary
        template = {}
        for field_name in dir(default_config):
            if not field_name.startswith('_'):
                value = getattr(default_config, field_name)
                if not callable(value):
                    template[field_name] = value
        
        return template
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type[BaseStrategy],
                         config_class: Type[StrategyConfig]) -> None:
        """Register a new strategy type"""
        cls.STRATEGY_REGISTRY[name] = {
            'strategy_class': strategy_class,
            'config_class': config_class
        }
```

## Testing Requirements

### Unit Tests

1. **Strategy Tests** (`tests/unit/test_strategies.py`)
   - Signal generation logic
   - Position sizing calculations
   - Exit condition evaluation
   - Risk limit validation

2. **Signal Tests** (`tests/unit/test_signals.py`)
   - Technical indicator calculations
   - Volatility analysis
   - Market regime detection

3. **Strategy Factory Tests** (`tests/unit/test_strategy_factory.py`)
   - Strategy creation from config
   - Configuration validation
   - Strategy registration

### Integration Tests

1. **Strategy Execution Tests** (`tests/integration/test_strategy_execution.py`)
   - End-to-end signal generation and execution
   - Multi-strategy portfolio management
   - Performance tracking

## Configuration Files

### Sell Put Configuration (`config/strategies/sell_put_config.yaml`)

```yaml
name: "sell_put"
max_position_size: 0.1
max_positions: 5
risk_per_trade: 0.02
min_dte: 30
max_dte: 45
target_delta: 0.3
delta_tolerance: 0.05
min_premium: 0.5
profit_target_pct: 0.5
dte_exit: 7
avoid_earnings: true
min_volume: 100
min_open_interest: 50
commission_per_contract: 1.0
```

### Covered Call Configuration (`config/strategies/covered_call_config.yaml`)

```yaml
name: "covered_call"
max_position_size: 1.0  # Can use all stock positions
max_positions: 10
risk_per_trade: 0.05
min_dte: 30
max_dte: 45
target_delta: 0.3
delta_tolerance: 0.05
min_premium: 1.0
profit_target_pct: 0.5
roll_dte: 21
roll_delta_threshold: 0.5
max_roll_attempts: 2
commission_per_contract: 1.0
```

## Performance Requirements

- **Signal Generation**: <100ms per symbol
- **Strategy Validation**: <10ms per signal
- **Position Sizing**: <5ms per calculation
- **Portfolio Greeks**: <50ms for 100 positions
- **Strategy Factory**: <50ms for strategy creation

## Success Criteria

- [ ] Flexible strategy framework supporting multiple strategies
- [ ] Realistic sell put and covered call implementations
- [ ] Comprehensive signal generation with market analysis
- [ ] Risk-based position sizing algorithms
- [ ] Configuration-driven strategy creation
- [ ] Comprehensive testing with >90% coverage
- [ ] Performance benchmarks met under realistic loads