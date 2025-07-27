"""
Core contract definitions for options and stock instruments.

This module defines the fundamental data structures for representing
financial contracts in the backtesting system, including options and stocks.
"""

import datetime
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any

from .exceptions import InvalidContractError, ExpiredContractError, ValidationError


class OptionType(Enum):
    """Enumeration for option types."""
    CALL = "CALL"
    PUT = "PUT"


class ContractStatus(Enum):
    """Enumeration for contract status."""
    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"
    DELISTED = "DELISTED"


@dataclass
class OptionContract:
    """
    Represents an options contract with all relevant attributes.
    
    This class encapsulates all information needed to identify and price
    an options contract, including market data and Greeks.
    """
    
    # Contract Identification
    symbol: str                    # Underlying symbol (e.g., "AAPL")
    strike: Decimal               # Strike price
    expiration: datetime.date     # Expiration date
    option_type: OptionType       # PUT or CALL
    exchange: str = "SMART"       # Exchange
    
    # Contract Specifications
    multiplier: int = 100         # Standard option multiplier
    currency: str = "USD"
    
    # Market Data
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    last: Optional[Decimal] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    
    # Greeks
    delta: Optional[Decimal] = None
    gamma: Optional[Decimal] = None
    theta: Optional[Decimal] = None
    vega: Optional[Decimal] = None
    rho: Optional[Decimal] = None
    implied_volatility: Optional[Decimal] = None
    
    # Metadata
    last_update: Optional[datetime.datetime] = None
    status: ContractStatus = ContractStatus.ACTIVE
    
    # Additional fields
    _contract_id: Optional[str] = field(default=None, init=False)

    def __post_init__(self):
        """Validate contract data after initialization."""
        self._validate_contract()
        
    def _validate_contract(self) -> None:
        """Validate contract parameters."""
        if not self.symbol or not isinstance(self.symbol, str):
            raise InvalidContractError("Symbol must be a non-empty string")
        
        if self.strike <= 0:
            raise InvalidContractError("Strike price must be positive")
        
        if self.expiration <= datetime.date.today():
            raise ExpiredContractError(
                f"Contract expired on {self.expiration}"
            )
        
        if self.multiplier <= 0:
            raise InvalidContractError("Multiplier must be positive")
        
        # Validate market data ranges
        if self.bid is not None and self.bid < 0:
            raise ValidationError("Bid price cannot be negative")
        
        if self.ask is not None and self.ask < 0:
            raise ValidationError("Ask price cannot be negative")
        
        if self.last is not None and self.last < 0:
            raise ValidationError("Last price cannot be negative")
        
        if self.bid and self.ask and self.bid > self.ask:
            raise ValidationError("Bid price cannot exceed ask price")
        
        if self.volume is not None and self.volume < 0:
            raise ValidationError("Volume cannot be negative")
        
        if self.open_interest is not None and self.open_interest < 0:
            raise ValidationError("Open interest cannot be negative")
        
        # Validate Greeks ranges
        if self.delta is not None and not (-1 <= self.delta <= 1):
            raise ValidationError("Delta must be between -1 and 1")
        
        if self.implied_volatility is not None and self.implied_volatility < 0:
            raise ValidationError("Implied volatility cannot be negative")

    @property
    def contract_id(self) -> str:
        """Generate unique contract identifier."""
        if self._contract_id is None:
            self._contract_id = (
                f"{self.symbol}_{self.expiration.strftime('%Y%m%d')}_"
                f"{self.option_type.value}_{self.strike}"
            )
        return self._contract_id
    
    @property
    def time_to_expiry(self) -> Decimal:
        """Calculate time to expiry in years."""
        if self.expiration <= datetime.date.today():
            return Decimal('0')
        
        days = (self.expiration - datetime.date.today()).days
        return Decimal(days) / Decimal('365')
    
    @property
    def days_to_expiry(self) -> int:
        """Calculate days to expiry."""
        if self.expiration <= datetime.date.today():
            return 0
        return (self.expiration - datetime.date.today()).days
    
    @property
    def mid_price(self) -> Optional[Decimal]:
        """Calculate mid price from bid/ask."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return None
    
    @property
    def spread(self) -> Optional[Decimal]:
        """Calculate bid-ask spread."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None
    
    @property
    def spread_pct(self) -> Optional[Decimal]:
        """Calculate bid-ask spread as percentage of mid price."""
        mid = self.mid_price
        spread = self.spread
        if mid and spread and mid > 0:
            return (spread / mid) * 100
        return None
    
    def is_itm(self, underlying_price: Decimal) -> bool:
        """
        Check if option is in-the-money given underlying price.
        
        Args:
            underlying_price: Current price of the underlying asset
            
        Returns:
            True if option is in-the-money, False otherwise
        """
        if self.option_type == OptionType.CALL:
            return underlying_price > self.strike
        else:  # PUT
            return underlying_price < self.strike
    
    def is_otm(self, underlying_price: Decimal) -> bool:
        """
        Check if option is out-of-the-money given underlying price.
        
        Args:
            underlying_price: Current price of the underlying asset
            
        Returns:
            True if option is out-of-the-money, False otherwise
        """
        return not self.is_itm(underlying_price)
    
    def is_atm(self, underlying_price: Decimal, tolerance: Decimal = Decimal('0.01')) -> bool:
        """
        Check if option is at-the-money given underlying price.
        
        Args:
            underlying_price: Current price of the underlying asset
            tolerance: Tolerance for ATM determination (as percentage)
            
        Returns:
            True if option is at-the-money, False otherwise
        """
        price_diff = abs(underlying_price - self.strike)
        threshold = self.strike * tolerance
        return price_diff <= threshold
    
    def intrinsic_value(self, underlying_price: Decimal) -> Decimal:
        """
        Calculate intrinsic value of the option.
        
        Args:
            underlying_price: Current price of the underlying asset
            
        Returns:
            Intrinsic value of the option
        """
        if self.option_type == OptionType.CALL:
            return max(Decimal('0'), underlying_price - self.strike)
        else:  # PUT
            return max(Decimal('0'), self.strike - underlying_price)
    
    def time_value(self, underlying_price: Decimal) -> Optional[Decimal]:
        """
        Calculate time value of the option.
        
        Args:
            underlying_price: Current price of the underlying asset
            
        Returns:
            Time value of the option, or None if no market price available
        """
        market_price = self.last or self.mid_price
        if market_price is None:
            return None
        
        intrinsic = self.intrinsic_value(underlying_price)
        return max(Decimal('0'), market_price - intrinsic)
    
    def update_market_data(self, market_data: Dict[str, Any]) -> None:
        """
        Update contract with new market data.
        
        Args:
            market_data: Dictionary containing market data fields
        """
        if 'bid' in market_data and market_data['bid'] is not None:
            self.bid = Decimal(str(market_data['bid']))
        
        if 'ask' in market_data and market_data['ask'] is not None:
            self.ask = Decimal(str(market_data['ask']))
        
        if 'last' in market_data and market_data['last'] is not None:
            self.last = Decimal(str(market_data['last']))
        
        if 'volume' in market_data:
            self.volume = market_data['volume']
        
        if 'open_interest' in market_data:
            self.open_interest = market_data['open_interest']
        
        # Update Greeks
        for greek in ['delta', 'gamma', 'theta', 'vega', 'rho', 'implied_volatility']:
            if greek in market_data and market_data[greek] is not None:
                setattr(self, greek, Decimal(str(market_data[greek])))
        
        self.last_update = datetime.datetime.now()
        
        # Re-validate after update
        self._validate_contract()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert contract to dictionary representation."""
        return {
            'contract_id': self.contract_id,
            'symbol': self.symbol,
            'strike': float(self.strike),
            'expiration': self.expiration.isoformat(),
            'option_type': self.option_type.value,
            'exchange': self.exchange,
            'multiplier': self.multiplier,
            'currency': self.currency,
            'bid': float(self.bid) if self.bid else None,
            'ask': float(self.ask) if self.ask else None,
            'last': float(self.last) if self.last else None,
            'volume': self.volume,
            'open_interest': self.open_interest,
            'delta': float(self.delta) if self.delta else None,
            'gamma': float(self.gamma) if self.gamma else None,
            'theta': float(self.theta) if self.theta else None,
            'vega': float(self.vega) if self.vega else None,
            'rho': float(self.rho) if self.rho else None,
            'implied_volatility': float(self.implied_volatility) if self.implied_volatility else None,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'status': self.status.value
        }


@dataclass
class StockContract:
    """
    Represents a stock contract.
    
    This class encapsulates information needed to identify and track
    a stock instrument used as underlying for options.
    """
    
    # Contract Identification
    symbol: str
    exchange: str = "SMART"
    currency: str = "USD"
    
    # Market Data
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    last: Optional[Decimal] = None
    volume: Optional[int] = None
    
    # Fundamental Data
    market_cap: Optional[Decimal] = None
    pe_ratio: Optional[Decimal] = None
    dividend_yield: Optional[Decimal] = None
    
    # Metadata
    last_update: Optional[datetime.datetime] = None
    status: ContractStatus = ContractStatus.ACTIVE
    
    # Additional fields
    _contract_id: Optional[str] = field(default=None, init=False)

    def __post_init__(self):
        """Validate contract data after initialization."""
        self._validate_contract()

    def _validate_contract(self) -> None:
        """Validate contract parameters."""
        if not self.symbol or not isinstance(self.symbol, str):
            raise InvalidContractError("Symbol must be a non-empty string")
        
        # Validate market data
        if self.bid is not None and self.bid < 0:
            raise ValidationError("Bid price cannot be negative")
        
        if self.ask is not None and self.ask < 0:
            raise ValidationError("Ask price cannot be negative")
        
        if self.last is not None and self.last < 0:
            raise ValidationError("Last price cannot be negative")
        
        if self.bid and self.ask and self.bid > self.ask:
            raise ValidationError("Bid price cannot exceed ask price")
        
        if self.volume is not None and self.volume < 0:
            raise ValidationError("Volume cannot be negative")
        
        # Validate fundamental data
        if self.market_cap is not None and self.market_cap < 0:
            raise ValidationError("Market cap cannot be negative")
        
        if self.pe_ratio is not None and self.pe_ratio < 0:
            raise ValidationError("P/E ratio cannot be negative")
        
        if self.dividend_yield is not None and self.dividend_yield < 0:
            raise ValidationError("Dividend yield cannot be negative")

    @property
    def contract_id(self) -> str:
        """Generate unique contract identifier."""
        if self._contract_id is None:
            self._contract_id = f"STK_{self.symbol}_{self.exchange}_{self.currency}"
        return self._contract_id
    
    @property
    def mid_price(self) -> Optional[Decimal]:
        """Calculate mid price from bid/ask."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return None
    
    @property
    def spread(self) -> Optional[Decimal]:
        """Calculate bid-ask spread."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None
    
    @property
    def spread_pct(self) -> Optional[Decimal]:
        """Calculate bid-ask spread as percentage of mid price."""
        mid = self.mid_price
        spread = self.spread
        if mid and spread and mid > 0:
            return (spread / mid) * 100
        return None
    
    def update_market_data(self, market_data: Dict[str, Any]) -> None:
        """
        Update contract with new market data.
        
        Args:
            market_data: Dictionary containing market data fields
        """
        if 'bid' in market_data and market_data['bid'] is not None:
            self.bid = Decimal(str(market_data['bid']))
        
        if 'ask' in market_data and market_data['ask'] is not None:
            self.ask = Decimal(str(market_data['ask']))
        
        if 'last' in market_data and market_data['last'] is not None:
            self.last = Decimal(str(market_data['last']))
        
        if 'volume' in market_data:
            self.volume = market_data['volume']
        
        # Update fundamental data
        if 'market_cap' in market_data and market_data['market_cap'] is not None:
            self.market_cap = Decimal(str(market_data['market_cap']))
        
        if 'pe_ratio' in market_data and market_data['pe_ratio'] is not None:
            self.pe_ratio = Decimal(str(market_data['pe_ratio']))
        
        if 'dividend_yield' in market_data and market_data['dividend_yield'] is not None:
            self.dividend_yield = Decimal(str(market_data['dividend_yield']))
        
        self.last_update = datetime.datetime.now()
        
        # Re-validate after update
        self._validate_contract()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert contract to dictionary representation."""
        return {
            'contract_id': self.contract_id,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'currency': self.currency,
            'bid': float(self.bid) if self.bid else None,
            'ask': float(self.ask) if self.ask else None,
            'last': float(self.last) if self.last else None,
            'volume': self.volume,
            'market_cap': float(self.market_cap) if self.market_cap else None,
            'pe_ratio': float(self.pe_ratio) if self.pe_ratio else None,
            'dividend_yield': float(self.dividend_yield) if self.dividend_yield else None,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'status': self.status.value
        }