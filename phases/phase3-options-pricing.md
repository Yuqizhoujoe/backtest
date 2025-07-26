# Phase 3: Options Pricing & Greeks (Week 4)

## Overview

Implement comprehensive options pricing engine with accurate Black-Scholes calculations and Greeks. This phase establishes the mathematical foundation for realistic options backtesting with proper risk calculations.

## Objectives

- Implement Black-Scholes option pricing model with Greeks calculations
- Create implied volatility calculation system
- Build market data processing pipeline with bid-ask spread modeling
- Handle American vs European option differences
- Develop comprehensive pricing validation and testing framework

## Technical Deliverables

### 1. Enhanced Project Structure

```
src/
├── pricing/
│   ├── __init__.py
│   ├── black_scholes.py        # Black-Scholes implementation
│   ├── greeks_calculator.py    # Greeks calculations
│   ├── implied_volatility.py   # IV calculation methods
│   ├── volatility_models.py    # Historical & implied vol models
│   ├── interest_rates.py       # Risk-free rate handling
│   └── dividend_adjustments.py # Dividend adjustment logic
├── market/
│   ├── __init__.py
│   ├── bid_ask_processor.py    # Bid-ask spread modeling
│   ├── liquidity_analyzer.py   # Volume and liquidity analysis
│   └── corporate_actions.py    # Stock splits, dividends, etc.
└── utils/
    ├── math_utils.py           # Mathematical utilities
    └── date_utils.py           # Date/time utilities

tests/
├── unit/
│   ├── test_black_scholes.py
│   ├── test_greeks.py
│   ├── test_implied_volatility.py
│   └── test_market_processing.py
└── integration/
    ├── test_pricing_accuracy.py
    └── test_market_data_pipeline.py
```

### 2. Black-Scholes Implementation (`src/pricing/black_scholes.py`)

```python
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from decimal import Decimal, getcontext
import math
from typing import Union, Optional
from datetime import datetime, date
import logging

from ..core.contracts import OptionContract
from ..core.exceptions import PricingError
from .interest_rates import RiskFreeRateProvider
from .dividend_adjustments import DividendAdjuster

# Set high precision for decimal calculations
getcontext().prec = 10

class BlackScholesCalculator:
    """Black-Scholes option pricing calculator with Greeks"""
    
    def __init__(self, rate_provider: Optional[RiskFreeRateProvider] = None,
                 dividend_adjuster: Optional[DividendAdjuster] = None):
        self.rate_provider = rate_provider or RiskFreeRateProvider()
        self.dividend_adjuster = dividend_adjuster or DividendAdjuster()
        self.logger = logging.getLogger(__name__)
    
    def calculate_option_price(self, 
                             underlying_price: Union[float, Decimal],
                             strike_price: Union[float, Decimal],
                             time_to_expiry: Union[float, Decimal],
                             volatility: Union[float, Decimal],
                             risk_free_rate: Union[float, Decimal],
                             dividend_yield: Union[float, Decimal] = 0,
                             option_type: str = 'CALL') -> Decimal:
        """
        Calculate Black-Scholes option price
        
        Args:
            underlying_price: Current price of underlying asset
            strike_price: Strike price of option
            time_to_expiry: Time to expiry in years
            volatility: Implied volatility (annualized)
            risk_free_rate: Risk-free interest rate (annualized)
            dividend_yield: Dividend yield (annualized)
            option_type: 'CALL' or 'PUT'
            
        Returns:
            Option price as Decimal
        """
        try:
            # Convert inputs to float for calculations
            S = float(underlying_price)
            K = float(strike_price)
            T = float(time_to_expiry)
            sigma = float(volatility)
            r = float(risk_free_rate)
            q = float(dividend_yield)
            
            # Input validation
            if S <= 0:
                raise PricingError("Underlying price must be positive")
            if K <= 0:
                raise PricingError("Strike price must be positive")
            if T < 0:
                raise PricingError("Time to expiry cannot be negative")
            if T == 0:
                # At expiry, option value is intrinsic value
                if option_type.upper() == 'CALL':
                    return Decimal(max(S - K, 0))
                else:
                    return Decimal(max(K - S, 0))
            if sigma < 0:
                raise PricingError("Volatility cannot be negative")
            
            # Handle edge cases
            if sigma == 0:
                # Zero volatility case
                discount_factor = math.exp(-r * T)
                if option_type.upper() == 'CALL':
                    return Decimal(max((S * math.exp(-q * T) - K) * discount_factor, 0))
                else:
                    return Decimal(max((K - S * math.exp(-q * T)) * discount_factor, 0))
            
            # Calculate d1 and d2
            d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            
            # Calculate option price
            if option_type.upper() == 'CALL':
                price = (S * math.exp(-q * T) * norm.cdf(d1) - 
                        K * math.exp(-r * T) * norm.cdf(d2))
            elif option_type.upper() == 'PUT':
                price = (K * math.exp(-r * T) * norm.cdf(-d2) - 
                        S * math.exp(-q * T) * norm.cdf(-d1))
            else:
                raise PricingError(f"Invalid option type: {option_type}")
            
            return Decimal(str(round(price, 4)))
            
        except Exception as e:
            self.logger.error(f"Error calculating option price: {e}")
            raise PricingError(f"Failed to calculate option price: {e}")
    
    def calculate_greeks(self,
                        underlying_price: Union[float, Decimal],
                        strike_price: Union[float, Decimal],
                        time_to_expiry: Union[float, Decimal],
                        volatility: Union[float, Decimal],
                        risk_free_rate: Union[float, Decimal],
                        dividend_yield: Union[float, Decimal] = 0,
                        option_type: str = 'CALL') -> dict:
        """
        Calculate all Greeks for an option
        
        Returns:
            Dictionary containing delta, gamma, theta, vega, rho
        """
        try:
            # Convert inputs to float
            S = float(underlying_price)
            K = float(strike_price)
            T = float(time_to_expiry)
            sigma = float(volatility)
            r = float(risk_free_rate)
            q = float(dividend_yield)
            
            # Input validation
            if T <= 0:
                return {
                    'delta': Decimal('0'),
                    'gamma': Decimal('0'),
                    'theta': Decimal('0'),
                    'vega': Decimal('0'),
                    'rho': Decimal('0')
                }
            
            if sigma == 0:
                # Handle zero volatility case
                return self._calculate_greeks_zero_vol(S, K, T, r, q, option_type)
            
            # Calculate d1 and d2
            d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            
            # Standard normal PDF and CDF values
            pdf_d1 = norm.pdf(d1)
            cdf_d1 = norm.cdf(d1)
            cdf_d2 = norm.cdf(d2)
            cdf_minus_d1 = norm.cdf(-d1)
            cdf_minus_d2 = norm.cdf(-d2)
            
            # Calculate Greeks
            greeks = {}
            
            # Delta
            if option_type.upper() == 'CALL':
                greeks['delta'] = Decimal(str(round(math.exp(-q * T) * cdf_d1, 6)))
            else:
                greeks['delta'] = Decimal(str(round(-math.exp(-q * T) * cdf_minus_d1, 6)))
            
            # Gamma (same for calls and puts)
            gamma = (math.exp(-q * T) * pdf_d1) / (S * sigma * math.sqrt(T))
            greeks['gamma'] = Decimal(str(round(gamma, 6)))
            
            # Theta
            theta_term1 = -(S * pdf_d1 * sigma * math.exp(-q * T)) / (2 * math.sqrt(T))
            theta_term2 = r * K * math.exp(-r * T)
            theta_term3 = q * S * math.exp(-q * T)
            
            if option_type.upper() == 'CALL':
                theta = theta_term1 - theta_term2 * cdf_d2 + theta_term3 * cdf_d1
            else:
                theta = theta_term1 + theta_term2 * cdf_minus_d2 - theta_term3 * cdf_minus_d1
            
            # Convert to per-day theta
            greeks['theta'] = Decimal(str(round(theta / 365, 6)))
            
            # Vega (same for calls and puts)
            vega = S * math.exp(-q * T) * pdf_d1 * math.sqrt(T)
            greeks['vega'] = Decimal(str(round(vega / 100, 6)))  # Per 1% vol change
            
            # Rho
            if option_type.upper() == 'CALL':
                rho = K * T * math.exp(-r * T) * cdf_d2
            else:
                rho = -K * T * math.exp(-r * T) * cdf_minus_d2
            
            greeks['rho'] = Decimal(str(round(rho / 100, 6)))  # Per 1% rate change
            
            return greeks
            
        except Exception as e:
            self.logger.error(f"Error calculating Greeks: {e}")
            raise PricingError(f"Failed to calculate Greeks: {e}")
    
    def _calculate_greeks_zero_vol(self, S: float, K: float, T: float, 
                                  r: float, q: float, option_type: str) -> dict:
        """Calculate Greeks for zero volatility case"""
        # For zero vol, Greeks are discontinuous at S=K
        epsilon = 1e-6
        
        if abs(S - K * math.exp(-(r - q) * T)) < epsilon:
            # At the money case
            return {
                'delta': Decimal('0.5') if option_type.upper() == 'CALL' else Decimal('-0.5'),
                'gamma': Decimal('0'),
                'theta': Decimal('0'),
                'vega': Decimal('0'),
                'rho': Decimal('0')
            }
        else:
            # In or out of the money
            if option_type.upper() == 'CALL':
                delta = Decimal('1') if S > K * math.exp(-(r - q) * T) else Decimal('0')
            else:
                delta = Decimal('-1') if S < K * math.exp(-(r - q) * T) else Decimal('0')
            
            return {
                'delta': delta,
                'gamma': Decimal('0'),
                'theta': Decimal('0'),
                'vega': Decimal('0'),
                'rho': Decimal('0')
            }
    
    def price_option_contract(self, contract: OptionContract, 
                             underlying_price: Union[float, Decimal],
                             market_data: Optional[dict] = None) -> dict:
        """
        Price an option contract with all Greeks
        
        Args:
            contract: OptionContract instance
            underlying_price: Current underlying price
            market_data: Optional market data (volatility, rates, etc.)
            
        Returns:
            Dictionary with price and Greeks
        """
        try:
            # Calculate time to expiry
            time_to_expiry = contract.time_to_expiry
            
            # Get market parameters
            if market_data:
                volatility = market_data.get('volatility', contract.implied_volatility or 0.2)
                risk_free_rate = market_data.get('risk_free_rate')
                dividend_yield = market_data.get('dividend_yield', 0)
            else:
                volatility = contract.implied_volatility or 0.2
                risk_free_rate = None
                dividend_yield = 0
            
            # Get risk-free rate if not provided
            if risk_free_rate is None:
                risk_free_rate = self.rate_provider.get_rate(time_to_expiry)
            
            # Adjust for dividends if applicable
            adjusted_params = self.dividend_adjuster.adjust_for_dividends(
                underlying_price, contract.strike, time_to_expiry, 
                dividend_yield, contract.symbol
            )
            
            # Calculate price
            price = self.calculate_option_price(
                underlying_price=adjusted_params['underlying_price'],
                strike_price=contract.strike,
                time_to_expiry=time_to_expiry,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                option_type=contract.option_type.value
            )
            
            # Calculate Greeks
            greeks = self.calculate_greeks(
                underlying_price=adjusted_params['underlying_price'],
                strike_price=contract.strike,
                time_to_expiry=time_to_expiry,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                option_type=contract.option_type.value
            )
            
            return {
                'theoretical_price': price,
                'delta': greeks['delta'],
                'gamma': greeks['gamma'],
                'theta': greeks['theta'],
                'vega': greeks['vega'],
                'rho': greeks['rho'],
                'time_to_expiry': Decimal(str(time_to_expiry)),
                'volatility': Decimal(str(volatility)),
                'risk_free_rate': Decimal(str(risk_free_rate)),
                'dividend_yield': Decimal(str(dividend_yield))
            }
            
        except Exception as e:
            self.logger.error(f"Error pricing option contract: {e}")
            raise PricingError(f"Failed to price option contract: {e}")
```

### 3. Implied Volatility Calculator (`src/pricing/implied_volatility.py`)

```python
import math
from scipy.optimize import brentq, minimize_scalar
from decimal import Decimal
import numpy as np
from typing import Union, Optional
import logging

from .black_scholes import BlackScholesCalculator
from ..core.exceptions import PricingError

class ImpliedVolatilityCalculator:
    """Calculate implied volatility from market prices"""
    
    def __init__(self):
        self.bs_calculator = BlackScholesCalculator()
        self.logger = logging.getLogger(__name__)
        
        # Calculation parameters
        self.min_vol = 0.001    # 0.1%
        self.max_vol = 5.0      # 500%
        self.tolerance = 1e-6
        self.max_iterations = 100
    
    def calculate_implied_volatility(self,
                                   market_price: Union[float, Decimal],
                                   underlying_price: Union[float, Decimal],
                                   strike_price: Union[float, Decimal],
                                   time_to_expiry: Union[float, Decimal],
                                   risk_free_rate: Union[float, Decimal],
                                   dividend_yield: Union[float, Decimal] = 0,
                                   option_type: str = 'CALL') -> Optional[Decimal]:
        """
        Calculate implied volatility using Brent's method
        
        Args:
            market_price: Observed market price
            underlying_price: Current underlying price
            strike_price: Strike price
            time_to_expiry: Time to expiry in years
            risk_free_rate: Risk-free rate
            dividend_yield: Dividend yield
            option_type: 'CALL' or 'PUT'
            
        Returns:
            Implied volatility as Decimal, or None if calculation fails
        """
        try:
            market_price = float(market_price)
            
            # Check if option has any time value
            if float(time_to_expiry) <= 0:
                return None
            
            # Calculate intrinsic value
            S = float(underlying_price)
            K = float(strike_price)
            
            if option_type.upper() == 'CALL':
                intrinsic = max(S - K, 0)
            else:
                intrinsic = max(K - S, 0)
            
            # Check if market price is below intrinsic value
            if market_price < intrinsic * 0.99:  # Allow small tolerance
                self.logger.warning(f"Market price {market_price} below intrinsic value {intrinsic}")
                return None
            
            # Define objective function
            def objective(vol):
                try:
                    theoretical_price = self.bs_calculator.calculate_option_price(
                        underlying_price=underlying_price,
                        strike_price=strike_price,
                        time_to_expiry=time_to_expiry,
                        volatility=vol,
                        risk_free_rate=risk_free_rate,
                        dividend_yield=dividend_yield,
                        option_type=option_type
                    )
                    return float(theoretical_price) - market_price
                except:
                    return float('inf')
            
            # Check bounds
            obj_min = objective(self.min_vol)
            obj_max = objective(self.max_vol)
            
            # If both bounds have same sign, no solution exists
            if obj_min * obj_max > 0:
                # Try expanding search range
                if obj_min > 0:  # Market price too low
                    return Decimal(str(self.min_vol))
                else:  # Market price too high
                    return Decimal(str(self.max_vol))
            
            # Use Brent's method to find root
            try:
                implied_vol = brentq(
                    objective, 
                    self.min_vol, 
                    self.max_vol,
                    xtol=self.tolerance,
                    maxiter=self.max_iterations
                )
                
                return Decimal(str(round(implied_vol, 6)))
                
            except ValueError:
                # Fallback to minimize_scalar
                result = minimize_scalar(
                    lambda vol: abs(objective(vol)),
                    bounds=(self.min_vol, self.max_vol),
                    method='bounded'
                )
                
                if result.success:
                    return Decimal(str(round(result.x, 6)))
                else:
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error calculating implied volatility: {e}")
            return None
    
    def calculate_vega_weighted_iv(self, option_chain: list) -> Optional[Decimal]:
        """
        Calculate vega-weighted implied volatility for option chain
        
        Args:
            option_chain: List of option data with IV and vega values
            
        Returns:
            Vega-weighted implied volatility
        """
        try:
            total_vega = Decimal('0')
            weighted_iv = Decimal('0')
            
            for option in option_chain:
                iv = option.get('implied_volatility')
                vega = option.get('vega')
                
                if iv is not None and vega is not None and vega > 0:
                    iv = Decimal(str(iv))
                    vega = Decimal(str(vega))
                    
                    weighted_iv += iv * vega
                    total_vega += vega
            
            if total_vega > 0:
                return weighted_iv / total_vega
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error calculating vega-weighted IV: {e}")
            return None
    
    def calculate_iv_surface(self, option_chains: dict) -> dict:
        """
        Calculate implied volatility surface across strikes and expirations
        
        Args:
            option_chains: Dict with expiration dates as keys, option data as values
            
        Returns:
            IV surface data structure
        """
        iv_surface = {}
        
        for expiration, chain_data in option_chains.items():
            if not chain_data:
                continue
                
            expiration_data = {
                'calls': {},
                'puts': {},
                'atm_iv': None
            }
            
            # Calculate IV for each option
            for option in chain_data:
                strike = option['strike']
                option_type = option['option_type']
                market_price = option.get('mid_price') or option.get('last')
                
                if market_price is None or market_price <= 0:
                    continue
                
                iv = self.calculate_implied_volatility(
                    market_price=market_price,
                    underlying_price=option['underlying_price'],
                    strike_price=strike,
                    time_to_expiry=option['time_to_expiry'],
                    risk_free_rate=option['risk_free_rate'],
                    dividend_yield=option.get('dividend_yield', 0),
                    option_type=option_type
                )
                
                if iv is not None:
                    if option_type == 'CALL':
                        expiration_data['calls'][strike] = iv
                    else:
                        expiration_data['puts'][strike] = iv
            
            # Find ATM implied volatility
            atm_strikes = self._find_atm_strikes(chain_data)
            if atm_strikes:
                atm_ivs = []
                for strike in atm_strikes:
                    call_iv = expiration_data['calls'].get(strike)
                    put_iv = expiration_data['puts'].get(strike)
                    
                    if call_iv is not None:
                        atm_ivs.append(call_iv)
                    if put_iv is not None:
                        atm_ivs.append(put_iv)
                
                if atm_ivs:
                    expiration_data['atm_iv'] = sum(atm_ivs) / len(atm_ivs)
            
            iv_surface[expiration] = expiration_data
        
        return iv_surface
    
    def _find_atm_strikes(self, chain_data: list) -> list:
        """Find at-the-money strike prices"""
        if not chain_data:
            return []
        
        underlying_price = chain_data[0].get('underlying_price')
        if underlying_price is None:
            return []
        
        # Find strikes closest to underlying price
        strikes = list(set(option['strike'] for option in chain_data))
        strikes.sort()
        
        # Find strike closest to underlying
        closest_strike = min(strikes, key=lambda x: abs(x - underlying_price))
        
        # Return closest strike and adjacent strikes
        idx = strikes.index(closest_strike)
        atm_strikes = [closest_strike]
        
        if idx > 0:
            atm_strikes.append(strikes[idx - 1])
        if idx < len(strikes) - 1:
            atm_strikes.append(strikes[idx + 1])
        
        return atm_strikes
```

### 4. Greeks Calculator (`src/pricing/greeks_calculator.py`)

```python
import numpy as np
from decimal import Decimal
from typing import Dict, Union, Optional
import logging

from .black_scholes import BlackScholesCalculator
from ..core.contracts import OptionContract
from ..core.exceptions import PricingError

class GreeksCalculator:
    """Advanced Greeks calculations and portfolio-level Greeks"""
    
    def __init__(self):
        self.bs_calculator = BlackScholesCalculator()
        self.logger = logging.getLogger(__name__)
    
    def calculate_portfolio_greeks(self, positions: list) -> Dict[str, Decimal]:
        """
        Calculate portfolio-level Greeks by summing position Greeks
        
        Args:
            positions: List of position dictionaries with contract and quantity
            
        Returns:
            Portfolio Greeks dictionary
        """
        try:
            portfolio_greeks = {
                'delta': Decimal('0'),
                'gamma': Decimal('0'),
                'theta': Decimal('0'),
                'vega': Decimal('0'),
                'rho': Decimal('0')
            }
            
            for position in positions:
                contract = position['contract']
                quantity = Decimal(str(position['quantity']))
                underlying_price = position['underlying_price']
                
                # Calculate position Greeks
                if isinstance(contract, OptionContract):
                    greeks = self.bs_calculator.calculate_greeks(
                        underlying_price=underlying_price,
                        strike_price=contract.strike,
                        time_to_expiry=contract.time_to_expiry,
                        volatility=contract.implied_volatility or Decimal('0.2'),
                        risk_free_rate=position.get('risk_free_rate', Decimal('0.02')),
                        dividend_yield=position.get('dividend_yield', Decimal('0')),
                        option_type=contract.option_type.value
                    )
                    
                    # Multiply by position size and option multiplier
                    multiplier = Decimal(str(contract.multiplier))
                    
                    for greek_name, greek_value in greeks.items():
                        portfolio_greeks[greek_name] += greek_value * quantity * multiplier
                
                else:
                    # Stock position contributes only to delta
                    portfolio_greeks['delta'] += quantity
            
            return portfolio_greeks
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio Greeks: {e}")
            raise PricingError(f"Failed to calculate portfolio Greeks: {e}")
    
    def calculate_second_order_greeks(self,
                                    underlying_price: Union[float, Decimal],
                                    strike_price: Union[float, Decimal],
                                    time_to_expiry: Union[float, Decimal],
                                    volatility: Union[float, Decimal],
                                    risk_free_rate: Union[float, Decimal],
                                    dividend_yield: Union[float, Decimal] = 0,
                                    option_type: str = 'CALL') -> Dict[str, Decimal]:
        """
        Calculate second-order Greeks (gamma, vanna, volga, etc.)
        
        Returns:
            Dictionary with second-order Greeks
        """
        try:
            # Use finite difference method for numerical derivatives
            epsilon_s = 0.01  # 1% change in underlying
            epsilon_v = 0.01  # 1% change in volatility
            epsilon_t = 1.0 / 365  # 1 day change in time
            
            S = float(underlying_price)
            base_greeks = self.bs_calculator.calculate_greeks(
                underlying_price, strike_price, time_to_expiry,
                volatility, risk_free_rate, dividend_yield, option_type
            )
            
            # Gamma (already calculated in base_greeks)
            gamma = base_greeks['gamma']
            
            # Vanna (sensitivity of delta to volatility)
            delta_up = self.bs_calculator.calculate_greeks(
                underlying_price, strike_price, time_to_expiry,
                float(volatility) + epsilon_v, risk_free_rate, dividend_yield, option_type
            )['delta']
            
            delta_down = self.bs_calculator.calculate_greeks(
                underlying_price, strike_price, time_to_expiry,
                float(volatility) - epsilon_v, risk_free_rate, dividend_yield, option_type
            )['delta']
            
            vanna = (delta_up - delta_down) / (2 * epsilon_v)
            
            # Volga (sensitivity of vega to volatility)
            vega_up = self.bs_calculator.calculate_greeks(
                underlying_price, strike_price, time_to_expiry,
                float(volatility) + epsilon_v, risk_free_rate, dividend_yield, option_type
            )['vega']
            
            vega_down = self.bs_calculator.calculate_greeks(
                underlying_price, strike_price, time_to_expiry,
                float(volatility) - epsilon_v, risk_free_rate, dividend_yield, option_type
            )['vega']
            
            volga = (vega_up - vega_down) / (2 * epsilon_v)
            
            # Charm (sensitivity of delta to time)
            if float(time_to_expiry) > epsilon_t:
                delta_future = self.bs_calculator.calculate_greeks(
                    underlying_price, strike_price, float(time_to_expiry) - epsilon_t,
                    volatility, risk_free_rate, dividend_yield, option_type
                )['delta']
                
                charm = (base_greeks['delta'] - delta_future) / epsilon_t
            else:
                charm = Decimal('0')
            
            return {
                'gamma': gamma,
                'vanna': Decimal(str(round(float(vanna), 8))),
                'volga': Decimal(str(round(float(volga), 8))),
                'charm': Decimal(str(round(float(charm), 8)))
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating second-order Greeks: {e}")
            raise PricingError(f"Failed to calculate second-order Greeks: {e}")
    
    def calculate_risk_metrics(self, positions: list, 
                             underlying_price_scenarios: list) -> Dict:
        """
        Calculate risk metrics for different underlying price scenarios
        
        Args:
            positions: List of positions
            underlying_price_scenarios: List of price scenarios to test
            
        Returns:
            Risk metrics dictionary
        """
        try:
            scenario_pnl = []
            
            for scenario_price in underlying_price_scenarios:
                total_pnl = Decimal('0')
                
                for position in positions:
                    contract = position['contract']
                    quantity = Decimal(str(position['quantity']))
                    entry_price = Decimal(str(position['entry_price']))
                    
                    if isinstance(contract, OptionContract):
                        # Calculate option value at scenario price
                        current_value = self.bs_calculator.calculate_option_price(
                            underlying_price=scenario_price,
                            strike_price=contract.strike,
                            time_to_expiry=contract.time_to_expiry,
                            volatility=contract.implied_volatility or Decimal('0.2'),
                            risk_free_rate=position.get('risk_free_rate', Decimal('0.02')),
                            dividend_yield=position.get('dividend_yield', Decimal('0')),
                            option_type=contract.option_type.value
                        )
                        
                        position_pnl = (current_value - entry_price) * quantity * contract.multiplier
                    else:
                        # Stock position
                        position_pnl = (Decimal(str(scenario_price)) - entry_price) * quantity
                    
                    total_pnl += position_pnl
                
                scenario_pnl.append(float(total_pnl))
            
            # Calculate risk metrics
            scenario_pnl = np.array(scenario_pnl)
            
            return {
                'max_profit': Decimal(str(np.max(scenario_pnl))),
                'max_loss': Decimal(str(np.min(scenario_pnl))),
                'profit_probability': len(scenario_pnl[scenario_pnl > 0]) / len(scenario_pnl),
                'expected_pnl': Decimal(str(np.mean(scenario_pnl))),
                'pnl_std': Decimal(str(np.std(scenario_pnl))),
                'var_95': Decimal(str(np.percentile(scenario_pnl, 5))),  # 95% VaR
                'scenario_prices': underlying_price_scenarios,
                'scenario_pnl': [Decimal(str(pnl)) for pnl in scenario_pnl]
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            raise PricingError(f"Failed to calculate risk metrics: {e}")
```

### 5. Market Data Processing (`src/market/bid_ask_processor.py`)

```python
import pandas as pd
import numpy as np
from decimal import Decimal
from typing import Dict, Optional, Tuple
import logging

from ..core.exceptions import DataValidationError

class BidAskProcessor:
    """Process bid-ask spread data and estimate fair values"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Spread modeling parameters
        self.max_spread_ratio = 0.5  # Maximum spread as ratio of mid price
        self.min_tick_size = Decimal('0.01')
        
    def calculate_mid_price(self, bid: Optional[Decimal], 
                          ask: Optional[Decimal]) -> Optional[Decimal]:
        """Calculate mid price from bid and ask"""
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            return (bid + ask) / 2
        return None
    
    def calculate_spread_metrics(self, bid: Decimal, ask: Decimal) -> Dict:
        """Calculate bid-ask spread metrics"""
        if bid <= 0 or ask <= 0 or bid > ask:
            raise DataValidationError("Invalid bid-ask prices")
        
        mid_price = (bid + ask) / 2
        spread = ask - bid
        spread_ratio = spread / mid_price
        
        return {
            'mid_price': mid_price,
            'spread': spread,
            'spread_ratio': spread_ratio,
            'spread_bps': spread_ratio * 10000  # Basis points
        }
    
    def estimate_transaction_costs(self, bid: Decimal, ask: Decimal,
                                  volume: Optional[int] = None,
                                  order_size: int = 1) -> Dict:
        """Estimate transaction costs for given order size"""
        spread_metrics = self.calculate_spread_metrics(bid, ask)
        
        # Basic spread cost (half spread)
        spread_cost = spread_metrics['spread'] / 2
        
        # Market impact based on order size and volume
        market_impact = Decimal('0')
        if volume and volume > 0:
            volume_ratio = order_size / volume
            # Simple linear market impact model
            market_impact = spread_cost * Decimal(str(min(volume_ratio * 10, 1.0)))
        
        total_cost = spread_cost + market_impact
        
        return {
            'spread_cost': spread_cost,
            'market_impact': market_impact,
            'total_cost': total_cost,
            'cost_ratio': total_cost / spread_metrics['mid_price']
        }
    
    def process_option_quotes(self, quotes_df: pd.DataFrame) -> pd.DataFrame:
        """Process option quotes and add derived fields"""
        processed_df = quotes_df.copy()
        
        # Calculate mid prices
        processed_df['mid_price'] = processed_df.apply(
            lambda row: self.calculate_mid_price(row.get('bid'), row.get('ask')),
            axis=1
        )
        
        # Calculate spread metrics where possible
        valid_quotes = (processed_df['bid'].notna() & 
                       processed_df['ask'].notna() & 
                       (processed_df['bid'] > 0) & 
                       (processed_df['ask'] > 0) &
                       (processed_df['bid'] <= processed_df['ask']))
        
        for idx in processed_df[valid_quotes].index:
            try:
                spread_metrics = self.calculate_spread_metrics(
                    processed_df.loc[idx, 'bid'],
                    processed_df.loc[idx, 'ask']
                )
                
                processed_df.loc[idx, 'spread'] = spread_metrics['spread']
                processed_df.loc[idx, 'spread_ratio'] = spread_metrics['spread_ratio']
                processed_df.loc[idx, 'spread_bps'] = spread_metrics['spread_bps']
                
            except Exception as e:
                self.logger.warning(f"Error processing quote at index {idx}: {e}")
                continue
        
        # Flag wide spreads
        if 'spread_ratio' in processed_df.columns:
            processed_df['wide_spread'] = (
                processed_df['spread_ratio'] > self.max_spread_ratio
            )
        
        return processed_df
    
    def filter_tradeable_quotes(self, quotes_df: pd.DataFrame,
                               max_spread_ratio: float = 0.2,
                               min_volume: int = 1) -> pd.DataFrame:
        """Filter for tradeable option quotes"""
        filtered_df = quotes_df.copy()
        
        # Remove quotes with invalid bid/ask
        valid_mask = (
            filtered_df['bid'].notna() & 
            filtered_df['ask'].notna() & 
            (filtered_df['bid'] > 0) & 
            (filtered_df['ask'] > 0) &
            (filtered_df['bid'] <= filtered_df['ask'])
        )
        
        # Remove wide spreads
        if 'spread_ratio' in filtered_df.columns:
            valid_mask &= (filtered_df['spread_ratio'] <= max_spread_ratio)
        
        # Minimum volume filter
        if 'volume' in filtered_df.columns and min_volume > 0:
            valid_mask &= (filtered_df['volume'] >= min_volume)
        
        tradeable_df = filtered_df[valid_mask].copy()
        
        removed_count = len(filtered_df) - len(tradeable_df)
        if removed_count > 0:
            self.logger.info(f"Filtered out {removed_count} non-tradeable quotes")
        
        return tradeable_df
    
    def estimate_execution_price(self, bid: Decimal, ask: Decimal,
                                side: str, aggressive: bool = False) -> Decimal:
        """
        Estimate execution price for market orders
        
        Args:
            bid: Current bid price
            ask: Current ask price
            side: 'BUY' or 'SELL'
            aggressive: Whether to use aggressive pricing
            
        Returns:
            Estimated execution price
        """
        if side.upper() == 'BUY':
            if aggressive:
                return ask  # Pay the ask
            else:
                # Estimate getting filled between mid and ask
                mid_price = (bid + ask) / 2
                return mid_price + (ask - mid_price) * Decimal('0.6')
        
        elif side.upper() == 'SELL':
            if aggressive:
                return bid  # Hit the bid
            else:
                # Estimate getting filled between bid and mid
                mid_price = (bid + ask) / 2
                return mid_price - (mid_price - bid) * Decimal('0.6')
        
        else:
            raise ValueError(f"Invalid side: {side}")
```

## Testing Requirements

### Unit Tests

1. **Black-Scholes Tests** (`tests/unit/test_black_scholes.py`)
   - Price calculation accuracy against known values
   - Greeks calculation validation
   - Edge cases (zero volatility, at expiry, etc.)
   - Input validation and error handling

2. **Implied Volatility Tests** (`tests/unit/test_implied_volatility.py`)
   - IV calculation accuracy
   - Convergence for different market conditions
   - Boundary condition handling

3. **Market Processing Tests** (`tests/unit/test_market_processing.py`)
   - Bid-ask spread calculations
   - Quote filtering logic
   - Transaction cost estimation

### Integration Tests

1. **Pricing Accuracy Tests** (`tests/integration/test_pricing_accuracy.py`)
   - Compare calculated prices with market data
   - Greeks validation against market Greeks
   - Performance benchmarking

## Performance Requirements

- **Option Pricing**: <1ms per option
- **Greeks Calculation**: <2ms per option
- **Implied Volatility**: <10ms per option
- **Portfolio Greeks**: <50ms for 100 positions
- **Pricing Accuracy**: Within 1% of market prices for liquid options

## Configuration

### Pricing Configuration (`config/pricing.yaml`)

```yaml
black_scholes:
  precision: 6  # Decimal places
  max_iterations: 100
  tolerance: 1e-6

implied_volatility:
  min_vol: 0.001
  max_vol: 5.0
  tolerance: 1e-6
  max_iterations: 100

market_data:
  max_spread_ratio: 0.2
  min_volume: 1
  stale_quote_seconds: 300

risk_free_rates:
  default_rate: 0.02
  curve_source: "treasury"  # treasury, libor, fed_funds
```

## Success Criteria

- [ ] Black-Scholes pricing within 1% accuracy for liquid options
- [ ] Greeks calculations match market standards
- [ ] Implied volatility convergence >95% success rate
- [ ] Market data processing handles edge cases gracefully
- [ ] Performance benchmarks met under load
- [ ] Comprehensive test coverage >90%
- [ ] Integration with existing data layer seamless