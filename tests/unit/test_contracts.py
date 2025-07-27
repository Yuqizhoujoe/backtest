"""
Unit tests for the contracts module.

This module tests the OptionContract and StockContract classes including
validation, properties, and methods.
"""

import datetime
import pytest
from decimal import Decimal

from src.core.contracts import OptionContract, StockContract, OptionType, ContractStatus
from src.core.exceptions import InvalidContractError, ExpiredContractError, ValidationError


class TestOptionContract:
    """Test cases for OptionContract class."""
    
    def test_option_contract_creation(self, sample_option_contract):
        """Test basic option contract creation."""
        contract = sample_option_contract
        
        assert contract.symbol == "AAPL"
        assert contract.strike == Decimal('150.00')
        assert contract.option_type == OptionType.CALL
        assert contract.multiplier == 100
        assert contract.currency == "USD"
        assert contract.status == ContractStatus.ACTIVE
    
    def test_option_contract_id_generation(self, sample_option_contract):
        """Test contract ID generation."""
        contract = sample_option_contract
        
        expected_id = f"AAPL_{contract.expiration.strftime('%Y%m%d')}_CALL_150.00"
        assert contract.contract_id == expected_id
    
    def test_option_contract_validation(self):
        """Test option contract validation."""
        future_date = datetime.date.today() + datetime.timedelta(days=30)
        
        # Valid contract should not raise exceptions
        contract = OptionContract(
            symbol="AAPL",
            strike=Decimal('150.00'),
            expiration=future_date,
            option_type=OptionType.CALL
        )
        assert contract.symbol == "AAPL"
    
    def test_invalid_symbol_raises_error(self):
        """Test that invalid symbol raises error."""
        future_date = datetime.date.today() + datetime.timedelta(days=30)
        
        with pytest.raises(InvalidContractError, match="Symbol must be a non-empty string"):
            OptionContract(
                symbol="",
                strike=Decimal('150.00'),
                expiration=future_date,
                option_type=OptionType.CALL
            )
    
    def test_negative_strike_raises_error(self):
        """Test that negative strike price raises error."""
        future_date = datetime.date.today() + datetime.timedelta(days=30)
        
        with pytest.raises(InvalidContractError, match="Strike price must be positive"):
            OptionContract(
                symbol="AAPL",
                strike=Decimal('-150.00'),
                expiration=future_date,
                option_type=OptionType.CALL
            )
    
    def test_expired_contract_raises_error(self, expired_option_contract):
        """Test that expired contract raises error."""
        with pytest.raises(ExpiredContractError):
            OptionContract(**expired_option_contract)
    
    def test_negative_bid_raises_error(self):
        """Test that negative bid raises validation error."""
        future_date = datetime.date.today() + datetime.timedelta(days=30)
        
        with pytest.raises(ValidationError, match="Bid price cannot be negative"):
            OptionContract(
                symbol="AAPL",
                strike=Decimal('150.00'),
                expiration=future_date,
                option_type=OptionType.CALL,
                bid=Decimal('-1.00')
            )
    
    def test_bid_greater_than_ask_raises_error(self):
        """Test that bid > ask raises validation error."""
        future_date = datetime.date.today() + datetime.timedelta(days=30)
        
        with pytest.raises(ValidationError, match="Bid price cannot exceed ask price"):
            OptionContract(
                symbol="AAPL",
                strike=Decimal('150.00'),
                expiration=future_date,
                option_type=OptionType.CALL,
                bid=Decimal('6.00'),
                ask=Decimal('5.50')
            )
    
    def test_invalid_delta_range_raises_error(self):
        """Test that delta outside [-1, 1] raises error."""
        future_date = datetime.date.today() + datetime.timedelta(days=30)
        
        with pytest.raises(ValidationError, match="Delta must be between -1 and 1"):
            OptionContract(
                symbol="AAPL",
                strike=Decimal('150.00'),
                expiration=future_date,
                option_type=OptionType.CALL,
                delta=Decimal('1.5')
            )
    
    def test_time_to_expiry_calculation(self, sample_option_contract):
        """Test time to expiry calculation."""
        contract = sample_option_contract
        
        # Should be approximately 30/365 years
        tte = contract.time_to_expiry
        assert Decimal('0.08') <= tte <= Decimal('0.09')  # Roughly 30 days
    
    def test_days_to_expiry_calculation(self, sample_option_contract):
        """Test days to expiry calculation."""
        contract = sample_option_contract
        
        # Should be approximately 30 days
        dte = contract.days_to_expiry
        assert 29 <= dte <= 31  # Allow for slight variation
    
    def test_mid_price_calculation(self, sample_option_contract):
        """Test mid price calculation."""
        contract = sample_option_contract
        
        expected_mid = (contract.bid + contract.ask) / 2
        assert contract.mid_price == expected_mid
    
    def test_mid_price_with_no_bid_ask(self):
        """Test mid price when bid/ask are None."""
        future_date = datetime.date.today() + datetime.timedelta(days=30)
        
        contract = OptionContract(
            symbol="AAPL",
            strike=Decimal('150.00'),
            expiration=future_date,
            option_type=OptionType.CALL
        )
        
        assert contract.mid_price is None
    
    def test_spread_calculation(self, sample_option_contract):
        """Test bid-ask spread calculation."""
        contract = sample_option_contract
        
        expected_spread = contract.ask - contract.bid
        assert contract.spread == expected_spread
    
    def test_spread_percentage_calculation(self, sample_option_contract):
        """Test spread percentage calculation."""
        contract = sample_option_contract
        
        expected_spread_pct = (contract.spread / contract.mid_price) * 100
        assert abs(contract.spread_pct - expected_spread_pct) < Decimal('0.01')
    
    def test_is_itm_call_option(self, sample_option_contract):
        """Test in-the-money check for call option."""
        contract = sample_option_contract  # Strike = 150.00
        
        # Call is ITM when underlying > strike
        assert contract.is_itm(Decimal('155.00')) == True
        assert contract.is_itm(Decimal('145.00')) == False
        assert contract.is_itm(Decimal('150.00')) == False  # ATM is not ITM
    
    def test_is_itm_put_option(self, sample_put_contract):
        """Test in-the-money check for put option."""
        contract = sample_put_contract  # Strike = 145.00
        
        # Put is ITM when underlying < strike
        assert contract.is_itm(Decimal('140.00')) == True
        assert contract.is_itm(Decimal('150.00')) == False
        assert contract.is_itm(Decimal('145.00')) == False  # ATM is not ITM
    
    def test_is_otm_call_option(self, sample_option_contract):
        """Test out-of-the-money check for call option."""
        contract = sample_option_contract  # Strike = 150.00
        
        # Call is OTM when underlying <= strike
        assert contract.is_otm(Decimal('145.00')) == True
        assert contract.is_otm(Decimal('150.00')) == True  # ATM is OTM
        assert contract.is_otm(Decimal('155.00')) == False
    
    def test_is_atm_option(self, sample_option_contract):
        """Test at-the-money check."""
        contract = sample_option_contract  # Strike = 150.00
        
        # Default tolerance is 1%
        assert contract.is_atm(Decimal('150.00')) == True
        assert contract.is_atm(Decimal('149.50')) == True  # Within 1%
        assert contract.is_atm(Decimal('150.50')) == True  # Within 1%
        assert contract.is_atm(Decimal('148.00')) == False  # Outside 1%
        
        # Custom tolerance test
        assert contract.is_atm(Decimal('148.00'), tolerance=Decimal('0.02')) == True  # Within 2%
    
    def test_intrinsic_value_call(self, sample_option_contract):
        """Test intrinsic value calculation for call option."""
        contract = sample_option_contract  # Strike = 150.00
        
        # Call intrinsic value = max(0, underlying - strike)
        assert contract.intrinsic_value(Decimal('155.00')) == Decimal('5.00')
        assert contract.intrinsic_value(Decimal('150.00')) == Decimal('0.00')
        assert contract.intrinsic_value(Decimal('145.00')) == Decimal('0.00')
    
    def test_intrinsic_value_put(self, sample_put_contract):
        """Test intrinsic value calculation for put option."""
        contract = sample_put_contract  # Strike = 145.00
        
        # Put intrinsic value = max(0, strike - underlying)
        assert contract.intrinsic_value(Decimal('140.00')) == Decimal('5.00')
        assert contract.intrinsic_value(Decimal('145.00')) == Decimal('0.00')
        assert contract.intrinsic_value(Decimal('150.00')) == Decimal('0.00')
    
    def test_time_value_calculation(self, sample_option_contract):
        """Test time value calculation."""
        contract = sample_option_contract  # Last = 5.55
        underlying_price = Decimal('150.00')  # ATM
        
        intrinsic = contract.intrinsic_value(underlying_price)  # 0 for ATM
        expected_time_value = contract.last - intrinsic
        
        assert contract.time_value(underlying_price) == expected_time_value
    
    def test_time_value_with_no_market_price(self):
        """Test time value when no market price available."""
        future_date = datetime.date.today() + datetime.timedelta(days=30)
        
        contract = OptionContract(
            symbol="AAPL",
            strike=Decimal('150.00'),
            expiration=future_date,
            option_type=OptionType.CALL
        )
        
        assert contract.time_value(Decimal('150.00')) is None
    
    def test_update_market_data(self, sample_option_contract):
        """Test market data update."""
        contract = sample_option_contract
        
        new_data = {
            'bid': 5.80,
            'ask': 5.90,
            'last': 5.85,
            'volume': 750,
            'delta': 0.55,
            'implied_volatility': 0.28
        }
        
        contract.update_market_data(new_data)
        
        assert contract.bid == Decimal('5.80')
        assert contract.ask == Decimal('5.90')
        assert contract.last == Decimal('5.85')
        assert contract.volume == 750
        assert contract.delta == Decimal('0.55')
        assert contract.implied_volatility == Decimal('0.28')
        assert contract.last_update is not None
    
    def test_update_market_data_with_invalid_data_raises_error(self, sample_option_contract):
        """Test that updating with invalid data raises validation error."""
        contract = sample_option_contract
        
        invalid_data = {
            'bid': 6.00,
            'ask': 5.50  # Bid > ask
        }
        
        with pytest.raises(ValidationError):
            contract.update_market_data(invalid_data)
    
    def test_to_dict_conversion(self, sample_option_contract):
        """Test contract to dictionary conversion."""
        contract = sample_option_contract
        contract_dict = contract.to_dict()
        
        assert contract_dict['symbol'] == "AAPL"
        assert contract_dict['strike'] == 150.00
        assert contract_dict['option_type'] == "CALL"
        assert contract_dict['bid'] == 5.50
        assert contract_dict['delta'] == 0.5
        assert 'contract_id' in contract_dict
        assert 'last_update' in contract_dict


class TestStockContract:
    """Test cases for StockContract class."""
    
    def test_stock_contract_creation(self, sample_stock_contract):
        """Test basic stock contract creation."""
        contract = sample_stock_contract
        
        assert contract.symbol == "AAPL"
        assert contract.exchange == "SMART"
        assert contract.currency == "USD"
        assert contract.status == ContractStatus.ACTIVE
    
    def test_stock_contract_id_generation(self, sample_stock_contract):
        """Test contract ID generation."""
        contract = sample_stock_contract
        
        expected_id = "STK_AAPL_SMART_USD"
        assert contract.contract_id == expected_id
    
    def test_invalid_stock_symbol_raises_error(self):
        """Test that invalid symbol raises error."""
        with pytest.raises(InvalidContractError, match="Symbol must be a non-empty string"):
            StockContract(symbol="")
    
    def test_negative_stock_price_raises_error(self):
        """Test that negative prices raise validation error."""
        with pytest.raises(ValidationError, match="Bid price cannot be negative"):
            StockContract(
                symbol="AAPL",
                bid=Decimal('-150.00')
            )
    
    def test_bid_greater_than_ask_raises_error(self):
        """Test that bid > ask raises validation error."""
        with pytest.raises(ValidationError, match="Bid price cannot exceed ask price"):
            StockContract(
                symbol="AAPL",
                bid=Decimal('150.00'),
                ask=Decimal('149.50')
            )
    
    def test_negative_volume_raises_error(self):
        """Test that negative volume raises validation error."""
        with pytest.raises(ValidationError, match="Volume cannot be negative"):
            StockContract(
                symbol="AAPL",
                volume=-1000
            )
    
    def test_negative_market_cap_raises_error(self):
        """Test that negative market cap raises validation error."""
        with pytest.raises(ValidationError, match="Market cap cannot be negative"):
            StockContract(
                symbol="AAPL",
                market_cap=Decimal('-1000000')
            )
    
    def test_stock_mid_price_calculation(self, sample_stock_contract):
        """Test mid price calculation."""
        contract = sample_stock_contract
        
        expected_mid = (contract.bid + contract.ask) / 2
        assert contract.mid_price == expected_mid
    
    def test_stock_spread_calculation(self, sample_stock_contract):
        """Test bid-ask spread calculation."""
        contract = sample_stock_contract
        
        expected_spread = contract.ask - contract.bid
        assert contract.spread == expected_spread
    
    def test_stock_spread_percentage(self, sample_stock_contract):
        """Test spread percentage calculation."""
        contract = sample_stock_contract
        
        expected_spread_pct = (contract.spread / contract.mid_price) * 100
        assert abs(contract.spread_pct - expected_spread_pct) < Decimal('0.01')
    
    def test_stock_update_market_data(self, sample_stock_contract):
        """Test market data update."""
        contract = sample_stock_contract
        
        new_data = {
            'bid': 151.00,
            'ask': 151.10,
            'last': 151.05,
            'volume': 2000000,
            'market_cap': 2600000000000,
            'pe_ratio': 26.0
        }
        
        contract.update_market_data(new_data)
        
        assert contract.bid == Decimal('151.00')
        assert contract.ask == Decimal('151.10')
        assert contract.last == Decimal('151.05')
        assert contract.volume == 2000000
        assert contract.market_cap == Decimal('2600000000000')
        assert contract.pe_ratio == Decimal('26.0')
        assert contract.last_update is not None
    
    def test_stock_to_dict_conversion(self, sample_stock_contract):
        """Test contract to dictionary conversion."""
        contract = sample_stock_contract
        contract_dict = contract.to_dict()
        
        assert contract_dict['symbol'] == "AAPL"
        assert contract_dict['exchange'] == "SMART"
        assert contract_dict['currency'] == "USD"
        assert contract_dict['bid'] == 150.00
        assert contract_dict['market_cap'] == 2500000000000.0
        assert 'contract_id' in contract_dict
        assert 'last_update' in contract_dict


class TestContractEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_option_with_zero_strike_raises_error(self):
        """Test that zero strike raises error."""
        future_date = datetime.date.today() + datetime.timedelta(days=30)
        
        with pytest.raises(InvalidContractError):
            OptionContract(
                symbol="AAPL",
                strike=Decimal('0'),
                expiration=future_date,
                option_type=OptionType.CALL
            )
    
    def test_option_with_zero_multiplier_raises_error(self):
        """Test that zero multiplier raises error."""
        future_date = datetime.date.today() + datetime.timedelta(days=30)
        
        with pytest.raises(InvalidContractError):
            OptionContract(
                symbol="AAPL",
                strike=Decimal('150.00'),
                expiration=future_date,
                option_type=OptionType.CALL,
                multiplier=0
            )
    
    def test_option_expiring_today(self):
        """Test option expiring today raises error."""
        today = datetime.date.today()
        
        with pytest.raises(ExpiredContractError):
            OptionContract(
                symbol="AAPL",
                strike=Decimal('150.00'),
                expiration=today,
                option_type=OptionType.CALL
            )
    
    def test_very_small_bid_ask_spread(self):
        """Test handling of very small bid-ask spreads."""
        future_date = datetime.date.today() + datetime.timedelta(days=30)
        
        contract = OptionContract(
            symbol="AAPL",
            strike=Decimal('150.00'),
            expiration=future_date,
            option_type=OptionType.CALL,
            bid=Decimal('5.0000'),
            ask=Decimal('5.0001')
        )
        
        assert contract.spread == Decimal('0.0001')
        assert contract.spread_pct is not None
    
    def test_extreme_greeks_values(self):
        """Test handling of extreme Greeks values."""
        future_date = datetime.date.today() + datetime.timedelta(days=30)
        
        # Test boundary values
        contract = OptionContract(
            symbol="AAPL",
            strike=Decimal('150.00'),
            expiration=future_date,
            option_type=OptionType.CALL,
            delta=Decimal('1.0'),  # Maximum delta
            gamma=Decimal('0.0'),  # Minimum gamma
            theta=Decimal('-10.0'),  # Large theta
            vega=Decimal('100.0'),  # Large vega
            implied_volatility=Decimal('2.0')  # 200% IV
        )
        
        assert contract.delta == Decimal('1.0')
        assert contract.gamma == Decimal('0.0')
        assert contract.theta == Decimal('-10.0')
        assert contract.vega == Decimal('100.0')
        assert contract.implied_volatility == Decimal('2.0')


@pytest.mark.performance
class TestContractPerformance:
    """Performance tests for contract operations."""
    
    def test_contract_creation_performance(self, performance_timer):
        """Test contract creation performance."""
        future_date = datetime.date.today() + datetime.timedelta(days=30)
        
        performance_timer.start()
        
        # Create 1000 contracts
        for i in range(1000):
            contract = OptionContract(
                symbol="AAPL",
                strike=Decimal(f'{150 + i}'),
                expiration=future_date,
                option_type=OptionType.CALL
            )
        
        performance_timer.stop()
        
        # Should complete in less than 1 second
        assert performance_timer.elapsed < 1.0
    
    def test_contract_validation_performance(self, performance_timer):
        """Test contract validation performance."""
        future_date = datetime.date.today() + datetime.timedelta(days=30)
        
        contracts = []
        for i in range(100):
            contract = OptionContract(
                symbol="AAPL",
                strike=Decimal(f'{150 + i}'),
                expiration=future_date,
                option_type=OptionType.CALL,
                bid=Decimal(f'{5 + i * 0.1}'),
                ask=Decimal(f'{5.1 + i * 0.1}'),
                delta=Decimal(f'{0.5 + i * 0.001}'),
                implied_volatility=Decimal(f'{0.2 + i * 0.001}')
            )
            contracts.append(contract)
        
        performance_timer.start()
        
        # Update market data for all contracts
        for i, contract in enumerate(contracts):
            contract.update_market_data({
                'bid': 6.0 + i * 0.1,
                'ask': 6.1 + i * 0.1,
                'delta': 0.6 + i * 0.001,
                'implied_volatility': 0.25 + i * 0.001
            })
        
        performance_timer.stop()
        
        # Should complete in less than 0.1 seconds
        assert performance_timer.elapsed < 0.1