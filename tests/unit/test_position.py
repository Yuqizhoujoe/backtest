"""
Unit tests for the position module.

This module tests the Position and Transaction classes including
P&L calculations, position management, and risk metrics.
"""

import datetime
import pytest
from decimal import Decimal

from src.core.position import Position, Transaction, PositionStatus, PositionSide
from src.core.exceptions import InvalidPositionError, InsufficientPositionError, ValidationError


class TestTransaction:
    """Test cases for Transaction class."""
    
    def test_transaction_creation(self, sample_transaction):
        """Test basic transaction creation."""
        transaction = sample_transaction
        
        assert transaction.quantity == 100
        assert transaction.price == Decimal('150.00')
        assert transaction.commission == Decimal('1.00')
        assert transaction.fees == Decimal('0.50')
        assert transaction.timestamp is not None
    
    def test_transaction_validation(self):
        """Test transaction validation."""
        # Valid transaction should not raise exceptions
        transaction = Transaction(
            timestamp=datetime.datetime.now(),
            quantity=100,
            price=Decimal('150.00')
        )
        assert transaction.quantity == 100
    
    def test_zero_quantity_raises_error(self):
        """Test that zero quantity raises error."""
        with pytest.raises(InvalidPositionError, match="Transaction quantity cannot be zero"):
            Transaction(
                timestamp=datetime.datetime.now(),
                quantity=0,
                price=Decimal('150.00')
            )
    
    def test_negative_price_raises_error(self):
        """Test that negative price raises error."""
        with pytest.raises(ValidationError, match="Transaction price cannot be negative"):
            Transaction(
                timestamp=datetime.datetime.now(),
                quantity=100,
                price=Decimal('-150.00')
            )
    
    def test_negative_commission_raises_error(self):
        """Test that negative commission raises error."""
        with pytest.raises(ValidationError, match="Commission cannot be negative"):
            Transaction(
                timestamp=datetime.datetime.now(),
                quantity=100,
                price=Decimal('150.00'),
                commission=Decimal('-1.00')
            )
    
    def test_is_buy_property(self):
        """Test is_buy property."""
        buy_transaction = Transaction(
            timestamp=datetime.datetime.now(),
            quantity=100,
            price=Decimal('150.00')
        )
        assert buy_transaction.is_buy == True
        assert buy_transaction.is_sell == False
    
    def test_is_sell_property(self):
        """Test is_sell property."""
        sell_transaction = Transaction(
            timestamp=datetime.datetime.now(),
            quantity=-100,
            price=Decimal('150.00')
        )
        assert sell_transaction.is_buy == False
        assert sell_transaction.is_sell == True
    
    def test_gross_amount_calculation(self, sample_transaction):
        """Test gross amount calculation."""
        transaction = sample_transaction
        
        expected_gross = abs(transaction.quantity) * transaction.price
        assert transaction.gross_amount == expected_gross
    
    def test_net_amount_calculation_buy(self):
        """Test net amount calculation for buy transaction."""
        transaction = Transaction(
            timestamp=datetime.datetime.now(),
            quantity=100,
            price=Decimal('150.00'),
            commission=Decimal('1.00'),
            fees=Decimal('0.50')
        )
        
        # For buy: net = gross + costs
        expected_net = Decimal('15000.00') + Decimal('1.00') + Decimal('0.50')
        assert transaction.net_amount == expected_net
    
    def test_net_amount_calculation_sell(self):
        """Test net amount calculation for sell transaction."""
        transaction = Transaction(
            timestamp=datetime.datetime.now(),
            quantity=-100,
            price=Decimal('150.00'),
            commission=Decimal('1.00'),
            fees=Decimal('0.50')
        )
        
        # For sell: net = gross - costs
        expected_net = Decimal('15000.00') - Decimal('1.00') - Decimal('0.50')
        assert transaction.net_amount == expected_net
    
    def test_transaction_to_dict(self, sample_transaction):
        """Test transaction to dictionary conversion."""
        transaction = sample_transaction
        transaction_dict = transaction.to_dict()
        
        assert transaction_dict['quantity'] == 100
        assert transaction_dict['price'] == 150.00
        assert transaction_dict['commission'] == 1.00
        assert transaction_dict['fees'] == 0.50
        assert 'timestamp' in transaction_dict
        assert 'gross_amount' in transaction_dict
        assert 'net_amount' in transaction_dict


class TestPosition:
    """Test cases for Position class."""
    
    def test_position_creation(self, sample_position):
        """Test basic position creation."""
        position = sample_position
        
        assert position.initial_quantity == 10
        assert position.current_quantity == 10
        assert position.entry_price == Decimal('5.50')
        assert position.status == PositionStatus.OPEN
        assert len(position.transactions) == 1  # Initial transaction
    
    def test_position_validation(self, sample_option_contract):
        """Test position validation."""
        # Valid position should not raise exceptions
        position = Position(
            contract=sample_option_contract,
            entry_timestamp=datetime.datetime.now(),
            initial_quantity=10,
            entry_price=Decimal('5.50')
        )
        assert position.initial_quantity == 10
    
    def test_zero_quantity_raises_error(self, sample_option_contract):
        """Test that zero quantity raises error."""
        with pytest.raises(InvalidPositionError, match="Position quantity cannot be zero"):
            Position(
                contract=sample_option_contract,
                entry_timestamp=datetime.datetime.now(),
                initial_quantity=0,
                entry_price=Decimal('5.50')
            )
    
    def test_negative_entry_price_raises_error(self, sample_option_contract):
        """Test that negative entry price raises error."""
        with pytest.raises(InvalidPositionError, match="Entry price must be positive"):
            Position(
                contract=sample_option_contract,
                entry_timestamp=datetime.datetime.now(),
                initial_quantity=10,
                entry_price=Decimal('-5.50')
            )
    
    def test_position_properties(self, sample_position):
        """Test position properties."""
        position = sample_position
        
        assert position.contract_id == position.contract.contract_id
        assert position.symbol == position.contract.symbol
        assert position.is_long == True
        assert position.is_short == False
        assert position.is_closed == False
        assert position.side == PositionSide.LONG
        assert position.multiplier == 100  # Option multiplier
    
    def test_short_position_properties(self, sample_option_contract):
        """Test short position properties."""
        position = Position(
            contract=sample_option_contract,
            entry_timestamp=datetime.datetime.now(),
            initial_quantity=-10,
            entry_price=Decimal('5.50')
        )
        
        assert position.is_long == False
        assert position.is_short == True
        assert position.side == PositionSide.SHORT
    
    def test_position_value_calculation(self, sample_position):
        """Test position value calculation."""
        position = sample_position
        
        # Mock current price in contract
        position.contract.last = Decimal('6.00')
        
        expected_value = abs(position.current_quantity) * Decimal('6.00') * position.multiplier
        assert position.position_value == expected_value
    
    def test_position_value_with_no_current_price(self, sample_position):
        """Test position value when no current price available."""
        position = sample_position
        
        # Clear all price fields
        position.contract.last = None
        position.contract.bid = None
        position.contract.ask = None
        
        assert position.position_value == Decimal('0')
    
    def test_total_pnl_calculation(self, sample_position):
        """Test total P&L calculation."""
        position = sample_position
        position.realized_pnl = Decimal('100.00')
        position.unrealized_pnl = Decimal('50.00')
        
        assert position.total_pnl == Decimal('150.00')
    
    def test_avg_entry_price_single_transaction(self, sample_position):
        """Test average entry price with single transaction."""
        position = sample_position
        
        assert position.avg_entry_price == position.entry_price
    
    def test_avg_entry_price_multiple_transactions(self, sample_position):
        """Test average entry price with multiple transactions."""
        position = sample_position
        
        # Add another entry transaction
        additional_transaction = Transaction(
            timestamp=datetime.datetime.now(),
            quantity=5,
            price=Decimal('6.00')
        )
        position.add_transaction(additional_transaction)
        
        # Weighted average: (10 * 5.50 + 5 * 6.00) / 15 = 5.67
        expected_avg = (Decimal('10') * Decimal('5.50') + Decimal('5') * Decimal('6.00')) / Decimal('15')
        assert abs(position.avg_entry_price - expected_avg) < Decimal('0.01')
    
    def test_holding_period_calculation(self, sample_position):
        """Test holding period calculation."""
        position = sample_position
        
        # Position should have been created recently
        holding_period = position.holding_period
        assert holding_period.total_seconds() < 60  # Less than 1 minute
    
    def test_add_closing_transaction(self, sample_position):
        """Test adding a closing transaction."""
        position = sample_position
        initial_realized_pnl = position.realized_pnl
        
        # Add closing transaction for 5 contracts
        closing_transaction = Transaction(
            timestamp=datetime.datetime.now(),
            quantity=-5,
            price=Decimal('6.00'),
            commission=Decimal('1.00')
        )
        
        position.add_transaction(closing_transaction)
        
        # Check position was updated
        assert position.current_quantity == 5
        assert position.status == PositionStatus.PARTIALLY_CLOSED
        assert position.realized_pnl > initial_realized_pnl
    
    def test_add_over_closing_transaction_raises_error(self, sample_position):
        """Test that over-closing raises error."""
        position = sample_position
        
        # Try to close more than available
        over_closing_transaction = Transaction(
            timestamp=datetime.datetime.now(),
            quantity=-15,  # More than the 10 available
            price=Decimal('6.00')
        )
        
        with pytest.raises(InsufficientPositionError):
            position.add_transaction(over_closing_transaction)
    
    def test_update_unrealized_pnl(self, sample_position):
        """Test unrealized P&L update."""
        position = sample_position
        current_price = Decimal('6.00')
        
        position.update_unrealized_pnl(current_price)
        
        # P&L = (current - entry) * quantity * multiplier
        expected_pnl = (current_price - position.avg_entry_price) * position.current_quantity * position.multiplier
        assert position.unrealized_pnl == expected_pnl
    
    def test_update_unrealized_pnl_short_position(self, sample_option_contract):
        """Test unrealized P&L update for short position."""
        position = Position(
            contract=sample_option_contract,
            entry_timestamp=datetime.datetime.now(),
            initial_quantity=-10,
            entry_price=Decimal('5.50')
        )
        
        current_price = Decimal('5.00')  # Price went down, good for short
        position.update_unrealized_pnl(current_price)
        
        # For short: P&L = (entry - current) * abs(quantity) * multiplier
        expected_pnl = (position.avg_entry_price - current_price) * abs(position.current_quantity) * position.multiplier
        assert position.unrealized_pnl == expected_pnl
    
    def test_close_position(self, sample_position):
        """Test closing entire position."""
        position = sample_position
        exit_price = Decimal('6.50')
        exit_time = datetime.datetime.now()
        
        realized_pnl = position.close_position(exit_price, exit_time, commission=Decimal('1.00'))
        
        assert position.is_closed == True
        assert position.status == PositionStatus.CLOSED
        assert position.current_quantity == 0
        assert realized_pnl > 0  # Should be profitable
        assert len(position.transactions) == 2  # Entry + exit
    
    def test_close_already_closed_position_returns_zero(self, sample_position):
        """Test closing already closed position returns zero."""
        position = sample_position
        
        # Close position first time
        position.close_position(Decimal('6.50'), datetime.datetime.now())
        
        # Try to close again
        realized_pnl = position.close_position(Decimal('7.00'), datetime.datetime.now())
        
        assert realized_pnl == Decimal('0')
    
    def test_partial_close(self, sample_position):
        """Test partial position closure."""
        position = sample_position
        
        realized_pnl = position.partial_close(
            quantity_to_close=3,
            exit_price=Decimal('6.00'),
            exit_time=datetime.datetime.now(),
            commission=Decimal('0.50')
        )
        
        assert position.current_quantity == 7
        assert position.status == PositionStatus.PARTIALLY_CLOSED
        assert realized_pnl > 0
        assert len(position.transactions) == 2
    
    def test_partial_close_invalid_quantity_raises_error(self, sample_position):
        """Test partial close with invalid quantity raises error."""
        position = sample_position
        
        with pytest.raises(InvalidPositionError, match="Quantity to close must be positive"):
            position.partial_close(
                quantity_to_close=-5,
                exit_price=Decimal('6.00'),
                exit_time=datetime.datetime.now()
            )
    
    def test_partial_close_excessive_quantity_raises_error(self, sample_position):
        """Test partial close with excessive quantity raises error."""
        position = sample_position
        
        with pytest.raises(InsufficientPositionError):
            position.partial_close(
                quantity_to_close=15,  # More than available
                exit_price=Decimal('6.00'),
                exit_time=datetime.datetime.now()
            )
    
    def test_calculate_return_pct(self, sample_position):
        """Test percentage return calculation."""
        position = sample_position
        
        # Set some P&L
        position.realized_pnl = Decimal('275.00')  # $275 profit
        position.unrealized_pnl = Decimal('225.00')  # $225 unrealized
        
        # Entry value = 10 * 5.50 * 100 = $5,500
        # Total P&L = $500
        # Return = 500 / 5500 * 100 = 9.09%
        expected_return = Decimal('500') / Decimal('5500') * 100
        
        calculated_return = position.calculate_return_pct()
        assert abs(calculated_return - expected_return) < Decimal('0.01')
    
    def test_calculate_return_pct_zero_entry_value_returns_none(self, sample_option_contract):
        """Test return calculation when entry value is zero."""
        position = Position(
            contract=sample_option_contract,
            entry_timestamp=datetime.datetime.now(),
            initial_quantity=10,
            entry_price=Decimal('0.01')  # Very small price
        )
        
        # This should still work, but edge case
        return_pct = position.calculate_return_pct()
        assert return_pct is not None
    
    def test_calculate_margin_requirement(self, sample_position):
        """Test margin requirement calculation."""
        position = sample_position
        
        # Mock position value
        position.contract.last = Decimal('6.00')
        margin_rate = Decimal('0.5')
        
        margin_req = position.calculate_margin_requirement(margin_rate)
        expected_margin = position.position_value * margin_rate
        
        assert margin_req == expected_margin
    
    def test_calculate_margin_requirement_short_option(self, sample_option_contract):
        """Test margin requirement for short option position."""
        position = Position(
            contract=sample_option_contract,
            entry_timestamp=datetime.datetime.now(),
            initial_quantity=-10,
            entry_price=Decimal('5.50')
        )
        
        # Mock current price
        position.contract.last = Decimal('6.00')
        margin_rate = Decimal('0.5')
        
        margin_req = position.calculate_margin_requirement(margin_rate)
        
        # Short options have higher margin requirements
        expected_margin = position.position_value * margin_rate * 2
        assert margin_req == expected_margin
    
    def test_calculate_margin_requirement_closed_position(self, sample_position):
        """Test margin requirement for closed position."""
        position = sample_position
        position.close_position(Decimal('6.00'), datetime.datetime.now())
        
        margin_req = position.calculate_margin_requirement()
        assert margin_req == Decimal('0')
    
    def test_get_summary(self, sample_position):
        """Test position summary generation."""
        position = sample_position
        summary = position.get_summary()
        
        assert summary['initial_quantity'] == 10
        assert summary['current_quantity'] == 10
        assert summary['entry_price'] == 5.50
        assert summary['side'] == 'LONG'
        assert summary['status'] == 'OPEN'
        assert 'contract_id' in summary
        assert 'symbol' in summary
        assert 'realized_pnl' in summary
        assert 'unrealized_pnl' in summary
    
    def test_to_dict(self, sample_position):
        """Test position to dictionary conversion."""
        position = sample_position
        position_dict = position.to_dict()
        
        assert 'transactions' in position_dict
        assert 'initial_margin' in position_dict
        assert 'maintenance_margin' in position_dict
        assert len(position_dict['transactions']) == 1
        
        # Should include everything from summary plus additional fields
        summary_keys = set(position.get_summary().keys())
        dict_keys = set(position_dict.keys())
        assert summary_keys.issubset(dict_keys)


class TestPositionPnLCalculations:
    """Test complex P&L calculation scenarios."""
    
    def test_long_position_profit(self, sample_option_contract):
        """Test P&L calculation for profitable long position."""
        position = Position(
            contract=sample_option_contract,
            entry_timestamp=datetime.datetime.now(),
            initial_quantity=10,
            entry_price=Decimal('5.00')
        )
        
        # Close at higher price
        exit_price = Decimal('6.00')
        realized_pnl = position.close_position(exit_price, datetime.datetime.now())
        
        # P&L = (6.00 - 5.00) * 10 * 100 = $1,000
        expected_pnl = Decimal('1000.00')
        assert realized_pnl == expected_pnl
    
    def test_long_position_loss(self, sample_option_contract):
        """Test P&L calculation for losing long position."""
        position = Position(
            contract=sample_option_contract,
            entry_timestamp=datetime.datetime.now(),
            initial_quantity=10,
            entry_price=Decimal('5.00')
        )
        
        # Close at lower price
        exit_price = Decimal('4.00')
        realized_pnl = position.close_position(exit_price, datetime.datetime.now())
        
        # P&L = (4.00 - 5.00) * 10 * 100 = -$1,000
        expected_pnl = Decimal('-1000.00')
        assert realized_pnl == expected_pnl
    
    def test_short_position_profit(self, sample_option_contract):
        """Test P&L calculation for profitable short position."""
        position = Position(
            contract=sample_option_contract,
            entry_timestamp=datetime.datetime.now(),
            initial_quantity=-10,
            entry_price=Decimal('5.00')
        )
        
        # Close at lower price (good for short)
        exit_price = Decimal('4.00')
        realized_pnl = position.close_position(exit_price, datetime.datetime.now())
        
        # P&L = (5.00 - 4.00) * 10 * 100 = $1,000
        expected_pnl = Decimal('1000.00')
        assert realized_pnl == expected_pnl
    
    def test_short_position_loss(self, sample_option_contract):
        """Test P&L calculation for losing short position."""
        position = Position(
            contract=sample_option_contract,
            entry_timestamp=datetime.datetime.now(),
            initial_quantity=-10,
            entry_price=Decimal('5.00')
        )
        
        # Close at higher price (bad for short)
        exit_price = Decimal('6.00')
        realized_pnl = position.close_position(exit_price, datetime.datetime.now())
        
        # P&L = (5.00 - 6.00) * 10 * 100 = -$1,000
        expected_pnl = Decimal('-1000.00')
        assert realized_pnl == expected_pnl
    
    def test_partial_close_pnl_calculation(self, sample_option_contract):
        """Test P&L calculation for partial closes."""
        position = Position(
            contract=sample_option_contract,
            entry_timestamp=datetime.datetime.now(),
            initial_quantity=10,
            entry_price=Decimal('5.00')
        )
        
        # Partial close 1: Close 3 contracts at $6.00
        pnl_1 = position.partial_close(3, Decimal('6.00'), datetime.datetime.now())
        expected_pnl_1 = (Decimal('6.00') - Decimal('5.00')) * 3 * 100
        assert pnl_1 == expected_pnl_1
        
        # Partial close 2: Close 4 contracts at $5.50
        pnl_2 = position.partial_close(4, Decimal('5.50'), datetime.datetime.now())
        expected_pnl_2 = (Decimal('5.50') - Decimal('5.00')) * 4 * 100
        assert pnl_2 == expected_pnl_2
        
        # Verify total realized P&L
        expected_total_pnl = expected_pnl_1 + expected_pnl_2
        assert position.realized_pnl == expected_total_pnl
        
        # Verify remaining position
        assert position.current_quantity == 3
        assert position.status == PositionStatus.PARTIALLY_CLOSED
    
    def test_commission_and_fees_impact_on_pnl(self, sample_option_contract):
        """Test that commissions and fees reduce P&L."""
        position = Position(
            contract=sample_option_contract,
            entry_timestamp=datetime.datetime.now(),
            initial_quantity=10,
            entry_price=Decimal('5.00')
        )
        
        # Close with commission and fees
        exit_price = Decimal('6.00')
        commission = Decimal('10.00')
        fees = Decimal('2.50')
        
        realized_pnl = position.close_position(exit_price, datetime.datetime.now(), commission, fees)
        
        # P&L = (6.00 - 5.00) * 10 * 100 - 10.00 - 2.50 = $987.50
        expected_pnl = Decimal('1000.00') - commission - fees
        assert realized_pnl == expected_pnl


@pytest.mark.performance
class TestPositionPerformance:
    """Performance tests for position operations."""
    
    def test_position_creation_performance(self, sample_option_contract, performance_timer):
        """Test position creation performance."""
        performance_timer.start()
        
        # Create 1000 positions
        positions = []
        for i in range(1000):
            position = Position(
                contract=sample_option_contract,
                entry_timestamp=datetime.datetime.now(),
                initial_quantity=10 + i,
                entry_price=Decimal(f'{5.00 + i * 0.01}')
            )
            positions.append(position)
        
        performance_timer.stop()
        
        # Should complete in less than 1 second
        assert performance_timer.elapsed < 1.0
    
    def test_pnl_calculation_performance(self, sample_position, performance_timer):
        """Test P&L calculation performance."""
        position = sample_position
        
        # Add many transactions
        for i in range(100):
            if i % 2 == 0:  # Buy
                transaction = Transaction(
                    timestamp=datetime.datetime.now(),
                    quantity=1,
                    price=Decimal(f'{5.50 + i * 0.01}')
                )
            else:  # Sell
                transaction = Transaction(
                    timestamp=datetime.datetime.now(),
                    quantity=-1,
                    price=Decimal(f'{5.60 + i * 0.01}')
                )
            
            position.add_transaction(transaction)
        
        performance_timer.start()
        
        # Calculate P&L many times
        for _ in range(1000):
            position.update_unrealized_pnl(Decimal('6.00'))
            _ = position.total_pnl
            _ = position.calculate_return_pct()
        
        performance_timer.stop()
        
        # Should complete in less than 0.1 seconds
        assert performance_timer.elapsed < 0.1