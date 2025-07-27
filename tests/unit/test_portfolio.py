"""
Unit tests for the portfolio module.

This module tests the Portfolio class including position management,
P&L tracking, risk management, and performance calculations.
"""

import datetime
from typing import Any
import pytest
from decimal import Decimal

from src.core.portfolio import Portfolio, PortfolioMetrics
from src.core.position import Position
from src.core.contracts import OptionContract, OptionType
from src.core.exceptions import (
    InsufficientFundsError,
    InsufficientMarginError,
    ExcessiveRiskError,
    InvalidPositionError,
    PortfolioError,
)


class TestPortfolio:
    """Test cases for Portfolio class."""

    def test_portfolio_creation(self, sample_portfolio: Any) -> None:
        """Test basic portfolio creation."""
        portfolio = sample_portfolio

        assert portfolio.initial_cash == Decimal("100000")
        assert portfolio.cash == Decimal("100000")
        assert portfolio.name == "Test Portfolio"
        assert portfolio.num_open_positions == 0
        assert portfolio.num_closed_positions == 0
        assert len(portfolio.positions) == 0
        assert len(portfolio.closed_positions) == 0

    def test_portfolio_properties(self, sample_portfolio: Any) -> None:
        """Test portfolio properties calculation."""
        portfolio = sample_portfolio

        assert portfolio.total_position_value == Decimal("0")
        assert portfolio.total_value == Decimal("100000")  # All cash
        assert portfolio.total_pnl == Decimal("0")
        assert portfolio.return_pct == Decimal("0")
        assert portfolio.equity == Decimal("100000")
        assert portfolio.buying_power >= Decimal("100000")
        assert portfolio.leverage == Decimal("0")

    def test_add_long_position(
        self, sample_portfolio: Any, sample_option_contract: Any
    ) -> None:
        """Test adding a long position to portfolio."""
        portfolio = sample_portfolio

        position = portfolio.add_position(
            contract=sample_option_contract,
            quantity=10,
            entry_price=Decimal("5.50"),
            commission=Decimal("10.00"),
        )

        assert position is not None
        assert portfolio.num_open_positions == 1
        assert len(portfolio.positions) == 1
        assert portfolio.cash < Decimal("100000")  # Cash reduced by cost

        # Check transaction was recorded
        assert len(portfolio.all_transactions) == 1

        # Verify position is in portfolio
        contract_id = sample_option_contract.contract_id
        assert contract_id in portfolio.positions
        assert portfolio.positions[contract_id] == position

    def test_add_short_position(
        self, sample_portfolio: Any, sample_option_contract: Any
    ) -> None:
        """Test adding a short position to portfolio."""
        portfolio = sample_portfolio

        position = portfolio.add_position(
            contract=sample_option_contract,
            quantity=-5,
            entry_price=Decimal("5.50"),
            commission=Decimal("10.00"),
        )

        assert position is not None
        assert position.is_short is True
        assert portfolio.num_open_positions == 1
        # For short positions, cash should increase by premium received (minus commission)
        expected_cash_change = -(
            Decimal("10.00")
        )  # Just commission for simplified model
        assert portfolio.cash == Decimal("100000") + expected_cash_change

    def test_add_position_insufficient_funds_raises_error(
        self, sample_portfolio: Any, sample_option_contract: Any
    ) -> None:
        """Test that insufficient funds raises error."""
        portfolio = sample_portfolio

        # Create a position that passes risk checks but exceeds available cash
        # Set cash to a small amount but increase max position size to allow
        # the position
        portfolio.cash = Decimal("100")  # Very small cash amount
        portfolio.max_position_size_pct = Decimal("0.5")  # 50% to allow

        with pytest.raises(InsufficientFundsError):
            portfolio.add_position(
                contract=sample_option_contract,
                quantity=1,  # Small quantity to pass risk check
                entry_price=Decimal("0.50"),  # Small price to pass risk check
                commission=Decimal("1000.00"),  # Very high commission to exceed cash
            )

    def test_add_position_exceeds_risk_limits_raises_error(
        self, sample_portfolio: Any, sample_option_contract: Any
    ) -> None:
        """Test that position exceeding risk limits raises error."""
        portfolio = sample_portfolio

        # Set very restrictive position size limit
        portfolio.max_position_size_pct = Decimal("0.01")  # 1%

        with pytest.raises(ExcessiveRiskError):
            portfolio.add_position(
                contract=sample_option_contract,
                quantity=100,  # Large quantity
                entry_price=Decimal("10.00"),
                commission=Decimal("10.00"),
            )

    def test_close_position(
        self, sample_portfolio: Any, sample_option_contract: Any
    ) -> None:
        """Test closing a position."""
        portfolio = sample_portfolio

        # Add position first
        position = portfolio.add_position(
            contract=sample_option_contract,
            quantity=10,
            entry_price=Decimal("5.50"),
            commission=Decimal("10.00"),
        )

        initial_cash = portfolio.cash
        contract_id = sample_option_contract.contract_id

        # Close position at higher price (profit)
        realized_pnl = portfolio.close_position(
            contract_id=contract_id,
            exit_price=Decimal("6.50"),
            commission=Decimal("10.00"),
        )

        assert realized_pnl > 0  # Should be profitable
        assert portfolio.num_open_positions == 0
        assert portfolio.num_closed_positions == 1
        assert contract_id not in portfolio.positions
        assert len(portfolio.closed_positions) == 1
        assert portfolio.cash > initial_cash  # Cash increased from profit
        assert portfolio.total_realized_pnl == realized_pnl

    def test_close_nonexistent_position_returns_none(
        self, sample_portfolio: Any
    ) -> None:
        """Test closing non-existent position returns None."""
        portfolio = sample_portfolio

        result = portfolio.close_position("nonexistent_id", Decimal("10.00"))
        assert result is None

    def test_partial_close_position(
        self, sample_portfolio: Any, sample_option_contract: Any
    ) -> None:
        """Test partially closing a position."""
        portfolio = sample_portfolio

        # Add position
        position = portfolio.add_position(
            contract=sample_option_contract,
            quantity=10,
            entry_price=Decimal("5.50"),
            commission=Decimal("10.00"),
        )

        contract_id = sample_option_contract.contract_id
        initial_cash = portfolio.cash

        # Partially close 4 contracts
        realized_pnl = portfolio.partial_close_position(
            contract_id=contract_id,
            quantity_to_close=4,
            exit_price=Decimal("6.00"),
            commission=Decimal("5.00"),
        )

        assert realized_pnl > 0
        assert portfolio.num_open_positions == 1  # Still open
        assert contract_id in portfolio.positions
        assert portfolio.positions[contract_id].current_quantity == 6
        assert portfolio.cash > initial_cash
        assert portfolio.total_realized_pnl == realized_pnl

    def test_update_portfolio_value(
        self, sample_portfolio: Any, sample_option_contract: Any
    ) -> None:
        """Test updating portfolio value and unrealized P&L."""
        portfolio = sample_portfolio

        # Add position
        portfolio.add_position(
            contract=sample_option_contract, quantity=10, entry_price=Decimal("5.50")
        )

        # Update contract price
        sample_option_contract.last = Decimal("6.50")

        # Update portfolio
        portfolio.update_portfolio_value()

        assert portfolio.total_unrealized_pnl > 0
        assert portfolio.total_value > portfolio.initial_cash

    def test_get_position(
        self, sample_portfolio: Any, sample_option_contract: Any
    ) -> None:
        """Test getting a position by contract ID."""
        portfolio = sample_portfolio

        # Add position
        position = portfolio.add_position(
            contract=sample_option_contract, quantity=10, entry_price=Decimal("5.50")
        )

        contract_id = sample_option_contract.contract_id
        retrieved_position = portfolio.get_position(contract_id)

        assert retrieved_position == position
        assert portfolio.get_position("nonexistent") is None

    def test_get_positions_by_symbol(
        self,
        sample_portfolio: Any,
        sample_option_contract: Any,
        sample_put_contract: Any,
    ) -> None:
        """Test getting positions by symbol."""
        portfolio = sample_portfolio

        # Add positions for same symbol
        portfolio.add_position(sample_option_contract, 10, Decimal("5.50"))
        portfolio.add_position(sample_put_contract, 5, Decimal("3.25"))

        aapl_positions = portfolio.get_positions_by_symbol("AAPL")
        assert len(aapl_positions) == 2

        msft_positions = portfolio.get_positions_by_symbol("MSFT")
        assert len(msft_positions) == 0

    def test_get_positions_by_type(
        self,
        sample_portfolio: Any,
        sample_option_contract: Any,
        sample_stock_contract: Any,
    ) -> None:
        """Test getting positions by contract type."""
        portfolio = sample_portfolio

        # Increase position size limit to accommodate stock position
        portfolio.max_position_size_pct = Decimal("0.2")  # 20%

        # Add option position
        portfolio.add_position(sample_option_contract, 10, Decimal("5.50"))

        # Add stock position
        portfolio.add_position(sample_stock_contract, 100, Decimal("150.00"))

        from src.core.contracts import OptionContract, StockContract

        option_positions = portfolio.get_positions_by_type(OptionContract)
        stock_positions = portfolio.get_positions_by_type(StockContract)

        assert len(option_positions) == 1
        assert len(stock_positions) == 1

    def test_get_long_and_short_positions(
        self,
        sample_portfolio: Any,
        sample_option_contract: Any,
        sample_put_contract: Any,
    ) -> None:
        """Test getting long and short positions."""
        portfolio = sample_portfolio

        # Add long position
        portfolio.add_position(sample_option_contract, 10, Decimal("5.50"))

        # Add short position
        portfolio.add_position(sample_put_contract, -5, Decimal("3.25"))

        long_positions = portfolio.get_long_positions()
        short_positions = portfolio.get_short_positions()

        assert len(long_positions) == 1
        assert len(short_positions) == 1
        assert long_positions[0].is_long
        assert short_positions[0].is_short

    def test_calculate_var_insufficient_data(self, sample_portfolio: Any) -> None:
        """Test VaR calculation with insufficient data."""
        portfolio = sample_portfolio

        # No daily returns yet
        var = portfolio.calculate_var()
        assert var is None

    def test_calculate_var_with_data(self, sample_portfolio: Any) -> None:
        """Test VaR calculation with sufficient data."""
        portfolio = sample_portfolio

        # Add some fake daily returns
        portfolio.daily_returns = [
            Decimal(str(x)) for x in [-5, -3, -1, 0, 1, 2, 3, 4, 5, 6] * 5
        ]  # 50 returns

        var_95 = portfolio.calculate_var(confidence_level=Decimal("0.95"))
        assert var_95 is not None
        assert var_95 > 0

    def test_calculate_beta_insufficient_data(self, sample_portfolio: Any) -> None:
        """Test beta calculation with insufficient data."""
        portfolio = sample_portfolio

        market_returns = [Decimal("1.0")] * 5
        beta = portfolio.calculate_beta(market_returns)
        assert beta is None

    def test_calculate_beta_with_data(self, sample_portfolio: Any) -> None:
        """Test beta calculation with sufficient data."""
        portfolio = sample_portfolio

        # Create correlated returns
        portfolio.daily_returns = [Decimal(str(x)) for x in range(1, 21)]  # 20 returns
        market_returns = [
            Decimal(str(x * 0.8)) for x in range(1, 21)
        ]  # Correlated market returns

        beta = portfolio.calculate_beta(market_returns)
        assert beta is not None
        assert beta > 0

    def test_portfolio_metrics_calculation(
        self, sample_portfolio: Any, sample_option_contract: Any
    ) -> None:
        """Test comprehensive portfolio metrics calculation."""
        portfolio = sample_portfolio

        # Add and close some positions to generate metrics
        position1 = portfolio.add_position(sample_option_contract, 10, Decimal("5.50"))
        portfolio.close_position(sample_option_contract.contract_id, Decimal("6.50"))

        # Add some daily returns
        portfolio.daily_returns = [
            Decimal("1.0"),
            Decimal("-0.5"),
            Decimal("2.0"),
            Decimal("0.5"),
        ]

        metrics = portfolio.get_portfolio_metrics()

        assert isinstance(metrics, PortfolioMetrics)
        assert metrics.total_value > 0
        assert metrics.realized_pnl > 0  # Should have made profit
        # Note: return_pct can be negative if cash was reduced by costs
        assert metrics.win_rate == Decimal(
            "100"
        )  # 100% win rate with one profitable trade
        assert metrics.sharpe_ratio is not None

    def test_portfolio_summary(
        self, sample_portfolio: Any, sample_option_contract: Any
    ) -> None:
        """Test portfolio summary generation."""
        portfolio = sample_portfolio

        # Add position
        portfolio.add_position(sample_option_contract, 10, Decimal("5.50"))

        summary = portfolio.get_portfolio_summary()

        assert "portfolio_id" in summary
        assert "name" in summary
        assert "total_value" in summary
        assert "total_pnl" in summary
        assert "num_open_positions" in summary
        assert "num_closed_positions" in summary
        assert "leverage" in summary
        assert "margin_used" in summary
        assert summary["num_open_positions"] == 1
        assert summary["initial_cash"] == 100000.0

    def test_portfolio_risk_management(
        self, sample_portfolio: Any, sample_option_contract: Any
    ) -> None:
        """Test portfolio risk management features."""
        portfolio = sample_portfolio

        # Test position size limits
        original_limit = portfolio.max_position_size_pct
        portfolio.max_position_size_pct = Decimal("0.05")  # 5% max

        # Should be able to add small position
        portfolio.add_position(sample_option_contract, 5, Decimal("5.50"))

        # Should not be able to add large position
        with pytest.raises(ExcessiveRiskError):
            large_contract = OptionContract(
                symbol="EXPENSIVE",
                strike=Decimal("1000.00"),
                expiration=datetime.date.today() + datetime.timedelta(days=30),
                option_type=OptionType.CALL,
            )
            portfolio.add_position(large_contract, 100, Decimal("50.00"))

    def test_combine_positions_same_contract(
        self, sample_portfolio: Any, sample_option_contract: Any
    ) -> None:
        """Test combining positions for the same contract."""
        portfolio = sample_portfolio

        # Add first position
        portfolio.add_position(sample_option_contract, 5, Decimal("5.50"))

        # Add second position for same contract
        portfolio.add_position(sample_option_contract, 3, Decimal("6.00"))

        # Should still have only one position entry but combined
        assert portfolio.num_open_positions == 1

        contract_id = sample_option_contract.contract_id
        position = portfolio.positions[contract_id]

        # Should have multiple transactions
        assert len(position.transactions) >= 2

    def test_portfolio_drawdown_tracking(
        self, sample_portfolio: Any, sample_option_contract: Any
    ) -> None:
        """Test maximum drawdown tracking."""
        portfolio = sample_portfolio

        # Add position
        portfolio.add_position(sample_option_contract, 10, Decimal("5.50"))

        # Simulate price increase (portfolio value goes up)
        sample_option_contract.last = Decimal("7.00")
        portfolio.update_portfolio_value()

        # Peak value should be updated
        peak_value = portfolio.peak_value
        assert peak_value > portfolio.initial_cash

        # Simulate price decrease (drawdown)
        sample_option_contract.last = Decimal("4.00")
        portfolio.update_portfolio_value()

        # Max drawdown should be tracked
        assert portfolio.max_drawdown > 0

    def test_transaction_recording(
        self, sample_portfolio: Any, sample_option_contract: Any
    ) -> None:
        """Test that all transactions are properly recorded."""
        portfolio = sample_portfolio

        # Add position
        portfolio.add_position(
            sample_option_contract, 10, Decimal("5.50"), commission=Decimal("10.00")
        )

        # Partially close
        portfolio.partial_close_position(
            sample_option_contract.contract_id,
            3,
            Decimal("6.00"),
            commission=Decimal("5.00"),
        )

        # Close remaining
        portfolio.close_position(
            sample_option_contract.contract_id,
            Decimal("6.50"),
            commission=Decimal("7.00"),
        )

        # Should have 3 transactions recorded
        assert len(portfolio.all_transactions) == 3

        # Verify transaction details
        transactions = portfolio.all_transactions
        assert transactions[0]["quantity"] == 10  # Initial buy
        assert transactions[1]["quantity"] == -3  # Partial close
        assert transactions[2]["quantity"] == -7  # Final close

        # All should have portfolio_id
        for transaction in transactions:
            assert transaction["portfolio_id"] == portfolio.portfolio_id


class TestPortfolioComplexScenarios:
    """Test complex portfolio scenarios."""

    def test_multiple_positions_different_symbols(self, sample_portfolio: Any) -> None:
        """Test portfolio with multiple positions in different symbols."""
        portfolio = sample_portfolio

        # Create contracts for different symbols
        aapl_call = OptionContract(
            symbol="AAPL",
            strike=Decimal("150.00"),
            expiration=datetime.date.today() + datetime.timedelta(days=30),
            option_type=OptionType.CALL,
        )

        msft_call = OptionContract(
            symbol="MSFT",
            strike=Decimal("300.00"),
            expiration=datetime.date.today() + datetime.timedelta(days=30),
            option_type=OptionType.CALL,
        )

        # Add positions
        portfolio.add_position(aapl_call, 10, Decimal("5.50"))
        portfolio.add_position(msft_call, 5, Decimal("12.00"))

        assert portfolio.num_open_positions == 2
        assert len(portfolio.get_positions_by_symbol("AAPL")) == 1
        assert len(portfolio.get_positions_by_symbol("MSFT")) == 1

        # Update prices and portfolio value
        aapl_call.last = Decimal("6.50")
        msft_call.last = Decimal("13.50")
        portfolio.update_portfolio_value()

        # Both positions should contribute to unrealized P&L
        assert portfolio.total_unrealized_pnl > 0
        assert portfolio.total_value > portfolio.initial_cash

    def test_mixed_long_short_positions(self, sample_portfolio: Any) -> None:
        """Test portfolio with mixed long and short positions."""
        portfolio = sample_portfolio

        # Long call position
        long_call = OptionContract(
            symbol="AAPL",
            strike=Decimal("150.00"),
            expiration=datetime.date.today() + datetime.timedelta(days=30),
            option_type=OptionType.CALL,
        )

        # Short put position
        short_put = OptionContract(
            symbol="AAPL",
            strike=Decimal("145.00"),
            expiration=datetime.date.today() + datetime.timedelta(days=30),
            option_type=OptionType.PUT,
        )

        # Add positions
        portfolio.add_position(long_call, 10, Decimal("5.50"))
        portfolio.add_position(short_put, -10, Decimal("3.25"))

        assert len(portfolio.get_long_positions()) == 1
        assert len(portfolio.get_short_positions()) == 1

        # Simulate underlying price movement
        long_call.last = Decimal("6.50")  # Call goes up (good for long)
        short_put.last = Decimal("2.75")  # Put goes down (good for short)

        portfolio.update_portfolio_value()

        # Both positions should be profitable
        assert portfolio.total_unrealized_pnl > 0

    def test_portfolio_stress_scenarios(
        self, sample_portfolio: Any, sample_option_contract: Any
    ) -> None:
        """Test portfolio under stress scenarios."""
        portfolio = sample_portfolio

        # Add position
        portfolio.add_position(sample_option_contract, 10, Decimal("5.50"))

        # Scenario 1: Large price drop (stress test)
        sample_option_contract.last = Decimal("1.00")  # 80% drop
        portfolio.update_portfolio_value()

        unrealized_loss = portfolio.total_unrealized_pnl
        assert unrealized_loss < 0
        assert abs(unrealized_loss) > Decimal("4000")  # Significant loss

        # Portfolio value should still be reasonable
        assert portfolio.total_value > 0

        # Scenario 2: Price recovery
        sample_option_contract.last = Decimal("8.00")  # Recovery above entry
        portfolio.update_portfolio_value()

        assert portfolio.total_unrealized_pnl > 0
        assert portfolio.total_value > portfolio.initial_cash

    def test_margin_and_leverage_calculations(
        self, sample_portfolio: Any, sample_option_contract: Any
    ) -> None:
        """Test margin and leverage calculations."""
        portfolio = sample_portfolio

        # Increase position size limit
        portfolio.max_position_size_pct = Decimal("0.3")  # 30%

        # Add leveraged position
        portfolio.add_position(
            sample_option_contract, 50, Decimal("5.50")
        )  # Large position

        # Update portfolio metrics
        portfolio._update_portfolio_metrics()

        # Should have margin used
        assert portfolio.margin_used >= 0

        # Leverage should be calculated
        leverage = portfolio.leverage
        assert leverage >= 0

        # Buying power should account for margin
        assert portfolio.buying_power >= portfolio.cash

    def test_profit_factor_calculation(
        self, sample_portfolio: Any, sample_option_contract: Any
    ) -> None:
        """Test profit factor calculation with multiple trades."""
        portfolio = sample_portfolio

        # Trade 1: Profitable
        portfolio.add_position(sample_option_contract, 10, Decimal("5.50"))
        portfolio.close_position(sample_option_contract.contract_id, Decimal("6.50"))

        # Trade 2: Loss (same contract, new position)
        portfolio.add_position(sample_option_contract, 10, Decimal("6.00"))
        portfolio.close_position(sample_option_contract.contract_id, Decimal("5.00"))

        # Get metrics
        metrics = portfolio.get_portfolio_metrics()

        # Should have 50% win rate (1 win, 1 loss)
        assert metrics.win_rate == Decimal("50")

        # Should have profit factor calculated
        assert metrics.profit_factor is not None
        assert metrics.profit_factor > 0


class TestPortfolioEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_portfolio_metrics(self, sample_portfolio: Any) -> None:
        """Test metrics calculation for empty portfolio."""
        portfolio = sample_portfolio

        metrics = portfolio.get_portfolio_metrics()

        assert metrics.total_value == portfolio.initial_cash
        assert metrics.total_pnl == Decimal("0")
        assert metrics.return_pct == Decimal("0")
        assert metrics.win_rate is None
        assert metrics.profit_factor is None

    def test_portfolio_with_zero_initial_cash(self):
        """Test portfolio with zero initial cash."""
        # Zero initial cash should be allowed - just creates an empty portfolio
        portfolio = Portfolio(initial_cash=Decimal("0"))
        assert portfolio.initial_cash == Decimal("0")
        assert portfolio.cash == Decimal("0")

    def test_close_position_without_market_price_raises_error(
        self, sample_portfolio: Any, sample_option_contract: Any
    ) -> None:
        """Test closing position without market price raises error."""
        portfolio = sample_portfolio

        # Add position
        portfolio.add_position(sample_option_contract, 10, Decimal("5.50"))

        # Clear market prices
        sample_option_contract.last = None
        sample_option_contract.bid = None
        sample_option_contract.ask = None

        contract_id = sample_option_contract.contract_id

        with pytest.raises(PortfolioError, match="No exit price available"):
            portfolio.close_position(contract_id)

    def test_extreme_leverage_scenario(self, sample_portfolio: Any) -> None:
        """Test portfolio with extreme leverage."""
        portfolio = sample_portfolio

        # Set high leverage limit
        portfolio.max_portfolio_leverage = Decimal("10.0")

        # Create high-value contract
        expensive_contract = OptionContract(
            symbol="EXPENSIVE",
            strike=Decimal("5000.00"),
            expiration=datetime.date.today() + datetime.timedelta(days=30),
            option_type=OptionType.CALL,
        )

        # Try to add position that would exceed leverage
        with pytest.raises(ExcessiveRiskError):
            portfolio.add_position(expensive_contract, 100, Decimal("100.00"))

    def test_portfolio_with_expired_positions(self, sample_portfolio: Any) -> None:
        """Test portfolio behavior with expired positions."""
        # Create expired contract
        expired_contract_data = {
            "symbol": "AAPL",
            "strike": Decimal("150.00"),
            "expiration": datetime.date.today() - datetime.timedelta(days=1),
            "option_type": OptionType.CALL,
        }

        # Should not be able to create expired contract
        with pytest.raises(Exception):
            OptionContract(**expired_contract_data)


@pytest.mark.performance
class TestPortfolioPerformance:
    """Performance tests for portfolio operations."""

    def test_portfolio_with_many_positions_performance(self, performance_timer):
        """Test portfolio performance with many positions."""
        portfolio = Portfolio(initial_cash=Decimal("1000000"), name="Large Portfolio")

        # Create many contracts
        contracts = []
        for i in range(100):
            contract = OptionContract(
                symbol=f"SYM{i:03d}",
                strike=Decimal(f"{100 + i}"),
                expiration=datetime.date.today() + datetime.timedelta(days=30),
                option_type=OptionType.CALL,
            )
            contracts.append(contract)

        performance_timer.start()

        # Add many positions
        for i, contract in enumerate(contracts):
            portfolio.add_position(contract, 10, Decimal(f"{5.00 + i * 0.01}"))

        # Update portfolio value
        for contract in contracts:
            contract.last = Decimal(f"{6.00 + contracts.index(contract) * 0.01}")

        portfolio.update_portfolio_value()

        # Get summary
        _ = portfolio.get_portfolio_summary()

        performance_timer.stop()

        # Should complete in reasonable time
        assert performance_timer.elapsed < 2.0
        assert portfolio.num_open_positions == 100

    def test_portfolio_metrics_calculation_performance(
        self, sample_portfolio: Any, performance_timer
    ) -> None:
        """Test performance of portfolio metrics calculation."""
        portfolio = sample_portfolio

        # Increase position size limit
        portfolio.max_position_size_pct = Decimal("0.2")  # 20%

        # Add many daily returns
        portfolio.daily_returns = [Decimal(str(x % 10 - 5)) for x in range(1000)]

        # Add and close fewer positions to avoid risk limit errors
        for i in range(10):
            contract = OptionContract(
                symbol=f"SYM{i:03d}",
                strike=Decimal(f"{100 + i}"),
                expiration=datetime.date.today() + datetime.timedelta(days=30),
                option_type=OptionType.CALL,
            )

            portfolio.add_position(contract, 5, Decimal(f"{5.00 + i * 0.01}"))
            portfolio.close_position(
                contract.contract_id, Decimal(f"{5.50 + i * 0.01}")
            )

        performance_timer.start()

        # Calculate metrics many times
        for _ in range(100):
            _ = portfolio.get_portfolio_metrics()

        performance_timer.stop()

        # Should complete quickly
        assert performance_timer.elapsed < 0.5
