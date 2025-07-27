"""
Unit tests for the data manager module.

This module tests the DatabaseManager class including CRUD operations,
data validation, and error handling.
"""

import datetime
from typing import Any
import pytest
from decimal import Decimal

from src.core.contracts import OptionContract, OptionType
from src.core.position import Position, Transaction

from src.data.data_manager import DatabaseManager


class TestDatabaseManager:
    """Test cases for DatabaseManager class."""

    def test_database_manager_initialization(
        self, test_db_manager: DatabaseManager
    ) -> None:
        """Test database manager initialization."""
        assert test_db_manager.connection_string is not None
        assert test_db_manager.engine is not None
        assert test_db_manager.SessionLocal is not None

    def test_database_connection(self, test_db_manager: Any) -> None:
        """Test database connection."""
        assert test_db_manager.test_connection()

    def test_database_info(self, test_db_manager: Any) -> None:
        """Test database info retrieval."""
        info = test_db_manager.get_database_info()

        assert "connection_string" in info
        assert "table_count" in info
        assert "tables" in info
        assert info["table_count"] > 0

    def test_stock_contract_operations(
        self, test_db_manager: Any, sample_stock_contract: Any
    ) -> None:
        """Test stock contract CRUD operations."""
        # Save contract
        contract_id = test_db_manager.save_stock_contract(sample_stock_contract)
        assert contract_id > 0

        # Retrieve contract
        retrieved = test_db_manager.get_stock_contract(sample_stock_contract.symbol)
        assert retrieved is not None
        assert retrieved.symbol == sample_stock_contract.symbol
        assert retrieved.bid == sample_stock_contract.bid
        assert retrieved.ask == sample_stock_contract.ask

    def test_option_contract_operations(
        self, test_db_manager: Any, sample_option_contract: Any
    ) -> None:
        """Test option contract CRUD operations."""
        # Save contract
        contract_id = test_db_manager.save_option_contract(sample_option_contract)
        assert contract_id > 0

        # Retrieve contract
        retrieved = test_db_manager.get_option_contract(
            sample_option_contract.contract_id
        )
        assert retrieved is not None
        assert retrieved.contract_id == sample_option_contract.contract_id
        assert retrieved.strike == sample_option_contract.strike
        assert retrieved.option_type == sample_option_contract.option_type

    def test_get_options_by_symbol(
        self, test_db_manager: Any, sample_option_contract: Any
    ) -> None:
        """Test getting options by symbol."""
        # Save contract
        test_db_manager.save_option_contract(sample_option_contract)

        # Get options by symbol
        options = test_db_manager.get_options_by_symbol(sample_option_contract.symbol)
        assert len(options) > 0
        assert options[0].symbol == sample_option_contract.symbol

    def test_market_data_operations(
        self, test_db_manager: Any, sample_stock_contract: Any
    ) -> None:
        """Test market data operations."""
        # Save stock contract first
        test_db_manager.save_stock_contract(sample_stock_contract)

        # Save market data
        timestamp = datetime.datetime.now()
        market_data = {
            "open": 150.00,
            "high": 151.50,
            "low": 149.50,
            "close": 151.00,
            "volume": 1000000,
            "bid": 150.95,
            "ask": 151.05,
        }

        test_db_manager.save_stock_market_data(
            sample_stock_contract.symbol, timestamp, market_data
        )

        # Retrieve market data
        start_date = timestamp - datetime.timedelta(hours=1)
        end_date = timestamp + datetime.timedelta(hours=1)

        data_records = test_db_manager.get_stock_market_data_range(
            sample_stock_contract.symbol, start_date, end_date
        )

        assert len(data_records) > 0
        assert data_records[0]["close"] == 151.00

    def test_portfolio_operations(
        self, test_db_manager: Any, sample_portfolio: Any
    ) -> None:
        """Test portfolio operations."""
        # Save portfolio
        portfolio_id = test_db_manager.save_portfolio(sample_portfolio)
        assert portfolio_id > 0

        # Retrieve portfolio
        retrieved = test_db_manager.get_portfolio(sample_portfolio.portfolio_id)
        assert retrieved is not None
        assert retrieved.portfolio_id == sample_portfolio.portfolio_id
        assert retrieved.name == sample_portfolio.name

    def test_position_operations(
        self, test_db_manager: Any, sample_portfolio: Any, sample_option_contract: Any
    ) -> None:
        """Test position operations."""
        # Save portfolio first
        portfolio_db_id = test_db_manager.save_portfolio(sample_portfolio)

        # Create position
        position = Position(
            contract=sample_option_contract,
            entry_timestamp=datetime.datetime.now(),
            initial_quantity=10,
            entry_price=Decimal("5.50"),
            position_id="test_position_001",
        )

        # Save position
        position_db_id = test_db_manager.save_position(position, portfolio_db_id)
        assert position_db_id > 0

    def test_transaction_operations(
        self, test_db_manager: Any, sample_portfolio: Any, sample_option_contract: Any
    ) -> None:
        """Test transaction operations."""
        # Save portfolio and position first
        portfolio_db_id = test_db_manager.save_portfolio(sample_portfolio)

        position = Position(
            contract=sample_option_contract,
            entry_timestamp=datetime.datetime.now(),
            initial_quantity=10,
            entry_price=Decimal("5.50"),
            position_id="test_position_002",
        )
        position_db_id = test_db_manager.save_position(position, portfolio_db_id)

        # Create transaction
        transaction = Transaction(
            timestamp=datetime.datetime.now(),
            quantity=10,
            price=Decimal("5.50"),
            commission=Decimal("10.00"),
        )

        # Save transaction
        transaction_db_id = test_db_manager.save_transaction(
            transaction, position_db_id, portfolio_db_id
        )
        assert transaction_db_id > 0

    def test_cleanup_operations(self, test_db_manager: Any) -> None:
        """Test cleanup operations."""
        deleted_counts = test_db_manager.cleanup_old_data(days_to_keep=0)
        assert isinstance(deleted_counts, dict)
        assert "stock_market_data" in deleted_counts
        assert "option_market_data" in deleted_counts

    def test_error_handling(self, test_db_manager: Any) -> None:
        """Test error handling."""
        # Test getting non-existent contract
        non_existent = test_db_manager.get_stock_contract("NONEXISTENT")
        assert non_existent is None

        # Test getting non-existent option
        non_existent_option = test_db_manager.get_option_contract("NONEXISTENT")
        assert non_existent_option is None

    def test_session_management(self, test_db_manager: Any) -> None:
        """Test session management."""
        with test_db_manager.get_session() as session:
            # Perform some operations
            from sqlalchemy import text

            result = session.execute(text("SELECT 1"))
            assert result is not None

        # Session should be closed after context manager exits


class TestDatabaseManagerEdgeCases:
    """Test edge cases for DatabaseManager."""

    def test_duplicate_contract_handling(
        self, test_db_manager: Any, sample_option_contract: Any
    ) -> None:
        """Test handling of duplicate contracts."""
        # Save contract twice
        id1 = test_db_manager.save_option_contract(sample_option_contract)
        id2 = test_db_manager.save_option_contract(sample_option_contract)

        # Should return the same ID (update, not insert)
        assert id1 == id2

    def test_large_data_handling(self, test_db_manager: Any) -> None:
        """Test handling of large datasets."""
        # Create many contracts
        contracts = []
        for i in range(50):
            expiration = datetime.date.today() + datetime.timedelta(days=30)
            contract = OptionContract(
                symbol=f"TEST{i}",
                strike=Decimal(f"{100 + i}"),
                expiration=expiration,
                option_type=OptionType.CALL,
            )
            contracts.append(contract)

        # Save all contracts
        for contract in contracts:
            contract_id = test_db_manager.save_option_contract(contract)
            assert contract_id > 0

        # Verify they can be retrieved
        for contract in contracts:
            retrieved = test_db_manager.get_option_contract(contract.contract_id)
            assert retrieved is not None
            assert retrieved.contract_id == contract.contract_id


@pytest.mark.performance
class TestDatabaseManagerPerformance:
    """Performance tests for DatabaseManager."""

    def test_bulk_insert_performance(
        self, test_db_manager: Any, performance_timer: Any
    ) -> None:
        """Test bulk insert performance."""
        contracts = []
        for i in range(100):
            expiration = datetime.date.today() + datetime.timedelta(days=30)
            contract = OptionContract(
                symbol=f"PERF{i}",
                strike=Decimal(f"{100 + i}"),
                expiration=expiration,
                option_type=OptionType.CALL,
            )
            contracts.append(contract)

        performance_timer.start()

        for contract in contracts:
            test_db_manager.save_option_contract(contract)

        performance_timer.stop()

        # Should complete in reasonable time
        assert performance_timer.elapsed < 5.0

    def test_query_performance(
        self, test_db_manager: Any, sample_option_contract: Any, performance_timer: Any
    ) -> None:
        """Test query performance."""
        # Save contract first
        test_db_manager.save_option_contract(sample_option_contract)

        performance_timer.start()

        # Perform multiple queries
        for _ in range(100):
            test_db_manager.get_option_contract(sample_option_contract.contract_id)

        performance_timer.stop()

        # Should complete quickly
        assert performance_timer.elapsed < 1.0
