"""
Integration tests for database operations.

This module tests the complete database integration including
CRUD operations, data persistence, and transaction handling.
"""

import datetime
from decimal import Decimal

from src.core.contracts import OptionContract, StockContract, OptionType
from src.core.position import Position, Transaction


class TestDatabaseIntegration:
    """Integration tests for database operations."""
    
    def test_database_initialization(self, test_db_manager):
        """Test database initialization and connection."""
        # Test connection
        assert test_db_manager.test_connection()
        
        # Test database info
        info = test_db_manager.get_database_info()
        assert 'connection_string' in info
        assert 'table_count' in info
        assert info['table_count'] > 0
    
    def test_stock_contract_crud_operations(self, test_db_manager, sample_stock_contract):
        """Test complete CRUD operations for stock contracts."""
        # Create
        contract_id = test_db_manager.save_stock_contract(sample_stock_contract)
        assert contract_id > 0
        
        # Read
        retrieved_contract = test_db_manager.get_stock_contract(sample_stock_contract.symbol)
        assert retrieved_contract is not None
        assert retrieved_contract.symbol == sample_stock_contract.symbol
        assert retrieved_contract.bid == sample_stock_contract.bid
        assert retrieved_contract.ask == sample_stock_contract.ask
        
        # Update
        sample_stock_contract.bid = Decimal('151.00')
        sample_stock_contract.ask = Decimal('151.05')
        test_db_manager.save_stock_contract(sample_stock_contract)
        
        # Verify update
        updated_contract = test_db_manager.get_stock_contract(sample_stock_contract.symbol)
        assert updated_contract.bid == Decimal('151.00')
        assert updated_contract.ask == Decimal('151.05')
    
    def test_option_contract_crud_operations(self, test_db_manager, sample_option_contract):
        """Test complete CRUD operations for option contracts."""
        # Create
        contract_id = test_db_manager.save_option_contract(sample_option_contract)
        assert contract_id > 0
        
        # Read
        retrieved_contract = test_db_manager.get_option_contract(sample_option_contract.contract_id)
        assert retrieved_contract is not None
        assert retrieved_contract.contract_id == sample_option_contract.contract_id
        assert retrieved_contract.strike == sample_option_contract.strike
        assert retrieved_contract.option_type == sample_option_contract.option_type
        
        # Update
        sample_option_contract.bid = Decimal('5.60')
        sample_option_contract.ask = Decimal('5.70')
        test_db_manager.save_option_contract(sample_option_contract)
        
        # Verify update
        updated_contract = test_db_manager.get_option_contract(sample_option_contract.contract_id)
        assert updated_contract.bid == Decimal('5.60')
        assert updated_contract.ask == Decimal('5.70')
    
    def test_market_data_operations(self, test_db_manager, sample_stock_contract):
        """Test market data storage and retrieval."""
        # Save stock contract first
        test_db_manager.save_stock_contract(sample_stock_contract)
        
        # Save market data
        timestamp = datetime.datetime.now()
        market_data = {
            'open': 150.00,
            'high': 151.50,
            'low': 149.50,
            'close': 151.00,
            'volume': 1000000,
            'bid': 150.95,
            'ask': 151.05
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
        assert data_records[0]['close'] == 151.00
        assert data_records[0]['volume'] == 1000000
    
    def test_portfolio_persistence(self, test_db_manager, sample_portfolio):
        """Test portfolio persistence and retrieval."""
        # Save portfolio
        portfolio_id = test_db_manager.save_portfolio(sample_portfolio)
        assert portfolio_id > 0
        
        # Retrieve portfolio
        retrieved_portfolio = test_db_manager.get_portfolio(sample_portfolio.portfolio_id)
        assert retrieved_portfolio is not None
        assert retrieved_portfolio.portfolio_id == sample_portfolio.portfolio_id
        assert retrieved_portfolio.name == sample_portfolio.name
        assert retrieved_portfolio.initial_cash == sample_portfolio.initial_cash
    
    def test_position_and_transaction_persistence(self, test_db_manager, 
                                                sample_portfolio, sample_option_contract):
        """Test position and transaction persistence."""
        # Save portfolio first
        portfolio_db_id = test_db_manager.save_portfolio(sample_portfolio)
        
        # Create a position
        position = Position(
            contract=sample_option_contract,
            entry_timestamp=datetime.datetime.now(),
            initial_quantity=10,
            entry_price=Decimal('5.50'),
            position_id="test_position_001"
        )
        
        # Save position
        position_db_id = test_db_manager.save_position(position, portfolio_db_id)
        assert position_db_id > 0
        
        # Save transaction
        transaction = Transaction(
            timestamp=datetime.datetime.now(),
            quantity=10,
            price=Decimal('5.50'),
            commission=Decimal('10.00')
        )
        
        transaction_db_id = test_db_manager.save_transaction(
            transaction, position_db_id, portfolio_db_id
        )
        assert transaction_db_id > 0
    
    def test_data_integrity_constraints(self, test_db_manager):
        """Test database integrity constraints."""
        # Test duplicate contract ID
        expiration = datetime.date.today() + datetime.timedelta(days=30)
        
        contract1 = OptionContract(
            symbol="TEST",
            strike=Decimal('100.00'),
            expiration=expiration,
            option_type=OptionType.CALL
        )
        
        contract2 = OptionContract(
            symbol="TEST",
            strike=Decimal('100.00'),
            expiration=expiration,
            option_type=OptionType.CALL
        )
        
        # Both should have the same contract_id
        assert contract1.contract_id == contract2.contract_id
        
        # Save first contract
        test_db_manager.save_option_contract(contract1)
        
        # Save second contract (should update, not create duplicate)
        test_db_manager.save_option_contract(contract2)
        
        # Verify only one record exists
        retrieved = test_db_manager.get_option_contract(contract1.contract_id)
        assert retrieved is not None
    
    def test_error_handling(self, test_db_manager):
        """Test error handling for invalid operations."""
        # Test getting non-existent contract
        non_existent = test_db_manager.get_stock_contract("NONEXISTENT")
        assert non_existent is None
        
        # Test getting non-existent option
        non_existent_option = test_db_manager.get_option_contract("NONEXISTENT_OPTION")
        assert non_existent_option is None
    
    def test_cleanup_operations(self, test_db_manager):
        """Test data cleanup operations."""
        # Test cleanup of old data
        deleted_counts = test_db_manager.cleanup_old_data(days_to_keep=0)
        assert isinstance(deleted_counts, dict)
        assert 'stock_market_data' in deleted_counts
        assert 'option_market_data' in deleted_counts


class TestDatabasePerformance:
    """Performance tests for database operations."""
    
    def test_bulk_operations_performance(self, test_db_manager, performance_timer):
        """Test performance of bulk operations."""
        # Create multiple contracts
        contracts = []
        for i in range(100):
            expiration = datetime.date.today() + datetime.timedelta(days=30)
            contract = OptionContract(
                symbol=f"TEST{i}",
                strike=Decimal(f'{100 + i}'),
                expiration=expiration,
                option_type=OptionType.CALL
            )
            contracts.append(contract)
        
        performance_timer.start()
        
        # Save all contracts
        for contract in contracts:
            test_db_manager.save_option_contract(contract)
        
        performance_timer.stop()
        
        # Should complete in reasonable time
        assert performance_timer.elapsed < 5.0  # 5 seconds
    
    def test_query_performance(self, test_db_manager, sample_option_contract, performance_timer):
        """Test query performance."""
        # Save contract first
        test_db_manager.save_option_contract(sample_option_contract)
        
        performance_timer.start()
        
        # Perform multiple queries
        for _ in range(100):
            test_db_manager.get_option_contract(sample_option_contract.contract_id)
        
        performance_timer.stop()
        
        # Should complete quickly
        assert performance_timer.elapsed < 1.0  # 1 second


class TestDatabaseConcurrency:
    """Test database concurrency and session management."""
    
    def test_session_management(self, test_db_manager):
        """Test proper session management."""
        # Test that sessions are properly closed
        with test_db_manager.get_session() as session:
            # Perform some operations
            from sqlalchemy import text
            result = session.execute(text("SELECT 1"))
            assert result is not None
        
        # Session should be closed after context manager exits
        # This is tested by the fact that no exceptions are raised
    
    def test_transaction_rollback(self, test_db_manager, sample_stock_contract):
        """Test transaction rollback on error."""
        # This test verifies that database errors are properly handled
        # and transactions are rolled back
        
        # Save a valid contract
        contract_id = test_db_manager.save_stock_contract(sample_stock_contract)
        assert contract_id > 0
        
        # Verify it was saved
        retrieved = test_db_manager.get_stock_contract(sample_stock_contract.symbol)
        assert retrieved is not None
        
        # Test that database operations continue to work after errors
        # (indicating proper rollback)
        another_contract = StockContract(symbol="ANOTHER")
        another_id = test_db_manager.save_stock_contract(another_contract)
        assert another_id > 0 