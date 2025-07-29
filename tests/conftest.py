"""
Pytest configuration and fixtures for the options backtesting system tests.

This module provides common test fixtures and configuration used across
all test modules in the test suite.
"""

import datetime
import pytest
import tempfile
import os
from decimal import Decimal
from pathlib import Path
from typing import Generator, Dict, Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.core.contracts import OptionContract, StockContract, OptionType, ContractStatus
from src.core.position import Position, Transaction
from src.core.portfolio import Portfolio
from src.data.schemas import Base
from src.data.data_manager import DatabaseManager
from src.utils.config import ConfigManager
from src.utils.logger import setup_application_logging


@pytest.fixture(scope="session")
def test_config():
    """Create test configuration."""
    # Use temporary directory for test database
    temp_dir = tempfile.mkdtemp()
    
    config = ConfigManager()
    
    # Override with test settings
    config.database.type = "sqlite"
    config.database.path = os.path.join(temp_dir, "test_backtest.db")
    config.logging.level = "WARNING"  # Reduce log noise during tests
    config.logging.file = os.path.join(temp_dir, "test.log")
    
    return config


@pytest.fixture(scope="session")
def test_db_manager(test_config):
    """Create test database manager."""
    db_manager = DatabaseManager(test_config.database.connection_string, echo=False)
    db_manager.initialize_database(drop_existing=True)
    
    yield db_manager
    
    # Cleanup
    db_manager.close()


@pytest.fixture
def db_session(test_db_manager):
    """Create a database session for testing."""
    with test_db_manager.get_session() as session:
        yield session


@pytest.fixture
def sample_stock_contract():
    """Create a sample stock contract for testing."""
    return StockContract(
        symbol="AAPL",
        exchange="SMART",
        currency="USD",
        bid=Decimal('150.00'),
        ask=Decimal('150.05'),
        last=Decimal('150.02'),
        volume=1000000,
        market_cap=Decimal('2500000000000'),  # $2.5T
        pe_ratio=Decimal('25.5'),
        dividend_yield=Decimal('0.0065')  # 0.65%
    )


@pytest.fixture
def sample_option_contract():
    """Create a sample option contract for testing."""
    expiration = datetime.date.today() + datetime.timedelta(days=30)
    
    return OptionContract(
        symbol="AAPL",
        strike=Decimal('150.00'),
        expiration=expiration,
        option_type=OptionType.CALL,
        exchange="SMART",
        multiplier=100,
        currency="USD",
        bid=Decimal('5.50'),
        ask=Decimal('5.60'),
        last=Decimal('5.55'),
        volume=500,
        open_interest=1000,
        delta=Decimal('0.5'),
        gamma=Decimal('0.02'),
        theta=Decimal('-0.05'),
        vega=Decimal('0.15'),
        rho=Decimal('0.08'),
        implied_volatility=Decimal('0.25')
    )


@pytest.fixture
def sample_put_contract():
    """Create a sample put option contract for testing."""
    expiration = datetime.date.today() + datetime.timedelta(days=30)
    
    return OptionContract(
        symbol="AAPL",
        strike=Decimal('145.00'),
        expiration=expiration,
        option_type=OptionType.PUT,
        exchange="SMART",
        multiplier=100,
        currency="USD",
        bid=Decimal('3.20'),
        ask=Decimal('3.30'),
        last=Decimal('3.25'),
        volume=300,
        open_interest=800,
        delta=Decimal('-0.3'),
        gamma=Decimal('0.02'),
        theta=Decimal('-0.04'),
        vega=Decimal('0.12'),
        rho=Decimal('-0.06'),
        implied_volatility=Decimal('0.22')
    )


@pytest.fixture
def sample_transaction():
    """Create a sample transaction for testing."""
    return Transaction(
        timestamp=datetime.datetime.now(),
        quantity=100,
        price=Decimal('150.00'),
        commission=Decimal('1.00'),
        fees=Decimal('0.50')
    )


@pytest.fixture
def sample_position(sample_option_contract):
    """Create a sample position for testing."""
    return Position(
        contract=sample_option_contract,
        entry_timestamp=datetime.datetime.now(),
        initial_quantity=10,
        entry_price=Decimal('5.50')
    )


@pytest.fixture
def sample_portfolio():
    """Create a sample portfolio for testing."""
    return Portfolio(
        initial_cash=Decimal('100000'),
        name="Test Portfolio"
    )


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    return {
        'timestamp': datetime.datetime.now(),
        'open': Decimal('149.50'),
        'high': Decimal('151.20'),
        'low': Decimal('149.00'),
        'close': Decimal('150.75'),
        'volume': 1500000,
        'bid': Decimal('150.70'),
        'ask': Decimal('150.80')
    }


@pytest.fixture
def sample_option_market_data():
    """Create sample option market data for testing."""
    return {
        'timestamp': datetime.datetime.now(),
        'bid': Decimal('5.40'),
        'ask': Decimal('5.50'),
        'last': Decimal('5.45'),
        'volume': 250,
        'open_interest': 1200,
        'delta': Decimal('0.52'),
        'gamma': Decimal('0.025'),
        'theta': Decimal('-0.06'),
        'vega': Decimal('0.18'),
        'rho': Decimal('0.09'),
        'implied_volatility': Decimal('0.27'),
        'underlying_price': Decimal('151.00')
    }


@pytest.fixture
def expired_option_contract():
    """Create an expired option contract for testing error conditions."""
    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    
    # This should raise an exception when created due to expiration
    return {
        'symbol': 'AAPL',
        'strike': Decimal('150.00'),
        'expiration': yesterday,
        'option_type': OptionType.CALL,
        'exchange': 'SMART'
    }


@pytest.fixture
def complex_portfolio_setup(sample_portfolio, sample_option_contract, sample_put_contract, sample_stock_contract):
    """Create a portfolio with multiple positions for complex testing."""
    portfolio = sample_portfolio
    
    # Add a long call position
    call_position = Position(
        contract=sample_option_contract,
        entry_timestamp=datetime.datetime.now() - datetime.timedelta(days=5),
        initial_quantity=5,
        entry_price=Decimal('5.00')
    )
    
    # Add a short put position
    put_position = Position(
        contract=sample_put_contract,
        entry_timestamp=datetime.datetime.now() - datetime.timedelta(days=3),
        initial_quantity=-3,
        entry_price=Decimal('3.50')
    )
    
    # Add positions to portfolio (simplified - normally would use portfolio.add_position)
    portfolio.positions[sample_option_contract.contract_id] = call_position
    portfolio.positions[sample_put_contract.contract_id] = put_position
    
    return portfolio


# Test data generators

def generate_option_chain(symbol: str, underlying_price: Decimal, expiration: datetime.date, 
                         num_strikes: int = 10) -> list[OptionContract]:
    """Generate a chain of option contracts for testing."""
    contracts = []
    
    # Generate strikes around the underlying price
    strike_range = underlying_price * Decimal('0.2')  # 20% range
    strikes = []
    
    for i in range(num_strikes):
        strike_offset = (i - num_strikes // 2) * (strike_range / num_strikes)
        strike = underlying_price + strike_offset
        strikes.append(strike.quantize(Decimal('0.01')))
    
    # Create calls and puts for each strike
    for strike in strikes:
        # Call option
        call = OptionContract(
            symbol=symbol,
            strike=strike,
            expiration=expiration,
            option_type=OptionType.CALL,
            bid=max(Decimal('0.01'), underlying_price - strike + Decimal('2.00')),
            ask=max(Decimal('0.02'), underlying_price - strike + Decimal('2.10')),
            last=max(Decimal('0.01'), underlying_price - strike + Decimal('2.05')),
            volume=100 + int(strike) % 500,
            open_interest=500 + int(strike) % 2000,
            delta=min(Decimal('0.99'), max(Decimal('0.01'), (underlying_price - strike) / 20 + Decimal('0.5'))),
            implied_volatility=Decimal('0.20') + (abs(strike - underlying_price) / underlying_price) * Decimal('0.1')
        )
        contracts.append(call)
        
        # Put option
        put = OptionContract(
            symbol=symbol,
            strike=strike,
            expiration=expiration,
            option_type=OptionType.PUT,
            bid=max(Decimal('0.01'), strike - underlying_price + Decimal('2.00')),
            ask=max(Decimal('0.02'), strike - underlying_price + Decimal('2.10')),
            last=max(Decimal('0.01'), strike - underlying_price + Decimal('2.05')),
            volume=100 + int(strike) % 500,
            open_interest=500 + int(strike) % 2000,
            delta=max(Decimal('-0.99'), min(Decimal('-0.01'), -(strike - underlying_price) / 20 - Decimal('0.5'))),
            implied_volatility=Decimal('0.20') + (abs(strike - underlying_price) / underlying_price) * Decimal('0.1')
        )
        contracts.append(put)
    
    return contracts


@pytest.fixture
def option_chain(sample_stock_contract):
    """Generate a full option chain for testing."""
    expiration = datetime.date.today() + datetime.timedelta(days=30)
    underlying_price = sample_stock_contract.last or Decimal('150.00')
    
    return generate_option_chain(
        symbol=sample_stock_contract.symbol,
        underlying_price=underlying_price,
        expiration=expiration,
        num_strikes=20
    )


# Performance testing helpers

@pytest.fixture
def performance_timer():
    """Timer fixture for performance testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()


# Logging setup for tests
def pytest_configure(config):
    """Configure pytest."""
    # Set up minimal logging for tests
    setup_application_logging(
        log_level="WARNING",
        log_file=None,  # No file logging during tests
        console_output=False  # No console output during tests
    )


# Test markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


# Mock data helpers

class MockMarketDataProvider:
    """Mock market data provider for testing."""
    
    def __init__(self):
        self.data = {}
    
    def add_stock_data(self, symbol: str, price: Decimal, volume: int = 1000):
        """Add mock stock data."""
        self.data[symbol] = {
            'type': 'stock',
            'price': price,
            'volume': volume,
            'timestamp': datetime.datetime.now()
        }
    
    def add_option_data(self, contract_id: str, price: Decimal, greeks: Dict[str, Decimal]):
        """Add mock option data."""
        self.data[contract_id] = {
            'type': 'option',
            'price': price,
            'greeks': greeks,
            'timestamp': datetime.datetime.now()
        }
    
    def get_data(self, identifier: str):
        """Get mock data."""
        return self.data.get(identifier)


@pytest.fixture
def mock_market_data():
    """Mock market data provider fixture."""
    provider = MockMarketDataProvider()
    
    # Add some default data
    provider.add_stock_data("AAPL", Decimal('150.00'))
    provider.add_stock_data("MSFT", Decimal('300.00'))
    provider.add_stock_data("GOOGL", Decimal('2500.00'))
    
    return provider


# Cleanup helpers

@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary files after tests."""
    temp_files = []
    
    yield temp_files
    
    # Cleanup any temporary files created during tests
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception:
            pass  # Ignore cleanup errors