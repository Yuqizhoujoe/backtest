"""
Database manager for the options backtesting system.

This module provides comprehensive database operations including connection
management, CRUD operations, and data integrity maintenance.
"""

import datetime
import os
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple
from decimal import Decimal

from sqlalchemy import create_engine, func, and_, or_, desc, asc
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.pool import StaticPool

from .schemas import (
    Base, StockContract, OptionContract, StockMarketData, OptionMarketData,
    Portfolio, Position, Transaction, PortfolioPerformance, BacktestRun,
    create_additional_indexes, get_table_names
)
from ..core.contracts import OptionContract as CoreOptionContract, StockContract as CoreStockContract
from ..core.position import Position as CorePosition, Transaction as CoreTransaction
from ..core.portfolio import Portfolio as CorePortfolio
from ..core.exceptions import (
    DatabaseError, DatabaseConnectionError, DatabaseIntegrityError,
    DatabaseOperationError, DataNotFoundError
)
from ..utils.logger import get_logger
from ..utils.config import get_config


class DatabaseManager:
    """
    Comprehensive database manager for the backtesting system.
    
    This class handles all database operations including connection management,
    CRUD operations, data validation, and performance optimization.
    """
    
    def __init__(self, connection_string: Optional[str] = None, echo: bool = False):
        """
        Initialize the database manager.
        
        Args:
            connection_string: Database connection string
            echo: Whether to echo SQL statements (for debugging)
        """
        self.logger = get_logger('data.manager')
        
        # Get connection string from config if not provided
        if connection_string is None:
            config = get_config()
            connection_string = config.database.connection_string
            echo = config.database.echo
        
        self.connection_string = connection_string
        self.echo = echo
        
        # Create engine with appropriate settings
        engine_kwargs = {
            'echo': echo,
            'pool_pre_ping': True,  # Verify connections before use
        }
        
        # SQLite-specific settings
        if connection_string.startswith('sqlite'):
            engine_kwargs.update({
                'poolclass': StaticPool,
                'connect_args': {
                    'check_same_thread': False,
                    'timeout': 30
                }
            })
        
        try:
            self.engine = create_engine(connection_string, **engine_kwargs)
            self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)
            self.logger.info(f"Database engine created: {connection_string}")
        except Exception as e:
            self.logger.error(f"Failed to create database engine: {e}")
            raise DatabaseConnectionError(f"Failed to create database engine: {e}")
    
    def initialize_database(self, drop_existing: bool = False) -> None:
        """
        Initialize the database schema.
        
        Args:
            drop_existing: Whether to drop existing tables first
        """
        try:
            if drop_existing:
                self.logger.warning("Dropping existing database tables")
                Base.metadata.drop_all(bind=self.engine)
            
            # Create all tables
            Base.metadata.create_all(bind=self.engine)
            
            # Create additional indexes
            create_additional_indexes(self.engine)
            
            # Create data directories if using SQLite
            if self.connection_string.startswith('sqlite'):
                db_path = self.connection_string.replace('sqlite:///', '')
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            
            self.logger.info("Database schema initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise DatabaseError(f"Failed to initialize database: {e}")
    
    @contextmanager
    def get_session(self):
        """
        Get a database session with automatic cleanup.
        
        Yields:
            SQLAlchemy session
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def test_connection(self) -> bool:
        """
        Test database connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            with self.get_session() as session:
                session.execute(func.now())
            self.logger.info("Database connection test successful")
            return True
        except Exception as e:
            self.logger.error(f"Database connection test failed: {e}")
            return False
    
    # Stock Contract Operations
    
    def save_stock_contract(self, contract: CoreStockContract) -> int:
        """
        Save a stock contract to the database.
        
        Args:
            contract: Stock contract to save
            
        Returns:
            Database ID of the saved contract
        """
        try:
            with self.get_session() as session:
                # Check if contract already exists
                existing = session.query(StockContract).filter_by(symbol=contract.symbol).first()
                
                if existing:
                    # Update existing contract
                    self._update_stock_contract_from_core(existing, contract)
                    contract_id = existing.id
                    self.logger.debug(f"Updated stock contract: {contract.symbol}")
                else:
                    # Create new contract
                    db_contract = self._create_stock_contract_from_core(contract)
                    session.add(db_contract)
                    session.flush()  # Get the ID
                    contract_id = db_contract.id
                    self.logger.debug(f"Created stock contract: {contract.symbol}")
                
                return contract_id
                
        except IntegrityError as e:
            raise DatabaseIntegrityError(f"Failed to save stock contract: {e}")
        except Exception as e:
            raise DatabaseOperationError(f"Failed to save stock contract: {e}")
    
    def get_stock_contract(self, symbol: str) -> Optional[CoreStockContract]:
        """
        Get a stock contract by symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Stock contract or None if not found
        """
        try:
            with self.get_session() as session:
                db_contract = session.query(StockContract).filter_by(symbol=symbol.upper()).first()
                
                if db_contract:
                    return self._create_core_stock_contract(db_contract)
                return None
                
        except Exception as e:
            raise DatabaseOperationError(f"Failed to get stock contract: {e}")
    
    def get_all_stock_contracts(self) -> List[CoreStockContract]:
        """Get all stock contracts."""
        try:
            with self.get_session() as session:
                db_contracts = session.query(StockContract).all()
                return [self._create_core_stock_contract(db_contract) for db_contract in db_contracts]
                
        except Exception as e:
            raise DatabaseOperationError(f"Failed to get stock contracts: {e}")
    
    # Option Contract Operations
    
    def save_option_contract(self, contract: CoreOptionContract) -> int:
        """
        Save an option contract to the database.
        
        Args:
            contract: Option contract to save
            
        Returns:
            Database ID of the saved contract
        """
        try:
            with self.get_session() as session:
                # Check if contract already exists
                existing = session.query(OptionContract).filter_by(contract_id=contract.contract_id).first()
                
                if existing:
                    # Update existing contract
                    self._update_option_contract_from_core(existing, contract)
                    contract_id = existing.id
                    self.logger.debug(f"Updated option contract: {contract.contract_id}")
                else:
                    # Create new contract
                    db_contract = self._create_option_contract_from_core(contract)
                    session.add(db_contract)
                    session.flush()  # Get the ID
                    contract_id = db_contract.id
                    self.logger.debug(f"Created option contract: {contract.contract_id}")
                
                return contract_id
                
        except IntegrityError as e:
            raise DatabaseIntegrityError(f"Failed to save option contract: {e}")
        except Exception as e:
            raise DatabaseOperationError(f"Failed to save option contract: {e}")
    
    def get_option_contract(self, contract_id: str) -> Optional[CoreOptionContract]:
        """
        Get an option contract by contract ID.
        
        Args:
            contract_id: Option contract ID
            
        Returns:
            Option contract or None if not found
        """
        try:
            with self.get_session() as session:
                db_contract = session.query(OptionContract).filter_by(contract_id=contract_id).first()
                
                if db_contract:
                    return self._create_core_option_contract(db_contract)
                return None
                
        except Exception as e:
            raise DatabaseOperationError(f"Failed to get option contract: {e}")
    
    def get_options_by_symbol(self, symbol: str, expiration_date: Optional[datetime.date] = None) -> List[CoreOptionContract]:
        """
        Get option contracts by symbol and optional expiration.
        
        Args:
            symbol: Underlying symbol
            expiration_date: Optional expiration date filter
            
        Returns:
            List of option contracts
        """
        try:
            with self.get_session() as session:
                query = session.query(OptionContract).filter_by(symbol=symbol.upper())
                
                if expiration_date:
                    query = query.filter_by(expiration=expiration_date)
                
                db_contracts = query.all()
                return [self._create_core_option_contract(db_contract) for db_contract in db_contracts]
                
        except Exception as e:
            raise DatabaseOperationError(f"Failed to get option contracts: {e}")
    
    # Market Data Operations
    
    def save_stock_market_data(self, symbol: str, timestamp: datetime.datetime, data: Dict[str, Any]) -> None:
        """
        Save stock market data.
        
        Args:
            symbol: Stock symbol
            timestamp: Data timestamp
            data: Market data dictionary
        """
        try:
            with self.get_session() as session:
                # Get stock contract
                stock_contract = session.query(StockContract).filter_by(symbol=symbol.upper()).first()
                if not stock_contract:
                    raise DataNotFoundError(f"Stock contract not found: {symbol}")
                
                # Create market data record
                market_data = StockMarketData(
                    stock_contract_id=stock_contract.id,
                    timestamp=timestamp,
                    open_price=data.get('open'),
                    high_price=data.get('high'),
                    low_price=data.get('low'),
                    close_price=data.get('close'),
                    volume=data.get('volume'),
                    bid=data.get('bid'),
                    ask=data.get('ask')
                )
                
                session.merge(market_data)  # Use merge to handle duplicates
                
                # Update stock contract with latest data
                if 'bid' in data:
                    stock_contract.bid = data['bid']
                if 'ask' in data:
                    stock_contract.ask = data['ask']
                if 'close' in data:
                    stock_contract.last = data['close']
                if 'volume' in data:
                    stock_contract.volume = data['volume']
                
                stock_contract.last_data_update = timestamp
                
                self.logger.debug(f"Saved market data for {symbol} at {timestamp}")
                
        except Exception as e:
            raise DatabaseOperationError(f"Failed to save stock market data: {e}")
    
    def save_option_market_data(self, contract_id: str, timestamp: datetime.datetime, data: Dict[str, Any]) -> None:
        """
        Save option market data.
        
        Args:
            contract_id: Option contract ID
            timestamp: Data timestamp
            data: Market data dictionary
        """
        try:
            with self.get_session() as session:
                # Get option contract
                option_contract = session.query(OptionContract).filter_by(contract_id=contract_id).first()
                if not option_contract:
                    raise DataNotFoundError(f"Option contract not found: {contract_id}")
                
                # Create market data record
                market_data = OptionMarketData(
                    option_contract_id=option_contract.id,
                    timestamp=timestamp,
                    bid=data.get('bid'),
                    ask=data.get('ask'),
                    last=data.get('last'),
                    volume=data.get('volume'),
                    open_interest=data.get('open_interest'),
                    delta=data.get('delta'),
                    gamma=data.get('gamma'),
                    theta=data.get('theta'),
                    vega=data.get('vega'),
                    rho=data.get('rho'),
                    implied_volatility=data.get('implied_volatility'),
                    underlying_price=data.get('underlying_price')
                )
                
                session.merge(market_data)  # Use merge to handle duplicates
                
                # Update option contract with latest data
                for field in ['bid', 'ask', 'last', 'volume', 'open_interest',
                             'delta', 'gamma', 'theta', 'vega', 'rho', 'implied_volatility']:
                    if field in data:
                        setattr(option_contract, field, data[field])
                
                option_contract.last_data_update = timestamp
                
                self.logger.debug(f"Saved market data for {contract_id} at {timestamp}")
                
        except Exception as e:
            raise DatabaseOperationError(f"Failed to save option market data: {e}")
    
    def get_stock_market_data_range(self, symbol: str, start_date: datetime.datetime, 
                                   end_date: datetime.datetime) -> List[Dict[str, Any]]:
        """
        Get stock market data for a date range.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            List of market data records
        """
        try:
            with self.get_session() as session:
                stock_contract = session.query(StockContract).filter_by(symbol=symbol.upper()).first()
                if not stock_contract:
                    raise DataNotFoundError(f"Stock contract not found: {symbol}")
                
                market_data = session.query(StockMarketData).filter(
                    and_(
                        StockMarketData.stock_contract_id == stock_contract.id,
                        StockMarketData.timestamp >= start_date,
                        StockMarketData.timestamp <= end_date
                    )
                ).order_by(StockMarketData.timestamp).all()
                
                return [self._stock_market_data_to_dict(data) for data in market_data]
                
        except Exception as e:
            raise DatabaseOperationError(f"Failed to get stock market data: {e}")
    
    # Portfolio Operations
    
    def save_portfolio(self, portfolio: CorePortfolio) -> int:
        """
        Save a portfolio to the database.
        
        Args:
            portfolio: Portfolio to save
            
        Returns:
            Database ID of the saved portfolio
        """
        try:
            with self.get_session() as session:
                # Check if portfolio already exists
                existing = session.query(Portfolio).filter_by(portfolio_id=portfolio.portfolio_id).first()
                
                if existing:
                    # Update existing portfolio
                    self._update_portfolio_from_core(existing, portfolio)
                    portfolio_id = existing.id
                    self.logger.debug(f"Updated portfolio: {portfolio.portfolio_id}")
                else:
                    # Create new portfolio
                    db_portfolio = self._create_portfolio_from_core(portfolio)
                    session.add(db_portfolio)
                    session.flush()  # Get the ID
                    portfolio_id = db_portfolio.id
                    self.logger.debug(f"Created portfolio: {portfolio.portfolio_id}")
                
                return portfolio_id
                
        except IntegrityError as e:
            raise DatabaseIntegrityError(f"Failed to save portfolio: {e}")
        except Exception as e:
            raise DatabaseOperationError(f"Failed to save portfolio: {e}")
    
    def get_portfolio(self, portfolio_id: str) -> Optional[CorePortfolio]:
        """
        Get a portfolio by ID.
        
        Args:
            portfolio_id: Portfolio ID
            
        Returns:
            Portfolio or None if not found
        """
        try:
            with self.get_session() as session:
                db_portfolio = session.query(Portfolio).filter_by(portfolio_id=portfolio_id).first()
                
                if db_portfolio:
                    return self._create_core_portfolio(db_portfolio, session)
                return None
                
        except Exception as e:
            raise DatabaseOperationError(f"Failed to get portfolio: {e}")
    
    def save_position(self, position: CorePosition, portfolio_db_id: int) -> int:
        """
        Save a position to the database.
        
        Args:
            position: Position to save
            portfolio_db_id: Database ID of the portfolio
            
        Returns:
            Database ID of the saved position
        """
        try:
            with self.get_session() as session:
                # Check if position already exists
                existing = session.query(Position).filter_by(position_id=position.position_id).first()
                
                if existing:
                    # Update existing position
                    self._update_position_from_core(existing, position)
                    position_id = existing.id
                    self.logger.debug(f"Updated position: {position.position_id}")
                else:
                    # Create new position
                    db_position = self._create_position_from_core(position, portfolio_db_id)
                    session.add(db_position)
                    session.flush()  # Get the ID
                    position_id = db_position.id
                    self.logger.debug(f"Created position: {position.position_id}")
                
                return position_id
                
        except IntegrityError as e:
            raise DatabaseIntegrityError(f"Failed to save position: {e}")
        except Exception as e:
            raise DatabaseOperationError(f"Failed to save position: {e}")
    
    def save_transaction(self, transaction: CoreTransaction, position_db_id: int, portfolio_db_id: int) -> int:
        """
        Save a transaction to the database.
        
        Args:
            transaction: Transaction to save
            position_db_id: Database ID of the position
            portfolio_db_id: Database ID of the portfolio
            
        Returns:
            Database ID of the saved transaction
        """
        try:
            with self.get_session() as session:
                db_transaction = self._create_transaction_from_core(transaction, position_db_id, portfolio_db_id)
                session.add(db_transaction)
                session.flush()  # Get the ID
                
                self.logger.debug(f"Created transaction: {transaction.transaction_id}")
                return db_transaction.id
                
        except IntegrityError as e:
            raise DatabaseIntegrityError(f"Failed to save transaction: {e}")
        except Exception as e:
            raise DatabaseOperationError(f"Failed to save transaction: {e}")
    
    # Helper methods for object conversion
    
    def _create_stock_contract_from_core(self, contract: CoreStockContract) -> StockContract:
        """Create database StockContract from core StockContract."""
        return StockContract(
            symbol=contract.symbol,
            exchange=contract.exchange,
            currency=contract.currency,
            bid=contract.bid,
            ask=contract.ask,
            last=contract.last,
            volume=contract.volume,
            market_cap=contract.market_cap,
            pe_ratio=contract.pe_ratio,
            dividend_yield=contract.dividend_yield,
            status=contract.status.value,
            last_data_update=contract.last_update
        )
    
    def _update_stock_contract_from_core(self, db_contract: StockContract, contract: CoreStockContract) -> None:
        """Update database StockContract from core StockContract."""
        db_contract.exchange = contract.exchange
        db_contract.currency = contract.currency
        db_contract.bid = contract.bid
        db_contract.ask = contract.ask
        db_contract.last = contract.last
        db_contract.volume = contract.volume
        db_contract.market_cap = contract.market_cap
        db_contract.pe_ratio = contract.pe_ratio
        db_contract.dividend_yield = contract.dividend_yield
        db_contract.status = contract.status.value
        db_contract.last_data_update = contract.last_update
    
    def _create_core_stock_contract(self, db_contract: StockContract) -> CoreStockContract:
        """Create core StockContract from database StockContract."""
        from ..core.contracts import ContractStatus
        
        contract = CoreStockContract(
            symbol=db_contract.symbol,
            exchange=db_contract.exchange,
            currency=db_contract.currency,
            bid=db_contract.bid,
            ask=db_contract.ask,
            last=db_contract.last,
            volume=db_contract.volume,
            market_cap=db_contract.market_cap,
            pe_ratio=db_contract.pe_ratio,
            dividend_yield=db_contract.dividend_yield,
            last_update=db_contract.last_data_update,
            status=ContractStatus(db_contract.status)
        )
        return contract
    
    def _create_option_contract_from_core(self, contract: CoreOptionContract) -> OptionContract:
        """Create database OptionContract from core OptionContract."""
        return OptionContract(
            contract_id=contract.contract_id,
            symbol=contract.symbol,
            strike=contract.strike,
            expiration=contract.expiration,
            option_type=contract.option_type.value,
            exchange=contract.exchange,
            multiplier=contract.multiplier,
            currency=contract.currency,
            bid=contract.bid,
            ask=contract.ask,
            last=contract.last,
            volume=contract.volume,
            open_interest=contract.open_interest,
            delta=contract.delta,
            gamma=contract.gamma,
            theta=contract.theta,
            vega=contract.vega,
            rho=contract.rho,
            implied_volatility=contract.implied_volatility,
            status=contract.status.value,
            last_data_update=contract.last_update
        )
    
    def _update_option_contract_from_core(self, db_contract: OptionContract, contract: CoreOptionContract) -> None:
        """Update database OptionContract from core OptionContract."""
        db_contract.bid = contract.bid
        db_contract.ask = contract.ask
        db_contract.last = contract.last
        db_contract.volume = contract.volume
        db_contract.open_interest = contract.open_interest
        db_contract.delta = contract.delta
        db_contract.gamma = contract.gamma
        db_contract.theta = contract.theta
        db_contract.vega = contract.vega
        db_contract.rho = contract.rho
        db_contract.implied_volatility = contract.implied_volatility
        db_contract.status = contract.status.value
        db_contract.last_data_update = contract.last_update
    
    def _create_core_option_contract(self, db_contract: OptionContract) -> CoreOptionContract:
        """Create core OptionContract from database OptionContract."""
        from ..core.contracts import OptionType, ContractStatus
        
        contract = CoreOptionContract(
            symbol=db_contract.symbol,
            strike=db_contract.strike,
            expiration=db_contract.expiration,
            option_type=OptionType(db_contract.option_type),
            exchange=db_contract.exchange,
            multiplier=db_contract.multiplier,
            currency=db_contract.currency,
            bid=db_contract.bid,
            ask=db_contract.ask,
            last=db_contract.last,
            volume=db_contract.volume,
            open_interest=db_contract.open_interest,
            delta=db_contract.delta,
            gamma=db_contract.gamma,
            theta=db_contract.theta,
            vega=db_contract.vega,
            rho=db_contract.rho,
            implied_volatility=db_contract.implied_volatility,
            last_update=db_contract.last_data_update,
            status=ContractStatus(db_contract.status)
        )
        # Set the private contract_id field
        contract._contract_id = db_contract.contract_id
        return contract
    
    def _create_portfolio_from_core(self, portfolio: CorePortfolio) -> Portfolio:
        """Create database Portfolio from core Portfolio."""
        return Portfolio(
            portfolio_id=portfolio.portfolio_id,
            name=portfolio.name,
            initial_cash=portfolio.initial_cash,
            current_cash=portfolio.cash,
            total_realized_pnl=portfolio.total_realized_pnl,
            total_unrealized_pnl=portfolio.total_unrealized_pnl,
            max_drawdown=portfolio.max_drawdown,
            peak_value=portfolio.peak_value,
            margin_used=portfolio.margin_used,
            margin_available=portfolio.margin_available,
            max_position_size_pct=portfolio.max_position_size_pct,
            max_portfolio_leverage=portfolio.max_portfolio_leverage
        )
    
    def _update_portfolio_from_core(self, db_portfolio: Portfolio, portfolio: CorePortfolio) -> None:
        """Update database Portfolio from core Portfolio."""
        db_portfolio.name = portfolio.name
        db_portfolio.current_cash = portfolio.cash
        db_portfolio.total_realized_pnl = portfolio.total_realized_pnl
        db_portfolio.total_unrealized_pnl = portfolio.total_unrealized_pnl
        db_portfolio.max_drawdown = portfolio.max_drawdown
        db_portfolio.peak_value = portfolio.peak_value
        db_portfolio.margin_used = portfolio.margin_used
        db_portfolio.margin_available = portfolio.margin_available
        db_portfolio.max_position_size_pct = portfolio.max_position_size_pct
        db_portfolio.max_portfolio_leverage = portfolio.max_portfolio_leverage
    
    def _create_core_portfolio(self, db_portfolio: Portfolio, session: Session) -> CorePortfolio:
        """Create core Portfolio from database Portfolio."""
        # This is a simplified version - in practice, you'd need to load all positions and transactions
        portfolio = CorePortfolio(
            initial_cash=db_portfolio.initial_cash,
            name=db_portfolio.name,
            portfolio_id=db_portfolio.portfolio_id
        )
        
        # Update portfolio state
        portfolio.cash = db_portfolio.current_cash
        portfolio.total_realized_pnl = db_portfolio.total_realized_pnl
        portfolio.total_unrealized_pnl = db_portfolio.total_unrealized_pnl
        portfolio.max_drawdown = db_portfolio.max_drawdown
        portfolio.peak_value = db_portfolio.peak_value
        portfolio.margin_used = db_portfolio.margin_used
        portfolio.margin_available = db_portfolio.margin_available
        portfolio.max_position_size_pct = db_portfolio.max_position_size_pct
        portfolio.max_portfolio_leverage = db_portfolio.max_portfolio_leverage
        
        return portfolio
    
    def _create_position_from_core(self, position: CorePosition, portfolio_db_id: int) -> Position:
        """Create database Position from core Position."""
        from ..core.contracts import OptionContract as CoreOptionContract
        
        return Position(
            position_id=position.position_id,
            portfolio_id=portfolio_db_id,
            contract_type="option" if isinstance(position.contract, CoreOptionContract) else "stock",
            contract_symbol=position.contract.symbol,
            contract_id=position.contract.contract_id,
            initial_quantity=position.initial_quantity,
            current_quantity=position.current_quantity,
            entry_price=position.entry_price,
            entry_timestamp=position.entry_timestamp,
            realized_pnl=position.realized_pnl,
            unrealized_pnl=position.unrealized_pnl,
            initial_margin=position.initial_margin,
            maintenance_margin=position.maintenance_margin,
            status=position.status.value,
            side=position.side.value,
            closed_at=None if position.status.value != "CLOSED" else datetime.datetime.now()
        )
    
    def _update_position_from_core(self, db_position: Position, position: CorePosition) -> None:
        """Update database Position from core Position."""
        db_position.current_quantity = position.current_quantity
        db_position.realized_pnl = position.realized_pnl
        db_position.unrealized_pnl = position.unrealized_pnl
        db_position.status = position.status.value
        db_position.side = position.side.value
        if position.status.value == "CLOSED" and db_position.closed_at is None:
            db_position.closed_at = datetime.datetime.now()
    
    def _create_transaction_from_core(self, transaction: CoreTransaction, position_db_id: int, portfolio_db_id: int) -> Transaction:
        """Create database Transaction from core Transaction."""
        return Transaction(
            transaction_id=transaction.transaction_id,
            portfolio_id=portfolio_db_id,
            position_id=position_db_id,
            timestamp=transaction.timestamp,
            transaction_type="BUY" if transaction.quantity > 0 else "SELL",
            quantity=transaction.quantity,
            price=transaction.price,
            commission=transaction.commission,
            fees=transaction.fees,
            gross_amount=transaction.gross_amount,
            net_amount=transaction.net_amount
        )
    
    def _stock_market_data_to_dict(self, data: StockMarketData) -> Dict[str, Any]:
        """Convert database StockMarketData to dictionary."""
        return {
            'timestamp': data.timestamp,
            'open': float(data.open_price) if data.open_price else None,
            'high': float(data.high_price) if data.high_price else None,
            'low': float(data.low_price) if data.low_price else None,
            'close': float(data.close_price) if data.close_price else None,
            'volume': data.volume,
            'bid': float(data.bid) if data.bid else None,
            'ask': float(data.ask) if data.ask else None
        }
    
    # Utility methods
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database information and statistics."""
        try:
            with self.get_session() as session:
                info = {
                    'connection_string': self.connection_string.replace(r'password=\w+', 'password=***'),
                    'table_count': len(get_table_names()),
                    'tables': {}
                }
                
                # Get record counts for each table
                for table_name in get_table_names():
                    try:
                        table_class = globals().get(table_name.title().replace('_', ''))
                        if table_class:
                            count = session.query(table_class).count()
                            info['tables'][table_name] = count
                    except Exception:
                        info['tables'][table_name] = 'Error counting records'
                
                return info
                
        except Exception as e:
            raise DatabaseOperationError(f"Failed to get database info: {e}")
    
    def cleanup_old_data(self, days_to_keep: int = 365) -> Dict[str, int]:
        """
        Clean up old market data to save space.
        
        Args:
            days_to_keep: Number of days of data to keep
            
        Returns:
            Dictionary with counts of deleted records
        """
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_to_keep)
        deleted_counts = {}
        
        try:
            with self.get_session() as session:
                # Delete old stock market data
                stock_deleted = session.query(StockMarketData).filter(
                    StockMarketData.timestamp < cutoff_date
                ).delete()
                deleted_counts['stock_market_data'] = stock_deleted
                
                # Delete old option market data
                option_deleted = session.query(OptionMarketData).filter(
                    OptionMarketData.timestamp < cutoff_date
                ).delete()
                deleted_counts['option_market_data'] = option_deleted
                
                self.logger.info(f"Cleaned up old data: {deleted_counts}")
                return deleted_counts
                
        except Exception as e:
            raise DatabaseOperationError(f"Failed to cleanup old data: {e}")
    
    def close(self) -> None:
        """Close database connections."""
        try:
            self.engine.dispose()
            self.logger.info("Database connections closed")
        except Exception as e:
            self.logger.error(f"Error closing database connections: {e}")


# Singleton instance for global access
_db_manager: Optional[DatabaseManager] = None


def get_database_manager(connection_string: Optional[str] = None) -> DatabaseManager:
    """
    Get the global database manager instance.
    
    Args:
        connection_string: Database connection string (only used on first call)
        
    Returns:
        DatabaseManager instance
    """
    global _db_manager
    
    if _db_manager is None:
        _db_manager = DatabaseManager(connection_string)
    
    return _db_manager


if __name__ == "__main__":
    # Test the database manager
    db = DatabaseManager("sqlite:///test_backtest.db")
    
    # Initialize database
    db.initialize_database(drop_existing=True)
    
    # Test connection
    if db.test_connection():
        print("Database connection successful!")
        
        # Get database info
        info = db.get_database_info()
        print(f"Database info: {info}")
    
    # Clean up
    db.close()