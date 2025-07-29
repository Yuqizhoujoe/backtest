"""
SQLAlchemy database schemas for the options backtesting system.

This module defines the database models that correspond to the core
business objects in the backtesting system.
"""

import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Date,
    Numeric,
    Text,
    ForeignKey,
    Index,
    CheckConstraint,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.types import TypeDecorator, VARCHAR

from ..core.contracts import OptionType, ContractStatus
from ..core.position import PositionStatus, PositionSide


Base = declarative_base()


class ChoiceType(TypeDecorator):
    """Custom type for storing enum values as strings."""

    impl = VARCHAR

    def __init__(self, choices, **kw):
        self.choices = tuple(choices)
        super().__init__(**kw)

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if hasattr(value, "value"):
            return value.value
        return str(value)

    def process_result_value(self, value, dialect):
        return value


class StockContract(Base):
    """Stock contract database model."""

    __tablename__ = "stock_contracts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), unique=True, nullable=False, index=True)
    exchange = Column(String(20), default="SMART", nullable=False)
    currency = Column(String(3), default="USD", nullable=False)

    # Market data (latest values)
    bid = Column(Numeric(10, 4), nullable=True)
    ask = Column(Numeric(10, 4), nullable=True)
    last = Column(Numeric(10, 4), nullable=True)
    volume = Column(Integer, nullable=True)

    # Fundamental data
    market_cap = Column(Numeric(15, 2), nullable=True)
    pe_ratio = Column(Numeric(8, 2), nullable=True)
    dividend_yield = Column(Numeric(5, 4), nullable=True)

    # Status and metadata
    status = Column(
        ChoiceType(ContractStatus), default=ContractStatus.ACTIVE.value, nullable=False
    )
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
        nullable=False,
    )
    last_data_update = Column(DateTime, nullable=True)

    # Relationships
    market_data_history = relationship(
        "StockMarketData", back_populates="stock_contract", cascade="all, delete-orphan"
    )

    # Constraints
    __table_args__ = (
        CheckConstraint("bid >= 0", name="ck_stock_bid_positive"),
        CheckConstraint("ask >= 0", name="ck_stock_ask_positive"),
        CheckConstraint("last >= 0", name="ck_stock_last_positive"),
        CheckConstraint("volume >= 0", name="ck_stock_volume_positive"),
        CheckConstraint("market_cap >= 0", name="ck_stock_market_cap_positive"),
    )

    @validates("symbol")
    def validate_symbol(self, key, symbol):
        if not symbol or len(symbol.strip()) == 0:
            raise ValueError("Symbol cannot be empty")
        return symbol.upper().strip()

    def __repr__(self):
        return f"<StockContract(symbol='{self.symbol}', exchange='{self.exchange}')>"


class OptionContract(Base):
    """Option contract database model."""

    __tablename__ = "option_contracts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    contract_id = Column(String(100), unique=True, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    strike = Column(Numeric(10, 2), nullable=False)
    expiration = Column(Date, nullable=False, index=True)
    option_type = Column(ChoiceType(OptionType), nullable=False)
    exchange = Column(String(20), default="SMART", nullable=False)
    multiplier = Column(Integer, default=100, nullable=False)
    currency = Column(String(3), default="USD", nullable=False)

    # Market data (latest values)
    bid = Column(Numeric(10, 4), nullable=True)
    ask = Column(Numeric(10, 4), nullable=True)
    last = Column(Numeric(10, 4), nullable=True)
    volume = Column(Integer, nullable=True)
    open_interest = Column(Integer, nullable=True)

    # Greeks (latest values)
    delta = Column(Numeric(8, 6), nullable=True)
    gamma = Column(Numeric(8, 6), nullable=True)
    theta = Column(Numeric(8, 6), nullable=True)
    vega = Column(Numeric(8, 6), nullable=True)
    rho = Column(Numeric(8, 6), nullable=True)
    implied_volatility = Column(Numeric(8, 6), nullable=True)

    # Status and metadata
    status = Column(
        ChoiceType(ContractStatus), default=ContractStatus.ACTIVE.value, nullable=False
    )
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
        nullable=False,
    )
    last_data_update = Column(DateTime, nullable=True)

    # Relationships
    market_data_history = relationship(
        "OptionMarketData",
        back_populates="option_contract",
        cascade="all, delete-orphan",
    )

    # Constraints
    __table_args__ = (
        CheckConstraint("strike > 0", name="ck_option_strike_positive"),
        CheckConstraint("multiplier > 0", name="ck_option_multiplier_positive"),
        CheckConstraint("bid >= 0", name="ck_option_bid_positive"),
        CheckConstraint("ask >= 0", name="ck_option_ask_positive"),
        CheckConstraint("last >= 0", name="ck_option_last_positive"),
        CheckConstraint("volume >= 0", name="ck_option_volume_positive"),
        CheckConstraint("open_interest >= 0", name="ck_option_oi_positive"),
        CheckConstraint("delta >= -1 AND delta <= 1", name="ck_option_delta_range"),
        CheckConstraint("implied_volatility >= 0", name="ck_option_iv_positive"),
        UniqueConstraint(
            "symbol", "strike", "expiration", "option_type", name="uq_option_contract"
        ),
        Index("idx_option_symbol_expiration", "symbol", "expiration"),
        Index("idx_option_expiration_type", "expiration", "option_type"),
    )

    @validates("contract_id")
    def validate_contract_id(self, key, contract_id):
        if not contract_id or len(contract_id.strip()) == 0:
            raise ValueError("Contract ID cannot be empty")
        return contract_id

    @validates("symbol")
    def validate_symbol(self, key, symbol):
        if not symbol or len(symbol.strip()) == 0:
            raise ValueError("Symbol cannot be empty")
        return symbol.upper().strip()

    def __repr__(self):
        return (
            f"<OptionContract(symbol='{self.symbol}', strike={self.strike}, "
            f"expiration='{self.expiration}', type='{self.option_type}')>"
        )


class StockMarketData(Base):
    """Stock market data time series."""

    __tablename__ = "stock_market_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_contract_id = Column(
        Integer, ForeignKey("stock_contracts.id"), nullable=False
    )
    timestamp = Column(DateTime, nullable=False, index=True)

    # OHLCV data
    open_price = Column(Numeric(10, 4), nullable=True)
    high_price = Column(Numeric(10, 4), nullable=True)
    low_price = Column(Numeric(10, 4), nullable=True)
    close_price = Column(Numeric(10, 4), nullable=True)
    volume = Column(Integer, nullable=True)

    # Bid/Ask data
    bid = Column(Numeric(10, 4), nullable=True)
    ask = Column(Numeric(10, 4), nullable=True)

    # Relationships
    stock_contract = relationship("StockContract", back_populates="market_data_history")

    # Constraints
    __table_args__ = (
        UniqueConstraint("stock_contract_id", "timestamp", name="uq_stock_market_data"),
        CheckConstraint("open_price >= 0", name="ck_stock_md_open_positive"),
        CheckConstraint("high_price >= 0", name="ck_stock_md_high_positive"),
        CheckConstraint("low_price >= 0", name="ck_stock_md_low_positive"),
        CheckConstraint("close_price >= 0", name="ck_stock_md_close_positive"),
        CheckConstraint("volume >= 0", name="ck_stock_md_volume_positive"),
        Index("idx_stock_md_timestamp", "timestamp"),
    )

    def __repr__(self):
        return (
            f"<StockMarketData(contract_id={self.stock_contract_id}, "
            f"timestamp='{self.timestamp}', close={self.close_price})>"
        )


class OptionMarketData(Base):
    """Option market data time series."""

    __tablename__ = "option_market_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    option_contract_id = Column(
        Integer, ForeignKey("option_contracts.id"), nullable=False
    )
    timestamp = Column(DateTime, nullable=False, index=True)

    # Market data
    bid = Column(Numeric(10, 4), nullable=True)
    ask = Column(Numeric(10, 4), nullable=True)
    last = Column(Numeric(10, 4), nullable=True)
    volume = Column(Integer, nullable=True)
    open_interest = Column(Integer, nullable=True)

    # Greeks
    delta = Column(Numeric(8, 6), nullable=True)
    gamma = Column(Numeric(8, 6), nullable=True)
    theta = Column(Numeric(8, 6), nullable=True)
    vega = Column(Numeric(8, 6), nullable=True)
    rho = Column(Numeric(8, 6), nullable=True)
    implied_volatility = Column(Numeric(8, 6), nullable=True)

    # Underlying price at the time
    underlying_price = Column(Numeric(10, 4), nullable=True)

    # Relationships
    option_contract = relationship(
        "OptionContract", back_populates="market_data_history"
    )

    # Constraints
    __table_args__ = (
        UniqueConstraint(
            "option_contract_id", "timestamp", name="uq_option_market_data"
        ),
        CheckConstraint("bid >= 0", name="ck_option_md_bid_positive"),
        CheckConstraint("ask >= 0", name="ck_option_md_ask_positive"),
        CheckConstraint("last >= 0", name="ck_option_md_last_positive"),
        CheckConstraint("volume >= 0", name="ck_option_md_volume_positive"),
        CheckConstraint("open_interest >= 0", name="ck_option_md_oi_positive"),
        CheckConstraint("delta >= -1 AND delta <= 1", name="ck_option_md_delta_range"),
        CheckConstraint("implied_volatility >= 0", name="ck_option_md_iv_positive"),
        Index("idx_option_md_timestamp", "timestamp"),
    )

    def __repr__(self):
        return (
            f"<OptionMarketData(contract_id={self.option_contract_id}, "
            f"timestamp='{self.timestamp}', last={self.last})>"
        )


class Portfolio(Base):
    """Portfolio database model."""

    __tablename__ = "portfolios"

    id = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_id = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    initial_cash = Column(Numeric(15, 2), nullable=False)

    # Current state
    current_cash = Column(Numeric(15, 2), nullable=False)
    total_realized_pnl = Column(Numeric(15, 2), default=0, nullable=False)
    total_unrealized_pnl = Column(Numeric(15, 2), default=0, nullable=False)
    max_drawdown = Column(Numeric(8, 4), default=0, nullable=False)
    peak_value = Column(Numeric(15, 2), nullable=False)

    # Risk metrics
    margin_used = Column(Numeric(15, 2), default=0, nullable=False)
    margin_available = Column(Numeric(15, 2), default=0, nullable=False)

    # Configuration
    max_position_size_pct = Column(Numeric(5, 4), default=0.1, nullable=False)
    max_portfolio_leverage = Column(Numeric(8, 4), default=2.0, nullable=False)

    # Metadata
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
        nullable=False,
    )

    # Relationships
    positions = relationship(
        "Position", back_populates="portfolio", cascade="all, delete-orphan"
    )
    transactions = relationship(
        "Transaction", back_populates="portfolio", cascade="all, delete-orphan"
    )
    performance_snapshots = relationship(
        "PortfolioPerformance", back_populates="portfolio", cascade="all, delete-orphan"
    )

    # Constraints
    __table_args__ = (
        CheckConstraint("initial_cash > 0", name="ck_portfolio_initial_cash_positive"),
        CheckConstraint(
            "max_position_size_pct > 0 AND max_position_size_pct <= 1",
            name="ck_portfolio_max_position_size",
        ),
        CheckConstraint(
            "max_portfolio_leverage > 0", name="ck_portfolio_max_leverage_positive"
        ),
    )

    @validates("portfolio_id")
    def validate_portfolio_id(self, key, portfolio_id):
        if not portfolio_id or len(portfolio_id.strip()) == 0:
            raise ValueError("Portfolio ID cannot be empty")
        return portfolio_id

    def __repr__(self):
        return (
            f"<Portfolio(id='{self.portfolio_id}', name='{self.name}', "
            f"cash={self.current_cash})>"
        )


class Position(Base):
    """Position database model."""

    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    position_id = Column(String(100), unique=True, nullable=False, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)

    # Contract reference (polymorphic - can be stock or option)
    contract_type = Column(String(20), nullable=False)  # 'stock' or 'option'
    contract_symbol = Column(String(20), nullable=False, index=True)
    contract_id = Column(String(100), nullable=False, index=True)

    # Position details
    initial_quantity = Column(Integer, nullable=False)
    current_quantity = Column(Integer, nullable=False)
    entry_price = Column(Numeric(10, 4), nullable=False)
    entry_timestamp = Column(DateTime, nullable=False)

    # P&L tracking
    realized_pnl = Column(Numeric(15, 2), default=0, nullable=False)
    unrealized_pnl = Column(Numeric(15, 2), default=0, nullable=False)

    # Risk metrics
    initial_margin = Column(Numeric(15, 2), nullable=True)
    maintenance_margin = Column(Numeric(15, 2), nullable=True)

    # Status
    status = Column(
        ChoiceType(PositionStatus), default=PositionStatus.OPEN.value, nullable=False
    )
    side = Column(ChoiceType(PositionSide), nullable=False)

    # Metadata
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
        nullable=False,
    )
    closed_at = Column(DateTime, nullable=True)

    # Relationships
    portfolio = relationship("Portfolio", back_populates="positions")
    transactions = relationship(
        "Transaction", back_populates="position", cascade="all, delete-orphan"
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "initial_quantity != 0", name="ck_position_initial_quantity_nonzero"
        ),
        CheckConstraint("entry_price > 0", name="ck_position_entry_price_positive"),
        Index("idx_position_portfolio_symbol", "portfolio_id", "contract_symbol"),
        Index("idx_position_status", "status"),
    )

    @validates("position_id")
    def validate_position_id(self, key, position_id):
        if not position_id or len(position_id.strip()) == 0:
            raise ValueError("Position ID cannot be empty")
        return position_id

    def __repr__(self):
        return (
            f"<Position(id='{self.position_id}', symbol='{self.contract_symbol}', "
            f"qty={self.current_quantity}, status='{self.status}')>"
        )


class Transaction(Base):
    """Transaction database model."""

    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(100), unique=True, nullable=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    position_id = Column(Integer, ForeignKey("positions.id"), nullable=False)

    # Transaction details
    timestamp = Column(DateTime, nullable=False, index=True)
    transaction_type = Column(String(10), nullable=False)  # BUY, SELL
    quantity = Column(Integer, nullable=False)
    price = Column(Numeric(10, 4), nullable=False)

    # Costs
    commission = Column(Numeric(8, 2), default=0, nullable=False)
    fees = Column(Numeric(8, 2), default=0, nullable=False)

    # Calculated fields
    gross_amount = Column(Numeric(15, 2), nullable=False)
    net_amount = Column(Numeric(15, 2), nullable=False)

    # Metadata
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    # Relationships
    portfolio = relationship("Portfolio", back_populates="transactions")
    position = relationship("Position", back_populates="transactions")

    # Constraints
    __table_args__ = (
        CheckConstraint("quantity != 0", name="ck_transaction_quantity_nonzero"),
        CheckConstraint("price > 0", name="ck_transaction_price_positive"),
        CheckConstraint("commission >= 0", name="ck_transaction_commission_positive"),
        CheckConstraint("fees >= 0", name="ck_transaction_fees_positive"),
        Index("idx_transaction_portfolio_timestamp", "portfolio_id", "timestamp"),
        Index("idx_transaction_position_timestamp", "position_id", "timestamp"),
    )

    def __repr__(self):
        return (
            f"<Transaction(id='{self.transaction_id}', "
            f"type='{self.transaction_type}', qty={self.quantity}, "
            f"price={self.price})>"
        )


class PortfolioPerformance(Base):
    """Portfolio performance snapshots."""

    __tablename__ = "portfolio_performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)

    # Portfolio values
    total_value = Column(Numeric(15, 2), nullable=False)
    cash = Column(Numeric(15, 2), nullable=False)
    position_value = Column(Numeric(15, 2), nullable=False)
    realized_pnl = Column(Numeric(15, 2), nullable=False)
    unrealized_pnl = Column(Numeric(15, 2), nullable=False)

    # Performance metrics
    return_pct = Column(Numeric(8, 4), nullable=False)
    daily_pnl = Column(Numeric(15, 2), nullable=True)
    daily_return_pct = Column(Numeric(8, 4), nullable=True)
    max_drawdown = Column(Numeric(8, 4), nullable=False)

    # Risk metrics
    leverage = Column(Numeric(8, 4), default=0, nullable=False)
    margin_used = Column(Numeric(15, 2), default=0, nullable=False)
    num_positions = Column(Integer, default=0, nullable=False)

    # Relationships
    portfolio = relationship("Portfolio", back_populates="performance_snapshots")

    # Constraints
    __table_args__ = (
        UniqueConstraint("portfolio_id", "timestamp", name="uq_portfolio_performance"),
        CheckConstraint("total_value >= 0", name="ck_perf_total_value_positive"),
        CheckConstraint("cash >= 0", name="ck_perf_cash_positive"),
        CheckConstraint("leverage >= 0", name="ck_perf_leverage_positive"),
        CheckConstraint("num_positions >= 0", name="ck_perf_num_positions_positive"),
        Index("idx_perf_portfolio_timestamp", "portfolio_id", "timestamp"),
    )

    def __repr__(self):
        return (
            f"<PortfolioPerformance(portfolio_id={self.portfolio_id}, "
            f"timestamp='{self.timestamp}', value={self.total_value})>"
        )


class BacktestRun(Base):
    """Backtest execution metadata."""

    __tablename__ = "backtest_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)

    # Backtest parameters
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    initial_cash = Column(Numeric(15, 2), nullable=False)
    strategy_name = Column(String(100), nullable=False)
    strategy_config = Column(Text, nullable=True)  # JSON configuration

    # Results summary
    final_value = Column(Numeric(15, 2), nullable=True)
    total_return_pct = Column(Numeric(8, 4), nullable=True)
    sharpe_ratio = Column(Numeric(8, 4), nullable=True)
    max_drawdown = Column(Numeric(8, 4), nullable=True)
    num_trades = Column(Integer, nullable=True)
    win_rate = Column(Numeric(5, 4), nullable=True)

    # Execution metadata
    status = Column(
        String(20), default="RUNNING", nullable=False
    )  # RUNNING, COMPLETED, FAILED
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    execution_time_seconds = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)

    # Configuration
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    # Constraints
    __table_args__ = (
        CheckConstraint("initial_cash > 0", name="ck_backtest_initial_cash_positive"),
        CheckConstraint("start_date <= end_date", name="ck_backtest_date_range"),
        Index("idx_backtest_strategy_start", "strategy_name", "start_date"),
        Index("idx_backtest_status", "status"),
    )

    def __repr__(self):
        return (
            f"<BacktestRun(id='{self.run_id}', name='{self.name}', "
            f"status='{self.status}')>"
        )


# Create indexes for better performance
def create_additional_indexes(engine):
    """Create additional database indexes for better query performance."""
    from sqlalchemy import text

    additional_indexes = [
        (
            "CREATE INDEX IF NOT EXISTS idx_option_contracts_symbol_expiration_strike "
            "ON option_contracts(symbol, expiration, strike)"
        ),
        (
            "CREATE INDEX IF NOT EXISTS idx_stock_market_data_contract_timestamp_desc "
            "ON stock_market_data(stock_contract_id, timestamp DESC)"
        ),
        (
            "CREATE INDEX IF NOT EXISTS idx_option_market_data_contract_timestamp_desc "
            "ON option_market_data(option_contract_id, timestamp DESC)"
        ),
        (
            "CREATE INDEX IF NOT EXISTS idx_positions_portfolio_status "
            "ON positions(portfolio_id, status)"
        ),
        (
            "CREATE INDEX IF NOT EXISTS idx_transactions_portfolio_timestamp_desc "
            "ON transactions(portfolio_id, timestamp DESC)"
        ),
        (
            "CREATE INDEX IF NOT EXISTS idx_portfolio_performance_timestamp_desc "
            "ON portfolio_performance(portfolio_id, timestamp DESC)"
        ),
    ]

    with engine.connect() as conn:
        for index_sql in additional_indexes:
            try:
                conn.execute(text(index_sql))
                conn.commit()
            except Exception as e:
                print(f"Warning: Could not create index: {e}")


# Utility functions for database operations
def get_table_names():
    """Get all table names defined in this schema."""
    return [table.name for table in Base.metadata.tables.values()]


def get_table_by_name(table_name: str):
    """Get table class by name."""
    for table in Base.metadata.tables.values():
        if table.name == table_name:
            return table
    return None
