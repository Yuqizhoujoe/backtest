# Phase 1: Foundation & Core Models (Weeks 1-2)

## Overview

Establish the foundational architecture and core data models for the options backtesting system. This phase focuses on creating robust, well-tested building blocks that will support the entire application.

## Objectives

- Set up professional project structure and development environment
- Implement core data models for options and stock contracts
- Create position and portfolio management systems
- Establish database layer with proper schema design
- Set up comprehensive testing framework

## Technical Deliverables

### 1. Project Structure Setup

```
src/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── contracts.py        # Option & Stock contract classes
│   ├── position.py         # Position management
│   ├── portfolio.py        # Portfolio state and tracking
│   └── exceptions.py       # Custom exceptions
├── data/
│   ├── __init__.py
│   ├── data_manager.py     # Database operations
│   └── schemas.py          # Database schema definitions
├── utils/
│   ├── __init__.py
│   ├── logger.py          # Logging configuration
│   └── config.py          # Configuration management
└── config/
    ├── __init__.py
    ├── settings.py         # Global settings
    └── database.py         # Database configuration

tests/
├── __init__.py
├── conftest.py            # Pytest configuration
├── unit/
│   ├── test_contracts.py
│   ├── test_position.py
│   ├── test_portfolio.py
│   └── test_data_manager.py
└── fixtures/
    └── sample_data.py     # Test data fixtures

config/
├── logging.yaml           # Logging configuration
└── settings.yaml          # Application settings
```

### 2. Core Data Models

#### OptionContract Class (`src/core/contracts.py`)

```python
@dataclass
class OptionContract:
    """Represents an options contract with all relevant attributes"""
    
    # Contract Identification
    symbol: str                    # Underlying symbol (e.g., "AAPL")
    strike: Decimal               # Strike price
    expiration: datetime.date     # Expiration date
    option_type: OptionType       # PUT or CALL
    exchange: str                 # Exchange (e.g., "SMART")
    
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
    
    # Contract Specifications
    multiplier: int = 100         # Standard option multiplier
    currency: str = "USD"
    
    # Metadata
    last_update: Optional[datetime.datetime] = None
    
    def __post_init__(self):
        """Validate contract data after initialization"""
        self._validate_contract()
    
    def _validate_contract(self) -> None:
        """Validate contract parameters"""
        if self.strike <= 0:
            raise ValueError("Strike price must be positive")
        if self.expiration <= datetime.date.today():
            raise ValueError("Expiration must be in the future")
        # Additional validations...
    
    @property
    def contract_id(self) -> str:
        """Generate unique contract identifier"""
        return f"{self.symbol}_{self.expiration}_{self.option_type.value}_{self.strike}"
    
    @property
    def time_to_expiry(self) -> Decimal:
        """Calculate time to expiry in years"""
        days = (self.expiration - datetime.date.today()).days
        return Decimal(days) / Decimal(365)
    
    @property
    def is_itm(self) -> Optional[bool]:
        """Check if option is in-the-money (requires underlying price)"""
        # Implementation depends on current underlying price
        pass
    
    def theoretical_price(self, underlying_price: Decimal, risk_free_rate: Decimal, 
                         volatility: Decimal) -> Decimal:
        """Calculate theoretical option price using Black-Scholes"""
        # Implementation will be added in Phase 4
        pass
```

#### StockContract Class (`src/core/contracts.py`)

```python
@dataclass
class StockContract:
    """Represents a stock contract"""
    
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
    
    @property
    def mid_price(self) -> Optional[Decimal]:
        """Calculate mid price from bid/ask"""
        if self.bid and self.ask:
            return (self.bid + self.ask) / 2
        return None
```

#### Position Class (`src/core/position.py`)

```python
@dataclass
class Position:
    """Represents a trading position (long or short)"""
    
    contract: Union[OptionContract, StockContract]
    quantity: int                 # Positive for long, negative for short
    avg_price: Decimal           # Average entry price
    timestamp: datetime.datetime  # Position entry time
    
    # P&L Tracking
    realized_pnl: Decimal = Decimal('0')
    unrealized_pnl: Decimal = Decimal('0')
    
    # Risk Metrics
    initial_margin: Optional[Decimal] = None
    maintenance_margin: Optional[Decimal] = None
    
    def __post_init__(self):
        if self.quantity == 0:
            raise ValueError("Position quantity cannot be zero")
    
    @property
    def position_value(self) -> Decimal:
        """Calculate current position value"""
        current_price = self._get_current_price()
        if current_price:
            return abs(self.quantity) * current_price * self._get_multiplier()
        return Decimal('0')
    
    @property
    def is_long(self) -> bool:
        """Check if position is long"""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short"""
        return self.quantity < 0
    
    def update_unrealized_pnl(self, current_price: Decimal) -> None:
        """Update unrealized P&L based on current market price"""
        multiplier = self._get_multiplier()
        self.unrealized_pnl = (current_price - self.avg_price) * self.quantity * multiplier
    
    def close_position(self, exit_price: Decimal, exit_time: datetime.datetime) -> Decimal:
        """Close position and calculate realized P&L"""
        multiplier = self._get_multiplier()
        pnl = (exit_price - self.avg_price) * self.quantity * multiplier
        self.realized_pnl += pnl
        self.quantity = 0
        return pnl
    
    def _get_multiplier(self) -> int:
        """Get contract multiplier"""
        if isinstance(self.contract, OptionContract):
            return self.contract.multiplier
        return 1
    
    def _get_current_price(self) -> Optional[Decimal]:
        """Get current market price"""
        if hasattr(self.contract, 'last') and self.contract.last:
            return self.contract.last
        return None
```

#### Portfolio Class (`src/core/portfolio.py`)

```python
class Portfolio:
    """Manages a collection of positions and tracks portfolio-level metrics"""
    
    def __init__(self, initial_cash: Decimal, name: str = "Default Portfolio"):
        self.name = name
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        
        # Portfolio Metrics
        self.total_realized_pnl = Decimal('0')
        self.total_unrealized_pnl = Decimal('0')
        self.max_drawdown = Decimal('0')
        self.peak_value = initial_cash
        
        # Risk Metrics
        self.margin_used = Decimal('0')
        self.margin_available = Decimal('0')
        
        # Transaction History
        self.transactions: List[Transaction] = []
        
    @property
    def total_value(self) -> Decimal:
        """Calculate total portfolio value"""
        position_value = sum(pos.position_value for pos in self.positions.values())
        return self.cash + position_value
    
    @property
    def total_pnl(self) -> Decimal:
        """Calculate total P&L"""
        return self.total_realized_pnl + self.total_unrealized_pnl
    
    @property
    def return_pct(self) -> Decimal:
        """Calculate portfolio return percentage"""
        if self.initial_cash == 0:
            return Decimal('0')
        return (self.total_value - self.initial_cash) / self.initial_cash * 100
    
    def add_position(self, position: Position) -> None:
        """Add a new position to the portfolio"""
        contract_id = position.contract.contract_id
        
        if contract_id in self.positions:
            # Combine with existing position
            self._combine_positions(contract_id, position)
        else:
            self.positions[contract_id] = position
        
        # Update cash and margin
        self._update_portfolio_metrics()
    
    def close_position(self, contract_id: str, exit_price: Decimal, 
                      exit_time: datetime.datetime) -> Optional[Decimal]:
        """Close a position and update portfolio"""
        if contract_id not in self.positions:
            return None
        
        position = self.positions[contract_id]
        realized_pnl = position.close_position(exit_price, exit_time)
        
        # Update portfolio cash
        self.cash += realized_pnl
        self.total_realized_pnl += realized_pnl
        
        # Remove closed position
        del self.positions[contract_id]
        
        self._update_portfolio_metrics()
        return realized_pnl
    
    def update_portfolio_value(self) -> None:
        """Update unrealized P&L for all positions"""
        self.total_unrealized_pnl = Decimal('0')
        
        for position in self.positions.values():
            current_price = position._get_current_price()
            if current_price:
                position.update_unrealized_pnl(current_price)
                self.total_unrealized_pnl += position.unrealized_pnl
        
        # Update drawdown
        current_value = self.total_value
        if current_value > self.peak_value:
            self.peak_value = current_value
        else:
            drawdown = (self.peak_value - current_value) / self.peak_value
            self.max_drawdown = max(self.max_drawdown, drawdown)
    
    def get_positions_by_type(self, contract_type: type) -> List[Position]:
        """Get all positions of a specific contract type"""
        return [pos for pos in self.positions.values() 
                if isinstance(pos.contract, contract_type)]
    
    def get_portfolio_summary(self) -> Dict:
        """Generate portfolio summary"""
        return {
            'name': self.name,
            'total_value': self.total_value,
            'cash': self.cash,
            'total_pnl': self.total_pnl,
            'return_pct': self.return_pct,
            'max_drawdown': self.max_drawdown,
            'num_positions': len(self.positions),
            'margin_used': self.margin_used,
            'margin_available': self.margin_available
        }
```

### 3. Database Schema (`src/data/schemas.py`)

```sql
-- Options contracts table
CREATE TABLE option_contracts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    contract_id VARCHAR(100) UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    strike DECIMAL(10,2) NOT NULL,
    expiration DATE NOT NULL,
    option_type VARCHAR(4) NOT NULL, -- CALL or PUT
    exchange VARCHAR(20) DEFAULT 'SMART',
    multiplier INTEGER DEFAULT 100,
    currency VARCHAR(3) DEFAULT 'USD',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Stock contracts table
CREATE TABLE stock_contracts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(20) UNIQUE NOT NULL,
    exchange VARCHAR(20) DEFAULT 'SMART',
    currency VARCHAR(3) DEFAULT 'USD',
    market_cap DECIMAL(15,2),
    pe_ratio DECIMAL(8,2),
    dividend_yield DECIMAL(5,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Market data table (time series)
CREATE TABLE market_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    contract_id VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    bid DECIMAL(10,4),
    ask DECIMAL(10,4),
    last DECIMAL(10,4),
    volume INTEGER,
    open_interest INTEGER,
    -- Greeks (for options)
    delta DECIMAL(8,6),
    gamma DECIMAL(8,6),
    theta DECIMAL(8,6),
    vega DECIMAL(8,6),
    rho DECIMAL(8,6),
    implied_volatility DECIMAL(8,6),
    FOREIGN KEY (contract_id) REFERENCES option_contracts(contract_id)
);

-- Portfolios table
CREATE TABLE portfolios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) NOT NULL,
    initial_cash DECIMAL(15,2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Positions table
CREATE TABLE positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id INTEGER NOT NULL,
    contract_id VARCHAR(100) NOT NULL,
    quantity INTEGER NOT NULL,
    avg_price DECIMAL(10,4) NOT NULL,
    entry_timestamp TIMESTAMP NOT NULL,
    exit_timestamp TIMESTAMP,
    realized_pnl DECIMAL(15,2) DEFAULT 0,
    status VARCHAR(20) DEFAULT 'OPEN', -- OPEN, CLOSED
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
);

-- Transactions table
CREATE TABLE transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id INTEGER NOT NULL,
    contract_id VARCHAR(100) NOT NULL,
    transaction_type VARCHAR(10) NOT NULL, -- BUY, SELL
    quantity INTEGER NOT NULL,
    price DECIMAL(10,4) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    commission DECIMAL(8,2) DEFAULT 0,
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
);

-- Indexes for performance
CREATE INDEX idx_market_data_contract_timestamp ON market_data(contract_id, timestamp);
CREATE INDEX idx_positions_portfolio ON positions(portfolio_id);
CREATE INDEX idx_transactions_portfolio ON transactions(portfolio_id);
```

### 4. Database Manager (`src/data/data_manager.py`)

```python
class DatabaseManager:
    """Handles all database operations for the backtesting system"""
    
    def __init__(self, db_path: str = "data/backtest.db"):
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}")
        self.SessionLocal = sessionmaker(bind=self.engine)
        
    def create_tables(self) -> None:
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    # Contract Operations
    def save_option_contract(self, contract: OptionContract) -> None:
        """Save option contract to database"""
        with self.get_session() as session:
            # Implementation...
            pass
    
    def save_stock_contract(self, contract: StockContract) -> None:
        """Save stock contract to database"""
        with self.get_session() as session:
            # Implementation...
            pass
    
    def get_option_contract(self, contract_id: str) -> Optional[OptionContract]:
        """Retrieve option contract by ID"""
        with self.get_session() as session:
            # Implementation...
            pass
    
    # Market Data Operations
    def save_market_data(self, contract_id: str, data: Dict) -> None:
        """Save market data snapshot"""
        with self.get_session() as session:
            # Implementation...
            pass
    
    def get_market_data_range(self, contract_id: str, 
                            start_date: datetime.date, 
                            end_date: datetime.date) -> List[Dict]:
        """Get market data for date range"""
        with self.get_session() as session:
            # Implementation...
            pass
    
    # Portfolio Operations
    def save_portfolio(self, portfolio: Portfolio) -> int:
        """Save portfolio and return ID"""
        with self.get_session() as session:
            # Implementation...
            pass
    
    def save_position(self, portfolio_id: int, position: Position) -> None:
        """Save position to database"""
        with self.get_session() as session:
            # Implementation...
            pass
    
    def save_transaction(self, portfolio_id: int, transaction: Dict) -> None:
        """Save transaction record"""
        with self.get_session() as session:
            # Implementation...
            pass
```

## Testing Requirements

### Unit Tests Coverage Target: >90%

1. **Contract Tests** (`tests/unit/test_contracts.py`)
   - OptionContract validation and properties
   - StockContract functionality
   - Contract ID generation
   - Time to expiry calculations

2. **Position Tests** (`tests/unit/test_position.py`)
   - Position creation and validation
   - P&L calculations
   - Position updates and closures

3. **Portfolio Tests** (`tests/unit/test_portfolio.py`)
   - Portfolio initialization
   - Position management
   - Portfolio metrics calculations
   - Risk management functions

4. **Database Tests** (`tests/unit/test_data_manager.py`)
   - CRUD operations for all entities
   - Data integrity and constraints
   - Query performance

### Integration Tests

1. **Database Integration**
   - End-to-end database operations
   - Data persistence and retrieval
   - Transaction rollback scenarios

## Configuration Management

### Settings Structure (`config/settings.yaml`)

```yaml
# Database Configuration
database:
  type: "sqlite"  # sqlite, postgresql
  path: "data/backtest.db"
  pool_size: 5
  echo: false

# Logging Configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/backtest.log"
  max_bytes: 10485760  # 10MB
  backup_count: 5

# Portfolio Defaults
portfolio:
  initial_cash: 100000
  commission: 1.0  # Per contract
  margin_rate: 0.5

# Risk Management
risk:
  max_position_size: 0.1  # 10% of portfolio
  max_portfolio_leverage: 2.0
  stop_loss_pct: 0.2  # 20%
```

## Performance Requirements

- **Database Operations**: <100ms for single record operations
- **Portfolio Calculations**: <50ms for 100 positions
- **Memory Usage**: <100MB for core models
- **Test Execution**: Complete test suite in <30 seconds

## Quality Gates

1. **Code Coverage**: Minimum 90% line coverage
2. **Type Checking**: All code must pass mypy strict mode
3. **Linting**: All code must pass black and flake8
4. **Documentation**: All public methods must have docstrings
5. **Testing**: All edge cases must have dedicated tests

## Dependencies

Create `pyproject.toml` for uv dependency management:

```toml
[project]
name = "options-backtest"
version = "0.1.0"
description = "Options backtesting system with IBKR integration"
authors = [{name = "Your Name", email = "your.email@example.com"}]
requires-python = ">=3.9"
dependencies = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "sqlalchemy>=2.0.0",
    "pydantic>=2.0.0",
    "scipy>=1.10.0",
    "ib-insync>=0.9.86",
    "psycopg2-binary>=2.9.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "mypy>=1.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "pre-commit>=3.0.0",
]
viz = [
    "matplotlib>=3.7.0",
    "plotly>=5.17.0",
    "streamlit>=1.28.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

[tool.mypy]
python_version = "3.9"
strict = true
ignore_missing_imports = true

[tool.black]
line-length = 88
target-version = ['py39']

[tool.ruff]
line-length = 88
target-version = "py39"
```

## Success Criteria

- [ ] All core data models implemented with comprehensive validation
- [ ] Database schema created with proper relationships and indexes
- [ ] Portfolio and position management working with accurate P&L calculations
- [ ] 90%+ test coverage achieved
- [ ] All performance benchmarks met
- [ ] Clean code architecture following SOLID principles
- [ ] Comprehensive documentation and type hints