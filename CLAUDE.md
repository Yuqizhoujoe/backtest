# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an options backtesting system for analyzing trading strategies (cash-secured puts, covered calls, iron condors). The system integrates with Interactive Brokers (IBKR) for real market data and provides comprehensive backtesting capabilities with options Greeks calculations.

## Technology Stack

- **Python 3.9+** with pandas, numpy, scipy
- **ib_insync** for IBKR API integration
- **SQLite/PostgreSQL** for data storage
- **pytest** for testing with >90% coverage target
- **matplotlib/plotly** for visualization
- **streamlit** for optional web dashboard
- **Docker** for containerization

## Development Commands

Based on plan.md specifications:

```bash
# Setup (when implemented)
python setup.py install
pip install -r requirements.txt

# Testing
pytest tests/                    # All tests
pytest tests/unit/              # Unit tests only
pytest tests/integration/       # Integration tests
pytest --cov=src               # Coverage report

# Code Quality
black src/                     # Code formatting
mypy src/                      # Type checking
pre-commit run --all-files     # Run all quality checks

# Data Operations
python scripts/setup_database.py     # Initialize database
python scripts/collect_data.py       # Fetch market data
python scripts/run_backtest.py       # Run backtesting
python scripts/analyze_results.py    # Analyze results
```

## Architecture Overview

The system follows a modular architecture with clear separation of concerns:

### Core Components (`src/core/`)

- **contracts.py**: OptionContract and StockContract classes with pricing models
- **position.py**: Position tracking and P&L calculations
- **portfolio.py**: Portfolio state management and risk tracking
- **exceptions.py**: Custom exception hierarchy

### Data Layer (`src/data/`)

- **ibkr_fetcher.py**: IBKR API integration with rate limiting and error handling
- **data_manager.py**: Database operations and schema management
- **data_validator.py**: Data quality checks and cleaning
- **simulators.py**: Test data generation for edge cases

### Strategy Framework (`src/strategies/`)

- **base_strategy.py**: Abstract strategy interface with signal generation
- **sell_put.py / covered_call.py**: Core strategy implementations
- **strategy_factory.py**: Configuration-driven strategy creation

### Backtesting Engine (`src/backtesting/`)

- **engine.py**: Event-driven backtesting orchestrator
- **simulator.py**: Market simulation with multi-timeframe support
- **execution.py**: Order execution modeling with slippage and liquidity constraints
- **risk_manager.py**: Portfolio risk controls and margin requirements

### Analytics (`src/analytics/`)

- **performance.py**: Returns, Sharpe/Sortino ratios, risk metrics
- **greeks.py**: Black-Scholes implementation with Greeks calculations
- **reports.py**: Automated report generation (PDF/HTML)

## Key Implementation Notes

- **Event-driven architecture** for backtesting to ensure realistic simulation
- **Options pricing** uses Black-Scholes with implied volatility calculations
- **Risk management** includes position size limits, drawdown controls, margin simulation
- **Data validation** ensures quality with missing data interpolation
- **Performance targets**: <30 seconds for 1 year daily backtests, <2GB memory usage

## IBKR Integration

- Uses ib_insync library for API connectivity
- Implements connection management with automatic reconnection
- Rate limiting and API quota management
- Handles both historical and real-time data feeds
- Corporate actions and dividend adjustments

## Testing Strategy

- **Unit tests**: Individual component testing with mocked dependencies
- **Integration tests**: IBKR connectivity and database operations
- **Edge case testing**: ITM/OTM options, assignment scenarios, market holidays
- **Performance tests**: Large dataset processing and memory usage
- **Data simulation**: Comprehensive test data for various market conditions

## Development Workflow

8-week development plan with phased approach:

1. **Weeks 1-2**: Core data models and database layer
2. **Week 3**: Data fetching and IBKR integration
3. **Week 4**: Options pricing and Greeks calculations
4. **Week 5**: Strategy framework implementation
5. **Week 6**: Backtesting engine development
6. **Week 7**: Analytics and performance metrics
7. **Week 8**: Visualization and user interface
