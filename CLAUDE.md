# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an options backtesting system for analyzing trading strategies (cash-secured puts, covered calls, iron condors). The system will integrate with Interactive Brokers (IBKR) for real market data and provide comprehensive backtesting capabilities with options Greeks calculations.

**Current State**: Planning phase - only documentation files exist (README.md, plan.md, CLAUDE.md). No implementation has started yet.

## Project Documentation

- **plan.md**: Comprehensive 8-week development plan with detailed architecture, timeline, and technical specifications
- **README.md**: Basic project description

## Development Commands

Once implementation begins, these commands will be used:

```bash
# Setup
uv sync                        # Install all dependencies
uv add <package>              # Add new dependency
uv remove <package>           # Remove dependency

# Testing
uv run pytest tests/                    # All tests
uv run pytest tests/unit/              # Unit tests only
uv run pytest tests/integration/       # Integration tests
uv run pytest --cov=src               # Coverage report

# Code Quality
uv run ruff format src/               # Code formatting
uv run ruff check src/                # Linting
uv run ruff check --fix src/          # Auto-fix linting issues
uv run mypy src/                      # Type checking
uv run pre-commit run --all-files     # Run all quality checks

# Data Operations
uv run python scripts/setup_database.py     # Initialize database
uv run python scripts/collect_data.py       # Fetch market data
uv run python scripts/run_backtest.py       # Run backtesting
uv run python scripts/analyze_results.py    # Analyze results
```

## Planned Architecture

The system will follow a modular, event-driven architecture with clear separation of concerns:

### Core Layer
- **contracts.py**: OptionContract and StockContract classes with pricing models
- **position.py**: Position tracking and P&L calculations
- **portfolio.py**: Portfolio state management and risk tracking
- **exceptions.py**: Custom exception hierarchy

### Data Layer
- **ibkr_fetcher.py**: IBKR API integration with rate limiting and error handling
- **data_manager.py**: Database operations and schema management
- **data_validator.py**: Data quality checks and cleaning
- **simulators.py**: Test data generation for edge cases

### Strategy Framework
- **base_strategy.py**: Abstract strategy interface with signal generation
- **sell_put.py / covered_call.py**: Core strategy implementations
- **strategy_factory.py**: Configuration-driven strategy creation

### Backtesting Engine
- **engine.py**: Event-driven backtesting orchestrator
- **simulator.py**: Market simulation with multi-timeframe support
- **execution.py**: Order execution modeling with slippage and liquidity constraints
- **risk_manager.py**: Portfolio risk controls and margin requirements

### Analytics
- **performance.py**: Returns, Sharpe/Sortino ratios, risk metrics
- **greeks.py**: Black-Scholes implementation with Greeks calculations
- **reports.py**: Automated report generation (PDF/HTML)

## Technology Stack

- **Python 3.9+** with pandas, numpy, scipy
- **ib_insync** for IBKR API integration
- **SQLite/PostgreSQL** for data storage
- **pytest** for testing with >90% coverage target
- **matplotlib/plotly** for visualization
- **streamlit** for optional web dashboard
- **Docker** for containerization

## Implementation Principles

- **Event-driven architecture** for realistic backtesting simulation
- **Options pricing** using Black-Scholes with implied volatility calculations
- **Risk management** with position size limits, drawdown controls, margin simulation
- **Data validation** ensuring quality with missing data interpolation
- **Performance targets**: <30 seconds for 1 year daily backtests, <2GB memory usage

## Development Plan

8-week phased development approach:

1. **Weeks 1-2**: Foundation and core data models
2. **Week 3**: Data layer and IBKR integration
3. **Week 4**: Options pricing and Greeks calculations
4. **Week 5**: Strategy framework implementation
5. **Week 6**: Backtesting engine development
6. **Week 7**: Analytics and performance metrics
7. **Week 8**: Visualization and user interface

Refer to plan.md for detailed specifications and implementation timeline.
