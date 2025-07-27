# Options Backtesting System

A comprehensive Python-based system for backtesting options trading strategies with real market data from Interactive Brokers.

## Overview

This project implements a professional-grade options backtesting framework designed to analyze trading strategies including:

- Cash-secured puts
- Covered calls
- Iron condors (future implementation)

## Key Features

- **Real Market Data**: Integration with Interactive Brokers API for historical and real-time data
- **Options Pricing**: Black-Scholes implementation with Greeks calculations (delta, gamma, theta, vega, rho)
- **Event-Driven Backtesting**: Realistic simulation with proper execution modeling
- **Risk Management**: Portfolio risk controls, position sizing, and margin requirements
- **Performance Analytics**: Comprehensive metrics including Sharpe ratios, drawdown analysis, and trade statistics
- **Visualization**: Charts and reports for strategy analysis

## Project Status

**Current Phase**: Planning and Documentation

- 📋 Comprehensive development plan completed
- 🏗️ Architecture design finalized
- ⏳ Implementation pending

## Development Plan

This project follows an 8-week phased development approach:

| Phase   | Duration  | Focus                         | Status     |
| ------- | --------- | ----------------------------- | ---------- |
| Phase 1 | Weeks 1-2 | Foundation & Core Models      | 📋 Planned |
| Phase 2 | Week 3    | Data Layer & IBKR Integration | 📋 Planned |
| Phase 3 | Week 4    | Options Pricing & Greeks      | 📋 Planned |
| Phase 4 | Week 5    | Strategy Framework            | 📋 Planned |
| Phase 5 | Week 6    | Backtesting Engine            | 📋 Planned |
| Phase 6 | Week 7    | Analytics & Performance       | 📋 Planned |
| Phase 7 | Week 8    | Visualization & UI            | 📋 Planned |

## Documentation

- 📋 **[plan.md](plan.md)**: Complete development plan with architecture overview
- 🔧 **[CLAUDE.md](CLAUDE.md)**: Developer guidance for AI assistants
- 📁 **phases/**: Detailed technical specifications for each development phase

## Technology Stack

- **Python 3.9+** with pandas, numpy, scipy
- **ib_insync** for Interactive Brokers API
- **SQLite/PostgreSQL** for data storage
- **pytest** for comprehensive testing
- **matplotlib/plotly** for visualization
- **streamlit** for web dashboard (optional)
- **Docker** for containerization

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Interactive Brokers account (for real data)
- Git for version control

### Development Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd backtest

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create project and install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Run tests (when implemented)
uv run pytest tests/
```

### Code Formatting and Linting

This project uses **ruff** for both code formatting and linting to maintain consistent code quality and catch potential issues.

#### Manual Commands

```bash
# Format code with ruff
uv run ruff format src/

# Check linting with ruff
uv run ruff check src/

# Fix auto-fixable linting issues
uv run ruff check --fix src/

# Run both formatting and linting
uv run ruff format src/ && uv run ruff check src/
```

#### IDE Setup for Auto-Formatting

Configure your IDE to automatically format and lint code:

**VS Code:**

1. Install the Ruff extension (`charliermarsh.ruff`)
2. Add to your `.vscode/settings.json`:

```json
{
  "[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll.ruff": true,
      "source.organizeImports.ruff": true
    },
    "editor.defaultFormatter": "charliermarsh.ruff"
  },
  "ruff.enable": true,
  "ruff.lint.enable": true,
  "ruff.format.enable": true
}
```

**PyCharm:**

1. Install the Ruff plugin from the marketplace
2. Go to File → Settings → Tools → Ruff
3. Enable "Use ruff" and configure paths as needed
4. Enable "Format on save" in File → Settings → Tools → Actions on Save

**Pre-commit Hook (Recommended):**

```bash
# Install pre-commit hooks to run formatting and linting automatically
uv add pre-commit
echo 'repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.12.5
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format' > .pre-commit-config.yaml
uv run pre-commit install
```

#### Ruff Configuration

Ruff is configured via `pyproject.toml`. Key settings include:

- **Line length**: 88 characters (matching Black)
- **Target Python version**: 3.9+
- **Enabled rules**: Comprehensive linting including imports, complexity, and style
- **Auto-fixes**: Import sorting, unused imports, and code style issues

## Performance Targets

- **Speed**: Backtest 1 year of daily data in <30 seconds
- **Memory**: <2GB usage for typical backtests
- **Coverage**: >90% test coverage
- **Capacity**: Support 10+ concurrent strategies
