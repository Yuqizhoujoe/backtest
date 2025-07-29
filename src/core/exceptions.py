"""
Custom exceptions for the options backtesting system.

This module defines a comprehensive exception hierarchy for handling
various error conditions that may occur during backtesting operations.
"""

from typing import Optional, Any, Dict


class BacktestError(Exception):
    """Base exception for all backtesting system errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(BacktestError):
    """Raised when data validation fails."""

    pass


class ContractError(BacktestError):
    """Base exception for contract-related errors."""

    pass


class InvalidContractError(ContractError):
    """Raised when a contract has invalid parameters."""

    pass


class ExpiredContractError(ContractError):
    """Raised when attempting to use an expired contract."""

    pass


class ContractNotFoundError(ContractError):
    """Raised when a requested contract cannot be found."""

    pass


class PositionError(BacktestError):
    """Base exception for position-related errors."""

    pass


class InvalidPositionError(PositionError):
    """Raised when a position has invalid parameters."""

    pass


class InsufficientPositionError(PositionError):
    """Raised when trying to close more than the available position."""

    pass


class PortfolioError(BacktestError):
    """Base exception for portfolio-related errors."""

    pass


class InsufficientFundsError(PortfolioError):
    """Raised when attempting to trade with insufficient funds."""

    pass


class InsufficientMarginError(PortfolioError):
    """Raised when margin requirements are not met."""

    pass


class ExcessiveRiskError(PortfolioError):
    """Raised when a trade would exceed risk limits."""

    pass


class DataError(BacktestError):
    """Base exception for data-related errors."""

    pass


class DataNotFoundError(DataError):
    """Raised when requested data is not available."""

    pass


class DataQualityError(DataError):
    """Raised when data quality checks fail."""

    pass


class DataCorruptionError(DataError):
    """Raised when data appears to be corrupted."""

    pass


class DatabaseError(BacktestError):
    """Base exception for database-related errors."""

    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""

    pass


class DatabaseIntegrityError(DatabaseError):
    """Raised when database integrity constraints are violated."""

    pass


class DatabaseOperationError(DatabaseError):
    """Raised when a database operation fails."""

    pass


class StrategyError(BacktestError):
    """Base exception for strategy-related errors."""

    pass


class StrategyConfigurationError(StrategyError):
    """Raised when strategy configuration is invalid."""

    pass


class StrategyExecutionError(StrategyError):
    """Raised when strategy execution fails."""

    pass


class BacktestingError(BacktestError):
    """Base exception for backtesting engine errors."""

    pass


class BacktestConfigurationError(BacktestingError):
    """Raised when backtest configuration is invalid."""

    pass


class BacktestExecutionError(BacktestingError):
    """Raised when backtest execution fails."""

    pass


class PricingError(BacktestError):
    """Base exception for options pricing errors."""

    pass


class InvalidPricingParametersError(PricingError):
    """Raised when pricing parameters are invalid."""

    pass


class PricingCalculationError(PricingError):
    """Raised when pricing calculation fails."""

    pass


class MarketDataError(DataError):
    """Raised when market data issues occur."""

    pass


class StaleDataError(MarketDataError):
    """Raised when market data is too old."""

    pass


class MissingDataError(MarketDataError):
    """Raised when required market data is missing."""

    pass


class ConfigurationError(BacktestError):
    """Raised when system configuration is invalid."""

    pass


class ResourceError(BacktestError):
    """Raised when system resources are insufficient."""

    pass


class MemoryError(ResourceError):
    """Raised when memory limits are exceeded."""

    pass


class TimeoutError(BacktestError):
    """Raised when operations timeout."""

    pass
