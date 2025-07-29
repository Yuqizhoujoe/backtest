"""
Logging configuration and utilities for the options backtesting system.

This module provides centralized logging configuration with support for
multiple log levels, file rotation, and structured logging formats.
"""

import logging
import logging.config
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


class BacktestLogger:
    """Centralized logger configuration for the backtesting system."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the logger configuration.

        Args:
            config_path: Path to logging configuration file
        """
        self.config_path = config_path
        self._configured = False

    def configure(
        self,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        console_output: bool = True,
        max_bytes: int = 10485760,
        backup_count: int = 5,
    ) -> None:
        """
        Configure logging for the application.

        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file (None for no file logging)
            console_output: Whether to output to console
            max_bytes: Maximum log file size before rotation
            backup_count: Number of backup files to keep
        """
        if self._configured:
            return

        # Create logs directory if it doesn't exist
        if log_file:
            log_dir = Path(log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)

        # Define logging configuration
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "detailed": {
                    "format": (
                        "%(asctime)s - %(name)s - %(levelname)s - "
                        "%(funcName)s:%(lineno)d - %(message)s"
                    ),
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
                "simple": {
                    "format": "%(asctime)s - %(levelname)s - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
                "json": {
                    "format": (
                        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
                        '"logger": "%(name)s", "function": "%(funcName)s", '
                        '"line": %(lineno)d, "message": "%(message)s"}'
                    ),
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {},
            "loggers": {
                "options_backtest": {
                    "level": log_level,
                    "handlers": [],
                    "propagate": False,
                }
            },
            "root": {"level": log_level, "handlers": []},
        }

        # Add console handler if requested
        if console_output:
            config["handlers"]["console"] = {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            }
            config["loggers"]["options_backtest"]["handlers"].append("console")
            config["root"]["handlers"].append("console")

        # Add file handler if requested
        if log_file:
            config["handlers"]["file"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "level": log_level,
                "formatter": "detailed",
                "filename": log_file,
                "maxBytes": max_bytes,
                "backupCount": backup_count,
                "encoding": "utf-8",
            }
            config["loggers"]["options_backtest"]["handlers"].append("file")
            config["root"]["handlers"].append("file")

        # Apply configuration
        logging.config.dictConfig(config)
        self._configured = True

        # Log initial message
        logger = logging.getLogger("options_backtest.logger")
        logger.info("Logging system initialized")
        logger.info(f"Log level: {log_level}")
        if log_file:
            logger.info(f"Log file: {log_file}")

    def configure_from_file(self, config_file: str) -> None:
        """
        Configure logging from a YAML configuration file.

        Args:
            config_file: Path to YAML configuration file
        """
        if self._configured:
            return

        try:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            # Ensure log directory exists
            for handler_name, handler_config in config.get("handlers", {}).items():
                if "filename" in handler_config:
                    log_dir = Path(handler_config["filename"]).parent
                    log_dir.mkdir(parents=True, exist_ok=True)

            logging.config.dictConfig(config)
            self._configured = True

            logger = logging.getLogger("options_backtest.logger")
            logger.info(f"Logging configured from file: {config_file}")

        except Exception as e:
            print(f"Error loading logging configuration from {config_file}: {e}")
            print("Falling back to default configuration")
            self.configure()

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger with the specified name.

        Args:
            name: Logger name

        Returns:
            Configured logger instance
        """
        if not self._configured:
            self.configure()

        return logging.getLogger(f"options_backtest.{name}")


# Global logger instance
_logger_instance = BacktestLogger()


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the specified module/component.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return _logger_instance.get_logger(name)


def configure_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
    config_file: Optional[str] = None,
) -> None:
    """
    Configure the global logging system.

    Args:
        log_level: Logging level
        log_file: Path to log file
        console_output: Whether to output to console
        config_file: Path to YAML configuration file (overrides other parameters)
    """
    if config_file and os.path.exists(config_file):
        _logger_instance.configure_from_file(config_file)
    else:
        _logger_instance.configure(log_level, log_file, console_output)


class PerformanceLogger:
    """Logger for performance monitoring and profiling."""

    def __init__(self, logger_name: str = "performance"):
        self.logger = get_logger(logger_name)

    def log_execution_time(
        self,
        func_name: str,
        execution_time: float,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log function execution time.

        Args:
            func_name: Name of the function
            execution_time: Execution time in seconds
            details: Additional details to log
        """
        message = f"Function '{func_name}' executed in {execution_time:.4f} seconds"
        if details:
            message += f" - Details: {details}"

        if execution_time > 1.0:  # Log as warning if slow
            self.logger.warning(message)
        else:
            self.logger.info(message)

    def log_memory_usage(self, operation: str, memory_mb: float) -> None:
        """
        Log memory usage for an operation.

        Args:
            operation: Description of the operation
            memory_mb: Memory usage in MB
        """
        message = f"Memory usage for '{operation}': {memory_mb:.2f} MB"

        if memory_mb > 1000:  # Log as warning if high memory usage
            self.logger.warning(message)
        else:
            self.logger.info(message)

    def log_database_query(
        self,
        query_type: str,
        execution_time: float,
        rows_affected: Optional[int] = None,
    ) -> None:
        """
        Log database query performance.

        Args:
            query_type: Type of query (SELECT, INSERT, UPDATE, etc.)
            execution_time: Query execution time in seconds
            rows_affected: Number of rows affected/returned
        """
        message = f"Database {query_type} executed in {execution_time:.4f} seconds"
        if rows_affected is not None:
            message += f" ({rows_affected} rows)"

        if execution_time > 0.1:  # Log as warning if slow query
            self.logger.warning(message)
        else:
            self.logger.debug(message)


class TradeLogger:
    """Specialized logger for trading operations."""

    def __init__(self, logger_name: str = "trading"):
        self.logger = get_logger(logger_name)

    def log_position_open(
        self,
        position_id: str,
        symbol: str,
        quantity: int,
        price: float,
        contract_type: str,
    ) -> None:
        """Log opening of a position."""
        self.logger.info(
            f"POSITION_OPEN: {position_id} | {symbol} | "
            f"{contract_type} | Qty: {quantity} | Price: ${price:.2f}"
        )

    def log_position_close(
        self,
        position_id: str,
        symbol: str,
        realized_pnl: float,
        holding_period_days: int,
    ) -> None:
        """Log closing of a position."""
        pnl_status = "PROFIT" if realized_pnl > 0 else "LOSS"
        self.logger.info(
            f"POSITION_CLOSE: {position_id} | {symbol} | "
            f"{pnl_status}: ${realized_pnl:.2f} | Held: {holding_period_days} days"
        )

    def log_trade_error(self, operation: str, symbol: str, error: str) -> None:
        """Log trading errors."""
        self.logger.error(f"TRADE_ERROR: {operation} | {symbol} | Error: {error}")

    def log_portfolio_update(
        self, portfolio_value: float, total_pnl: float, num_positions: int
    ) -> None:
        """Log portfolio status updates."""
        self.logger.info(
            f"PORTFOLIO_UPDATE: Value: ${portfolio_value:.2f} | "
            f"P&L: ${total_pnl:.2f} | Positions: {num_positions}"
        )


# Convenience function to set up logging for the entire application
def setup_application_logging(
    config_file: Optional[str] = None,
    log_level: str = "INFO",
    log_file: Optional[str] = "logs/backtest.log",
) -> None:
    """
    Set up logging for the entire application.

    Args:
        config_file: Path to logging configuration file
        log_level: Default log level if no config file
        log_file: Default log file if no config file
    """
    configure_logging(
        log_level=log_level,
        log_file=log_file,
        console_output=True,
        config_file=config_file,
    )

    # Log startup message
    logger = get_logger("main")
    logger.info("Options Backtesting System - Logging initialized")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")


if __name__ == "__main__":
    # Test the logging system
    setup_application_logging()

    logger = get_logger("test")
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    # Test performance logger
    perf_logger = PerformanceLogger()
    perf_logger.log_execution_time("test_function", 0.123)
    perf_logger.log_memory_usage("test_operation", 45.6)

    # Test trade logger
    trade_logger = TradeLogger()
    trade_logger.log_position_open("POS_001", "AAPL", 100, 150.50, "STOCK")
    trade_logger.log_position_close("POS_001", "AAPL", 250.75, 30)
