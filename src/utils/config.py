"""
Configuration management for the options backtesting system.

This module provides centralized configuration management with support for
YAML files, environment variables, and runtime configuration updates.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass, field
from decimal import Decimal

from .logger import get_logger


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    type: str = "sqlite"
    path: str = "data/backtest.db"
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    pool_size: int = 5
    echo: bool = False
    
    @property
    def connection_string(self) -> str:
        """Generate database connection string."""
        if self.type == "sqlite":
            return f"sqlite:///{self.path}"
        elif self.type == "postgresql":
            if not all([self.host, self.username, self.password]):
                raise ValueError("PostgreSQL requires host, username, and password")
            port_str = f":{self.port}" if self.port else ""
            return f"postgresql://{self.username}:{self.password}@{self.host}{port_str}/{self.path}"
        else:
            raise ValueError(f"Unsupported database type: {self.type}")


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = "logs/backtest.log"
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5
    console_output: bool = True


@dataclass
class PortfolioConfig:
    """Portfolio configuration settings."""
    initial_cash: Decimal = Decimal('100000')
    commission_per_contract: Decimal = Decimal('1.0')
    commission_per_share: Decimal = Decimal('0.005')
    margin_rate: Decimal = Decimal('0.5')
    max_position_size_pct: Decimal = Decimal('0.1')  # 10%
    max_portfolio_leverage: Decimal = Decimal('2.0')


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_position_size_pct: Decimal = Decimal('0.1')  # 10% of portfolio
    max_portfolio_leverage: Decimal = Decimal('2.0')
    stop_loss_pct: Decimal = Decimal('0.2')  # 20%
    max_drawdown_pct: Decimal = Decimal('0.15')  # 15%
    var_confidence_level: Decimal = Decimal('0.95')  # 95%
    concentration_limit_pct: Decimal = Decimal('0.25')  # 25% per symbol


@dataclass
class IBKRConfig:
    """Interactive Brokers configuration."""
    host: str = "127.0.0.1"
    port: int = 7497  # TWS paper trading port
    client_id: int = 1
    timeout: int = 30
    enable_logging: bool = True
    readonly: bool = True  # Start in read-only mode


@dataclass
class BacktestConfig:
    """Backtesting engine configuration."""
    start_date: Optional[str] = None  # Format: YYYY-MM-DD
    end_date: Optional[str] = None
    frequency: str = "daily"  # daily, hourly, minute
    slippage_bps: int = 5  # Basis points
    commission_model: str = "fixed"  # fixed, percentage
    fill_model: str = "midpoint"  # midpoint, aggressive, conservative


@dataclass
class PerformanceConfig:
    """Performance monitoring configuration."""
    benchmark_symbol: str = "SPY"
    risk_free_rate: Decimal = Decimal('0.02')  # 2% annual
    business_days_per_year: int = 252
    enable_profiling: bool = False
    memory_monitoring: bool = True


class ConfigManager:
    """
    Centralized configuration manager for the backtesting system.
    
    This class handles loading, merging, and accessing configuration settings
    from multiple sources including files, environment variables, and defaults.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Path to the main configuration file
        """
        self.logger = get_logger('config')
        self.config_file = config_file
        self._config_data: Dict[str, Any] = {}
        self._loaded = False
        
        # Configuration sections
        self.database = DatabaseConfig()
        self.logging = LoggingConfig()
        self.portfolio = PortfolioConfig()
        self.risk = RiskConfig()
        self.ibkr = IBKRConfig()
        self.backtest = BacktestConfig()
        self.performance = PerformanceConfig()
        
        # Load configuration
        self.load_config()
    
    def load_config(self, config_file: Optional[str] = None) -> None:
        """
        Load configuration from file and environment variables.
        
        Args:
            config_file: Override config file path
        """
        if config_file:
            self.config_file = config_file
        
        # Load from file if provided
        if self.config_file and os.path.exists(self.config_file):
            self._load_from_file(self.config_file)
        
        # Override with environment variables
        self._load_from_environment()
        
        # Apply loaded configuration to dataclass instances
        self._apply_config()
        
        self._loaded = True
        self.logger.info(f"Configuration loaded from: {self.config_file or 'defaults + environment'}")
    
    def _load_from_file(self, config_file: str) -> None:
        """Load configuration from YAML file."""
        try:
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f) or {}
            
            self._config_data.update(file_config)
            self.logger.info(f"Configuration loaded from file: {config_file}")
            
        except Exception as e:
            self.logger.error(f"Error loading configuration file {config_file}: {e}")
            raise
    
    def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        env_mappings = {
            'BACKTEST_DB_TYPE': ('database', 'type'),
            'BACKTEST_DB_PATH': ('database', 'path'),
            'BACKTEST_DB_HOST': ('database', 'host'),
            'BACKTEST_DB_PORT': ('database', 'port'),
            'BACKTEST_DB_USER': ('database', 'username'),
            'BACKTEST_DB_PASSWORD': ('database', 'password'),
            
            'BACKTEST_LOG_LEVEL': ('logging', 'level'),
            'BACKTEST_LOG_FILE': ('logging', 'file'),
            
            'BACKTEST_INITIAL_CASH': ('portfolio', 'initial_cash'),
            'BACKTEST_COMMISSION': ('portfolio', 'commission_per_contract'),
            
            'BACKTEST_IBKR_HOST': ('ibkr', 'host'),
            'BACKTEST_IBKR_PORT': ('ibkr', 'port'),
            'BACKTEST_IBKR_CLIENT_ID': ('ibkr', 'client_id'),
            
            'BACKTEST_START_DATE': ('backtest', 'start_date'),
            'BACKTEST_END_DATE': ('backtest', 'end_date'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if section not in self._config_data:
                    self._config_data[section] = {}
                
                # Convert to appropriate type
                converted_value = self._convert_env_value(value, section, key)
                self._config_data[section][key] = converted_value
                
                self.logger.debug(f"Environment variable {env_var} = {converted_value}")
    
    def _convert_env_value(self, value: str, section: str, key: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Boolean values
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Numeric values
        if section == 'database' and key == 'port':
            return int(value)
        if section == 'ibkr' and key in ('port', 'client_id'):
            return int(value)
        if section == 'portfolio' and key in ('initial_cash', 'commission_per_contract', 'commission_per_share'):
            return Decimal(value)
        if section == 'risk' and 'pct' in key:
            return Decimal(value)
        
        return value
    
    def _apply_config(self) -> None:
        """Apply loaded configuration to dataclass instances."""
        # Database configuration
        if 'database' in self._config_data:
            self._update_dataclass(self.database, self._config_data['database'])
        
        # Logging configuration
        if 'logging' in self._config_data:
            self._update_dataclass(self.logging, self._config_data['logging'])
        
        # Portfolio configuration
        if 'portfolio' in self._config_data:
            self._update_dataclass(self.portfolio, self._config_data['portfolio'])
        
        # Risk configuration
        if 'risk' in self._config_data:
            self._update_dataclass(self.risk, self._config_data['risk'])
        
        # IBKR configuration
        if 'ibkr' in self._config_data:
            self._update_dataclass(self.ibkr, self._config_data['ibkr'])
        
        # Backtest configuration
        if 'backtest' in self._config_data:
            self._update_dataclass(self.backtest, self._config_data['backtest'])
        
        # Performance configuration
        if 'performance' in self._config_data:
            self._update_dataclass(self.performance, self._config_data['performance'])
    
    def _update_dataclass(self, instance: Any, config_dict: Dict[str, Any]) -> None:
        """Update dataclass instance with configuration values."""
        for key, value in config_dict.items():
            if hasattr(instance, key):
                # Convert Decimal fields
                field_type = type(getattr(instance, key))
                if field_type == Decimal and not isinstance(value, Decimal):
                    value = Decimal(str(value))
                
                setattr(instance, key, value)
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            default: Default value if not found
            
        Returns:
            Configuration value or default
        """
        return self._config_data.get(section, {}).get(key, default)
    
    def set(self, section: str, key: str, value: Any) -> None:
        """
        Set a configuration value at runtime.
        
        Args:
            section: Configuration section
            key: Configuration key
            value: Value to set
        """
        if section not in self._config_data:
            self._config_data[section] = {}
        
        self._config_data[section][key] = value
        
        # Update corresponding dataclass if it exists
        if hasattr(self, section):
            dataclass_instance = getattr(self, section)
            if hasattr(dataclass_instance, key):
                setattr(dataclass_instance, key, value)
        
        self.logger.debug(f"Configuration updated: {section}.{key} = {value}")
    
    def validate_config(self) -> List[str]:
        """
        Validate the current configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate database configuration
        if self.database.type not in ('sqlite', 'postgresql'):
            errors.append(f"Invalid database type: {self.database.type}")
        
        if self.database.type == 'postgresql':
            if not self.database.host:
                errors.append("PostgreSQL requires host configuration")
            if not self.database.username:
                errors.append("PostgreSQL requires username configuration")
            if not self.database.password:
                errors.append("PostgreSQL requires password configuration")
        
        # Validate portfolio configuration
        if self.portfolio.initial_cash <= 0:
            errors.append("Initial cash must be positive")
        
        if self.portfolio.commission_per_contract < 0:
            errors.append("Commission per contract cannot be negative")
        
        # Validate risk configuration
        if not (0 < self.risk.max_position_size_pct <= 1):
            errors.append("Max position size must be between 0 and 1 (0-100%)")
        
        if self.risk.max_portfolio_leverage <= 0:
            errors.append("Max portfolio leverage must be positive")
        
        # Validate IBKR configuration
        if not (1 <= self.ibkr.port <= 65535):
            errors.append("IBKR port must be between 1 and 65535")
        
        if self.ibkr.client_id < 0:
            errors.append("IBKR client ID must be non-negative")
        
        # Validate logging configuration
        if self.logging.level not in ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'):
            errors.append(f"Invalid logging level: {self.logging.level}")
        
        return errors
    
    def save_config(self, output_file: str) -> None:
        """
        Save current configuration to a YAML file.
        
        Args:
            output_file: Path to output file
        """
        config_dict = {
            'database': {
                'type': self.database.type,
                'path': self.database.path,
                'host': self.database.host,
                'port': self.database.port,
                'username': self.database.username,
                'pool_size': self.database.pool_size,
                'echo': self.database.echo
            },
            'logging': {
                'level': self.logging.level,
                'format': self.logging.format,
                'file': self.logging.file,
                'max_bytes': self.logging.max_bytes,
                'backup_count': self.logging.backup_count,
                'console_output': self.logging.console_output
            },
            'portfolio': {
                'initial_cash': float(self.portfolio.initial_cash),
                'commission_per_contract': float(self.portfolio.commission_per_contract),
                'commission_per_share': float(self.portfolio.commission_per_share),
                'margin_rate': float(self.portfolio.margin_rate),
                'max_position_size_pct': float(self.portfolio.max_position_size_pct),
                'max_portfolio_leverage': float(self.portfolio.max_portfolio_leverage)
            },
            'risk': {
                'max_position_size_pct': float(self.risk.max_position_size_pct),
                'max_portfolio_leverage': float(self.risk.max_portfolio_leverage),
                'stop_loss_pct': float(self.risk.stop_loss_pct),
                'max_drawdown_pct': float(self.risk.max_drawdown_pct),
                'var_confidence_level': float(self.risk.var_confidence_level),
                'concentration_limit_pct': float(self.risk.concentration_limit_pct)
            },
            'ibkr': {
                'host': self.ibkr.host,
                'port': self.ibkr.port,
                'client_id': self.ibkr.client_id,
                'timeout': self.ibkr.timeout,
                'enable_logging': self.ibkr.enable_logging,
                'readonly': self.ibkr.readonly
            },
            'backtest': {
                'start_date': self.backtest.start_date,
                'end_date': self.backtest.end_date,
                'frequency': self.backtest.frequency,
                'slippage_bps': self.backtest.slippage_bps,
                'commission_model': self.backtest.commission_model,
                'fill_model': self.backtest.fill_model
            },
            'performance': {
                'benchmark_symbol': self.performance.benchmark_symbol,
                'risk_free_rate': float(self.performance.risk_free_rate),
                'business_days_per_year': self.performance.business_days_per_year,
                'enable_profiling': self.performance.enable_profiling,
                'memory_monitoring': self.performance.memory_monitoring
            }
        }
        
        # Create directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Configuration saved to: {output_file}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration."""
        return {
            'config_file': self.config_file,
            'loaded': self._loaded,
            'database_type': self.database.type,
            'database_path': self.database.path,
            'log_level': self.logging.level,
            'log_file': self.logging.file,
            'initial_cash': float(self.portfolio.initial_cash),
            'ibkr_host': self.ibkr.host,
            'ibkr_port': self.ibkr.port,
            'validation_errors': self.validate_config()
        }


# Global configuration instance
_config_instance: Optional[ConfigManager] = None


def get_config(config_file: Optional[str] = None) -> ConfigManager:
    """
    Get the global configuration instance.
    
    Args:
        config_file: Configuration file path (only used on first call)
        
    Returns:
        ConfigManager instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = ConfigManager(config_file)
    
    return _config_instance


def reload_config(config_file: Optional[str] = None) -> ConfigManager:
    """
    Reload the global configuration.
    
    Args:
        config_file: New configuration file path
        
    Returns:
        Reloaded ConfigManager instance
    """
    global _config_instance
    _config_instance = ConfigManager(config_file)
    return _config_instance


if __name__ == "__main__":
    # Test the configuration system
    config = ConfigManager()
    
    print("Configuration Summary:")
    print(yaml.dump(config.get_summary(), default_flow_style=False))
    
    # Validate configuration
    errors = config.validate_config()
    if errors:
        print("Validation Errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration is valid!")
    
    # Test saving configuration
    config.save_config("test_config.yaml")
    print("Test configuration saved to test_config.yaml")