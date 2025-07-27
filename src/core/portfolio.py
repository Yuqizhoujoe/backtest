"""
Portfolio management with position tracking and risk management.

This module defines the Portfolio class which manages a collection of
positions and tracks portfolio-level metrics, P&L, and risk measures.
"""

import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass

from .contracts import OptionContract, StockContract
from .position import Position, Transaction
from .exceptions import (
    InsufficientFundsError,
    ExcessiveRiskError,
    PortfolioError,
)


@dataclass
class PortfolioMetrics:
    """Container for portfolio performance metrics."""

    total_value: Decimal
    total_pnl: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    return_pct: Decimal
    max_drawdown: Decimal
    sharpe_ratio: Optional[Decimal] = None
    sortino_ratio: Optional[Decimal] = None
    win_rate: Optional[Decimal] = None
    profit_factor: Optional[Decimal] = None


class Portfolio:
    """
    Manages a collection of positions and tracks portfolio-level metrics.

    This class provides comprehensive portfolio management including position
    tracking, P&L calculations, risk management, and performance analytics.
    """

    def __init__(
        self,
        initial_cash: Decimal,
        name: str = "Default Portfolio",
        portfolio_id: Optional[str] = None,
    ):
        """
        Initialize a new portfolio.

        Args:
            initial_cash: Starting cash amount
            name: Portfolio name
            portfolio_id: Optional unique identifier
        """
        self.portfolio_id = (
            portfolio_id
            or f"portfolio_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.name = name
        self.initial_cash = initial_cash
        self.cash = initial_cash

        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []

        # Portfolio metrics
        self.total_realized_pnl = Decimal("0")
        self.total_unrealized_pnl = Decimal("0")
        self.max_drawdown = Decimal("0")
        self.peak_value = initial_cash

        # Risk management
        self.margin_used = Decimal("0")
        self.margin_available = initial_cash
        self.max_position_size_pct = Decimal("0.1")  # 10% max position size
        self.max_portfolio_leverage = Decimal("2.0")

        # Performance tracking
        self.daily_returns: List[Decimal] = []
        self.value_history: List[Tuple[datetime.datetime, Decimal]] = []

        # Transaction history
        self.all_transactions: List[Dict[str, Any]] = []

        # Metadata
        self.created_at = datetime.datetime.now()
        self.last_update = datetime.datetime.now()

    @property
    def total_position_value(self) -> Decimal:
        """Calculate total value of all open positions."""
        return sum(pos.position_value for pos in self.positions.values())

    @property
    def total_value(self) -> Decimal:
        """Calculate total portfolio value (cash + positions)."""
        return self.cash + self.total_position_value

    @property
    def total_pnl(self) -> Decimal:
        """Calculate total P&L (realized + unrealized)."""
        return self.total_realized_pnl + self.total_unrealized_pnl

    @property
    def return_pct(self) -> Decimal:
        """Calculate portfolio return percentage."""
        if self.initial_cash == 0:
            return Decimal("0")
        return (self.total_value - self.initial_cash) / self.initial_cash * 100

    @property
    def num_open_positions(self) -> int:
        """Get number of open positions."""
        return len(self.positions)

    @property
    def num_closed_positions(self) -> int:
        """Get number of closed positions."""
        return len(self.closed_positions)

    @property
    def equity(self) -> Decimal:
        """Calculate portfolio equity (total value)."""
        return self.total_value

    @property
    def buying_power(self) -> Decimal:
        """Calculate available buying power."""
        return self.cash + self.margin_available

    @property
    def leverage(self) -> Decimal:
        """Calculate portfolio leverage."""
        if self.total_value == 0:
            return Decimal("0")
        return self.total_position_value / self.total_value

    def add_position(
        self,
        contract: Union[OptionContract, StockContract],
        quantity: int,
        entry_price: Decimal,
        entry_time: Optional[datetime.datetime] = None,
        commission: Decimal = Decimal("0"),
        fees: Decimal = Decimal("0"),
        position_id: Optional[str] = None,
    ) -> Position:
        """
        Add a new position to the portfolio.

        Args:
            contract: The contract for the position
            quantity: Position quantity (positive for long, negative for short)
            entry_price: Entry price
            entry_time: Entry timestamp
            commission: Commission for opening the position
            fees: Additional fees
            position_id: Optional position identifier

        Returns:
            The created Position object

        Raises:
            InsufficientFundsError: If insufficient cash
            ExcessiveRiskError: If position exceeds risk limits
        """
        if entry_time is None:
            entry_time = datetime.datetime.now()

        # Validate position size against risk limits
        position_value = abs(quantity) * entry_price * self._get_multiplier(contract)
        max_position_value = self.total_value * self.max_position_size_pct

        if position_value > max_position_value:
            raise ExcessiveRiskError(
                f"Position value {position_value} exceeds maximum allowed "
                f"{max_position_value} ({self.max_position_size_pct * 100}%)"
            )

        # Calculate required cash/margin
        required_cash = self._calculate_required_cash(
            contract, quantity, entry_price, commission, fees
        )

        if required_cash > self.cash:
            raise InsufficientFundsError(
                f"Insufficient cash. Required: {required_cash}, Available: {self.cash}"
            )

        # Check leverage limits
        new_position_value = self.total_position_value + position_value
        if self.total_value > 0:
            new_leverage = new_position_value / self.total_value
            if new_leverage > self.max_portfolio_leverage:
                raise ExcessiveRiskError(
                    f"Position would create leverage of {new_leverage}, "
                    f"exceeding limit of {self.max_portfolio_leverage}"
                )

        # Create the position
        position = Position(
            contract=contract,
            entry_timestamp=entry_time,
            initial_quantity=quantity,
            entry_price=entry_price,
            position_id=position_id
            or f"{contract.contract_id}_{entry_time.strftime('%Y%m%d_%H%M%S')}",
        )

        # Add initial transaction with costs
        if commission > 0 or fees > 0:
            # Modify the initial transaction to include costs
            position.transactions[0].commission = commission
            position.transactions[0].fees = fees

        # Update portfolio
        contract_id = contract.contract_id

        if contract_id in self.positions:
            # Combine with existing position
            existing_position = self.positions[contract_id]
            self._combine_positions(existing_position, position)
        else:
            self.positions[contract_id] = position

        # Update cash and portfolio metrics
        self.cash -= required_cash
        self._update_portfolio_metrics()

        # Record transaction
        self._record_transaction(position, position.transactions[0])

        return position

    def close_position(
        self,
        contract_id: str,
        exit_price: Optional[Decimal] = None,
        exit_time: Optional[datetime.datetime] = None,
        commission: Decimal = Decimal("0"),
        fees: Decimal = Decimal("0"),
    ) -> Optional[Decimal]:
        """
        Close a position completely.

        Args:
            contract_id: Contract identifier
            exit_price: Exit price (uses market price if None)
            exit_time: Exit timestamp
            commission: Commission for closing
            fees: Additional fees

        Returns:
            Realized P&L from closing the position, or None if position not found
        """
        if contract_id not in self.positions:
            return None

        position = self.positions[contract_id]

        if exit_time is None:
            exit_time = datetime.datetime.now()

        if exit_price is None:
            exit_price = position._get_current_price()
            if exit_price is None:
                raise PortfolioError(f"No exit price available for {contract_id}")

        # Close the position
        realized_pnl = position.close_position(exit_price, exit_time, commission, fees)

        # Update portfolio cash
        closing_transaction = position.transactions[
            -1
        ]  # Last transaction is the closing one
        proceeds = closing_transaction.net_amount
        self.cash += proceeds

        # Update portfolio P&L
        self.total_realized_pnl += realized_pnl

        # Move to closed positions
        self.closed_positions.append(position)
        del self.positions[contract_id]

        # Record transaction
        self._record_transaction(position, closing_transaction)

        # Update portfolio metrics
        self._update_portfolio_metrics()

        return realized_pnl

    def partial_close_position(
        self,
        contract_id: str,
        quantity_to_close: int,
        exit_price: Optional[Decimal] = None,
        exit_time: Optional[datetime.datetime] = None,
        commission: Decimal = Decimal("0"),
        fees: Decimal = Decimal("0"),
    ) -> Optional[Decimal]:
        """
        Partially close a position.

        Args:
            contract_id: Contract identifier
            quantity_to_close: Quantity to close (positive number)
            exit_price: Exit price
            exit_time: Exit timestamp
            commission: Commission for partial closing
            fees: Additional fees

        Returns:
            Realized P&L from partial closure, or None if position not found
        """
        if contract_id not in self.positions:
            return None

        position = self.positions[contract_id]

        if exit_time is None:
            exit_time = datetime.datetime.now()

        if exit_price is None:
            exit_price = position._get_current_price()
            if exit_price is None:
                raise PortfolioError(f"No exit price available for {contract_id}")

        # Partially close the position
        realized_pnl = position.partial_close(
            quantity_to_close, exit_price, exit_time, commission, fees
        )

        # Update portfolio cash
        closing_transaction = position.transactions[
            -1
        ]  # Last transaction is the partial closing
        proceeds = closing_transaction.net_amount
        self.cash += proceeds

        # Update portfolio P&L
        self.total_realized_pnl += realized_pnl

        # Record transaction
        self._record_transaction(position, closing_transaction)

        # Check if position is now fully closed
        if position.is_closed:
            self.closed_positions.append(position)
            del self.positions[contract_id]

        # Update portfolio metrics
        self._update_portfolio_metrics()

        return realized_pnl

    def update_portfolio_value(self) -> None:
        """Update unrealized P&L for all positions and portfolio metrics."""
        self.total_unrealized_pnl = Decimal("0")

        for position in self.positions.values():
            position.update_unrealized_pnl()
            self.total_unrealized_pnl += position.unrealized_pnl

        # Update portfolio metrics
        self._update_portfolio_metrics()

        # Record daily return if it's a new day
        self._record_daily_return()

    def get_position(self, contract_id: str) -> Optional[Position]:
        """Get a position by contract ID."""
        return self.positions.get(contract_id)

    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """Get all positions for a specific symbol."""
        return [pos for pos in self.positions.values() if pos.symbol == symbol]

    def get_positions_by_type(self, contract_type: type) -> List[Position]:
        """Get all positions of a specific contract type."""
        return [
            pos
            for pos in self.positions.values()
            if isinstance(pos.contract, contract_type)
        ]

    def get_long_positions(self) -> List[Position]:
        """Get all long positions."""
        return [pos for pos in self.positions.values() if pos.is_long]

    def get_short_positions(self) -> List[Position]:
        """Get all short positions."""
        return [pos for pos in self.positions.values() if pos.is_short]

    def calculate_var(
        self, confidence_level: Decimal = Decimal("0.95"), time_horizon_days: int = 1
    ) -> Optional[Decimal]:
        """
        Calculate Value at Risk (VaR).

        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon_days: Time horizon in days

        Returns:
            VaR amount, or None if insufficient data
        """
        if len(self.daily_returns) < 30:  # Need at least 30 days of data
            return None

        # Sort returns in ascending order
        sorted_returns = sorted(self.daily_returns)

        # Calculate the percentile
        percentile_index = int((1 - confidence_level) * len(sorted_returns))
        var_return = sorted_returns[percentile_index]

        # Scale by time horizon (assuming returns are independent)
        var_return_scaled = var_return * Decimal(time_horizon_days).sqrt()

        # Convert to dollar amount
        return abs(var_return_scaled * self.total_value / 100)

    def calculate_beta(self, market_returns: List[Decimal]) -> Optional[Decimal]:
        """
        Calculate portfolio beta relative to market returns.

        Args:
            market_returns: List of market returns for the same periods

        Returns:
            Beta coefficient, or None if insufficient data
        """
        if (
            len(self.daily_returns) != len(market_returns)
            or len(self.daily_returns) < 10
        ):
            return None

        # Calculate means
        portfolio_mean = sum(self.daily_returns) / len(self.daily_returns)
        market_mean = sum(market_returns) / len(market_returns)

        # Calculate covariance and market variance
        covariance = Decimal("0")
        market_variance = Decimal("0")

        for i in range(len(self.daily_returns)):
            portfolio_dev = self.daily_returns[i] - portfolio_mean
            market_dev = market_returns[i] - market_mean

            covariance += portfolio_dev * market_dev
            market_variance += market_dev * market_dev

        if market_variance == 0:
            return None

        return covariance / market_variance

    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """Calculate and return comprehensive portfolio metrics."""
        # Calculate win rate and profit factor from closed positions
        if self.closed_positions:
            winning_trades = [
                pos for pos in self.closed_positions if pos.realized_pnl > 0
            ]
            losing_trades = [
                pos for pos in self.closed_positions if pos.realized_pnl < 0
            ]

            win_rate = (
                Decimal(len(winning_trades)) / Decimal(len(self.closed_positions)) * 100
            )

            if winning_trades and losing_trades:
                avg_win = sum(pos.realized_pnl for pos in winning_trades) / len(
                    winning_trades
                )
                avg_loss = abs(
                    sum(pos.realized_pnl for pos in losing_trades) / len(losing_trades)
                )
                profit_factor = avg_win / avg_loss if avg_loss > 0 else None
            else:
                profit_factor = None
        else:
            win_rate = None
            profit_factor = None

        # Calculate Sharpe ratio (simplified - assumes risk-free rate of 0)
        sharpe_ratio = None
        if len(self.daily_returns) > 1:
            mean_return = sum(self.daily_returns) / len(self.daily_returns)
            variance = sum((r - mean_return) ** 2 for r in self.daily_returns) / (
                len(self.daily_returns) - 1
            )
            std_dev = variance.sqrt() if variance > 0 else Decimal("0")

            if std_dev > 0:
                sharpe_ratio = (
                    mean_return / std_dev * Decimal("252").sqrt()
                )  # Annualized

        return PortfolioMetrics(
            total_value=self.total_value,
            total_pnl=self.total_pnl,
            realized_pnl=self.total_realized_pnl,
            unrealized_pnl=self.total_unrealized_pnl,
            return_pct=self.return_pct,
            max_drawdown=self.max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
        )

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Generate comprehensive portfolio summary."""
        metrics = self.get_portfolio_metrics()

        return {
            "portfolio_id": self.portfolio_id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "last_update": self.last_update.isoformat(),
            # Values
            "initial_cash": float(self.initial_cash),
            "current_cash": float(self.cash),
            "total_value": float(metrics.total_value),
            "total_position_value": float(self.total_position_value),
            # P&L
            "total_pnl": float(metrics.total_pnl),
            "realized_pnl": float(metrics.realized_pnl),
            "unrealized_pnl": float(metrics.unrealized_pnl),
            "return_pct": float(metrics.return_pct),
            "max_drawdown": float(metrics.max_drawdown),
            # Risk metrics
            "leverage": float(self.leverage),
            "margin_used": float(self.margin_used),
            "margin_available": float(self.margin_available),
            "buying_power": float(self.buying_power),
            # Performance metrics
            "sharpe_ratio": (
                float(metrics.sharpe_ratio) if metrics.sharpe_ratio else None
            ),
            "win_rate": float(metrics.win_rate) if metrics.win_rate else None,
            "profit_factor": (
                float(metrics.profit_factor) if metrics.profit_factor else None
            ),
            # Position counts
            "num_open_positions": self.num_open_positions,
            "num_closed_positions": self.num_closed_positions,
            "num_transactions": len(self.all_transactions),
        }

    def _get_multiplier(self, contract: Union[OptionContract, StockContract]) -> int:
        """Get contract multiplier."""
        if isinstance(contract, OptionContract):
            return contract.multiplier
        return 1

    def _calculate_required_cash(
        self,
        contract: Union[OptionContract, StockContract],
        quantity: int,
        price: Decimal,
        commission: Decimal,
        fees: Decimal,
    ) -> Decimal:
        """Calculate required cash for a position."""
        multiplier = self._get_multiplier(contract)
        position_value = abs(quantity) * price * multiplier
        transaction_costs = commission + fees

        if quantity > 0:
            # Long position - pay full amount plus costs
            return position_value + transaction_costs
        else:
            # Short position - receive premium minus costs (but need margin)
            # For now, simplified: just require transaction costs
            return transaction_costs

    def _combine_positions(self, existing: Position, new: Position) -> None:
        """Combine a new position with an existing one."""
        # This is a simplified version - in practice, this would be more complex
        # For now, we'll add the new transaction to the existing position
        new_transaction = Transaction(
            timestamp=new.entry_timestamp,
            quantity=new.initial_quantity,
            price=new.entry_price,
            commission=new.transactions[0].commission,
            fees=new.transactions[0].fees,
        )

        existing.add_transaction(new_transaction)

    def _update_portfolio_metrics(self) -> None:
        """Update portfolio-level metrics."""
        # Update drawdown
        current_value = self.total_value
        if current_value > self.peak_value:
            self.peak_value = current_value
        else:
            drawdown_pct = (self.peak_value - current_value) / self.peak_value * 100
            self.max_drawdown = max(self.max_drawdown, drawdown_pct)

        # Update margin calculations (simplified)
        self.margin_used = sum(
            pos.calculate_margin_requirement() for pos in self.positions.values()
        )
        self.margin_available = max(Decimal("0"), self.cash - self.margin_used)

        self.last_update = datetime.datetime.now()

    def _record_transaction(self, position: Position, transaction: Transaction) -> None:
        """Record a transaction in the portfolio history."""
        transaction_record = {
            "timestamp": transaction.timestamp.isoformat(),
            "portfolio_id": self.portfolio_id,
            "position_id": position.position_id,
            "contract_id": position.contract_id,
            "symbol": position.symbol,
            "quantity": transaction.quantity,
            "price": float(transaction.price),
            "commission": float(transaction.commission),
            "fees": float(transaction.fees),
            "gross_amount": float(transaction.gross_amount),
            "net_amount": float(transaction.net_amount),
        }

        self.all_transactions.append(transaction_record)

    def _record_daily_return(self) -> None:
        """Record daily return if it's a new day."""
        now = datetime.datetime.now()
        current_value = self.total_value

        # Record value for history
        self.value_history.append((now, current_value))

        # Calculate daily return if we have previous day's value
        if len(self.value_history) > 1:
            prev_value = self.value_history[-2][1]
            if prev_value > 0:
                daily_return = (current_value - prev_value) / prev_value * 100

                # Only add if it's a different day
                prev_date = self.value_history[-2][0].date()
                current_date = now.date()

                if current_date != prev_date:
                    self.daily_returns.append(daily_return)
