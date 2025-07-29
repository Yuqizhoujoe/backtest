"""
Position management and P&L tracking for trading positions.

This module defines the Position class which tracks individual trading
positions, including entry/exit details, P&L calculations, and risk metrics.
"""

import datetime
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Union, Optional, Dict, Any, List

from .contracts import OptionContract, StockContract
from .exceptions import InvalidPositionError, InsufficientPositionError, ValidationError


class PositionStatus(Enum):
    """Enumeration for position status."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PARTIALLY_CLOSED = "PARTIALLY_CLOSED"


class PositionSide(Enum):
    """Enumeration for position side."""

    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Transaction:
    """Represents a single transaction (buy/sell)."""

    timestamp: datetime.datetime
    quantity: int  # Positive for buy, negative for sell
    price: Decimal
    commission: Decimal = Decimal("0")
    fees: Decimal = Decimal("0")
    transaction_id: Optional[str] = None

    def __post_init__(self):
        if self.quantity == 0:
            raise InvalidPositionError("Transaction quantity cannot be zero")

        if self.price < 0:
            raise ValidationError("Transaction price cannot be negative")

        if self.commission < 0:
            raise ValidationError("Commission cannot be negative")

        if self.fees < 0:
            raise ValidationError("Fees cannot be negative")

    @property
    def is_buy(self) -> bool:
        """Check if this is a buy transaction."""
        return self.quantity > 0

    @property
    def is_sell(self) -> bool:
        """Check if this is a sell transaction."""
        return self.quantity < 0

    @property
    def gross_amount(self) -> Decimal:
        """Calculate gross transaction amount."""
        return abs(self.quantity) * self.price

    @property
    def net_amount(self) -> Decimal:
        """Calculate net transaction amount including costs."""
        gross = self.gross_amount
        costs = self.commission + self.fees

        if self.is_buy:
            return gross + costs  # Costs increase the amount paid
        else:
            return gross - costs  # Costs reduce the amount received

    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction to dictionary representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "quantity": self.quantity,
            "price": float(self.price),
            "commission": float(self.commission),
            "fees": float(self.fees),
            "transaction_id": self.transaction_id,
            "gross_amount": float(self.gross_amount),
            "net_amount": float(self.net_amount),
        }


@dataclass
class Position:
    """
    Represents a trading position with P&L tracking and risk metrics.

    This class tracks all aspects of a trading position including entry/exit
    transactions, realized and unrealized P&L, and position-level risk metrics.
    """

    contract: Union[OptionContract, StockContract]
    entry_timestamp: datetime.datetime
    initial_quantity: int  # Original position size
    entry_price: Decimal  # Average entry price

    # Position tracking
    current_quantity: int = field(init=False)
    transactions: List[Transaction] = field(default_factory=list, init=False)

    # P&L tracking
    realized_pnl: Decimal = field(default=Decimal("0"), init=False)
    unrealized_pnl: Decimal = field(default=Decimal("0"), init=False)

    # Risk metrics
    initial_margin: Optional[Decimal] = None
    maintenance_margin: Optional[Decimal] = None

    # Metadata
    position_id: Optional[str] = None
    status: PositionStatus = field(default=PositionStatus.OPEN, init=False)
    last_update: Optional[datetime.datetime] = field(default=None, init=False)

    def __post_init__(self):
        """Initialize position after creation."""
        if self.initial_quantity == 0:
            raise InvalidPositionError("Position quantity cannot be zero")

        if self.entry_price <= 0:
            raise InvalidPositionError("Entry price must be positive")

        # Initialize current quantity
        self.current_quantity = self.initial_quantity

        # Create initial transaction
        initial_transaction = Transaction(
            timestamp=self.entry_timestamp,
            quantity=self.initial_quantity,
            price=self.entry_price,
        )
        self.transactions.append(initial_transaction)

        self.last_update = datetime.datetime.now()

    @property
    def contract_id(self) -> str:
        """Get the contract identifier."""
        return self.contract.contract_id

    @property
    def symbol(self) -> str:
        """Get the underlying symbol."""
        return self.contract.symbol

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.current_quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.current_quantity < 0

    @property
    def is_closed(self) -> bool:
        """Check if position is completely closed."""
        return self.current_quantity == 0

    @property
    def side(self) -> PositionSide:
        """Get position side."""
        if self.current_quantity > 0:
            return PositionSide.LONG
        elif self.current_quantity < 0:
            return PositionSide.SHORT
        else:
            # Closed position - determine based on initial quantity
            return (
                PositionSide.LONG if self.initial_quantity > 0 else PositionSide.SHORT
            )

    @property
    def multiplier(self) -> int:
        """Get contract multiplier."""
        if isinstance(self.contract, OptionContract):
            return self.contract.multiplier
        return 1

    @property
    def position_value(self) -> Decimal:
        """Calculate current position value."""
        if self.is_closed:
            return Decimal("0")

        current_price = self._get_current_price()
        if current_price is None:
            return Decimal("0")

        return abs(self.current_quantity) * current_price * self.multiplier

    @property
    def total_pnl(self) -> Decimal:
        """Calculate total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def avg_entry_price(self) -> Decimal:
        """Calculate average entry price from all transactions."""
        if not self.transactions:
            return self.entry_price

        total_quantity = 0
        weighted_price_sum = Decimal("0")

        for transaction in self.transactions:
            if (transaction.quantity > 0 and self.initial_quantity > 0) or (
                transaction.quantity < 0 and self.initial_quantity < 0
            ):
                # Entry transaction (same direction as initial)
                total_quantity += abs(transaction.quantity)
                weighted_price_sum += abs(transaction.quantity) * transaction.price

        if total_quantity == 0:
            return self.entry_price

        return weighted_price_sum / total_quantity

    @property
    def holding_period(self) -> datetime.timedelta:
        """Calculate holding period."""
        end_time = datetime.datetime.now()
        if self.is_closed and self.transactions:
            # Find last closing transaction
            closing_transactions = [
                t
                for t in self.transactions
                if (t.quantity < 0 and self.initial_quantity > 0)
                or (t.quantity > 0 and self.initial_quantity < 0)
            ]
            if closing_transactions:
                end_time = max(t.timestamp for t in closing_transactions)

        return end_time - self.entry_timestamp

    def add_transaction(self, transaction: Transaction) -> None:
        """
        Add a new transaction to the position.

        Args:
            transaction: Transaction to add

        Raises:
            InsufficientPositionError: If trying to close more than available
        """
        # Check if this would over-close the position
        new_quantity = self.current_quantity + transaction.quantity

        # For positions, we need to ensure we don't reverse beyond zero
        if (self.current_quantity > 0 and new_quantity < 0) or (
            self.current_quantity < 0 and new_quantity > 0
        ):
            if abs(new_quantity) > 0:
                raise InsufficientPositionError(
                    f"Transaction would reverse position. Current: "
                    f"{self.current_quantity}, Transaction: {transaction.quantity}, "
                    f"New: {new_quantity}"
                )

        # Calculate P&L for closing transactions
        if (transaction.quantity < 0 and self.current_quantity > 0) or (
            transaction.quantity > 0 and self.current_quantity < 0
        ):
            # This is a closing transaction
            closed_quantity = min(abs(transaction.quantity), abs(self.current_quantity))
            avg_entry = self.avg_entry_price

            if self.current_quantity > 0:
                # Closing long position
                pnl = (
                    closed_quantity * (transaction.price - avg_entry) * self.multiplier
                )
            else:
                # Closing short position
                pnl = (
                    closed_quantity * (avg_entry - transaction.price) * self.multiplier
                )

            # Subtract transaction costs
            transaction_costs = transaction.commission + transaction.fees
            pnl -= transaction_costs

            self.realized_pnl += pnl

        # Update position
        self.current_quantity = new_quantity
        self.transactions.append(transaction)

        # Update status
        if self.current_quantity == 0:
            self.status = PositionStatus.CLOSED
        elif abs(self.current_quantity) < abs(self.initial_quantity):
            self.status = PositionStatus.PARTIALLY_CLOSED
        else:
            self.status = PositionStatus.OPEN

        self.last_update = datetime.datetime.now()

    def update_unrealized_pnl(self, current_price: Optional[Decimal] = None) -> None:
        """
        Update unrealized P&L based on current market price.

        Args:
            current_price: Current market price. If None, uses contract's last price.
        """
        if self.is_closed:
            self.unrealized_pnl = Decimal("0")
            return

        if current_price is None:
            current_price = self._get_current_price()

        if current_price is None:
            return

        avg_entry = self.avg_entry_price

        if self.current_quantity > 0:
            # Long position
            self.unrealized_pnl = (
                self.current_quantity * (current_price - avg_entry) * self.multiplier
            )
        else:
            # Short position
            self.unrealized_pnl = (
                abs(self.current_quantity)
                * (avg_entry - current_price)
                * self.multiplier
            )

        self.last_update = datetime.datetime.now()

    def close_position(
        self,
        exit_price: Decimal,
        exit_time: datetime.datetime,
        commission: Decimal = Decimal("0"),
        fees: Decimal = Decimal("0"),
    ) -> Decimal:
        """
        Close the entire position.

        Args:
            exit_price: Price at which to close the position
            exit_time: Time of position closure
            commission: Commission for the closing transaction
            fees: Additional fees for the closing transaction

        Returns:
            Realized P&L from closing the position
        """
        if self.is_closed:
            return Decimal("0")

        # Create closing transaction
        closing_quantity = -self.current_quantity  # Opposite of current quantity
        closing_transaction = Transaction(
            timestamp=exit_time,
            quantity=closing_quantity,
            price=exit_price,
            commission=commission,
            fees=fees,
        )

        # Calculate P&L before adding transaction
        initial_realized_pnl = self.realized_pnl

        # Add the closing transaction
        self.add_transaction(closing_transaction)

        # Return the P&L from this closing
        return self.realized_pnl - initial_realized_pnl

    def partial_close(
        self,
        quantity_to_close: int,
        exit_price: Decimal,
        exit_time: datetime.datetime,
        commission: Decimal = Decimal("0"),
        fees: Decimal = Decimal("0"),
    ) -> Decimal:
        """
        Partially close the position.

        Args:
            quantity_to_close: Number of contracts/shares to close (positive number)
            exit_price: Price at which to close
            exit_time: Time of partial closure
            commission: Commission for the transaction
            fees: Additional fees for the transaction

        Returns:
            Realized P&L from the partial closure
        """
        if self.is_closed:
            return Decimal("0")

        if quantity_to_close <= 0:
            raise InvalidPositionError("Quantity to close must be positive")

        if quantity_to_close > abs(self.current_quantity):
            raise InsufficientPositionError(
                f"Cannot close {quantity_to_close} contracts. "
                f"Position size: {abs(self.current_quantity)}"
            )

        # Create partial closing transaction
        if self.current_quantity > 0:
            transaction_quantity = -quantity_to_close  # Sell for long position
        else:
            transaction_quantity = quantity_to_close  # Buy to cover for short position

        closing_transaction = Transaction(
            timestamp=exit_time,
            quantity=transaction_quantity,
            price=exit_price,
            commission=commission,
            fees=fees,
        )

        # Calculate P&L before adding transaction
        initial_realized_pnl = self.realized_pnl

        # Add the partial closing transaction
        self.add_transaction(closing_transaction)

        # Return the P&L from this partial closing
        return self.realized_pnl - initial_realized_pnl

    def calculate_return_pct(self) -> Optional[Decimal]:
        """
        Calculate percentage return on the position.

        Returns:
            Percentage return, or None if cannot be calculated
        """
        entry_value = (
            abs(self.initial_quantity) * self.avg_entry_price * self.multiplier
        )
        if entry_value == 0:
            return None

        total_pnl = self.total_pnl
        return (total_pnl / entry_value) * 100

    def calculate_margin_requirement(
        self, margin_rate: Decimal = Decimal("0.5")
    ) -> Decimal:
        """
        Calculate margin requirement for the position.

        Args:
            margin_rate: Margin rate (e.g., 0.5 for 50%)

        Returns:
            Required margin amount
        """
        if self.is_closed:
            return Decimal("0")

        position_value = self.position_value
        if isinstance(self.contract, OptionContract) and self.current_quantity < 0:
            # Short options have different margin requirements
            # This is a simplified calculation - real margin requirements are
            # more complex
            return position_value * margin_rate * 2  # Higher margin for short options

        return position_value * margin_rate

    def _get_current_price(self) -> Optional[Decimal]:
        """Get current market price from the contract."""
        if hasattr(self.contract, "last") and self.contract.last:
            return self.contract.last
        elif hasattr(self.contract, "mid_price") and self.contract.mid_price:
            return self.contract.mid_price
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Generate position summary."""
        return {
            "position_id": self.position_id,
            "contract_id": self.contract_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "status": self.status.value,
            "initial_quantity": self.initial_quantity,
            "current_quantity": self.current_quantity,
            "entry_price": float(self.avg_entry_price),
            "entry_timestamp": self.entry_timestamp.isoformat(),
            "position_value": float(self.position_value),
            "realized_pnl": float(self.realized_pnl),
            "unrealized_pnl": float(self.unrealized_pnl),
            "total_pnl": float(self.total_pnl),
            "return_pct": (
                float(self.calculate_return_pct())
                if self.calculate_return_pct()
                else None
            ),
            "holding_period_days": self.holding_period.days,
            "num_transactions": len(self.transactions),
            "last_update": self.last_update.isoformat() if self.last_update else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary representation."""
        summary = self.get_summary()
        summary["transactions"] = [t.to_dict() for t in self.transactions]
        summary["initial_margin"] = (
            float(self.initial_margin) if self.initial_margin else None
        )
        summary["maintenance_margin"] = (
            float(self.maintenance_margin) if self.maintenance_margin else None
        )
        return summary
