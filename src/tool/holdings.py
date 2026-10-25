"""Loader/model for config/holdings.yaml — shared by covered_calls, screener, rolls, puts."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from pathlib import Path

import yaml

DEFAULT_HOLDINGS_PATH = Path("config/holdings.yaml")
LOT_SIZE = 100
ACCOUNTS = ("rh_taxable", "rh_ira", "ibkr")
DEFAULT_OPTION_ACCOUNT = "rh_taxable"  # options saved before they carried an account are read as this one


def option_account(option) -> str:
    """The account an option position is in (older rows without one are assumed to be the taxable account)."""
    return option.account or DEFAULT_OPTION_ACCOUNT


@dataclass(frozen=True)
class Lot:
    symbol: str
    account: str
    shares: float  # whole numbers stay int; fractional shares are allowed
    cost: float
    purchase_date: date | None = None
    note: str | None = None
    strategy: str | None = None
    no_covered_calls: bool = False


@dataclass(frozen=True)
class ShortOption:
    symbol: str
    type: str  # "call" | "put"
    strike: float
    expiry: date
    qty: int
    net_credit_total: float | None = None
    note: str | None = None
    account: str | None = None  # None = not recorded (read as DEFAULT_OPTION_ACCOUNT, marked assumed)


@dataclass(frozen=True)
class LongOption:
    symbol: str
    type: str  # "call" | "put"
    strike: float
    expiry: date
    qty: int
    avg_price: float | None = None
    note: str | None = None
    account: str | None = None


@dataclass(frozen=True)
class Holdings:
    lots: list[Lot] = field(default_factory=list)
    short_options: list[ShortOption] = field(default_factory=list)
    long_options: list[LongOption] = field(default_factory=list)
    watchlist_scan: list[str] = field(default_factory=list)
    alab_rules: dict = field(default_factory=dict)

    def positions(self) -> list[tuple[str, str]]:
        """Unique (symbol, account) pairs, in the order first seen."""
        seen: dict[tuple[str, str], None] = {}
        for lot in self.lots:
            seen.setdefault((lot.symbol, lot.account), None)
        return list(seen)

    def lots_for(self, symbol: str, account: str) -> list[Lot]:
        return [l for l in self.lots if l.symbol == symbol and l.account == account]

    def shares_for(self, symbol: str, account: str) -> int:
        return sum(l.shares for l in self.lots_for(symbol, account))

    def avg_cost_for(self, symbol: str, account: str) -> float:
        lots = self.lots_for(symbol, account)
        total_shares = sum(l.shares for l in lots)
        if total_shares == 0:
            return 0.0
        return sum(l.shares * l.cost for l in lots) / total_shares

    def no_covered_calls_for(self, symbol: str, account: str) -> bool:
        return any(l.no_covered_calls for l in self.lots_for(symbol, account))

    def strategy_for(self, symbol: str, account: str) -> str | None:
        return next((l.strategy for l in self.lots_for(symbol, account) if l.strategy), None)

    def short_options_for(self, symbol: str, account: str, type_: str | None = None) -> list[ShortOption]:
        # short_options in holdings.yaml aren't tagged by account; a symbol held
        # in only one account maps unambiguously, which covers today's data.
        matches = [so for so in self.short_options if so.symbol == symbol]
        if type_ is not None:
            matches = [so for so in matches if so.type == type_]
        return matches

    def covered_shares_for(self, symbol: str, account: str) -> int:
        return sum(so.qty for so in self.short_options_for(symbol, account, "call")) * LOT_SIZE

    def free_lots_for(self, symbol: str, account: str) -> int:
        free_shares = self.shares_for(symbol, account) - self.covered_shares_for(symbol, account)
        return max(int(free_shares // LOT_SIZE), 0)


def add_lot(holdings: Holdings, lot: Lot) -> Holdings:
    symbol = lot.symbol.strip().upper()
    if not symbol:
        raise ValueError("symbol is required")
    if lot.account not in ACCOUNTS:
        raise ValueError(f"account must be one of {', '.join(ACCOUNTS)}")
    if lot.shares <= 0:
        raise ValueError("shares must be positive")
    if lot.cost <= 0:
        raise ValueError("cost must be positive")
    return replace(holdings, lots=[*holdings.lots, replace(lot, symbol=symbol)])


def remove_lots(holdings: Holdings, symbol: str, account: str) -> Holdings:
    return replace(
        holdings,
        lots=[l for l in holdings.lots if not (l.symbol == symbol and l.account == account)],
    )


def _num(value: object) -> float:
    number = float(value)
    return int(number) if number == int(number) else number


def _as_date(value: object) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()
    return date.fromisoformat(str(value))


def load_holdings(path: str | Path = DEFAULT_HOLDINGS_PATH) -> Holdings:
    raw = yaml.safe_load(Path(path).read_text())
    lots = [
        Lot(
            symbol=h["symbol"],
            account=h["account"],
            shares=_num(h["shares"]),
            cost=float(h["cost"]),
            purchase_date=_as_date(h["purchase_date"]) if h.get("purchase_date") else None,
            note=h.get("note"),
            strategy=h.get("strategy"),
            no_covered_calls=bool(h.get("no_covered_calls", False)),
        )
        for h in raw.get("holdings", [])
    ]
    short_options = [
        ShortOption(
            symbol=so["symbol"],
            type=so["type"],
            strike=float(so["strike"]),
            expiry=_as_date(so["expiry"]),
            qty=int(so["qty"]),
            net_credit_total=(
                float(so["net_credit_total"]) if so.get("net_credit_total") is not None else None
            ),
            note=so.get("note"),
            account=so.get("account"),
        )
        for so in raw.get("short_options", [])
    ]
    long_options = [
        LongOption(
            symbol=lo["symbol"],
            type=lo["type"],
            strike=float(lo["strike"]),
            expiry=_as_date(lo["expiry"]),
            qty=int(lo["qty"]),
            avg_price=float(lo["avg_price"]) if lo.get("avg_price") is not None else None,
            note=lo.get("note"),
            account=lo.get("account"),
        )
        for lo in raw.get("long_options", [])
    ]
    return Holdings(
        lots=lots,
        short_options=short_options,
        long_options=long_options,
        watchlist_scan=list(raw.get("watchlist_scan", [])),
        alab_rules=dict(raw.get("alab_rules", {})),
    )


def _inline(fields: list[tuple[str, object]]) -> str:
    parts = []
    for key, value in fields:
        if value is None:
            continue
        if isinstance(value, bool):
            value = "true" if value else "false"
        elif isinstance(value, str) and key == "note":
            value = f'"{value}"'
        parts.append(f"{key}: {value}")
    return "{" + ", ".join(parts) + "}"


def save_holdings(holdings: Holdings, path: str | Path = DEFAULT_HOLDINGS_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copyfile(path, path.with_name(path.name + ".bak"))

    lines = [
        "# Holdings — gitignored (contains cost basis). Edited from the dashboard Portfolio tab.",
        "# account: rh_taxable | rh_ira | ibkr",
        "holdings:",
    ]
    for l in holdings.lots:
        lines.append(
            "  - "
            + _inline(
                [
                    ("symbol", l.symbol),
                    ("account", l.account),
                    ("shares", l.shares),
                    ("cost", l.cost),
                    ("purchase_date", l.purchase_date.isoformat() if l.purchase_date else None),
                    ("strategy", l.strategy),
                    ("no_covered_calls", l.no_covered_calls or None),
                    ("note", l.note),
                ]
            )
        )

    lines += ["", "short_options:" if holdings.short_options else "short_options: []"]
    for so in holdings.short_options:
        lines.append(
            "  - "
            + _inline(
                [
                    ("symbol", so.symbol),
                    ("type", so.type),
                    ("strike", so.strike),
                    ("expiry", so.expiry.isoformat()),
                    ("qty", so.qty),
                    ("account", so.account),
                    ("net_credit_total", so.net_credit_total),
                    ("note", so.note),
                ]
            )
        )

    lines += ["", "long_options:" if holdings.long_options else "long_options: []"]
    for lo in holdings.long_options:
        lines.append(
            "  - "
            + _inline(
                [
                    ("symbol", lo.symbol),
                    ("type", lo.type),
                    ("strike", lo.strike),
                    ("expiry", lo.expiry.isoformat()),
                    ("qty", lo.qty),
                    ("account", lo.account),
                    ("avg_price", lo.avg_price),
                    ("note", lo.note),
                ]
            )
        )

    lines += ["", "watchlist_scan: [" + ", ".join(holdings.watchlist_scan) + "]"]
    if holdings.alab_rules:
        lines += ["", "alab_rules:"]
        lines += [f"  {k}: {v}" for k, v in holdings.alab_rules.items()]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def replace_account_options(holdings: Holdings, account: str, shorts: list[ShortOption], longs: list[LongOption]) -> Holdings:
    """Swap one account's option positions for `shorts` / `longs` (tagged with the account); other accounts are untouched."""
    keep_short = [o for o in holdings.short_options if option_account(o) != account]
    keep_long = [o for o in holdings.long_options if option_account(o) != account]
    return replace(
        holdings,
        short_options=[*keep_short, *(replace(o, account=account) for o in shorts)],
        long_options=[*keep_long, *(replace(o, account=account) for o in longs)],
    )
