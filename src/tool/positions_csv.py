"""Import positions from a broker's positions CSV (Fidelity, Schwab, IBKR, Robinhood-style layouts, or anything similar).

Reads the header row by column aliases (symbol, quantity, average cost or cost-basis total, last price, account), turns option
rows into short or long option positions (OCC, Fidelity and Schwab/IBKR option symbols), and skips cash, totals and footers.
Anything it cannot read is returned in `skipped` with a reason, never guessed. `diff_account` previews what replacing one
account's positions would change and `apply_positions` does it, keeping per-symbol flags (strategy, no covered calls, notes).
Recommend-only. Nothing here places or stages an order.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from pathlib import Path

from tool.holdings import Holdings, LongOption, Lot, ShortOption, option_account

ALIASES = {
    "symbol": ("symbol", "ticker", "instrument", "financial instrument", "security symbol"),
    "description": ("description", "security description", "name", "security", "security name"),
    "qty": ("quantity", "qty", "shares", "position", "qty (quantity)", "current quantity"),
    "cost_total": ("cost basis total", "total cost basis", "cost basis", "cost basis ($)", "total cost", "cost basis amount"),
    "avg_cost": ("average cost basis", "avg cost basis", "average cost", "avg cost", "average price", "avg price",
                 "cost/share", "cost per share", "average cost/share"),
    "price": ("last price", "price", "last", "mark", "current price", "market price"),
    "account": ("account", "account name", "account number"),
}
SKIP_SYMBOLS = {"cash", "pending activity", "account total", "total", "totals", "cash & cash investments", "cash and cash equivalents"}
OCC = re.compile(r"^([A-Z.]{1,6})\s*(\d{6})([CP])(\d{8})$")
FIDELITY = re.compile(r"^-?([A-Z.]{1,6})(\d{6})([CP])(\d+(?:\.\d+)?)$")
SLASH = re.compile(r"^([A-Z.]{1,6})\s+(\d{1,2}/\d{1,2}/\d{4})\s+(\d+(?:\.\d+)?)\s+([CP])$", re.IGNORECASE)
IBKR = re.compile(r"^([A-Z.]{1,6})\s+(\d{1,2}[A-Z]{3}\d{2})\s+(\d+(?:\.\d+)?)\s+([CP])$", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedPositions:
    stocks: list[Lot] = field(default_factory=list)
    shorts: list[ShortOption] = field(default_factory=list)
    longs: list[LongOption] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (row text, reason)
    accounts_in_file: list[str] = field(default_factory=list)


def _num(text: str | None) -> float | None:
    if text is None:
        return None
    t = str(text).strip().replace("$", "").replace(",", "").replace("%", "")
    if not t or t in ("--", "n/a", "N/A"):
        return None
    neg = t.startswith("(") and t.endswith(")")
    try:
        value = float(t.strip("()"))
    except ValueError:
        return None
    return -value if neg else value


def parse_option_symbol(text: str) -> tuple[str, str, float, date] | None:
    """(underlying, 'call'|'put', strike, expiry) from an option symbol or description, else None."""
    t = (text or "").strip().upper()
    if m := OCC.match(t):
        return m.group(1), "call" if m.group(3) == "C" else "put", int(m.group(4)) / 1000, datetime.strptime(m.group(2), "%y%m%d").date()
    if m := FIDELITY.match(t):
        return m.group(1), "call" if m.group(3) == "C" else "put", float(m.group(4)), datetime.strptime(m.group(2), "%y%m%d").date()
    if m := SLASH.match(t):
        return m.group(1), "call" if m.group(4) == "C" else "put", float(m.group(3)), datetime.strptime(m.group(2), "%m/%d/%Y").date()
    if m := IBKR.match(t):
        return m.group(1), "call" if m.group(4) == "C" else "put", float(m.group(3)), datetime.strptime(m.group(2), "%d%b%y").date()
    return None


def _find_header(rows: list[list[str]]) -> tuple[int, dict[str, int]] | None:
    for i, row in enumerate(rows[:25]):
        cells = [c.strip().lower() for c in row]
        mapping: dict[str, int] = {}
        for name, names in ALIASES.items():
            for j, cell in enumerate(cells):
                if cell in names and name not in mapping:
                    mapping[name] = j
        if "symbol" in mapping and "qty" in mapping:
            return i, mapping
    return None


def parse_positions(source: str | bytes | Path, only_account: str | None = None) -> ParsedPositions:
    """Parse a positions CSV. `only_account` keeps just the rows whose account column equals it (files with several accounts)."""
    if isinstance(source, Path):
        text = source.read_bytes().decode("utf-8-sig", errors="replace")
    elif isinstance(source, bytes):
        text = source.decode("utf-8-sig", errors="replace")
    else:
        text = source
    rows = list(csv.reader(io.StringIO(text)))
    found = _find_header(rows)
    if found is None:
        raise ValueError("no header row with a Symbol column and a Quantity column was found in the first 25 lines")
    start, cols = found

    stocks, shorts, longs, skipped, accounts = [], [], [], [], []
    for row in rows[start + 1:]:
        if not any(c.strip() for c in row):
            continue

        def cell(name: str) -> str | None:
            j = cols.get(name)
            return row[j].strip() if j is not None and j < len(row) else None

        label = ",".join(c.strip() for c in row)[:90]
        account = cell("account")
        if account and account not in accounts:
            accounts.append(account)
        if only_account is not None and account != only_account:
            continue
        symbol = (cell("symbol") or "").strip()
        if not symbol or symbol.lower() in SKIP_SYMBOLS or symbol.endswith("**") or "money market" in (cell("description") or "").lower():
            if symbol:
                skipped.append((label, "cash, total or money-market line"))
            continue
        qty = _num(cell("qty"))
        if qty is None or qty == 0:
            skipped.append((label, "no quantity"))
            continue
        option = parse_option_symbol(symbol) or parse_option_symbol(cell("description") or "")
        avg, total, price = _num(cell("avg_cost")), _num(cell("cost_total")), _num(cell("price"))
        if option:
            under, kind, strike, expiry = option
            contracts = int(abs(round(qty)))
            if contracts == 0:
                skipped.append((label, "fractional option quantity"))
                continue
            if qty < 0:
                credit = abs(total) if total is not None else None
                shorts.append(ShortOption(under, kind, strike, expiry, contracts, net_credit_total=credit))
            else:
                per_share = avg if avg is not None else (total / (contracts * 100) if total else None)
                longs.append(LongOption(under, kind, strike, expiry, contracts,
                                        avg_price=round(per_share, 4) if per_share else None))
            continue
        if qty < 0:
            skipped.append((label, "short stock is not supported"))
            continue
        cost = avg if avg and avg > 0 else (total / qty if total and total > 0 else None)
        if cost is None:
            skipped.append((label, "no cost basis in the file (add this holding by hand)"))
            continue
        stocks.append(Lot(symbol=symbol.upper().replace(".", "/"), account="", shares=int(qty) if float(qty).is_integer() else qty,
                          cost=round(cost, 4)))
    return ParsedPositions(stocks, shorts, longs, skipped, accounts)


# ---------------------------------------------------------------- preview and apply
@dataclass(frozen=True)
class Change:
    kind: str  # stock | short | long
    key: str
    before: str
    after: str
    result: str  # add | update | remove | keep


def _stock_totals(lots: list[Lot]) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for l in lots:
        shares, cost = out.get(l.symbol, (0.0, 0.0))
        out[l.symbol] = (shares + l.shares, cost + l.shares * l.cost)
    return {s: (n, c / n) for s, (n, c) in out.items() if n}


def diff_account(holdings: Holdings, account: str, parsed: ParsedPositions) -> list[Change]:
    """What replacing `account`'s stored positions with the file would change."""
    changes: list[Change] = []
    old = _stock_totals([l for l in holdings.lots if l.account == account])
    new = _stock_totals(parsed.stocks)
    for sym in sorted(old.keys() | new.keys()):
        before = f"{old[sym][0]:g} @ {old[sym][1]:.2f}" if sym in old else ""
        after = f"{new[sym][0]:g} @ {new[sym][1]:.2f}" if sym in new else ""
        if sym not in old:
            result = "add"
        elif sym not in new:
            result = "remove"
        elif abs(old[sym][0] - new[sym][0]) < 1e-9 and abs(old[sym][1] - new[sym][1]) <= 0.005 * old[sym][1]:
            result = "keep"
        else:
            result = "update"
        changes.append(Change("stock", sym, before, after, result))

    def opt_key(o) -> tuple:
        return (o.symbol, o.type, o.strike, o.expiry)

    for kind, stored, incoming in (
        ("short", [o for o in holdings.short_options if option_account(o) == account], parsed.shorts),
        ("long", [o for o in holdings.long_options if option_account(o) == account], parsed.longs),
    ):
        old_o, new_o = {opt_key(o): o.qty for o in stored}, {opt_key(o): o.qty for o in incoming}
        for key in sorted(old_o.keys() | new_o.keys(), key=str):
            label = f"{key[0]} {key[1]} {key[2]:g} {key[3]}"
            result = "add" if key not in old_o else "remove" if key not in new_o else "keep" if old_o[key] == new_o[key] else "update"
            changes.append(Change(kind, label, str(old_o.get(key, "")), str(new_o.get(key, "")), result))
    return changes


def apply_positions(holdings: Holdings, account: str, parsed: ParsedPositions) -> Holdings:
    """Replace one account's stock lots and options with the file's. A stock whose shares and average cost already match is
    left exactly as stored (purchase dates, strategy, no-covered-calls flags); a changed one keeps its flags and notes."""
    from tool.holdings import replace_account_options

    old_lots = [l for l in holdings.lots if l.account == account]
    other = [l for l in holdings.lots if l.account != account]
    old_totals = _stock_totals(old_lots)
    new_lots: list[Lot] = []
    for lot in parsed.stocks:
        before = old_totals.get(lot.symbol)
        prior = [l for l in old_lots if l.symbol == lot.symbol]
        if before and abs(before[0] - lot.shares) < 1e-9 and abs(before[1] - lot.cost) <= 0.005 * before[1]:
            new_lots.extend(prior)  # unchanged: keep every stored lot as it is
            continue
        flags = prior[0] if prior else None
        new_lots.append(replace(
            lot, account=account,
            strategy=flags.strategy if flags else None, no_covered_calls=flags.no_covered_calls if flags else False,
            note=flags.note if flags else None,
        ))
    # a symbol appearing twice in the file (two lots) must not double-keep stored lots
    seen, deduped = set(), []
    for l in new_lots:
        k = (l.symbol, l.shares, l.cost, l.purchase_date)
        if k in seen:
            continue
        seen.add(k)
        deduped.append(l)
    updated = replace(holdings, lots=[*other, *deduped])
    return replace_account_options(updated, account, parsed.shorts, parsed.longs)
