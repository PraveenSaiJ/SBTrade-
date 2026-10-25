"""Local SQLite store for market-data snapshots and alerts (CLAUDE.md: SQLite for history)."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from tool.holdings import DEFAULT_HOLDINGS_PATH, Holdings, LongOption, Lot, ShortOption
from tool.holdings import load_holdings as load_holdings_yaml
from tool.holdings import save_holdings as save_holdings_yaml
from tool.levels import MARKET_CLOSE, NY

DEFAULT_DB_PATH = Path("data/sbtrade.sqlite")

SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_bars (
    symbol      TEXT NOT NULL,
    bar_date    TEXT NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    volume      REAL,
    source      TEXT NOT NULL,
    fetched_at  TEXT NOT NULL,
    PRIMARY KEY (symbol, bar_date)
);
CREATE TABLE IF NOT EXISTS spot_snapshots (
    symbol  TEXT NOT NULL,
    ts      TEXT NOT NULL,
    price   REAL NOT NULL,
    source  TEXT NOT NULL,
    PRIMARY KEY (symbol, ts)
);
CREATE TABLE IF NOT EXISTS option_snapshots (
    symbol         TEXT NOT NULL,
    ts             TEXT NOT NULL,
    expiry         TEXT NOT NULL,
    strike         REAL NOT NULL,
    opt_type       TEXT NOT NULL,
    bid            REAL,
    ask            REAL,
    mid            REAL,
    delta          REAL,
    iv             REAL,
    open_interest  REAL,
    source         TEXT NOT NULL,
    PRIMARY KEY (symbol, ts, expiry, strike, opt_type)
);
CREATE TABLE IF NOT EXISTS setups (
    scan_date   TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    kind        TEXT NOT NULL,
    spot        REAL,
    atr         REAL,
    sma20       REAL,
    prior_high  REAL,
    change_20d  REAL,
    entry       REAL,
    stop        REAL,
    t1          REAL,
    t2          REAL,
    rr          REAL,
    volume_ratio REAL,
    earnings_date TEXT,
    PRIMARY KEY (scan_date, symbol)
);
CREATE TABLE IF NOT EXISTS covered_calls (
    run_ts      TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    account     TEXT NOT NULL,
    status      TEXT NOT NULL,
    reason      TEXT,
    free_lots   INTEGER,
    expiry      TEXT,
    strike      REAL,
    bid         REAL,
    delta       REAL,
    notes       TEXT,
    mode        TEXT NOT NULL DEFAULT 'guardrail',
    tier        TEXT,
    net_pct     REAL,
    target_pct  REAL,
    prob_itm    REAL,
    flags       TEXT,
    meets_target INTEGER,
    generated_at TEXT,
    data_as_of   TEXT,
    source       TEXT,
    PRIMARY KEY (run_ts, symbol, account, mode)
);
CREATE TABLE IF NOT EXISTS option_recs (
    run_ts          TEXT NOT NULL,
    family          TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    status          TEXT NOT NULL,
    reason          TEXT,
    expiry_type     TEXT,
    expiry          TEXT,
    strike          REAL,
    bid             REAL,
    delta           REAL,
    premium         REAL,
    collateral      REAL,
    margin          REAL,
    return_pct      REAL,
    annualized      REAL,
    effective_cost  REAL,
    max_loss        REAL,
    high_risk       INTEGER,
    notes           TEXT,
    ask             REAL,
    cost            REAL,
    multiple        REAL,
    trigger         TEXT,
    entry_rule      TEXT,
    stop_rule       TEXT,
    target_rule     TEXT,
    entry_level     REAL,
    stop_level      REAL,
    stop_option     REAL,
    t1              REAL,
    t2              REAL,
    PRIMARY KEY (run_ts, family, symbol, expiry_type)
);
CREATE TABLE IF NOT EXISTS lots (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol            TEXT NOT NULL,
    account           TEXT NOT NULL,
    shares            INTEGER NOT NULL,
    cost              REAL NOT NULL,
    purchase_date     TEXT,
    strategy          TEXT,
    no_covered_calls  INTEGER NOT NULL DEFAULT 0,
    note              TEXT
);
CREATE TABLE IF NOT EXISTS short_options (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol            TEXT NOT NULL,
    opt_type          TEXT NOT NULL,
    strike            REAL NOT NULL,
    expiry            TEXT NOT NULL,
    qty               INTEGER NOT NULL,
    net_credit_total  REAL,
    note              TEXT
);
CREATE TABLE IF NOT EXISTS long_options (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol     TEXT NOT NULL,
    opt_type   TEXT NOT NULL,
    strike     REAL NOT NULL,
    expiry     TEXT NOT NULL,
    qty        INTEGER NOT NULL,
    avg_price  REAL,
    note       TEXT
);
CREATE TABLE IF NOT EXISTS settings (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS longterm_recs (
    run_ts       TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    status       TEXT NOT NULL,
    reason       TEXT,
    spot         REAL,
    entry        REAL,
    hold_months  INTEGER,
    stop         REAL,
    t1           REAL,
    t2           REAL,
    quality      INTEGER,
    trend        INTEGER,
    PRIMARY KEY (run_ts, symbol)
);
CREATE TABLE IF NOT EXISTS cc_snapshots (
    ts            TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    action        TEXT NOT NULL,
    strike        REAL,
    expiry        TEXT,
    contracts     INTEGER,
    spot          REAL,
    mid           REAL,
    delta         REAL,
    extrinsic     REAL,
    distance_pct  REAL,
    dte           INTEGER,
    reason        TEXT,
    detail        TEXT
);
CREATE INDEX IF NOT EXISTS idx_cc_snapshots_ts ON cc_snapshots (ts);
CREATE TABLE IF NOT EXISTS whale_flow (
    ts            TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    right         TEXT NOT NULL,
    expiry        TEXT NOT NULL,
    dte           INTEGER,
    strike        REAL,
    spot          REAL,
    otm_pct       REAL,
    volume        INTEGER,
    open_interest INTEGER,
    vol_oi        REAL,
    price         REAL,
    premium       REAL,
    side          TEXT,
    lean          TEXT,
    tier          TEXT,
    delta         REAL,
    iv            REAL,
    fresh         INTEGER
);
CREATE INDEX IF NOT EXISTS idx_whale_flow_ts ON whale_flow (ts);
CREATE TABLE IF NOT EXISTS whale_last (
    symbol  TEXT NOT NULL,
    right   TEXT NOT NULL,
    expiry  TEXT NOT NULL,
    strike  REAL NOT NULL,
    volume  INTEGER NOT NULL,
    ts      TEXT NOT NULL,
    PRIMARY KEY (symbol, right, expiry, strike)
);
CREATE TABLE IF NOT EXISTS signals (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_date    TEXT NOT NULL,
    tab            TEXT NOT NULL,
    symbol         TEXT NOT NULL,
    setup          TEXT NOT NULL,
    status         TEXT NOT NULL,
    excluded_code  TEXT,
    reason         TEXT,
    entry          REAL,
    stop           REAL,
    t1             REAL,
    t2             REAL,
    rr             REAL,
    score          REAL,
    resistance     REAL,
    shares         INTEGER,
    expected_move  REAL,
    expected_move_pct REAL,
    inputs         TEXT,
    generated_at   TEXT,
    data_as_of     TEXT,
    source         TEXT,
    verdict        TEXT,
    verdict_reason TEXT,
    UNIQUE (signal_date, tab, symbol)
);
CREATE TABLE IF NOT EXISTS portfolio_positions (
    as_of            TEXT NOT NULL,
    account          TEXT NOT NULL,
    symbol           TEXT NOT NULL,
    kind             TEXT NOT NULL,
    qty              REAL,
    strike           REAL,
    expiry           TEXT,
    price            REAL,
    option_mid       REAL,
    delta_shares     REAL,
    theta_day        REAL,
    beta             REAL,
    beta_assumed     INTEGER,
    spy_equiv_shares REAL,
    delta_dollars    REAL,
    value            REAL,
    status           TEXT
);
CREATE TABLE IF NOT EXISTS portfolio_info (
    as_of          TEXT NOT NULL,
    symbol         TEXT PRIMARY KEY,
    industry       TEXT,
    next_earnings  TEXT,
    ex_div         TEXT,
    dividend       REAL
);
CREATE TABLE IF NOT EXISTS portfolio_meta (
    as_of      TEXT NOT NULL,
    spy_price  REAL
);
CREATE TABLE IF NOT EXISTS holding_review (
    as_of         TEXT NOT NULL,
    symbol        TEXT PRIMARY KEY,
    shares        REAL,
    price         REAL,
    value         REAL,
    weight        REAL,
    gain_pct      REAL,
    rev_growth    REAL,
    eps_growth    REAL,
    forward_pe    REAL,
    pe_5y         REAL,
    pe_vs_avg     REAL,
    fcf_margin    REAL,
    drawdown      REAL,
    dist_sma200   REAL,
    quality       INTEGER,
    trend         INTEGER,
    conviction    REAL,
    fund          INTEGER,
    concentrated  INTEGER
);
CREATE TABLE IF NOT EXISTS holding_pairs (
    as_of  TEXT NOT NULL,
    a      TEXT NOT NULL,
    b      TEXT NOT NULL,
    corr   REAL NOT NULL,
    PRIMARY KEY (a, b)
);
CREATE TABLE IF NOT EXISTS iv_history (
    symbol  TEXT NOT NULL,
    day     TEXT NOT NULL,
    atm_iv  REAL NOT NULL,
    PRIMARY KEY (symbol, day)
);
CREATE TABLE IF NOT EXISTS company_size (
    symbol      TEXT PRIMARY KEY,
    market_cap  REAL,
    updated     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS bought_options (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    bought_at     TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    opt_type      TEXT NOT NULL,
    strike        REAL NOT NULL,
    expiry        TEXT NOT NULL,
    qty           INTEGER NOT NULL,
    price         REAL NOT NULL,
    family        TEXT,
    entry_level   REAL,
    stop_level    REAL,
    stop_option   REAL,
    t1            REAL,
    t2            REAL,
    status        TEXT NOT NULL DEFAULT 'open',
    closed_at     TEXT,
    closed_price  REAL,
    last_mid      REAL,
    last_spot     REAL,
    marked_at     TEXT
);
CREATE TABLE IF NOT EXISTS alerts (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT NOT NULL,
    day     TEXT NOT NULL,
    symbol  TEXT NOT NULL,
    kind    TEXT NOT NULL,
    detail  TEXT NOT NULL,
    UNIQUE (day, symbol, kind)
);
"""


COVERED_CALL_NEW_COLUMNS = [
    ("mode", "TEXT"), ("tier", "TEXT"), ("net_pct", "REAL"), ("target_pct", "REAL"), ("prob_itm", "REAL"),
    ("flags", "TEXT"), ("meets_target", "INTEGER"),
]
CC_SNAPSHOT_NEW_COLUMNS = [("rolls", "TEXT")]
OPTION_ACCOUNT_COLUMNS = [("account", "TEXT")]
WHALE_NEW_COLUMNS = [("intent", "TEXT"), ("burst_volume", "INTEGER"), ("burst_premium", "REAL"), ("noise", "TEXT")]
PROVENANCE_COLUMNS = [("generated_at", "TEXT"), ("data_as_of", "TEXT"), ("source", "TEXT")]
PROVENANCE_TABLES = ("setups", "option_recs", "covered_calls", "longterm_recs", "whale_flow", "cc_snapshots")
SETUP_NEW_COLUMNS = [
    ("volume_ratio", "REAL"), ("earnings_date", "TEXT"), ("rsi", "REAL"), ("resistance", "REAL"), ("score", "REAL"),
    ("excluded", "TEXT"), ("last_reported", "TEXT"), ("post_earnings", "INTEGER"), ("earnings_known", "INTEGER"),
    ("earnings_hour", "TEXT"), ("last_hour", "TEXT"),
    ("expected_move", "REAL"), ("expected_move_pct", "REAL"), ("em_expiry", "TEXT"),
    ("atr14", "REAL"), ("rechecked_at", "TEXT"),
    ("excluded_code", "TEXT"), ("shares", "INTEGER"), ("position_value", "REAL"), ("risk_dollars", "REAL"),
    ("flag_broken", "INTEGER"),
]
OPTION_REC_NEW_COLUMNS = [
    ("expected_move", "REAL"), ("expected_move_pct", "REAL"), ("breakeven", "REAL"), ("breakeven_pct", "REAL"),
    ("be_vs_move", "REAL"), ("theta_pct", "REAL"), ("iv", "REAL"), ("extrinsic_pct", "REAL"), ("contracts", "INTEGER"),
    ("flags", "TEXT"), ("idea_source", "TEXT"),
    ("dte", "INTEGER"), ("support", "REAL"), ("support_label", "TEXT"), ("rank", "REAL"), ("rank_source", "TEXT"),
    ("break_even", "REAL"), ("prob_otm", "REAL"),
    ("ask", "REAL"), ("cost", "REAL"), ("multiple", "REAL"), ("trigger", "TEXT"),
    ("entry_rule", "TEXT"), ("stop_rule", "TEXT"), ("target_rule", "TEXT"),
    ("entry_level", "REAL"), ("stop_level", "REAL"), ("stop_option", "REAL"), ("t1", "REAL"), ("t2", "REAL"),
]


def _now_iso() -> str:
    return datetime.now(NY).isoformat(timespec="seconds")


def _bar_date(index_value: object) -> str:
    ts = pd.Timestamp(index_value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(NY)
    return ts.date().isoformat()


def _opt(value: object) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


class Store:
    def __init__(self, path: Path | str = DEFAULT_DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.executescript(SCHEMA)
        migrations = [("option_recs", OPTION_REC_NEW_COLUMNS), ("setups", SETUP_NEW_COLUMNS)]
        migrations += [(table, PROVENANCE_COLUMNS) for table in PROVENANCE_TABLES]
        self._rebuild_covered_calls_key()
        migrations += [("covered_calls", COVERED_CALL_NEW_COLUMNS), ("cc_snapshots", CC_SNAPSHOT_NEW_COLUMNS)]
        migrations += [("whale_flow", WHALE_NEW_COLUMNS)]
        migrations += [("short_options", OPTION_ACCOUNT_COLUMNS), ("long_options", OPTION_ACCOUNT_COLUMNS)]
        for table, new_columns in migrations:
            existing = {row[1] for row in self._conn.execute(f"PRAGMA table_info({table})")}
            for column, kind in new_columns:
                if column not in existing:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
        self._conn.commit()

    def _rebuild_covered_calls_key(self) -> None:
        """One-time migration: covered_calls once had PRIMARY KEY (run_ts, symbol, account); an income and a guardrail
        row for the same position need `mode` in the key too."""
        pk = {r[1] for r in self._conn.execute("PRAGMA table_info(covered_calls)") if r[5]}
        if "mode" in pk:
            return
        old = [r[1] for r in self._conn.execute("PRAGMA table_info(covered_calls)")]
        self._conn.execute("ALTER TABLE covered_calls RENAME TO covered_calls_old")
        self._conn.executescript(SCHEMA.split("CREATE TABLE IF NOT EXISTS covered_calls", 1)[1].split(");", 1)[0].join(
            ["CREATE TABLE IF NOT EXISTS covered_calls", ");"]))
        new = {r[1] for r in self._conn.execute("PRAGMA table_info(covered_calls)")}
        common = [c for c in old if c in new]
        columns = ", ".join(common)
        select = ", ".join("COALESCE(mode, 'guardrail')" if c == "mode" else c for c in common)
        self._conn.execute(f"INSERT INTO covered_calls ({columns}) SELECT {select} FROM covered_calls_old")
        self._conn.execute("DROP TABLE covered_calls_old")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def save_daily_bars(self, symbol: str, bars: pd.DataFrame, source: str) -> int:
        """Save completed bars. Today's bar is skipped until the session is over (16:15 ET), so the database never
        holds an intraday snapshot that could later pass for a finished bar."""
        now = datetime.now(NY)
        fetched = now.isoformat(timespec="seconds")
        if now.time() < MARKET_CLOSE:
            bars = bars[[_bar_date(i) < now.date().isoformat() for i in bars.index]]
        rows = [
            (
                symbol,
                _bar_date(idx),
                _opt(r["open"]),
                _opt(r["high"]),
                _opt(r["low"]),
                _opt(r["close"]),
                _opt(r["volume"]),
                source,
                fetched,
            )
            for idx, r in bars.iterrows()
        ]
        self._conn.executemany("INSERT OR REPLACE INTO daily_bars VALUES (?,?,?,?,?,?,?,?,?)", rows)
        self._conn.commit()
        return len(rows)

    def load_trusted_daily_bars(self, symbol: str) -> pd.DataFrame:
        """Bars that were saved after their session ended, up to (not including) the first one that was not.

        Rows saved intraday by older code are therefore refetched instead of being served as completed bars.
        """
        df = pd.read_sql_query(
            "SELECT bar_date, open, high, low, close, volume, fetched_at FROM daily_bars WHERE symbol = ? "
            "ORDER BY bar_date",
            self._conn, params=(symbol,),
        )
        if df.empty:
            return df.drop(columns="fetched_at").set_index(pd.DatetimeIndex([], name="bar_date"))
        fetched_day, fetched_time = df["fetched_at"].str[:10], df["fetched_at"].str[11:16]
        close = MARKET_CLOSE.strftime("%H:%M")
        trusted = (fetched_day > df["bar_date"]) | ((fetched_day == df["bar_date"]) & (fetched_time >= close))
        bad = (~trusted).to_numpy().nonzero()[0]
        if len(bad):
            df = df.iloc[: bad[0]]
        out = df.drop(columns="fetched_at").set_index("bar_date")
        out.index = pd.to_datetime(out.index)
        return out

    def load_daily_bars(self, symbol: str) -> pd.DataFrame:
        df = pd.read_sql_query(
            "SELECT bar_date, open, high, low, close, volume FROM daily_bars WHERE symbol = ? ORDER BY bar_date",
            self._conn,
            params=(symbol,),
            index_col="bar_date",
        )
        df.index = pd.to_datetime(df.index)
        return df

    def save_spot(self, symbol: str, price: float, source: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO spot_snapshots VALUES (?,?,?,?)",
            (symbol, _now_iso(), float(price), source),
        )
        self._conn.commit()

    def save_option_chain(self, symbol: str, chain: pd.DataFrame, source: str) -> int:
        ts = _now_iso()
        rows = [
            (
                symbol,
                ts,
                pd.Timestamp(r["expiry"]).date().isoformat(),
                float(r["strike"]),
                r["right"],
                _opt(r["bid"]),
                _opt(r["ask"]),
                _opt(r["mid"]),
                _opt(r["delta"]),
                _opt(r["iv"]),
                _opt(r["open_interest"]),
                source,
            )
            for _, r in chain.iterrows()
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO option_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows
        )
        self._conn.commit()
        return len(rows)

    def save_alert(self, symbol: str, kind: str, detail: str, day: date | None = None) -> bool:
        """Insert an alert once per (day, symbol, kind). Returns True if it was new."""
        day_str = (day or datetime.now(NY).date()).isoformat()
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO alerts (ts, day, symbol, kind, detail) VALUES (?,?,?,?,?)",
            (_now_iso(), day_str, symbol, kind, detail),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def recent_alerts(self, limit: int = 200) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT ts, symbol, kind, detail FROM alerts ORDER BY id DESC LIMIT ?",
            self._conn,
            params=(limit,),
        )

    def save_setups(self, setups: list[object], day: date | None = None) -> int:
        day_str = (day or datetime.now(NY).date()).isoformat()
        generated = _now_iso()
        rows = [
            (
                day_str,
                s.symbol,
                s.kind,
                _opt(s.spot),
                _opt(s.atr),
                _opt(s.sma20),
                _opt(s.prior_high),
                _opt(s.change_20d),
                _opt(s.entry),
                _opt(s.stop),
                _opt(s.t1),
                _opt(s.t2),
                _opt(s.rr),
                _opt(getattr(s, "volume_ratio", None)),
                s.earnings.isoformat() if getattr(s, "earnings", None) else None,
                _opt(getattr(s, "rsi", None)),
                _opt(getattr(s, "resistance", None)),
                _opt(getattr(s, "score", None)),
                getattr(s, "excluded", None),
                s.last_reported.isoformat() if getattr(s, "last_reported", None) else None,
                int(bool(getattr(s, "post_earnings", False))),
                int(bool(getattr(s, "earnings_known", True))),
                getattr(s, "earnings_hour", None),
                getattr(s, "last_hour", None),
                _opt(getattr(s, "expected_move", None)),
                _opt(getattr(s, "expected_move_pct", None)),
                s.em_expiry.isoformat() if getattr(s, "em_expiry", None) else None,
                _opt(getattr(s, "atr14", None)),
                getattr(s, "excluded_code", None),
                getattr(s, "shares", None),
                _opt(getattr(s, "position_value", None)),
                _opt(getattr(s, "risk_dollars", None)),
                None if getattr(s, "flag_broken", None) is None else int(bool(s.flag_broken)),
                generated,
                s.data_as_of.isoformat() if getattr(s, "data_as_of", None) else None,
                "schwab,finnhub",
            )
            for s in setups
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO setups (scan_date, symbol, kind, spot, atr, sma20, prior_high, change_20d, entry, "
            "stop, t1, t2, rr, volume_ratio, earnings_date, rsi, resistance, score, excluded, last_reported, "
            "post_earnings, earnings_known, earnings_hour, last_hour, expected_move, expected_move_pct, em_expiry, "
            "atr14, excluded_code, shares, position_value, risk_dollars, flag_broken, generated_at, data_as_of, source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def save_signals(self, rows: list[dict]) -> int:
        """Log recommendations (recommended and excluded) with every input. One row per date, tab and symbol; a re-run
        updates the row but never overwrites a verdict recorded on it."""
        cols = ["signal_date", "tab", "symbol", "setup", "status", "excluded_code", "reason", "entry", "stop", "t1", "t2",
                "rr", "score", "resistance", "shares", "expected_move", "expected_move_pct", "inputs", "generated_at",
                "data_as_of", "source"]
        updates = ", ".join(f"{c} = excluded.{c}" for c in cols[3:])
        self._conn.executemany(
            f"INSERT INTO signals ({', '.join(cols)}) VALUES ({','.join('?' for _ in cols)}) "
            f"ON CONFLICT (signal_date, tab, symbol) DO UPDATE SET {updates}",
            [tuple(row.get(c) for c in cols) for row in rows],
        )
        self._conn.commit()
        return len(rows)

    def update_signal_status(self, signal_date: str, tab: str, symbol: str, setup: str, status: str,
                             excluded_code: str | None, reason: str | None, rr: float | None, score: float | None) -> None:
        self._conn.execute(
            "UPDATE signals SET setup = ?, status = ?, excluded_code = ?, reason = ?, rr = ?, score = ? "
            "WHERE signal_date = ? AND tab = ? AND symbol = ?",
            (setup, status, excluded_code, reason, _opt(rr), _opt(score), signal_date, tab, symbol),
        )
        self._conn.commit()

    def signals(self, tab: str | None = None, since: str | None = None) -> pd.DataFrame:
        sql, params = "SELECT * FROM signals WHERE 1 = 1", []
        if tab:
            sql, params = sql + " AND tab = ?", [*params, tab]
        if since:
            sql, params = sql + " AND signal_date >= ?", [*params, since]
        return pd.read_sql_query(sql + " ORDER BY signal_date, symbol", self._conn, params=params)

    def update_setup_recheck(self, scan_date: str, symbol: str, update: dict, rechecked_at: str) -> None:
        """Apply a live re-check to one saved setup: new spot, R:R, score and (possibly) failed/excluded."""
        self._conn.execute(
            "UPDATE setups SET kind = ?, excluded = ?, excluded_code = ?, rr = ?, score = ?, spot = ?, rechecked_at = ? "
            "WHERE scan_date = ? AND symbol = ?",
            (update["kind"], update["excluded"], update.get("excluded_code"), _opt(update["rr"]),
             _opt(update["score"]), _opt(update["spot"]), rechecked_at, scan_date, symbol),
        )
        self._conn.commit()

    def latest_setups(self) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT s.* FROM setups s JOIN (SELECT symbol, MAX(scan_date) AS d FROM setups GROUP BY symbol) m "
            "ON s.symbol = m.symbol AND s.scan_date = m.d",
            self._conn,
        )

    def save_covered_calls(self, recs: list[object]) -> int:
        run_ts = _now_iso()
        rows = [
            (
                run_ts,
                r.symbol,
                r.account,
                r.status,
                r.reason,
                r.free_lots,
                r.expiry.isoformat() if r.expiry else None,
                _opt(r.strike),
                _opt(r.bid),
                _opt(r.delta),
                " · ".join(r.notes),
                getattr(r, "mode", "guardrail"),
                getattr(r, "tier", None),
                _opt(getattr(r, "net_pct", None)),
                _opt(getattr(r, "target_pct", None)),
                _opt(getattr(r, "prob_itm", None)),
                " · ".join(getattr(r, "flags", []) or []),
                None if getattr(r, "meets_target", None) is None else int(bool(r.meets_target)),
            )
            for r in recs
        ]
        self._insert_with_provenance("covered_calls", rows, run_ts, run_ts, "schwab,finnhub")
        self._conn.commit()
        return len(rows)

    def latest_covered_calls(self) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM covered_calls WHERE run_ts = (SELECT MAX(run_ts) FROM covered_calls) ORDER BY status, symbol",
            self._conn,
        )

    def save_option_recs(self, family_recs: list[tuple[str, list[object]]]) -> int:
        run_ts = _now_iso()
        columns = (
            "run_ts, family, symbol, status, reason, expiry_type, expiry, strike, bid, delta, premium, collateral, "
            "margin, return_pct, annualized, effective_cost, max_loss, high_risk, notes, ask, cost, multiple, "
            "trigger, entry_rule, stop_rule, target_rule, entry_level, stop_level, stop_option, t1, t2, "
            "generated_at, data_as_of, source, dte, support, support_label, rank, rank_source, break_even, prob_otm, "
            "expected_move, expected_move_pct, breakeven, breakeven_pct, be_vs_move, theta_pct, iv, extrinsic_pct, "
            "contracts, flags, idea_source"
        )
        rows = []
        for family, recs in family_recs:
            for r in recs:
                def g(name: str, default: object = None, rec: object = r) -> object:
                    return getattr(rec, name, default)

                rows.append(
                    (
                        run_ts, family, r.symbol, r.status, r.reason, r.expiry_type,
                        r.expiry.isoformat() if r.expiry else None,
                        _opt(r.strike), _opt(g("bid")), _opt(r.delta),
                        _opt(g("premium_per_contract")), _opt(g("collateral_per_contract")),
                        _opt(g("margin_per_contract")),
                        _opt(g("return_on_margin") or g("return_on_collateral")),
                        _opt(g("annualized_return")), _opt(g("effective_cost")), _opt(g("max_loss_per_contract")),
                        int(g("high_risk", False)), " · ".join(r.notes),
                        _opt(g("ask")), _opt(g("cost_per_contract")), _opt(g("payoff_multiple")), g("trigger"),
                        g("entry_rule"), g("stop_rule"), g("target_rule"),
                        _opt(g("entry_level")), _opt(g("stop_level")), _opt(g("stop_option")),
                        _opt(g("t1")), _opt(g("t2")),
                        run_ts, run_ts, "schwab",
                        g("dte"), _opt(g("support")), g("support_label"), _opt(g("rank")), g("rank_source"),
                        _opt(g("break_even")), _opt(g("prob_otm")),
                        _opt(g("expected_move")), _opt(g("expected_move_pct")), _opt(g("breakeven")),
                        _opt(g("breakeven_pct")), _opt(g("be_vs_move")), _opt(g("theta_pct")), _opt(g("iv")),
                        _opt(g("extrinsic_pct")), g("contracts"), " · ".join(g("flags", []) or []), g("idea_source"),
                    )
                )
        placeholders = ",".join("?" * 52)
        self._conn.executemany(
            f"INSERT OR REPLACE INTO option_recs ({columns}) VALUES ({placeholders})", rows
        )
        self._conn.commit()
        return len(rows)

    def save_longterm(self, recs: list[object]) -> int:
        run_ts = _now_iso()
        rows = [
            (run_ts, r.symbol, r.status, r.reason, _opt(r.spot), _opt(r.entry), r.hold_months, _opt(r.stop),
             _opt(r.t1), _opt(r.t2), r.quality, r.trend)
            for r in recs
        ]
        self._insert_with_provenance("longterm_recs", rows, run_ts, run_ts, "schwab", verb="INSERT OR REPLACE")
        self._conn.commit()
        return len(rows)

    def latest_longterm(self) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM longterm_recs WHERE run_ts = (SELECT MAX(run_ts) FROM longterm_recs)", self._conn
        )

    def save_cc_snapshot(self, rows: list[object], keep_days: int = 7) -> int:
        """Save one monitor run and delete snapshots older than `keep_days`."""
        now = datetime.now(NY)
        ts = now.isoformat(timespec="seconds")
        data = [
            (ts, r.symbol, r.action, _opt(r.strike), r.expiry.isoformat() if r.expiry else None, r.contracts,
             _opt(r.spot), _opt(r.mid), _opt(r.delta), _opt(r.extrinsic), _opt(r.distance_pct), r.dte, r.reason,
             r.detail, json.dumps([asdict(x) | {"expiry": x.expiry.isoformat()} for x in getattr(r, "rolls", [])]))
            for r in rows
        ]
        self._insert_with_provenance("cc_snapshots", data, ts, ts, "schwab,finnhub")
        cutoff = (now - timedelta(days=keep_days)).isoformat(timespec="seconds")
        self._conn.execute("DELETE FROM cc_snapshots WHERE ts < ?", (cutoff,))
        self._conn.commit()
        return len(data)

    def latest_cc_snapshot(self) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM cc_snapshots WHERE ts = (SELECT MAX(ts) FROM cc_snapshots)", self._conn
        )

    def _insert_with_provenance(
        self, table: str, rows: list[tuple], generated_at: str, data_as_of: str, source: str, verb: str = "INSERT"
    ) -> None:
        """Positional rows for the table's original columns, plus generated_at / data_as_of / source."""
        if not rows:
            return
        names = [r[1] for r in self._conn.execute(f"PRAGMA table_info({table})")]
        base = [n for n in names if n not in ("generated_at", "data_as_of", "source")][: len(rows[0])]
        columns = ", ".join([*base, "generated_at", "data_as_of", "source"])
        marks = ",".join("?" for _ in range(len(base) + 3))
        self._conn.executemany(
            f"{verb} INTO {table} ({columns}) VALUES ({marks})",
            [(*row, generated_at, data_as_of, source) for row in rows],
        )

    def save_holding_review(self, reviews: list, pairs: pd.DataFrame, as_of: str | None = None) -> int:
        """Replace the saved holdings review and correlated pairs with this run's."""
        as_of = as_of or _now_iso()
        self._conn.execute("DELETE FROM holding_review")
        self._conn.execute("DELETE FROM holding_pairs")
        self._conn.executemany(
            "INSERT INTO holding_review VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(as_of, r.symbol, r.shares, _opt(r.price), _opt(r.value), _opt(r.weight), _opt(r.gain_pct), _opt(r.rev_growth),
              _opt(r.eps_growth), _opt(r.forward_pe), _opt(r.pe_5y), _opt(r.pe_vs_avg), _opt(r.fcf_margin), _opt(r.drawdown),
              _opt(r.dist_sma200), r.quality, r.trend, _opt(r.conviction), int(r.fund), int(r.concentrated)) for r in reviews],
        )
        self._conn.executemany(
            "INSERT INTO holding_pairs VALUES (?,?,?,?)",
            [(as_of, p.a, p.b, float(p.corr)) for p in pairs.itertuples()],
        )
        self._conn.commit()
        return len(reviews)

    def save_portfolio(self, positions: pd.DataFrame, info: pd.DataFrame, spy_price: float, as_of: str | None = None) -> int:
        """Replace the saved portfolio-risk snapshot (positions, per-symbol info, the SPY price used)."""
        as_of = as_of or _now_iso()
        for table in ("portfolio_positions", "portfolio_info", "portfolio_meta"):
            self._conn.execute(f"DELETE FROM {table}")
        cols = ["account", "symbol", "kind", "qty", "strike", "expiry", "price", "option_mid", "delta_shares", "theta_day",
                "beta", "beta_assumed", "spy_equiv_shares", "delta_dollars", "value", "status"]
        self._conn.executemany(
            f"INSERT INTO portfolio_positions VALUES (?,{','.join('?' for _ in cols)})",
            [(as_of, *[None if pd.isna(r[c]) else (int(r[c]) if c == "beta_assumed" else r[c]) for c in cols])
             for _, r in positions.iterrows()],
        )
        self._conn.executemany(
            "INSERT INTO portfolio_info VALUES (?,?,?,?,?,?)",
            [(as_of, r["symbol"], r.get("industry"), r.get("next_earnings"), r.get("ex_div"), _opt(r.get("dividend")))
             for _, r in info.iterrows()],
        )
        self._conn.execute("INSERT INTO portfolio_meta VALUES (?,?)", (as_of, float(spy_price)))
        self._conn.commit()
        return len(positions)

    def latest_portfolio(self) -> tuple[pd.DataFrame, pd.DataFrame, str | None, float | None]:
        """(positions, per-symbol info, as_of, SPY price) from the last portfolio run."""
        positions = pd.read_sql_query("SELECT * FROM portfolio_positions", self._conn)
        info = pd.read_sql_query("SELECT * FROM portfolio_info", self._conn)
        meta = self._conn.execute("SELECT as_of, spy_price FROM portfolio_meta").fetchone()
        if not positions.empty:
            positions["beta_assumed"] = positions["beta_assumed"].fillna(0).astype(bool)
        return positions.drop(columns="as_of", errors="ignore"), info.drop(columns="as_of", errors="ignore"), (meta[0] if meta else None), (meta[1] if meta else None)

    def latest_holding_review(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM holding_review ORDER BY weight DESC", self._conn)

    def latest_holding_pairs(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM holding_pairs ORDER BY corr DESC", self._conn)

    def save_iv(self, day: date, ivs: dict[str, float]) -> int:
        """One at-the-money implied-vol reading per symbol per day (the base for IV rank)."""
        rows = [(s, day.isoformat(), float(v)) for s, v in ivs.items() if v is not None and not pd.isna(v) and v > 0]
        self._conn.executemany("INSERT OR REPLACE INTO iv_history VALUES (?,?,?)", rows)
        self._conn.commit()
        return len(rows)

    def iv_history(self, symbol: str) -> list[float]:
        return [r[0] for r in self._conn.execute(
            "SELECT atm_iv FROM iv_history WHERE symbol = ? ORDER BY day", (symbol,))]

    def iv_symbols_on(self, day: date) -> set[str]:
        return {r[0] for r in self._conn.execute("SELECT symbol FROM iv_history WHERE day = ?", (day.isoformat(),))}

    def save_company_sizes(self, caps: dict[str, float | None]) -> int:
        """Remember each symbol's market cap (dollars). Symbols without one (ETFs) are skipped."""
        now = _now_iso()
        rows = [(s, float(c), now) for s, c in caps.items() if c is not None and not pd.isna(c) and c > 0]
        self._conn.executemany("INSERT OR REPLACE INTO company_size VALUES (?,?,?)", rows)
        self._conn.commit()
        return len(rows)

    def company_sizes(self) -> dict[str, float]:
        return dict(self._conn.execute("SELECT symbol, market_cap FROM company_size").fetchall())

    def add_bought(self, symbol: str, opt_type: str, strike: float, expiry: str, qty: int, price: float,
                   family: str | None = None, entry_level: float | None = None, stop_level: float | None = None,
                   stop_option: float | None = None, t1: float | None = None, t2: float | None = None) -> int:
        """Log an option you bought, with the idea's levels so it can be tracked against them."""
        cur = self._conn.execute(
            "INSERT INTO bought_options (bought_at, symbol, opt_type, strike, expiry, qty, price, family, entry_level, "
            "stop_level, stop_option, t1, t2) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (_now_iso(), symbol, opt_type, float(strike), expiry, int(qty), float(price), family, _opt(entry_level),
             _opt(stop_level), _opt(stop_option), _opt(t1), _opt(t2)),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def bought_options(self, status: str | None = None) -> pd.DataFrame:
        if status:
            return pd.read_sql_query(
                "SELECT * FROM bought_options WHERE status = ? ORDER BY expiry, symbol", self._conn, params=(status,)
            )
        return pd.read_sql_query("SELECT * FROM bought_options ORDER BY status DESC, expiry, symbol", self._conn)

    def mark_bought(self, option_id: int, mid: float | None, spot: float | None) -> None:
        self._conn.execute(
            "UPDATE bought_options SET last_mid = ?, last_spot = ?, marked_at = ? WHERE id = ?",
            (_opt(mid), _opt(spot), _now_iso(), option_id),
        )
        self._conn.commit()

    def close_bought(self, option_id: int, price: float) -> None:
        self._conn.execute(
            "UPDATE bought_options SET status = 'closed', closed_at = ?, closed_price = ? WHERE id = ?",
            (_now_iso(), float(price), option_id),
        )
        self._conn.commit()

    def delete_bought(self, option_id: int) -> None:
        self._conn.execute("DELETE FROM bought_options WHERE id = ?", (option_id,))
        self._conn.commit()

    def save_whales(self, rows: list[object], keep_days: int = 7) -> int:
        """Save one whale scan and delete scans older than `keep_days`."""
        now = datetime.now(NY)
        ts = now.isoformat(timespec="seconds")
        data = [
            (ts, r.symbol, r.right, r.expiry.isoformat(), r.dte, r.strike, _opt(r.spot), _opt(r.otm_pct), r.volume,
             r.open_interest, _opt(r.vol_oi), _opt(r.price), _opt(r.premium), r.side, r.lean, r.tier, _opt(r.delta),
             _opt(r.iv), int(r.fresh), getattr(r, "intent", "unknown"), getattr(r, "burst_volume", None),
             _opt(getattr(r, "burst_premium", None)), getattr(r, "noise", None))
            for r in rows
        ]
        self._insert_with_provenance("whale_flow", data, ts, ts, "schwab")
        cutoff = (now - timedelta(days=keep_days)).isoformat(timespec="seconds")
        self._conn.execute("DELETE FROM whale_flow WHERE ts < ?", (cutoff,))
        self._conn.commit()
        return len(data)

    def save_whale_snapshot(self, snapshot: list[tuple], ts: str | None = None) -> int:
        """Replace the previous scan's contract volumes with this scan's, so the next scan can see what was added."""
        ts = ts or _now_iso()
        self._conn.execute("DELETE FROM whale_last")
        self._conn.executemany("INSERT OR REPLACE INTO whale_last VALUES (?,?,?,?,?,?)", [(*row, ts) for row in snapshot])
        self._conn.commit()
        return len(snapshot)

    def load_whale_snapshot(self, now: datetime, max_gap_minutes: int = 45) -> dict | None:
        """Volumes from the last scan as {(symbol, right, expiry, strike): volume}, or None when there is no scan from the
        same day within `max_gap_minutes` (a burst needs a comparable previous scan)."""
        rows = self._conn.execute("SELECT symbol, right, expiry, strike, volume, ts FROM whale_last").fetchall()
        if not rows:
            return None
        ts = datetime.fromisoformat(rows[0][5])
        if ts.astimezone(NY).date() != now.astimezone(NY).date() or (now - ts).total_seconds() > max_gap_minutes * 60:
            return None
        return {(s, r, e, k): v for s, r, e, k, v, _ in rows}

    def whale_history(self, days: int = 5) -> pd.DataFrame:
        """Every saved scan row from the last `days` distinct days that have scans."""
        return pd.read_sql_query(
            "SELECT * FROM whale_flow WHERE substr(ts, 1, 10) IN "
            "(SELECT DISTINCT substr(ts, 1, 10) FROM whale_flow ORDER BY 1 DESC LIMIT ?)",
            self._conn, params=(days,),
        )

    def latest_whales(self) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM whale_flow WHERE ts = (SELECT MAX(ts) FROM whale_flow)", self._conn
        )

    def whale_runs(self, limit: int = 40) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT ts, COUNT(*) AS rows, SUM(premium) AS premium FROM whale_flow GROUP BY ts ORDER BY ts DESC LIMIT ?",
            self._conn, params=(limit,),
        )

    def cc_history(self, symbol: str) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM cc_snapshots WHERE symbol = ? ORDER BY ts DESC", self._conn, params=(symbol,)
        )

    def latest_option_recs(self, family: str) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM option_recs WHERE family = ? AND run_ts = "
            "(SELECT MAX(run_ts) FROM option_recs WHERE family = ?) ORDER BY status, symbol, expiry",
            self._conn,
            params=(family, family),
        )

    def save_holdings(self, h: Holdings) -> None:
        """Replace the stored portfolio with `h` in one transaction."""
        with self._conn:
            for table in ("lots", "short_options", "long_options"):
                self._conn.execute(f"DELETE FROM {table}")
            self._conn.executemany(
                "INSERT INTO lots (symbol, account, shares, cost, purchase_date, strategy, no_covered_calls, note) "
                "VALUES (?,?,?,?,?,?,?,?)",
                [
                    (l.symbol, l.account, l.shares, l.cost,
                     l.purchase_date.isoformat() if l.purchase_date else None,
                     l.strategy, int(l.no_covered_calls), l.note)
                    for l in h.lots
                ],
            )
            self._conn.executemany(
                "INSERT INTO short_options (symbol, opt_type, strike, expiry, qty, net_credit_total, note, account) "
                "VALUES (?,?,?,?,?,?,?,?)",
                [(o.symbol, o.type, o.strike, o.expiry.isoformat(), o.qty, o.net_credit_total, o.note, o.account)
                 for o in h.short_options],
            )
            self._conn.executemany(
                "INSERT INTO long_options (symbol, opt_type, strike, expiry, qty, avg_price, note, account) "
                "VALUES (?,?,?,?,?,?,?,?)",
                [(o.symbol, o.type, o.strike, o.expiry.isoformat(), o.qty, o.avg_price, o.note, o.account)
                 for o in h.long_options],
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO settings VALUES ('watchlist_scan', ?)", (json.dumps(h.watchlist_scan),)
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO settings VALUES ('alab_rules', ?)", (json.dumps(h.alab_rules),)
            )

    def load_holdings(self) -> Holdings:
        def day(value: str | None) -> date | None:
            return date.fromisoformat(value) if value else None

        lots = [
            Lot(symbol=r[0], account=r[1], shares=r[2], cost=r[3], purchase_date=day(r[4]),
                strategy=r[5], no_covered_calls=bool(r[6]), note=r[7])
            for r in self._conn.execute(
                "SELECT symbol, account, shares, cost, purchase_date, strategy, no_covered_calls, note "
                "FROM lots ORDER BY id"
            )
        ]
        shorts = [
            ShortOption(symbol=r[0], type=r[1], strike=r[2], expiry=date.fromisoformat(r[3]),
                        qty=r[4], net_credit_total=r[5], note=r[6], account=r[7])
            for r in self._conn.execute(
                "SELECT symbol, opt_type, strike, expiry, qty, net_credit_total, note, account FROM short_options ORDER BY id"
            )
        ]
        longs = [
            LongOption(symbol=r[0], type=r[1], strike=r[2], expiry=date.fromisoformat(r[3]),
                       qty=r[4], avg_price=r[5], note=r[6], account=r[7])
            for r in self._conn.execute(
                "SELECT symbol, opt_type, strike, expiry, qty, avg_price, note, account FROM long_options ORDER BY id"
            )
        ]
        settings = dict(self._conn.execute("SELECT key, value FROM settings"))
        return Holdings(
            lots=lots,
            short_options=shorts,
            long_options=longs,
            watchlist_scan=json.loads(settings.get("watchlist_scan", "[]")),
            alab_rules=json.loads(settings.get("alab_rules", "{}")),
        )

    def row_counts(self) -> dict[str, int]:
        tables = ["daily_bars", "spot_snapshots", "option_snapshots", "alerts", "setups", "lots", "short_options", "long_options"]
        return {t: self._conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}


def current_holdings(store: Store | None = None) -> Holdings:
    """The portfolio from the database; on first use, import config/holdings.yaml into it."""
    own = store is None
    store = store or Store()
    try:
        holdings = store.load_holdings()
        if not holdings.lots and DEFAULT_HOLDINGS_PATH.exists():
            holdings = load_holdings_yaml(DEFAULT_HOLDINGS_PATH)
            store.save_holdings(holdings)
        return holdings
    finally:
        if own:
            store.close()


def save_current_holdings(holdings: Holdings) -> None:
    """Save to the database, and mirror to config/holdings.yaml (with .bak) as a readable backup."""
    store = Store()
    try:
        store.save_holdings(holdings)
    finally:
        store.close()
    save_holdings_yaml(holdings)
