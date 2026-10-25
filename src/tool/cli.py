"""CLI entry point: `uv run tool levels MSFT AMZN NVDA`."""

from __future__ import annotations

import argparse
import sys
import time
import pandas as pd
from pathlib import Path
from datetime import date, datetime, timedelta

from tool.buy_options import INDEX_ETFS, MAG7, candidate_symbols, recommend_buys, recommend_special
from tool.cc_monitor import monitor_covered_calls
from tool.config import get_config, load_config
from tool.directional_calls import recommend_directional, recommend_leaps
from tool.ivhistory import make_rank_fn
from tool.support_puts import recommend_support_puts
from tool.earnings_move import attach_expected_moves
from tool.signals import signal_rows
from tool.covered_calls import recommend_covered_calls
from tool.income_calls import recommend_income_calls
from tool.data.earnings import EventsUnavailable, FinnhubEvents, split_reports
from tool.data.ibkr import IBKRDataSource
from tool.data.memo import MemoSource
from tool.data.schwab import REFRESH_TOKEN_DAYS, SchwabAuthRequired, SchwabDataSource, token_age_days
from tool.jobs import JOB_SPECS, JobManager
from tool.levels import NY, LevelsResult, compute_levels
from tool.longterm import MIN_BARS, LongTermRec, recommend_long_term
from tool.naked_calls import default_symbols, recommend_naked_calls
from tool.puts import csp_symbols, recommend_puts, wheel_symbols
from tool.screener import (
    ACTIONABLE,
    EARNINGS_WINDOW_DAYS,
    RANK,
    Setup,
    apply_filters,
    earnings_label,
    evaluate,
    flag_earnings,
    load_universe,
    sell_signals,
)
from tool.holdings import ACCOUNTS
from tool.store import Store, current_holdings, save_current_holdings


def _print_levels(symbol: str, result: LevelsResult) -> None:
    print(f"\n{symbol}")
    if result.spot is not None:
        print(f"  spot:   {result.spot:.2f}")

    d = result.daily
    dp = d.pivots
    print(f"  daily  [{d.bar_date}]  H {d.high:.2f}  L {d.low:.2f}  C {d.close:.2f}")
    print(
        f"         P {dp.p:.2f}  R1 {dp.r1:.2f}  R2 {dp.r2:.2f}  R3 {dp.r3:.2f}"
        f"  S1 {dp.s1:.2f}  S2 {dp.s2:.2f}  S3 {dp.s3:.2f}"
    )

    w = result.weekly
    wp = w.pivots
    print(f"  weekly [{w.week_start} .. {w.week_end}]  H {w.high:.2f}  L {w.low:.2f}  C {w.close:.2f}")
    print(
        f"         P {wp.p:.2f}  R1 {wp.r1:.2f}  R2 {wp.r2:.2f}  R3 {wp.r3:.2f}"
        f"  S1 {wp.s1:.2f}  S2 {wp.s2:.2f}  S3 {wp.s3:.2f}"
    )

    print(
        f"  ATR(10) {result.atr10:.2f}   SMA(20) {result.sma20:.2f}   "
        f"prior 20D high {result.prior_20d_high:.2f}"
    )


def cmd_levels(args: argparse.Namespace) -> int:
    source = IBKRDataSource(port=args.port)
    try:
        source.connect()
    except Exception as exc:
        print(
            f"Could not connect to IB Gateway at 127.0.0.1:{args.port}: {exc}\n"
            "Make sure IB Gateway is running and the API port is enabled.",
            file=sys.stderr,
        )
        return 1

    exit_code = 0
    try:
        for symbol in args.symbols:
            try:
                bars = source.daily_bars(symbol, lookback_days=args.lookback_days)
                spot = source.spot(symbol)
                result = compute_levels(bars, spot=spot, symbol=symbol)
            except Exception as exc:
                print(f"\n{symbol}: ERROR - {exc}", file=sys.stderr)
                exit_code = 1
                continue
            _print_levels(symbol, result)
    finally:
        source.disconnect()
    return exit_code


def earnings_text(s: Setup) -> str:
    return earnings_label(
        s.earnings, s.earnings_hour, s.last_reported, s.last_hour, s.post_earnings, s.earnings_known, date.today()
    )


def cmd_scan(args: argparse.Namespace) -> int:
    source = SchwabDataSource()
    store = Store()
    jobs = JobManager()
    held = current_holdings(store)
    extras = sorted({sym for sym, _ in held.positions()}) + held.watchlist_scan
    if args.symbols:
        universe = args.symbols
    elif args.light:
        universe = list(dict.fromkeys(extras))  # holdings + watchlist only
    else:
        universe = load_universe(extra=extras)
    setups: list[Setup] = []
    errors = 0
    cfg = load_config()

    earnings_by_symbol: dict = {}  # symbol -> (next report date, bmo/amc/dmh)
    reported_by_symbol: dict = {}  # symbol -> (most recent report date, bmo/amc/dmh)
    calendar_ok = False
    today = date.today()
    try:
        # one logical lookup, today-14d .. today+60d, cached on disk for the day (fetched a day at a time)
        events, cached = FinnhubEvents().cached_calendar(today, back=14, ahead=EARNINGS_WINDOW_DAYS)
        earnings_by_symbol, reported_by_symbol = split_reports(events, today)
        calendar_ok = True
        print(
            f"Earnings calendar ({'cache' if cached else 'fetched'}): {len(earnings_by_symbol)} companies report "
            f"in the next {EARNINGS_WINDOW_DAYS} days, {len(reported_by_symbol)} reported in the last 14"
        )
    except Exception as exc:  # no key, no network, or a rate limit: the scan still runs, earnings shown as unknown
        print(f"Earnings calendar unavailable ({exc}); earnings will show as unknown", file=sys.stderr)

    for index in range(args.start_index, len(universe)):
        symbol = universe[index]
        try:
            if index % 20 == 0:  # one batched quote request per 20 symbols instead of one per symbol
                source.prefetch_spots(universe[index:index + 20], ttl=300)
            bars = source.daily_bars(symbol, lookback_days=args.lookback_days)  # cached in SQLite by the data layer
            spot = source.spot(symbol)
            setup = flag_earnings(
                evaluate(symbol, bars, spot, cfg=cfg), earnings_by_symbol, today, reported_by_symbol, calendar_ok,
                cfg.post_earnings_days,
            )
            setups.append(apply_filters(setup, cfg, today, args.allow_earnings, args.show_failed))
        except Exception as exc:
            errors += 1
            print(f"{symbol}: ERROR - {exc}", file=sys.stderr)
        if args.job:
            jobs.write_state(args.job, next_index=index + 1, total=len(universe), errors=errors)
        time.sleep(args.pause)

    em_failures: dict[str, str] = {}
    if calendar_ok and not args.no_expected_move:
        try:
            setups, em_failures = attach_expected_moves(setups, source, cfg, today, args.allow_earnings, args.show_failed)
        except SchwabAuthRequired:
            raise
        n_moves = sum(1 for s in setups if s.expected_move is not None)
        print(f"Expected earnings moves: {n_moves} computed, {len(em_failures)} unavailable")
    store.save_setups(setups)
    store.save_signals(signal_rows(setups, today.isoformat(), cfg, datetime.now(NY).isoformat(timespec="seconds")))
    if args.job:
        jobs.write_state(args.job, status="done", next_index=len(universe), total=len(universe), errors=errors)
    symbols = universe[args.start_index:]

    tradable = [s for s in setups if s.kind in ACTIONABLE or s.kind == "failed"]
    actionable = sorted((s for s in tradable if s.excluded is None), key=lambda s: -(s.score or 0))
    excluded = [s for s in tradable if s.excluded is not None]
    print(f"\nActionable setups: {len(actionable)} of {len(setups)} scanned ({errors} errors, {len(excluded)} excluded)")
    print(
        f"  {'Symbol':6} {'Setup':9} {'Spot':>8} {'Entry':>8} {'Stop':>8} {'T1':>8} {'T2':>8} {'R:R':>5} "
        f"{'Resist':>8} {'Score':>5} {'Vol':>5} {'20D':>6}  Earnings"
    )
    for s in actionable:
        em = f"  move ±{s.expected_move_pct:.1%}" if s.expected_move_pct is not None else ""
        print(
            f"  {s.symbol:6} {s.kind:9} {s.spot:8.2f} {s.entry:8.2f} {s.stop:8.2f} {s.t1:8.2f} {s.t2:8.2f} "
            f"{s.rr if s.rr is not None else float('nan'):5.2f} "
            f"{s.resistance if s.resistance is not None else float('nan'):8.2f} {s.score:5.2f} "
            f"{s.volume_ratio if s.volume_ratio is not None else float('nan'):4.1f}x {s.change_20d:+6.1%}  "
            f"{earnings_text(s)}{em}"
        )
        store.save_alert(s.symbol, s.kind, f"entry {s.entry:.2f} stop {s.stop:.2f} T1 {s.t1:.2f} R:R {s.rr or 0:.2f}")
    if calendar_ok:
        gaps = sorted(
            s.symbol for s in setups
            if s.earnings is None and s.last_reported is None
            and s.symbol not in earnings_by_symbol and s.symbol not in reported_by_symbol
            and s.symbol.replace("/", ".") not in earnings_by_symbol
            and s.symbol.replace("/", ".") not in reported_by_symbol
        )
        full_scan = not args.symbols and not args.light  # a partial scan must not overwrite the full gaps list
        Path("data/cache").mkdir(parents=True, exist_ok=True)
        if full_scan:
            Path("data/cache/earnings_gaps.txt").write_text(
            f"{today.isoformat()}: {len(gaps)} of {len(setups)} scanned symbols have no earnings date from "
            f"{today - timedelta(days=14)} to {today + timedelta(days=EARNINGS_WINDOW_DAYS)}\n" + "\n".join(gaps) + "\n"
            + f"\nExpected move unavailable ({len(em_failures)}):\n"
            + "".join(f"{sym}: {why}\n" for sym, why in sorted(em_failures.items())),
            encoding="utf-8",
        )
        print(
            f"\nEarnings gaps: {len(gaps)} of {len(setups)} symbols have no report in the window "
            f"(listed in data/cache/earnings_gaps.txt): {', '.join(gaps[:25])}{' ...' if len(gaps) > 25 else ''}"
        )
    if excluded:
        print(f"\nExcluded ({len(excluded)}):")
        for s in sorted(excluded, key=lambda s: (s.excluded.split(" ")[0], s.symbol)):
            print(f"  {s.symbol:6} {s.kind:9} {s.excluded}")

    holdings = current_holdings()
    for sym, _account in holdings.positions():
        if sym not in symbols:
            continue
        try:
            signals = sell_signals(sym, store.load_daily_bars(sym), source.spot(sym))
        except SchwabAuthRequired:
            raise
        except Exception as exc:  # e.g. a holding with no quote (delisted or non-optionable): skip it, keep going
            print(f"  {sym}: sell signals skipped ({exc})", file=sys.stderr)
            continue
        for sig in signals:
            print(f"  SELL  {sig.symbol:6} {sig.kind}: {sig.detail}")
            store.save_alert(sig.symbol, sig.kind, sig.detail)

    print(f"\nDatabase rows: {store.row_counts()}")
    store.close()
    return 0 if errors == 0 else 1


def cmd_recheck(args: argparse.Namespace) -> int:
    """Re-check the saved swing setups against live prices: failed breakouts, overextension, fresh R:R."""
    from tool.recheck import recheck_setups

    source = SchwabDataSource()
    store = Store()
    jobs = JobManager()
    result = recheck_setups(store, source, load_config(), datetime.now(NY))
    print(f"Re-checked {result.checked} setups: {len(result.failed)} failed, {len(result.overextended)} stopped or overextended")
    for label, names in (("Failed", result.failed), ("Stopped or overextended", result.overextended), ("No quote", result.no_quote)):
        if names:
            print(f"  {label}: {', '.join(sorted(names))}")
    if args.job:
        jobs.write_state(args.job, status="done", next_index=result.checked, total=result.checked, errors=len(result.no_quote))
    store.close()
    return 0


def cmd_calls(args: argparse.Namespace) -> int:
    holdings = current_holdings()
    source = SchwabDataSource()
    store = Store()
    jobs = JobManager()
    try:
        events = FinnhubEvents()
    except EventsUnavailable:
        events = None

    recs = recommend_covered_calls(holdings, source, events=events)
    recommended = [r for r in recs if r.status == "recommend"]
    print(f"Covered calls: {len(recommended)} recommended of {len(recs)} positions")
    for r in recs:
        if r.status == "recommend":
            detail = f"{r.expiry} {r.strike:.2f} bid {r.bid:.2f} delta {r.delta:.2f} x{r.free_lots} lots"
            print(f"  {r.symbol:6} {r.account:10} {detail}")
            store.save_alert(r.symbol, "covered_call", detail)
        else:
            print(f"  {r.symbol:6} {r.account:10} skip: {r.reason}")
    store.save_covered_calls(recs)

    if args.job:
        jobs.write_state(args.job, status="done", next_index=len(recs), total=len(recs), errors=0)
    store.close()
    return 0


def _summarize(family: str, recs: list[object]) -> None:
    recommended = [r for r in recs if r.status == "recommend"]
    print(f"{family}: {len(recommended)} recommended of {len(recs)} rows")
    for r in recommended:
        print(f"  {r.symbol:6} {r.expiry_type:8} {r.expiry} strike {r.strike:.2f} bid {r.bid:.2f}")


def cmd_options(args: argparse.Namespace) -> int:
    holdings = current_holdings()
    source = SchwabDataSource()
    store = Store()
    jobs = JobManager()
    try:
        events = FinnhubEvents()
    except EventsUnavailable:
        events = None
    now = datetime.now(NY)

    call_symbols = args.symbols or default_symbols(holdings)
    put_symbols = args.symbols or csp_symbols(holdings)
    wheel = wheel_symbols(holdings)

    support_symbols = args.symbols or list(dict.fromkeys(
        [s for syms in get_config().covered_calls.tiers.values() for s in syms] + list(holdings.watchlist_scan) + wheel
    ))
    source.prefetch_spots(list(dict.fromkeys(call_symbols + put_symbols + support_symbols)), ttl=600)
    latest = store.latest_setups()
    kinds = dict(zip(latest["symbol"], latest["kind"])) if not latest.empty else {}
    families = [
        ("csp_support", recommend_support_puts(
            support_symbols, source, now=now, events=events, rank_fn=make_rank_fn(store), setup_kinds=kinds)),
        ("directional_call", recommend_directional(
            latest, source, now=now, events=events, rank_fn=make_rank_fn(store))),
        ("leaps_call", recommend_leaps(
            latest, store.latest_longterm(), source, now=now, events=events, rank_fn=make_rank_fn(store))),
        ("naked_call", recommend_naked_calls(call_symbols, source, now=now, events=events)),
        ("naked_call_breakout", recommend_naked_calls(call_symbols, source, now=now, events=events, breakout_ready=True)),
        ("csp", recommend_puts(put_symbols, "csp", source, now=now, events=events)),
        ("naked_put", recommend_puts(put_symbols, "naked_put", source, now=now, events=events)),
        ("wheel", recommend_puts(wheel, "wheel", source, now=now, events=events) if wheel else []),
    ]
    for family, recs in families:
        _summarize(family, recs)
    saved = store.save_option_recs(families)
    print(f"Saved {saved} option rows")

    if args.job:
        total = sum(len(recs) for _, recs in families)
        jobs.write_state(args.job, status="done", next_index=total, total=total, errors=0)
    store.close()
    return 0


def cmd_buys(args: argparse.Namespace) -> int:
    """Long calls/puts (swing) and weekly lottos, built from the latest swing scan."""
    source = SchwabDataSource()
    store = Store()
    jobs = JobManager()
    holdings = current_holdings(store)
    try:
        events = FinnhubEvents()
    except EventsUnavailable:
        events = None

    if args.symbols:
        symbols = args.symbols
    else:
        calls, puts = candidate_symbols(store.latest_setups(), holdings)
        symbols = list(dict.fromkeys(calls + puts))
    source.prefetch_spots(list(symbols) + list(MAG7 + INDEX_ETFS), ttl=300)
    print(f"Buy ideas for {len(symbols)} symbols")
    families = list(recommend_buys(symbols, source, events=events).items())
    families += list(recommend_special(source, events=events).items())  # 0DTE and MAG7 sections
    for family, recs in families:
        good = [r for r in recs if r.status == "recommend"]
        print(f"{family}: {len(good)} recommended of {len(recs)} rows")
        for r in good:
            print(f"  {r.symbol:6} {r.expiry_type:8} {r.expiry} strike {r.strike:.2f} ask {r.ask:.2f} {r.trigger}")
    saved = store.save_option_recs(families)
    print(f"Saved {saved} buy rows")
    try:  # one ATM-IV reading per symbol per day (a no-op after the first buys run of the day)
        from tool.expiries import monthly_expiry
        from tool.ivhistory import record_atm_iv

        today = datetime.now(NY).date()
        recorded = record_atm_iv(source, store, list(symbols) + MAG7 + INDEX_ETFS, today, monthly_expiry(today))
        if recorded:
            print(f"Recorded ATM implied vol for {recorded} symbols")
    except SchwabAuthRequired:
        raise
    except Exception as exc:
        print(f"IV history not recorded: {exc}", file=sys.stderr)
    from tool.tracker import refresh_marks

    marked, track_errors = refresh_marks(store, source)
    if marked or track_errors:
        print(f"Repriced {marked} bought options, {len(track_errors)} errors")

    if args.job:
        total = sum(len(recs) for _, recs in families)
        jobs.write_state(args.job, status="done", next_index=total, total=total, errors=0)
    store.close()
    return 0


def cmd_whales(args: argparse.Namespace) -> int:
    """Whale activity: large traded premium and fresh volume on 0DTE and weekly option chains."""
    from datetime import datetime as _dt

    from tool.levels import NY as _NY
    from tool.whales import WHALE_UNIVERSE, scan_whales

    source = SchwabDataSource()
    store = Store()
    jobs = JobManager()
    holdings = current_holdings(store)
    held = sorted({sym for sym, _ in holdings.positions()})
    symbols = args.symbols or list(dict.fromkeys(WHALE_UNIVERSE + held + holdings.watchlist_scan[:10]))
    source.prefetch_spots(symbols, ttl=300)
    now = _dt.now(_NY)
    previous = store.load_whale_snapshot(now, get_config().whales.burst_max_gap_minutes)  # None: no burst this scan
    try:  # ex-dividend dates for the dividend-capture noise filter
        fundamentals = source.fundamentals(symbols)
    except SchwabAuthRequired:
        raise
    except Exception as exc:
        print(f"Fundamentals unavailable ({exc}); dividend-capture filter skipped", file=sys.stderr)
        fundamentals = {}
    snapshot: list = []
    rows, errors = scan_whales(symbols, source, now.date(), previous=previous, fundamentals=fundamentals, snapshot=snapshot)
    saved = store.save_whales(rows)
    store.save_whale_snapshot(snapshot, now.isoformat(timespec="seconds"))
    shown = [r for r in rows if r.noise is None]
    print(
        f"Whales: {saved} contracts from {len(symbols)} symbols, {len(errors)} errors; {len(rows) - len(shown)} filtered as "
        f"noise, {sum(r.is_burst for r in rows)} bursts" + ("" if previous is not None else " (no comparable previous scan)")
    )
    rows = shown
    for r in sorted(rows, key=lambda r: -r.premium)[:15]:
        print(f"  {r.symbol:6} {r.right} {r.expiry} {r.strike:8.2f} vol {r.volume:6} ${r.premium/1e6:5.2f}M {r.side:3} {r.lean}")
    for e in errors[:5]:
        print("  error", e)
    if args.job:
        jobs.write_state(args.job, status="done", next_index=len(symbols), total=len(symbols), errors=len(errors))
    store.close()
    return 0


def run_holding_review(source, store, fundamentals: dict, pause: float = 1.1) -> None:
    """Review the holdings (growth, valuation, weight, conviction) and the pairs that move together; saved for the dashboard."""
    from tool.holding_review import correlated_pairs, returns_frame, review_holdings

    holdings = current_holdings(store)
    symbols = sorted({sym for sym, _ in holdings.positions()})
    if not symbols:
        return
    missing = [s for s in symbols if s not in fundamentals]
    if missing:
        fundamentals = {**fundamentals, **source.fundamentals(missing)}
    prices = source.quotes(symbols, ttl=0)
    try:
        metrics = FinnhubEvents().cached_metrics(symbols, date.today())
    except EventsUnavailable as exc:
        print(f"Finnhub metrics unavailable ({exc}); forward P/E and FCF margin left empty", file=sys.stderr)
        metrics = {}
    bars = {s: store.load_daily_bars(s) for s in symbols}
    cfg = get_config().longterm
    reviews = review_holdings(holdings, prices, fundamentals, metrics, bars, cfg)
    pairs = correlated_pairs(returns_frame(symbols, bars, cfg.corr_window), cfg.corr_store_min)
    store.save_holding_review(reviews, pairs)
    big = [r for r in reviews if r.concentrated]
    print(
        f"Holdings review: {len(reviews)} holdings, {sum(r.conviction is None for r in reviews)} without a conviction score, "
        f"{len(pairs[pairs['corr'] >= cfg.corr_threshold])} pairs at or above {cfg.corr_threshold:.0%} correlation"
        + (f"; over the {cfg.max_weight:.0%} weight limit: {', '.join(r.symbol for r in big)}" if big else "")
    )


def cmd_longterm(args: argparse.Namespace) -> int:
    """Long-term stock recommendations: Schwab fundamentals plus a year of daily bars."""
    source = SchwabDataSource()
    store = Store()
    jobs = JobManager()
    held = current_holdings(store)
    extras = sorted({sym for sym, _ in held.positions()}) + held.watchlist_scan
    universe = args.symbols or load_universe(extra=extras)
    fundamentals = source.fundamentals(universe)
    store.save_company_sizes({sym: f.get("marketCap") for sym, f in fundamentals.items()})
    recs: list[LongTermRec] = []
    errors = 0

    for index, symbol in enumerate(universe):
        bars = store.load_daily_bars(symbol)
        if len(bars) < MIN_BARS and symbol in fundamentals:
            try:
                bars = source.daily_bars(symbol, lookback_days=420)
                store.save_daily_bars(symbol, bars, "schwab")
            except Exception as exc:
                errors += 1
                recs.append(LongTermRec(symbol, "skip", f"data error: {exc}"))
                continue
            time.sleep(args.pause)
        recs.append(recommend_long_term(symbol, bars, fundamentals.get(symbol)))
        if args.job and index % 25 == 0:
            jobs.write_state(args.job, next_index=index + 1, total=len(universe), errors=errors)

    store.save_longterm(recs)
    if not args.no_review:
        run_holding_review(source, store, fundamentals, args.pause)
    good = [r for r in recs if r.status == "recommend"]
    waiting = [r for r in recs if r.status == "wait"]
    print(f"Long-term: {len(good)} recommended, {len(waiting)} waiting for a pullback, of {len(recs)} symbols")
    for r in sorted(good, key=lambda r: (-(r.quality or 0), -(r.trend or 0)))[:15]:
        print(f"  {r.symbol:6} entry {r.entry:8.2f} SL {r.stop:8.2f} T1 {r.t1:8.2f} T2 {r.t2:8.2f}  quality {r.quality}/6")
    if args.job:
        jobs.write_state(args.job, status="done", next_index=len(universe), total=len(universe), errors=errors)
    store.close()
    return 0


def cmd_ccmonitor(args: argparse.Namespace) -> int:
    """Covered-call monitor: close/roll warnings for open short calls, and new calls to open."""
    source = MemoSource(SchwabDataSource())  # the guardrail row, income row and roll search share chains and prices
    store = Store()
    jobs = JobManager()
    holdings = current_holdings(store)
    try:
        events = FinnhubEvents()
    except EventsUnavailable:
        events = None
    full_run: list = []
    symbols = sorted({sym for sym, _ in holdings.positions()} | {o.symbol for o in holdings.short_options})
    try:  # ex-dividend dates and amounts for the early-assignment flag
        fundamentals = source.fundamentals(symbols) if symbols else {}
    except SchwabAuthRequired:
        raise
    except Exception as exc:
        print(f"Fundamentals unavailable ({exc}); ex-dividend flags skipped", file=sys.stderr)
        fundamentals = {}
    rows = monitor_covered_calls(holdings, source, events=events, recs_out=full_run, fundamentals=fundamentals)
    income = recommend_income_calls(holdings, source, events=events, fundamentals=fundamentals)
    store.save_cc_snapshot(rows)
    if full_run or income:
        store.save_covered_calls(full_run + income)  # guardrail and income rows, plus the "why no call" reasons
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.action] = counts.get(r.action, 0) + 1
    print("Covered-call monitor:", ", ".join(f"{n} {a}" for a, n in sorted(counts.items())) or "nothing to show")
    for r in rows:
        if r.action in ("CLOSE", "ROLL"):
            print(f"  {r.action:5} {r.symbol:6} ${r.strike:.2f} {r.expiry}  {r.reason}" + (f" | {r.detail}" if r.detail else ""))
    if args.job:
        jobs.write_state(args.job, status="done", next_index=len(rows), total=len(rows), errors=0)
    store.close()
    return 0


def cmd_run_job(args: argparse.Namespace) -> int:
    """Start a dashboard job and wait for it to finish (used by Windows Task Scheduler)."""
    jobs = JobManager()
    age = token_age_days()
    if age is None or age >= REFRESH_TOKEN_DAYS:
        message = "Schwab login is missing or expired. Run: uv run tool schwab-login"
        print(message, file=sys.stderr)
        jobs.write_state(args.name, status="failed", error=message)
        return 2
    if age >= REFRESH_TOKEN_DAYS - 1:
        print(f"Warning: Schwab login is {age:.1f} days old and expires after {REFRESH_TOKEN_DAYS}. Run: uv run tool schwab-login")
    try:
        jobs.start(args.name)
    except RuntimeError as exc:
        print(exc)
        return 0
    while jobs.status(args.name).get("status") == "running":
        time.sleep(10)
    final = jobs.status(args.name)
    print(f"{args.name}: {final.get('status')} ({final.get('errors', 0)} errors)")
    return 0 if final.get("status") == "done" else 1


def cmd_portfolio(args: argparse.Namespace) -> int:
    """Portfolio risk across accounts: greeks, beta-weighted delta, sectors, this-week risk inputs."""
    from tool.portfolio_risk import build_position_rows, positions_frame, totals

    source = MemoSource(SchwabDataSource())
    store = Store()
    jobs = JobManager()
    holdings = current_holdings(store)
    today = date.today()
    symbols = sorted({l.symbol for l in holdings.lots} | {o.symbol for o in [*holdings.short_options, *holdings.long_options]})
    fundamentals = source.fundamentals(symbols + ["SPY"])
    spy = source.spot("SPY")
    rows = build_position_rows(holdings, source, fundamentals, spy, today)
    positions = positions_frame(rows)

    next_earnings: dict[str, str] = {}
    try:
        events, _ = FinnhubEvents().cached_calendar(today, back=14, ahead=EARNINGS_WINDOW_DAYS)
        upcoming, _last = split_reports(events, today)
        next_earnings = {s: d.isoformat() for s, (d, _h) in upcoming.items()}
    except Exception as exc:
        print(f"Earnings calendar unavailable ({exc}); earnings flags skipped", file=sys.stderr)
    try:
        industries = FinnhubEvents().cached_industries(symbols, today)
    except EventsUnavailable as exc:
        print(f"Finnhub industries unavailable ({exc}); sectors show Unknown", file=sys.stderr)
        industries = {}
    from tool.cc_flags import ex_dividend

    cc = get_config().covered_calls
    info_rows = []
    for sym in symbols:
        ex, quarterly = ex_dividend(sym, fundamentals, cc)
        info_rows.append({"symbol": sym, "industry": industries.get(sym), "next_earnings": next_earnings.get(sym),
                          "ex_div": ex.isoformat() if ex else None, "dividend": quarterly or None})
    info = pd.DataFrame(info_rows)
    store.save_portfolio(positions, info, spy)
    t = totals(positions, spy)
    print(
        f"Portfolio: {len(positions)} positions, stock value ${t['stock_value']:,.0f}, net delta ${t['delta_dollars']:,.0f}, "
        f"SPY-equivalent {t['spy_equiv_shares']:,.0f} shares (${t['spy_equiv_dollars']:,.0f}), theta ${t['theta_day']:,.0f}/day; "
        f"{t['no_quote']} without a quote, {t['unscaled_beta']} with an assumed beta"
    )
    if args.job:
        jobs.write_state(args.job, status="done", next_index=len(positions), total=len(positions), errors=t["no_quote"])
    store.close()
    return 0


def cmd_import_positions(args: argparse.Namespace) -> int:
    """Preview (or with --apply, replace) one account's positions from a broker positions CSV."""
    from tool.positions_csv import apply_positions, diff_account, parse_positions

    parsed = parse_positions(Path(args.file), args.file_account)
    if args.file_account is None and len(parsed.accounts_in_file) > 1:
        print(f"The file holds several accounts: {', '.join(parsed.accounts_in_file)}. Pass --file-account NAME.", file=sys.stderr)
        return 2
    holdings = current_holdings()
    changes = diff_account(holdings, args.account, parsed)
    shown = [c for c in changes if c.result != "keep"]
    print(f"{len(parsed.stocks)} stocks, {len(parsed.shorts)} short and {len(parsed.longs)} long options read; "
          f"{len(parsed.skipped)} rows skipped; {len(shown)} changes for {args.account}")
    for c in shown:
        print(f"  {c.result:7} {c.kind:6} {c.key:32} {c.before or '-':>18} -> {c.after or '-'}")
    for label, why in parsed.skipped[:15]:
        print(f"  skipped: {why}: {label}")
    if args.apply:
        save_current_holdings(apply_positions(holdings, args.account, parsed))
        print(f"Applied. The previous version is saved as a .bak next to config/holdings.yaml.")
    else:
        print("Preview only. Add --apply to replace this account's positions.")
    return 0


def cmd_run_all(args: argparse.Namespace) -> int:
    """Run every job one after another (fast ones first, the full scan last). `--login` signs in to Schwab first."""
    from tool.jobs import RUN_ALL_ORDER

    if args.login:
        SchwabDataSource().login()
        print("Schwab login complete; token saved.", flush=True)
    age = token_age_days()
    if age is None or age >= REFRESH_TOKEN_DAYS:
        print("Schwab login is missing or expired. Sign in first.", file=sys.stderr)
        return 2
    jobs = JobManager()
    for name in RUN_ALL_ORDER:
        try:
            jobs.start(name)
        except RuntimeError:
            print(f"{name}: already running; waiting for it", flush=True)
        time.sleep(5)
        while jobs.status(name).get("status") == "running":
            time.sleep(10)
        final = jobs.status(name)
        print(f"{name}: {final.get('status')} ({final.get('errors', 0)} errors)", flush=True)
    return 0


def cmd_schwab_login(_args: argparse.Namespace) -> int:
    SchwabDataSource().login()
    print("Schwab login complete; token saved.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tool", description="Personal trading recommendation tool (never places orders)")
    sub = parser.add_subparsers(dest="command", required=True)

    levels_p = sub.add_parser("levels", help="Print daily + weekly pivot levels for symbols")
    levels_p.add_argument("symbols", nargs="+", help="Ticker symbols, e.g. MSFT AMZN NVDA")
    levels_p.add_argument("--port", type=int, default=4001, help="IB Gateway port (4001 live, 4002 paper)")
    levels_p.add_argument("--lookback-days", type=int, default=90, help="Days of daily history to fetch")
    levels_p.set_defaults(func=cmd_levels)

    scan_p = sub.add_parser("scan", help="Scan S&P 500 (or given symbols) for swing setups and save bars")
    scan_p.add_argument("symbols", nargs="*", help="Optional symbols; default is the S&P 500 list")
    scan_p.add_argument("--lookback-days", type=int, default=400, help="Days of daily history to fetch")
    scan_p.add_argument("--pause", type=float, default=1.2, help="Seconds between symbols (Schwab rate limit)")
    scan_p.add_argument("--light", action="store_true", help="Scan only holdings and watchlist (quick refresh)")
    scan_p.add_argument("--start-index", type=int, default=0, help="Resume the universe from this index")
    scan_p.add_argument("--job", default=None, help="Job name for progress reporting (used by the dashboard)")
    scan_p.add_argument(
        "--no-expected-move", action="store_true", help="Skip the earnings expected-move lookup (fewer chain requests)"
    )
    scan_p.add_argument("--show-failed", action="store_true", help="Keep failed breakouts (spot back under the entry)")
    scan_p.add_argument(
        "--allow-earnings", action="store_true", help="Keep setups with earnings inside the hold window"
    )
    scan_p.set_defaults(func=cmd_scan)

    calls_p = sub.add_parser("calls", help="Covered-call recommendations for the portfolio")
    calls_p.add_argument("--job", default=None, help="Job name for progress reporting (used by the dashboard)")
    calls_p.set_defaults(func=cmd_calls)

    ccm_p = sub.add_parser("ccmonitor", help="Covered-call monitor: close/roll warnings and new calls to open")
    ccm_p.add_argument("--job", default=None, help="Job name for progress reporting (used by the dashboard)")
    ccm_p.set_defaults(func=cmd_ccmonitor)

    long_p = sub.add_parser("longterm", help="Long-term stock recommendations (quality plus trend)")
    long_p.add_argument("symbols", nargs="*", help="Optional symbols; default is S&P 500 + holdings + watchlist")
    long_p.add_argument("--pause", type=float, default=1.2, help="Seconds between history fetches")
    long_p.add_argument("--job", default=None, help="Job name for progress reporting (used by the dashboard)")
    long_p.add_argument("--no-review", action="store_true", help="Skip the holdings review (weights, conviction, overlaps)")
    long_p.set_defaults(func=cmd_longterm)

    buys_p = sub.add_parser("buys", help="Long calls/puts and weekly lottos with entry, stop, and target levels")
    buys_p.add_argument("symbols", nargs="*", help="Optional symbols; default is bullish/bearish names from the swing scan")
    buys_p.add_argument("--job", default=None, help="Job name for progress reporting (used by the dashboard)")
    buys_p.set_defaults(func=cmd_buys)

    whales_p = sub.add_parser("whales", help="Whale activity: large option premium and fresh volume")
    whales_p.add_argument("symbols", nargs="*", help="Optional symbols; default is indexes, MAG7, high-flow names, holdings")
    whales_p.add_argument("--job", default=None, help="Job name for progress reporting (used by the dashboard)")
    whales_p.set_defaults(func=cmd_whales)

    run_p = sub.add_parser("run-job", help="Run a dashboard job to completion (for scheduled tasks)")
    run_p.add_argument("name", choices=sorted(JOB_SPECS), help="Job name")
    run_p.set_defaults(func=cmd_run_job)

    recheck_p = sub.add_parser("recheck", help="Re-check saved swing setups against live prices")
    recheck_p.add_argument("--job", default=None, help="Job name for progress reporting (used by the dashboard)")
    recheck_p.set_defaults(func=cmd_recheck)

    pf_p = sub.add_parser("portfolio", help="Portfolio risk: greeks, beta-weighted delta, sectors, this-week risk")
    pf_p.add_argument("--job", default=None, help="Job name for progress reporting (used by the dashboard)")
    pf_p.set_defaults(func=cmd_portfolio)

    imp_p = sub.add_parser("import-positions", help="Preview or apply a broker positions CSV for one account")
    imp_p.add_argument("file", help="Path to the positions CSV")
    imp_p.add_argument("--account", required=True, choices=list(ACCOUNTS), help="Which account the file replaces")
    imp_p.add_argument("--file-account", default=None, help="Account name inside the file, when it holds several")
    imp_p.add_argument("--apply", action="store_true", help="Write the change (default is a preview)")
    imp_p.set_defaults(func=cmd_import_positions)

    all_p = sub.add_parser("run-all", help="Run every job in order; --login signs in to Schwab first")
    all_p.add_argument("--login", action="store_true", help="Open the Schwab sign-in first")
    all_p.set_defaults(func=cmd_run_all)

    login_p = sub.add_parser("schwab-login", help="Redo the Schwab browser login (needed every 7 days)")
    login_p.set_defaults(func=cmd_schwab_login)

    options_p = sub.add_parser("options", help="Naked calls, cash-secured puts, naked puts, and wheel puts")
    options_p.add_argument("symbols", nargs="*", help="Optional symbols; default is the watchlist minus held names")
    options_p.add_argument("--job", default=None, help="Job name for progress reporting (used by the dashboard)")
    options_p.set_defaults(func=cmd_options)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SchwabAuthRequired as exc:
        message = f"Schwab re-auth required: {exc}"
        print(message, file=sys.stderr)
        if getattr(args, "job", None):
            JobManager().write_state(args.job, status="failed", error=message)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

