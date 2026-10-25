"""Streamlit dashboard — levels + covered-call recommendations.

Recommend-only: nothing here places or stages an order. Run with:
    uv run streamlit run app.py
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from tool.screener import ACTIONABLE, LABELS as SETUP_LABELS, RANK, earnings_label
from tool.sections import bucket, chance_of_profit
from tool.themes import all_themes, theme_of
from tool.data.schwab import REFRESH_TOKEN_DAYS, SchwabDataSource, auth_status, token_age_days
from tool.data.robinhood_csv import (
    apply_stock_recon,
    build_stock_lots,
    open_options,
    options_from_statement,
    parse_activity,
    reconcile_stocks,
)
from tool.holdings import (
    ACCOUNTS, Holdings, LongOption, Lot, ShortOption, add_lot, option_account, remove_lots, replace_account_options,
)
from tool.jobs import JOB_SPECS, JobManager
from tool.tracker import assess_position, refresh_marks
from tool.config import get_config
from tool.size import size_label
from tool.why import buy_why, sell_why, swing_why
from tool.store import Store, current_holdings, save_current_holdings
from tool.freshness import assess, in_morning_window, last_session_date, parse_ts, previous_weekday
from tool.levels import MARKET_CLOSE, NY, LevelsResult, Pivots, compute_levels

st.set_page_config(page_title="SBTrade", layout="wide")


@st.cache_resource
def get_source() -> SchwabDataSource:
    return SchwabDataSource()


def load_holdings_safe() -> Holdings:
    return current_holdings()


@st.cache_data(ttl=600)
def _sizes() -> dict[str, float]:
    store = Store()
    try:
        return store.company_sizes()
    finally:
        store.close()


def size_of(symbol: str) -> str | None:
    """Company size label (bucket and market cap) for a ticker, None when unknown."""
    return size_label(_sizes().get(symbol))


def show_freshness(ts: str | None, kind: str, what: str) -> bool:
    """Show when a section's data was captured, and warn loudly if it is stale. Returns True when stale."""
    if not ts:
        return False
    date_only = len(ts) == 10  # the swing scan stores a date, not a time
    when = parse_ts(ts + "T12:00:00" if date_only else ts)
    now = datetime.now(NY)
    stale, age = assess(when, now, kind)
    stamp = f"{when:%a %b %d}" if date_only else f"{when:%a %b %d, %I:%M %p}"
    previous = previous_weekday(now.date())
    if when.date() == previous or (when.date() < now.date() and when.date() == last_session_date(now)):
        age = "previous trading session"  # overnight hours are not data age
    if stale and when.date() == previous and in_morning_window(now):
        st.info(
            f"Waiting for today's morning run. {what} data is still from {stamp} ({age}); the jobs refresh it "
            "between 9:35 and about 10:30 AM. Don't trade from it yet."
        )
        return True
    if stale:
        st.warning(
            f"STALE: {what} data is from {stamp} ({age}). Prices and levels have likely moved, so don't trade "
            "from this without a refresh. Start the job in the Jobs tab, or wait for the next scheduled run."
        )
    else:
        st.caption(f"{what} data as of {stamp} ({age})")
    return stale


def drop_expired(recs: pd.DataFrame) -> pd.DataFrame:
    """Option contracts past their expiry can no longer be traded; hide them and say so."""
    expired = recs["expiry"].notna() & (recs["expiry"] < date.today().isoformat())
    if expired.any():
        st.caption(f"{int(expired.sum())} row(s) hidden because their option contracts have already expired.")
    return recs[~expired]


def pivots_table(pivots: Pivots) -> pd.DataFrame:
    return pd.DataFrame(
        [{"S3": pivots.s3, "S2": pivots.s2, "S1": pivots.s1, "P": pivots.p, "R1": pivots.r1, "R2": pivots.r2, "R3": pivots.r3}]
    ).round(2)


def render_levels(levels: LevelsResult) -> None:
    cols = st.columns(3)
    cols[0].metric("Spot", f"{levels.spot:.2f}" if levels.spot is not None else "-")
    cols[1].metric("ATR(10)", f"{levels.atr10:.2f}")
    cols[2].metric("Prior 20D high", f"{levels.prior_20d_high:.2f}")

    st.caption(f"Daily [{levels.daily.bar_date}]  H {levels.daily.high:.2f}  L {levels.daily.low:.2f}  C {levels.daily.close:.2f}")
    st.dataframe(pivots_table(levels.daily.pivots), hide_index=True, width="stretch")

    st.caption(f"Weekly [{levels.weekly.week_start} .. {levels.weekly.week_end}]  H {levels.weekly.high:.2f}  L {levels.weekly.low:.2f}  C {levels.weekly.close:.2f}")
    st.dataframe(pivots_table(levels.weekly.pivots), hide_index=True, width="stretch")


def _whole(value: float) -> float:
    value = float(value)
    return int(value) if value == int(value) else value


def render_stocks(holdings: Holdings) -> None:
    st.subheader("Stocks")
    if holdings.lots:
        today = date.today()

        def held_days(lot: Lot) -> int | None:
            return (today - lot.purchase_date).days if lot.purchase_date else None

        total_cost = sum(l.shares * l.cost for l in holdings.lots)
        m1, m2, m3 = st.columns(3)
        m1.metric("Positions", len(holdings.positions()))
        m2.metric("Lots", len(holdings.lots))
        m3.metric("Total cost basis", f"${total_cost:,.0f}")

        table = pd.DataFrame(
            [
                {
                    "Symbol": l.symbol,
                    "Account": l.account,
                    "Shares": l.shares,
                    "Cost": l.cost,
                    "Purchase date": pd.Timestamp(l.purchase_date) if l.purchase_date else pd.NaT,
                    "Held (days)": held_days(l),
                    "Term": "—" if held_days(l) is None else ("Long-term" if held_days(l) > 365 else "Short-term"),
                    "Strategy": l.strategy or "",
                    "No covered calls": l.no_covered_calls,
                    "Note": l.note or "",
                }
                for l in holdings.lots
            ]
        )
        st.caption(
            "Edit shares, cost, purchase date, strategy, or note, then Save. A symbol can have several lots "
            "with different purchase dates. Term is long-term after more than one year held."
        )
        edited = st.data_editor(
            table,
            hide_index=True,
            width="stretch",
            key="lots_editor",
            disabled=["Symbol", "Account", "Held (days)", "Term"],
            column_config={
                "Purchase date": st.column_config.DateColumn("Purchase date", format="YYYY-MM-DD"),
                "Shares": st.column_config.NumberColumn("Shares", min_value=0.0001, format="%g"),
                "Cost": st.column_config.NumberColumn("Cost", min_value=0.0, format="%.2f"),
                "Strategy": st.column_config.SelectboxColumn("Strategy", options=["", "wheel"]),
            },
        )
        if st.button("Save changes", key="save_lots"):
            new_lots = []
            for lot, (_, row) in zip(holdings.lots, edited.iterrows()):
                bought = row["Purchase date"]
                new_lots.append(
                    replace(
                        lot,
                        shares=_whole(row["Shares"]),
                        cost=float(row["Cost"]),
                        purchase_date=None if pd.isna(bought) else pd.Timestamp(bought).date(),
                        strategy=row["Strategy"] or None,
                        no_covered_calls=bool(row["No covered calls"]),
                        note=row["Note"] or None,
                    )
                )
            save_current_holdings(replace(holdings, lots=new_lots))
            st.success("Saved.")
            st.rerun()
    else:
        st.info("No stock positions yet — add one below or upload a statement.")

    st.markdown("**Add a lot**")
    with st.form("add_position", clear_on_submit=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        symbol = c1.text_input("Symbol")
        account = c2.selectbox("Account", ACCOUNTS)
        shares = c3.number_input("Shares", min_value=0.0001, value=100.0, step=1.0, format="%g")
        cost = c4.number_input("Cost per share", min_value=0.01, value=100.0, step=0.01, format="%.2f")
        purchase_date = c5.date_input("Purchase date", value=None)
        c6, c7 = st.columns(2)
        strategy = c6.selectbox("Strategy", ["", "wheel"])
        no_cc = c7.checkbox("No covered calls")
        submitted = st.form_submit_button("Add lot")
    if submitted:
        try:
            updated = add_lot(
                holdings,
                Lot(
                    symbol=symbol,
                    account=account,
                    shares=_whole(shares),
                    cost=float(cost),
                    purchase_date=purchase_date,
                    strategy=strategy or None,
                    no_covered_calls=no_cc,
                ),
            )
        except ValueError as exc:
            st.error(str(exc))
        else:
            save_current_holdings(updated)
            st.success(f"Added {symbol.strip().upper()} ({account}).")
            st.rerun()

    st.markdown("**Remove a position** (all its lots in one account)")
    if holdings.positions():
        options = {f"{sym} — {acct}": (sym, acct) for sym, acct in holdings.positions()}
        choice = st.selectbox("Position", list(options))
        sym, acct = options[choice]
        if holdings.short_options_for(sym, acct, "call"):
            st.warning(f"{sym} has open short calls in your portfolio. Removing the shares does not close them.")
        if st.button("Remove position"):
            save_current_holdings(remove_lots(holdings, sym, acct))
            st.success(f"Removed all {sym} lots in {acct}.")
            st.rerun()
    else:
        st.caption("Nothing to remove.")


def _option_frame(options: list, with_price: bool) -> pd.DataFrame:
    rows = []
    for o in options:
        row = {
            "Symbol": o.symbol,
            "Type": o.type,
            "Strike": o.strike,
            "Expiry": pd.Timestamp(o.expiry),
            "Contracts": o.qty,
            "Account": option_account(o),
        }
        if with_price:
            row["Avg price"] = o.avg_price
        rows.append(row)
    columns = ["Symbol", "Type", "Strike", "Expiry", "Contracts", "Account"] + (["Avg price"] if with_price else [])
    frame = pd.DataFrame(rows, columns=columns)
    frame["Expiry"] = pd.to_datetime(frame["Expiry"])
    return frame


OPTION_TYPES = ["call", "put"]


def _option_editor(frame: pd.DataFrame, key: str, with_price: bool) -> pd.DataFrame:
    config = {
        "Type": st.column_config.SelectboxColumn("Type", options=OPTION_TYPES, required=True),
        "Strike": st.column_config.NumberColumn("Strike", min_value=0.01, format="%.2f", required=True),
        "Expiry": st.column_config.DateColumn("Expiry", format="YYYY-MM-DD", required=True),
        "Contracts": st.column_config.NumberColumn("Contracts", min_value=1, step=1, required=True),
        "Account": st.column_config.SelectboxColumn("Account", options=list(ACCOUNTS), required=True),
    }
    if with_price:
        config["Avg price"] = st.column_config.NumberColumn("Avg price", min_value=0.0, format="%.2f")
    return st.data_editor(frame, num_rows="dynamic", hide_index=True, width="stretch", key=key, column_config=config)


def _collect_options(frame: pd.DataFrame, factory, with_price: bool) -> list:
    out = []
    for _, row in frame.iterrows():
        if pd.isna(row["Symbol"]) or not str(row["Symbol"]).strip():
            continue
        incomplete = (
            row["Type"] not in OPTION_TYPES
            or pd.isna(row["Strike"])
            or pd.isna(row["Expiry"])
            or pd.isna(row["Contracts"])
        )
        if incomplete:
            raise ValueError(f"Row for {row['Symbol']} is incomplete (needs type, strike, expiry, contracts).")
        kwargs = {
            "symbol": str(row["Symbol"]).strip().upper(),
            "type": row["Type"],
            "strike": float(row["Strike"]),
            "expiry": pd.Timestamp(row["Expiry"]).date(),
            "qty": int(row["Contracts"]),
            "account": row["Account"] if row.get("Account") in ACCOUNTS else None,
        }
        if with_price:
            kwargs["avg_price"] = None if pd.isna(row["Avg price"]) else float(row["Avg price"])
        out.append(factory(**kwargs))
    return out


def render_options(holdings: Holdings) -> None:
    st.subheader("Options")
    st.caption(
        "Add rows with the + at the bottom of a table, delete rows by selecting them, then Save. "
        "Short options cover shares and reduce free lots in the covered-call engine; long options are "
        "stored for reference."
    )

    st.markdown(f"**Short options** · {len(holdings.short_options)}")
    short_edit = _option_editor(_option_frame(holdings.short_options, False), "short_options_editor", False)
    if st.button("Save short options"):
        try:
            shorts = _collect_options(short_edit, ShortOption, with_price=False)
        except ValueError as exc:
            st.error(str(exc))
        else:
            save_current_holdings(replace(holdings, short_options=shorts))
            st.success(f"Saved {len(shorts)} short option rows.")
            st.rerun()

    render_bought_tracker()

    st.markdown(f"**Long options** · {len(holdings.long_options)}")
    long_edit = _option_editor(_option_frame(holdings.long_options, True), "long_options_editor", True)
    if st.button("Save long options"):
        try:
            longs = _collect_options(long_edit, LongOption, with_price=True)
        except ValueError as exc:
            st.error(str(exc))
        else:
            save_current_holdings(replace(holdings, long_options=longs))
            st.success(f"Saved {len(longs)} long option rows.")
            st.rerun()


def render_statement_update(holdings: Holdings) -> None:
    st.subheader("Update from a Robinhood statement")
    st.caption(
        "Export your account activity as a CSV from Robinhood (Account, then Statements & history, then reports), "
        "then upload it here or pick a file from the data folder. The statement rebuilds your stock lots "
        "(with purchase dates, first-in first-out) and your open options. Nothing changes until you press Apply."
    )
    account = st.selectbox("Account this statement covers", ACCOUNTS, key="statement_account")
    uploaded = st.file_uploader("Upload account activity CSV", type="csv")
    data_files = sorted(Path("data").glob("*.csv"))
    picked = st.selectbox("…or use a file already in the data folder", ["—"] + [p.name for p in data_files])

    if uploaded is not None:
        source = uploaded.getvalue()
    elif picked != "—":
        source = Path("data") / picked
    else:
        st.info("Choose a statement to preview the changes.")
        return

    try:
        txns = parse_activity(source)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        st.error(f"Could not read that file as a Robinhood activity CSV: {exc}")
        return
    if not txns:
        st.error("No transactions found. This should be Robinhood's account activity export.")
        return

    recons = reconcile_stocks(build_stock_lots(txns), holdings, account)
    shorts, longs = options_from_statement(open_options(txns), account)
    st.caption(f"{len(txns):,} transactions from {txns[0].day} to {txns[-1].day}.")

    changed = [r for r in recons if r.lots]
    unchanged = [r for r in recons if not r.lots]
    matched = sum(1 for r in recons if r.status == "match")
    c1, c2, c3 = st.columns(3)
    c1.metric("Symbols matching", matched)
    c2.metric("Need attention", len(changed) - matched)
    c3.metric("Kept as is (no activity)", len(unchanged))

    st.markdown("**Stocks**")
    show_all = st.toggle("Show matching symbols too", value=False)
    rows = []
    for r in sorted(changed, key=lambda x: (x.status == "match", x.symbol)):
        if r.status == "match" and not show_all:
            continue
        dated = sorted(l.day for l in r.lots if l.day)
        rows.append(
            {
                "Symbol": r.symbol,
                "Stored shares": round(r.db_shares, 4),
                "Statement shares": round(r.statement_shares, 4),
                "Lots": len(r.lots),
                "Earliest purchase": dated[0] if dated else None,
                "Result": r.status,
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.caption("All matched symbols agree with the stored portfolio.")

    def key(o):
        return (o.symbol, o.type, o.strike, o.expiry)

    stored_short = {key(o): o.qty for o in holdings.short_options}
    stored_long = {key(o): o.qty for o in holdings.long_options}
    new_short = {key(o): o.qty for o in shorts}
    new_long = {key(o): o.qty for o in longs}
    short_changes = len({k for k in stored_short.keys() | new_short.keys() if stored_short.get(k) != new_short.get(k)})
    long_changes = len({k for k in stored_long.keys() | new_long.keys() if stored_long.get(k) != new_long.get(k)})
    st.markdown("**Options**")
    st.write(
        f"Statement shows {len(shorts)} open short and {len(longs)} open long option positions "
        f"({short_changes} short and {long_changes} long differ from the stored portfolio)."
    )
    replace_options = st.checkbox(
        f"Replace the stored option positions of {account} with the statement's open options (other accounts are untouched)",
        value=True,
    )
    st.caption(
        "Stock lots from the statement replace the stored lots for the symbols above, in the chosen account only. "
        "Shares bought before the statement starts, or transferred in, keep your stored average cost and an unknown date."
    )
    if st.button("Apply to portfolio", type="primary"):
        updated = apply_stock_recon(holdings, recons, account)
        if replace_options:
            updated = replace_account_options(updated, account, shorts, longs)
        save_current_holdings(updated)
        st.success("Portfolio updated. The previous version is saved in config/holdings.yaml.bak.")
        st.rerun()


def render_watchlist(holdings: Holdings) -> None:
    st.subheader("Watchlist")
    st.caption(
        "Used by the options job (naked calls, puts) and added to the swing scan alongside the S&P 500 "
        "and your holdings. Held names are covered-call territory and are skipped by the naked/put lists."
    )
    st.write(", ".join(holdings.watchlist_scan) or "Watchlist is empty.")
    with st.form("add_watch", clear_on_submit=True):
        new_text = st.text_input("Add tickers (comma or space separated)", placeholder="AAPL, TSLA, JPM")
        add_clicked = st.form_submit_button("Add to watchlist")
    if add_clicked:
        additions = [t.strip().upper() for t in new_text.replace(",", " ").split() if t.strip()]
        merged = list(dict.fromkeys([*holdings.watchlist_scan, *additions]))
        if len(merged) == len(holdings.watchlist_scan):
            st.info("Nothing new to add.")
        else:
            save_current_holdings(replace(holdings, watchlist_scan=merged))
            st.success(f"Added {len(merged) - len(holdings.watchlist_scan)} ticker(s).")
            st.rerun()
    if holdings.watchlist_scan:
        to_remove = st.multiselect("Remove from watchlist", holdings.watchlist_scan)
        if st.button("Remove selected", disabled=not to_remove):
            keep = [t for t in holdings.watchlist_scan if t not in to_remove]
            save_current_holdings(replace(holdings, watchlist_scan=keep))
            st.success(f"Removed {len(to_remove)} ticker(s).")
            st.rerun()


def render_positions_import(holdings: Holdings) -> None:
    from tool.positions_csv import apply_positions, diff_account, parse_positions

    st.subheader("Import positions from a broker CSV")
    st.caption(
        "Upload the positions export from Fidelity, Schwab, IBKR or similar. The file replaces the stocks and options of the one "
        "account you pick; other accounts are untouched. Rows it cannot read are listed, never guessed. Nothing changes until you press Replace."
    )
    account = st.selectbox("Account this file covers", ACCOUNTS, key="positions_account")
    uploaded = st.file_uploader("Upload positions CSV", type="csv", key="positions_upload")
    if uploaded is None:
        st.info("Choose a positions file to preview the changes.")
        return
    raw = uploaded.getvalue()
    try:
        names = parse_positions(raw).accounts_in_file
        file_account = None
        if len(names) > 1:
            file_account = st.selectbox("The file holds several accounts. Which one is this?", names)
        parsed = parse_positions(raw, file_account)
    except ValueError as exc:
        st.error(f"Could not read that file: {exc}")
        return
    st.write(
        f"Read {len(parsed.stocks)} stock positions, {len(parsed.shorts)} short and {len(parsed.longs)} long option positions; "
        f"{len(parsed.skipped)} rows skipped."
    )
    changes = [c for c in diff_account(holdings, account, parsed) if c.result != "keep"]
    if changes:
        st.dataframe(
            pd.DataFrame([{"Change": c.result, "Type": c.kind, "Position": c.key, "Stored": c.before, "File": c.after} for c in changes]),
            hide_index=True, width="stretch",
        )
    else:
        st.caption("The file matches what is stored for this account.")
    if parsed.skipped:
        with st.expander(f"{len(parsed.skipped)} rows skipped"):
            st.dataframe(pd.DataFrame(parsed.skipped, columns=["Row", "Why"]), hide_index=True, width="stretch")
    if st.button("Replace this account's positions", type="primary", disabled=not changes):
        save_current_holdings(apply_positions(holdings, account, parsed))
        st.success("Positions replaced. The previous version is saved as a .bak next to config/holdings.yaml.")
        st.rerun()


def render_risk() -> None:
    from tool.portfolio_risk import collateral_by_account, greeks_by_symbol, held_events, sector_exposure, totals, week_risk

    st.subheader("Portfolio risk")
    store = Store()
    try:
        positions, info, as_of, spy = store.latest_portfolio()
    finally:
        store.close()
    if st.button("Run portfolio job", key="run_portfolio_job"):
        try:
            JobManager().start("portfolio")
            st.info("Portfolio job started. Reload this tab in a minute.")
        except RuntimeError as exc:
            st.error(str(exc))
    if positions.empty:
        st.info("No portfolio snapshot yet. Press 'Run portfolio job' (needs a Schwab login).")
        return
    st.caption(f"Data as of {as_of}. SPY {spy:,.2f}. Recommend-only: nothing here places an order.")
    today = date.today()

    t = totals(positions, spy)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stock value", f"${t['stock_value']:,.0f}")
    c2.metric("Net delta $", f"${t['delta_dollars']:,.0f}")
    c3.metric("Beta-weighted (SPY shares)", f"{t['spy_equiv_shares']:,.0f}", help=f"About ${t['spy_equiv_dollars']:,.0f} of SPY")
    c4.metric("Theta / day", f"${t['theta_day']:,.0f}")
    if t["no_quote"] or t["unscaled_beta"]:
        st.caption(
            f"{t['no_quote']} positions have no price or quote and count as zero; {t['unscaled_beta']} have no beta and use 1.0. "
            "Leveraged funds are not scaled by their leverage. Beta comes from Schwab fundamentals."
        )

    st.markdown("**Cash vs cash-secured-put collateral**")
    coll = collateral_by_account(positions)
    if coll.empty:
        st.caption("No short puts and no cash configured. Set puts.cash per account in config/settings.yaml.")
    else:
        for _, r in coll.iterrows():
            if r["state"] == "over":
                st.error(f"{r['account']}: puts commit ${r['committed']:,.0f} but cash is ${r['cash']:,.0f}.")
            elif r["state"] == "unset":
                st.warning(f"{r['account']}: puts commit ${r['committed']:,.0f}; set puts.cash.{r['account']} in config/settings.yaml to check it.")
        st.dataframe(coll.rename(columns=str.title), hide_index=True, width="stretch")

    st.markdown("**This week**")
    risk = week_risk(positions, info, today)
    if risk.empty:
        st.caption("No short options expire within 7 days.")
    else:
        st.dataframe(risk, hide_index=True, width="stretch")
    events = held_events(positions, info, today)
    if not events.empty:
        st.caption("Earnings and ex-dividend dates within 7 days on held names")
        st.dataframe(events, hide_index=True, width="stretch")

    st.markdown("**Greeks by symbol**")
    st.dataframe(greeks_by_symbol(positions), hide_index=True, width="stretch")

    industry, themes = sector_exposure(positions, info)
    left, right = st.columns(2)
    with left:
        st.markdown("**Industry exposure (shares)**")
        st.dataframe(industry, hide_index=True, width="stretch", column_config={"weight": st.column_config.ProgressColumn("Weight", format="percent", min_value=0, max_value=1)})
    with right:
        st.markdown("**Theme exposure (shares)**")
        st.dataframe(themes, hide_index=True, width="stretch", column_config={"weight": st.column_config.ProgressColumn("Weight", format="percent", min_value=0, max_value=1)})
    with st.expander("All positions"):
        st.dataframe(positions, hide_index=True, width="stretch")


def render_portfolio() -> None:
    holdings = load_holdings_safe()
    tab_risk, tab_stocks, tab_options, tab_update, tab_import, tab_watch = st.tabs(
        ["Risk", "Stocks", "Options", "Update from statement", "Import positions", "Watchlist"]
    )
    with tab_risk:
        render_risk()
    with tab_import:
        render_positions_import(holdings)
    with tab_stocks:
        render_stocks(holdings)
    with tab_options:
        render_options(holdings)
    with tab_update:
        render_statement_update(holdings)
    with tab_watch:
        render_watchlist(holdings)


SETUP_COLUMNS = {
    "symbol": "Symbol",
    "kind": "Setup",
    "size": "Company size",
    "why": "Why recommended",
    "spot": "Spot",
    "entry": "Entry",
    "stop": "Stop",
    "t1": "T1",
    "t2": "T2",
    "rr": "R:R",
    "resistance": "Resistance",
    "score": "Score",
    "shares": "Shares",
    "position_value": "Position $",
    "risk_dollars": "Risk $",
    "change_20d": "20D change",
    "volume_ratio": "Volume vs avg",
    "earnings_date": "Earnings",
    "exp_move": "Exp. move",
}


def _earnings_label(row: pd.Series) -> str | None:
    """'Reported Oct 15 (amc)', 'Oct 28 (in 6d, bmo)', 'None', or 'Unknown' when the calendar was unavailable."""
    def when(key: str):
        value = row.get(key)
        return None if value is None or pd.isna(value) else pd.Timestamp(value).date()

    def text(key: str) -> str | None:
        value = row.get(key)
        return None if value is None or pd.isna(value) else str(value)

    known = row.get("earnings_known")
    if known is None or pd.isna(known):
        return None  # saved before the earnings columns existed
    post = row.get("post_earnings")
    return earnings_label(
        when("earnings_date"), text("earnings_hour"), when("last_reported"), text("last_hour"),
        bool(post) and not pd.isna(post), bool(int(known)), date.today(),
    )


def _setup_label(row: pd.Series) -> str | None:
    """'MA pullback (SMA20)', 'Bull flag (watching)', 'Breakout', ... for the Setup column."""
    kind = row.get("kind")
    label = SETUP_LABELS.get(kind, kind)
    if kind == "ma_pullback":
        entry, sma20 = row.get("entry"), row.get("sma20")
        if entry is not None and sma20 is not None and not pd.isna(entry) and not pd.isna(sma20):
            label += " (SMA20)" if abs(float(entry) - float(sma20)) < 1e-6 else " (SMA50)"
    elif kind == "bull_flag":
        broken = row.get("flag_broken")
        label += " (broke out)" if broken is not None and not pd.isna(broken) and int(broken) else " (watching)"
    return label


def _setup_view(df: pd.DataFrame) -> pd.DataFrame:
    def exp_move(r: pd.Series) -> str | None:
        value, pct = r.get("expected_move"), r.get("expected_move_pct")
        if value is None or pd.isna(value) or pct is None or pd.isna(pct):
            return None
        return f"${float(value):.2f} ({float(pct):.1%})"

    df = df.assign(
        why=[swing_why(r) for _, r in df.iterrows()], size=df["symbol"].map(size_of),
        exp_move=[exp_move(r) for _, r in df.iterrows()],
    )
    frame = df.reindex(columns=list(SETUP_COLUMNS))
    view = frame.rename(columns=SETUP_COLUMNS).copy()
    view["20D change"] = view["20D change"].map(lambda x: f"{x:+.1%}" if pd.notna(x) else None)
    view["Volume vs avg"] = pd.to_numeric(view["Volume vs avg"], errors="coerce").map(
        lambda x: f"{x:.1f}x" if pd.notna(x) else None
    )
    view["Earnings"] = [_earnings_label(r) for _, r in df.iterrows()]
    view["Setup"] = [_setup_label(r) for _, r in df.iterrows()]
    for col in ("Position $", "Risk $"):
        view[col] = pd.to_numeric(view[col], errors="coerce").round(0)
    view = view.drop(columns=[c for c in ("Shares", "Position $", "Risk $") if view[c].isna().all()])
    for col in ("Spot", "Entry", "Stop", "T1", "T2", "R:R", "Resistance", "Score"):
        view[col] = pd.to_numeric(view[col], errors="coerce").round(2)
    return view


OPTION_VIEW = {
    "symbol": "Symbol",
    "size": "Company size",
    "why": "Why recommended",
    "expiry_type": "Expiry type",
    "expiry": "Expiry",
    "strike": "Strike",
    "bid": "Bid",
    "delta": "Delta",
    "chance": "Chance of profit",
    "premium": "Premium/contract",
    "return_pct": "Return",
    "annualized": "Annualized",
    "entry_level": "ENTRY",
    "stop_level": "STOP-LOSS",
    "stop_option": "SL option at",
    "t1": "T1",
    "t2": "T2",
    "margin": "Margin (est.)",
    "collateral": "Collateral",
    "effective_cost": "Effective cost",
    "max_loss": "Max loss/contract",
    "theme": "Theme",
    "entry_rule": "Entry rule",
    "stop_rule": "Stop rule",
    "target_rule": "Target rule",
    "notes": "Notes",
}
LEVEL_COLUMNS = ("entry_level", "stop_level", "stop_option", "t1", "t2")


def _sell_labels(side: str) -> dict[str, str]:
    above = side == "put"  # a sold put wants the stock above its level; a sold call wants it below
    return {
        **OPTION_VIEW,
        "entry_level": "ENTRY: sell only while stock " + ("above" if above else "below"),
        "stop_level": "STOP-LOSS: stock closes " + ("below" if above else "above"),
        "stop_option": "SL: or option at",
        "t1": "T1: buy back at",
        "t2": "T2: buy back at",
    }


SECTIONS = [
    ("high", "High chances", "|delta| 0.20 or less: about an 80%+ chance of keeping the premium"),
    ("moderate", "Moderate", "|delta| 0.20 to 0.35"),
    ("lotto", "Aggressive", "|delta| above 0.35: big premium, close to the money, high chance of assignment"),
]

TRANSIENT_ERRORS = ("data error", "chain error")


def _option_table(rows: pd.DataFrame, side: str) -> pd.DataFrame:
    labels = _sell_labels(side)
    why = [sell_why(r, r.get("family", "")) for _, r in rows.iterrows()]
    view = rows.assign(
        chance=rows["delta"].map(chance_of_profit), why=why, size=rows["symbol"].map(size_of)
    ).rename(columns=labels)
    view["Delta"] = pd.to_numeric(view["Delta"], errors="coerce").round(2)
    for col, fmt in (("Chance of profit", "{:.0%}"), ("Return", "{:.2%}"), ("Annualized", "{:.1%}")):
        view[col] = view[col].map(lambda x, f=fmt: f.format(x) if pd.notna(x) else None)
    for key in LEVEL_COLUMNS:  # rows saved before these columns existed hold only empty values
        view[labels[key]] = pd.to_numeric(view[labels[key]], errors="coerce").round(2)
    keep = [c for c in labels.values() if c in view.columns and view[c].notna().any()]
    return view[keep]


EXPIRY_CHOICES = ["weekly", "monthly", "earnings"]


def option_filters(key: str) -> tuple[list[str], list[str], int | None]:
    """Expiry-type and theme filters plus a row cap. An empty selection means 'all'."""
    c1, c2, c3 = st.columns([2, 4, 1])
    expiries = c1.multiselect("Expiry type", EXPIRY_CHOICES, key=f"{key}_expiry", placeholder="All expiries")
    themes = c2.multiselect("Theme", all_themes(), key=f"{key}_themes", placeholder="All themes")
    top = c3.selectbox("Rows per section", [5, 10, 25, 50, "All"], index=1, key=f"{key}_top")
    return expiries, themes, (None if top == "All" else int(top))


def render_option_family(
    family: str,
    heading: str,
    hint: str,
    expiries: list[str] | None = None,
    themes: list[str] | None = None,
    top_n: int | None = 10,
) -> None:
    store = Store()
    try:
        recs = store.latest_option_recs(family)
    finally:
        store.close()

    st.subheader(heading)
    side = "call" if "call" in family else "put"
    if recs.empty:
        st.info(hint)
        return

    show_freshness(recs["run_ts"].max(), "hourly", "Options")
    recs = drop_expired(recs)
    recs = recs.assign(theme=recs["symbol"].map(theme_of))
    if expiries:
        recs = recs[recs["expiry_type"].isin(expiries)]
    if themes:
        recs = recs[recs["theme"].isin(themes)]

    recommended = recs[recs["status"] == "recommend"].copy()
    recommended["bucket"] = recommended["delta"].map(bucket)
    sort_key = recommended["annualized"].fillna(recommended["return_pct"])
    recommended = recommended.assign(_sort=sort_key).sort_values("_sort", ascending=False)

    st.caption(
        f"{len(recommended)} recommendations of {len(recs)} rows after filters. "
        f"Each section is sorted by return and shows {'all rows' if top_n is None else f'the top {top_n}'}."
    )
    missing = int(recommended["bucket"].isna().sum())
    if missing:
        st.warning(
            f"{missing} of {len(recommended)} recommendations have no delta, so they cannot be sorted into High / "
            "Moderate / Aggressive and are listed under 'Watch list' below. Schwab sometimes returns no greeks right "
            "after the open; run the options job again from the Jobs tab."
        )
    for key, title, definition in SECTIONS:
        part = recommended[recommended["bucket"] == key]
        shown = part if top_n is None else part.head(top_n)
        suffix = f" (showing {len(shown)})" if len(shown) < len(part) else ""
        st.markdown(f"**{title}** · {len(part)}{suffix}  —  {definition}")
        if part.empty:
            st.caption("None in this run.")
        else:
            st.dataframe(_option_table(shown, side), hide_index=True, width="stretch")

    skipped = recs[(recs["status"] != "recommend") & ~recs["reason"].fillna("").str.startswith(TRANSIENT_ERRORS)]
    no_delta = recommended[recommended["bucket"].isna()].assign(
        reason="no delta data from Schwab — cannot rate the chance of profit"
    )
    watch = pd.concat([skipped, no_delta])
    with st.expander(f"Watch list · {len(watch)} — waiting on a condition, or no delta data"):
        if watch.empty:
            st.caption("Nothing waiting.")
        else:
            view = watch[["symbol", "theme", "expiry_type", "expiry", "reason"]].rename(
                columns={
                    "symbol": "Symbol",
                    "theme": "Theme",
                    "expiry_type": "Expiry type",
                    "expiry": "Expiry",
                    "reason": "Why it's not a recommendation",
                }
            )
            st.dataframe(view.sort_values(["Theme", "Symbol"]), hide_index=True, width="stretch")


def render_support_puts(themes: list[str], top_n: int | None) -> None:
    """Cash-secured puts whose strike sits at or under the nearest support, with the collateral check."""
    from tool.support_puts import collateral_check, visible_rows

    store = Store()
    try:
        recs = store.latest_option_recs("csp_support")
    finally:
        store.close()
    cfg = get_config().puts
    st.markdown("#### Cash-secured puts at support")
    st.caption(
        f"Strike at or under the nearest support (weekly or daily S1, SMA50/100/200, or the last swing low) and under the "
        f"stock price, |delta| {cfg.support_delta_low:.2f}-{cfg.support_delta_high:.2f} preferred, {cfg.min_dte}-{cfg.max_dte} "
        "days out. Break-even is strike minus premium. The weekly-S1 guardrail rows are in the tabs to the right. "
        "Rank is the real IV rank once 60 days of implied vol are recorded; until then it is an HV rank, the 20-day "
        "realized volatility ranked against its own year."
    )
    if recs.empty or "support" not in recs or recs["support"].isna().all():
        st.info("No support-put run yet. Run the options job in the Jobs tab.")
        return
    show_freshness(recs["run_ts"].max(), "hourly", "Options")
    recs = drop_expired(recs)
    recs = recs.assign(theme=recs["symbol"].map(theme_of))
    if themes:
        recs = recs[recs["theme"].isin(themes)]
    earnings_play = st.checkbox(
        "Earnings play", key="puts_earnings_play",
        help=f"Show expiries that fall after an earnings date, only when the rank is above {cfg.earnings_rank_min:.0f}. "
             "Expect a gap and an IV crush.",
    )
    shown = visible_rows(recs, earnings_play).copy()
    shown["_order"] = pd.to_numeric(shown["annualized"], errors="coerce")
    shown = shown.sort_values("_order", ascending=False)
    cut = shown if top_n is None else shown.head(top_n)
    gated = int((recs["status"] == "earnings_gated").sum())
    st.caption(
        f"{len(shown)} ideas{'' if len(cut) == len(shown) else f' (showing {len(cut)})'}"
        + ("" if earnings_play or not gated else f". {gated} more expire after earnings: switch on Earnings play.")
    )
    if cut.empty:
        st.caption("None in this run.")
    else:
        bid = pd.to_numeric(cut["bid"], errors="coerce")
        support = pd.to_numeric(cut["support"], errors="coerce")
        strike = pd.to_numeric(cut["strike"], errors="coerce")
        view = pd.DataFrame({
            "Ticker": cut["symbol"], "Company size": cut["symbol"].map(size_of), "Expiry": cut["expiry"],
            "Days": cut["dte"], "Strike": strike, "Bid": bid.round(2),
            "Annualized": pd.to_numeric(cut["annualized"], errors="coerce").map(lambda x: f"{x:.0%}" if pd.notna(x) else None),
            "Return": pd.to_numeric(cut["return_pct"], errors="coerce").map(lambda x: f"{x:.2%}" if pd.notna(x) else None),
            "Delta": pd.to_numeric(cut["delta"], errors="coerce").round(2),
            "Prob OTM": pd.to_numeric(cut["prob_otm"], errors="coerce").map(lambda x: f"{x:.0%}" if pd.notna(x) else None),
            "Support": support.round(2), "Support is": cut["support_label"],
            "Strike under support": ((support - strike) / support).map(lambda x: f"{x:.1%}" if pd.notna(x) else None),
            "Break-even": pd.to_numeric(cut["break_even"], errors="coerce").round(2),
            "Rank": pd.to_numeric(cut["rank"], errors="coerce").round(0),
            "Rank is": cut["rank_source"],
            "Collateral": pd.to_numeric(cut["collateral"], errors="coerce").round(0),
            "Premium": pd.to_numeric(cut["premium"], errors="coerce").round(0),
            "Earnings": cut["status"].map({"earnings_gated": "AFTER EARNINGS"}).fillna(""),
            "Notes": cut["notes"].fillna(""),
        })
        event = st.dataframe(
            view, hide_index=True, width="stretch", on_select="rerun", selection_mode="multi-row", key="pick_csp_support"
        )
        picked = event.selection.rows if event is not None else []
        st.markdown("**Collateral check**")
        accounts = sorted(set(cfg.cash) | {"rh_taxable", "rh_ira", "ibkr"})
        account = st.selectbox("Account", accounts, key="puts_cash_account")
        selected = float(pd.to_numeric(cut.iloc[picked]["collateral"], errors="coerce").sum()) if picked else 0.0
        if not picked:
            st.caption(
                "Select one or more rows above to total their collateral. "
                f"Total for everything shown: ${pd.to_numeric(cut['collateral'], errors='coerce').sum():,.0f}."
            )
        else:
            state, message = collateral_check(selected, cfg.cash.get(account))
            {"ok": st.success, "over": st.error, "unset": st.info}[state](f"{account}: {message}")


def render_puts_tab() -> None:
    c1, c2 = st.columns([5, 1])
    themes = c1.multiselect("Theme", all_themes(), key="puts_themes", placeholder="All themes")
    top = c2.selectbox("Rows per section", [5, 10, 25, 50, "All"], index=1, key="puts_top")
    top_n = None if top == "All" else int(top)

    def sell(family: str, title: str, expiry_type: str) -> None:
        render_option_family(
            family, title, f"No {expiry_type} put run yet. Run the options job in the Jobs tab.",
            [expiry_type], themes, top_n,
        )

    buy_hint = "No buy run yet. Run the buys job in the Jobs tab."
    tab_support, tab_weekly, tab_lotto, tab_monthly, tab_earnings = st.tabs(
        ["At support (7-45 DTE)", "Weekly puts", "Lottos", "Monthly puts", "Earnings"]
    )
    with tab_support:
        render_support_puts(themes, top_n)
    with tab_weekly:
        st.caption("Selling puts that expire this week or next: collect premium, strike at or below weekly S1.")
        sell("csp", "Cash-secured puts (weekly)", "weekly")
        sell("naked_put", "Naked puts (weekly): loss to strike minus premium if the stock goes to zero", "weekly")
        sell("wheel", "Wheel put legs (weekly)", "weekly")
    with tab_lotto:
        st.warning(
            "Lottos are cheap out-of-the-money weekly puts you BUY. The most you can lose is the premium, and most "
            "expire worthless. Size them small, and buy only after the trigger."
        )
        render_buy_family(
            "lotto_put", "Weekly lotto puts (buy)",
            "Cheap out-of-the-money weekly puts on stocks in a downtrend, picked for the best payoff if the stock "
            "reaches its first target. Buy only below the trigger; exit if it closes back above it.",
            buy_hint, themes, [], top_n,
        )
    with tab_monthly:
        st.caption("Monthly expiries, both ways: swing puts you buy on downtrend names, and puts you sell.")
        render_buy_family(
            "long_put", "Swing puts (buy, monthly, about 0.50 delta)",
            "Monthly puts with delta near 0.50 on downtrend names, with the stock's trigger, stop, and two targets.",
            buy_hint, themes, [], top_n,
        )
        sell("csp", "Cash-secured puts (monthly)", "monthly")
        sell("naked_put", "Naked puts (monthly)", "monthly")
        sell("wheel", "Wheel put legs (monthly)", "monthly")
    with tab_earnings:
        st.warning(
            "These expire right after an earnings date. Expect a gap and an IV crush: the stock can move far past "
            "your strike overnight."
        )
        sell("csp", "Cash-secured puts (through earnings)", "earnings")
        sell("naked_put", "Naked puts (through earnings)", "earnings")


BUY_VIEW = {
    "symbol": "Symbol",
    "size": "Company size",
    "why": "Why recommended",
    "trigger": "Trigger",
    "expiry": "Expiry",
    "strike": "Strike",
    "ask": "Ask",
    "cost": "Cost/contract",
    "delta": "Delta",
    "entry_level": "ENTRY",
    "stop_level": "STOP-LOSS",
    "stop_option": "SL: or option at",
    "t1": "T1 (stock)",
    "t2": "T2 (stock)",
    "multiple": "Best-case payoff (T1 at expiry)",
    "theme": "Theme",
    "entry_rule": "Entry rule",
    "stop_rule": "Stop rule",
    "target_rule": "Target rule",
    "notes": "Notes",
}
TRIGGER_ORDER = {"TRIGGERED": 0, "WAITING": 1, "EXTENDED": 2}
TRIGGER_CHOICES = list(TRIGGER_ORDER)


def _buy_table(rows: pd.DataFrame, side: str) -> pd.DataFrame:
    bullish = side == "call"
    labels = {
        **BUY_VIEW,
        "entry_level": "ENTRY: buy " + ("above" if bullish else "below"),
        "stop_level": "STOP-LOSS: stock closes " + ("below" if bullish else "above"),
    }
    view = rows.rename(columns=labels)
    view["Delta"] = pd.to_numeric(view["Delta"], errors="coerce").round(2)
    payoff = labels["multiple"]
    view[payoff] = view[payoff].map(lambda x: f"{x:.1f}x" if pd.notna(x) else None)
    for key in ("entry_level", "stop_level", "stop_option", "t1", "t2"):
        view[labels[key]] = pd.to_numeric(view[labels[key]], errors="coerce").round(2)
    keep = [c for c in labels.values() if c in view.columns and view[c].notna().any()]
    return view[keep]


def render_bought_form(family: str, idea: pd.Series) -> None:
    """Log a purchase of the selected idea. This records it for tracking only; nothing is ordered."""
    ask = float(idea["ask"]) if pd.notna(idea.get("ask")) else 0.0
    with st.form(f"bought_{family}"):
        st.markdown(
            f"**Log purchase: {idea['symbol']} {idea['expiry']} ${float(idea['strike']):.2f} "
            f"{'call' if family.endswith('call') else 'put'}**"
        )
        c1, c2, c3 = st.columns(3)
        qty = c1.number_input("Contracts", min_value=1, value=1, step=1)
        price = c2.number_input("Price paid per share", min_value=0.01, value=max(ask, 0.01), step=0.05, format="%.2f")
        c3.metric("Cost", f"${qty * price * 100:,.0f}")
        if st.form_submit_button("I bought this"):
            store = Store()
            try:
                store.add_bought(
                    idea["symbol"], "call" if family.endswith("call") else "put", float(idea["strike"]),
                    str(idea["expiry"]), int(qty), float(price), family, idea.get("entry_level"),
                    idea.get("stop_level"), idea.get("stop_option"), idea.get("t1"), idea.get("t2"),
                )
            finally:
                store.close()
            st.success("Logged. Track it under Portfolio, Options, Bought options.")


def render_bought_tracker() -> None:
    store = Store()
    try:
        positions = store.bought_options()
    finally:
        store.close()
    openp = positions[positions["status"] == "open"] if not positions.empty else positions
    st.markdown(f"**Bought options (tracked)** · {len(openp)}")
    st.caption(
        "Options you logged with 'I bought this' in the Calls and Puts tabs. The buys job reprices them every 15 "
        "minutes. EXIT: the stock hit the idea's stop. TAKE PROFIT: it reached T1 or T2. Nothing is ordered."
    )
    if positions.empty:
        st.info("None yet. In Calls or Puts, select an idea row and press 'I bought this'.")
        return
    if st.button("Reprice now", key="reprice_bought"):
        store = Store()
        try:
            n, errs = refresh_marks(store, get_source())
        finally:
            store.close()
        for e in errs:
            st.error(e)
        st.rerun()
    today = datetime.now(NY).date()
    rows = []
    for _, p in openp.iterrows():
        mid, spot = p["last_mid"], p["last_spot"]
        mid = None if pd.isna(mid) else float(mid)
        spot = None if pd.isna(spot) else float(spot)
        signal, why = assess_position(p.to_dict(), spot, mid, today)
        pnl = None if mid is None else (mid - p["price"]) * p["qty"] * 100
        rows.append({
            "ID": int(p["id"]), "Signal": signal, "Ticker": p["symbol"], "Type": p["opt_type"],
            "Strike": p["strike"], "Expiry": p["expiry"], "Qty": int(p["qty"]), "Paid": p["price"],
            "Now": None if mid is None else round(mid, 2), "P&L $": None if pnl is None else round(pnl, 0),
            "P&L %": None if mid is None else round((mid / p["price"] - 1) * 100, 1),
            "Stock": spot, "Stop": p["stop_level"], "T1": p["t1"], "T2": p["t2"], "Why": why,
            "Priced": p["marked_at"],
        })
    table = pd.DataFrame(rows)
    show_freshness(openp["marked_at"].dropna().max() if openp["marked_at"].notna().any() else None, "intraday", "Bought-option")
    order = {"EXIT": 0, "TAKE PROFIT": 1, "EXPIRING": 2, "EXPIRED": 3, "HOLD": 4}
    st.dataframe(table.sort_values("Signal", key=lambda s: s.map(order)), hide_index=True, width="stretch")
    c1, c2, c3 = st.columns([2, 2, 2])
    option_id = c1.selectbox("Position", [r["ID"] for r in rows], format_func=lambda i: f"#{i} " + next(
        f"{r['Ticker']} {r['Expiry']} {r['Strike']:g} {r['Type']}" for r in rows if r["ID"] == i), key="bought_pick")
    sell_price = c2.number_input("Sold at (per share)", min_value=0.0, value=0.0, step=0.05, format="%.2f", key="bought_sell")
    if c3.button("Mark sold", key="bought_close", disabled=sell_price <= 0):
        store = Store()
        try:
            store.close_bought(int(option_id), sell_price)
        finally:
            store.close()
        st.rerun()
    if c3.button("Remove entry", key="bought_delete"):
        store = Store()
        try:
            store.delete_bought(int(option_id))
        finally:
            store.close()
        st.rerun()
    closed = positions[positions["status"] == "closed"]
    if not closed.empty:
        with st.expander(f"Closed · {len(closed)}"):
            c = closed.assign(**{"P&L $": ((closed["closed_price"] - closed["price"]) * closed["qty"] * 100).round(0)})
            st.dataframe(c[["symbol", "opt_type", "strike", "expiry", "qty", "price", "closed_price", "P&L $", "closed_at"]],
                         hide_index=True, width="stretch")


def render_buy_family(
    family: str, title: str, definition: str, hint: str, themes: list[str], triggers: list[str], top_n: int | None
) -> None:
    store = Store()
    try:
        recs = store.latest_option_recs(family)
    finally:
        store.close()

    st.markdown(f"#### {title}")
    side = "call" if family.endswith("call") else "put"
    if recs.empty:
        st.info(hint)
        return
    show_freshness(recs["run_ts"].max(), "intraday", "Buy-idea")
    recs = drop_expired(recs)
    recs = recs.assign(theme=recs["symbol"].map(theme_of))
    if themes:
        recs = recs[recs["theme"].isin(themes)]
    good = recs[recs["status"] == "recommend"]
    if triggers:
        good = good[good["trigger"].isin(triggers)]
    good = good.assign(_order=good["trigger"].map(TRIGGER_ORDER)).sort_values(
        ["_order", "multiple"], ascending=[True, False], na_position="last"
    )
    shown = good if top_n is None else good.head(top_n)
    suffix = f" (showing {len(shown)})" if len(shown) < len(good) else ""
    st.caption(f"{len(good)} ideas{suffix}. {definition}")
    if not shown.empty:
        store = Store()
        try:
            setup_rows = {r["symbol"]: r for _, r in store.latest_setups().iterrows()}
        finally:
            store.close()
        shown = shown.assign(
            why=[buy_why(r, setup_rows.get(r["symbol"]), family) for _, r in shown.iterrows()],
            size=shown["symbol"].map(size_of),
        )
    if good.empty:
        st.caption("None in this run.")
    else:
        event = st.dataframe(
            _buy_table(shown, side), hide_index=True, width="stretch", on_select="rerun",
            selection_mode="single-row", key=f"pick_{family}",
        )
        picked = event.selection.rows if event is not None else []
        if picked:
            render_bought_form(family, shown.iloc[picked[0]])
        else:
            st.caption("Select a row to log that option as bought and track it.")

    skipped = recs[(recs["status"] != "recommend") & ~recs["reason"].fillna("").str.startswith(TRANSIENT_ERRORS)]
    with st.expander(f"Watch list · {len(skipped)} — no contract or no setup"):
        if skipped.empty:
            st.caption("Nothing waiting.")
        else:
            view = skipped[["symbol", "theme", "expiry", "reason"]].rename(
                columns={"symbol": "Symbol", "theme": "Theme", "expiry": "Expiry", "reason": "Why there is no idea"}
            )
            st.dataframe(view.sort_values(["Theme", "Symbol"]), hide_index=True, width="stretch")


def render_call_ideas(
    family: str, title: str, definition: str, themes: list[str], triggers: list[str], top_n: int | None
) -> None:
    """Directional or LEAPS calls: break-even against the expected move, IV rank, theta, max loss and position size."""
    store = Store()
    try:
        recs = store.latest_option_recs(family)
        setup_rows = {r["symbol"]: r for _, r in store.latest_setups().iterrows()}
    finally:
        store.close()
    st.markdown(f"#### {title}")
    if recs.empty or "be_vs_move" not in recs:
        st.info("No run yet. Run the options job in the Jobs tab.")
        return
    show_freshness(recs["run_ts"].max(), "hourly", "Options")
    recs = drop_expired(recs)
    recs = recs.assign(theme=recs["symbol"].map(theme_of))
    if themes:
        recs = recs[recs["theme"].isin(themes)]
    good = recs[recs["status"] == "recommend"]
    if triggers:
        good = good[good["trigger"].isin(triggers)]
    good = good.assign(
        _order=good["trigger"].map(TRIGGER_ORDER), _ratio=pd.to_numeric(good["be_vs_move"], errors="coerce")
    ).sort_values(["_order", "_ratio"], ascending=[True, True], na_position="last")
    shown = good if top_n is None else good.head(top_n)
    st.caption(
        f"{len(good)} ideas{'' if len(shown) == len(good) else f' (showing {len(shown)})'}. {definition} "
        "Break-even / move is how far the stock must go to break even, as a multiple of the market's expected move to "
        "expiry (above 1.0 needs more than expected). Theta/day is the daily time decay as a share of the premium. "
        "Max loss is the whole premium. Rank is the real IV rank once 60 days of IV exist, otherwise an HV stand-in."
    )
    if shown.empty:
        st.caption("None in this run.")
    else:
        num = lambda col: pd.to_numeric(shown[col], errors="coerce")  # noqa: E731
        pct = lambda col, fmt="{:.1%}": num(col).map(lambda x: fmt.format(x) if pd.notna(x) else None)  # noqa: E731
        shown = shown.assign(
            why=[buy_why(r, setup_rows.get(r["symbol"]), family) for _, r in shown.iterrows()],
            size=shown["symbol"].map(size_of),
        )
        view = pd.DataFrame({
            "Ticker": shown["symbol"], "Company size": shown["size"], "Why recommended": shown["why"],
            "Trigger": shown["trigger"], "Expiry": shown["expiry"], "Days": shown["dte"], "Strike": shown["strike"],
            "Ask": num("ask").round(2), "Cost/contract": num("cost").round(0), "Delta": num("delta").round(2),
            "Break-even": num("breakeven").round(2), "Break-even over stock": pct("breakeven_pct"),
            "Expected move": num("expected_move").round(2), "Expected move %": pct("expected_move_pct"),
            "Break-even / move": num("be_vs_move").round(2), "Rank": num("rank").round(0), "Rank is": shown["rank_source"],
            "Theta/day of premium": pct("theta_pct", "{:.2%}"), "Time value": pct("extrinsic_pct", "{:.0%}"),
            "Max loss": num("max_loss").round(0), "Contracts": num("contracts"),
            "ENTRY": num("entry_level").round(2), "STOP (stock)": num("stop_level").round(2),
            "SL: or option at": num("stop_option").round(2), "T1 (stock)": num("t1").round(2), "T2 (stock)": num("t2").round(2),
            "Flags": shown["flags"].fillna(""),
        })
        event = st.dataframe(
            view.dropna(axis=1, how="all"), hide_index=True, width="stretch", on_select="rerun",
            selection_mode="single-row", key=f"pick_{family}",
        )
        picked = event.selection.rows if event is not None else []
        if picked:
            render_bought_form(family, shown.iloc[picked[0]])
        else:
            st.caption("Select a row to log that option as bought and track it.")
        if "contracts" in shown and shown["contracts"].isna().all():
            st.caption("Position size is off: set `account: {account_size: ...}` in config/settings.yaml to see contracts.")
    skipped = recs[(recs["status"] != "recommend") & ~recs["reason"].fillna("").str.startswith(TRANSIENT_ERRORS)]
    with st.expander(f"Watch list · {len(skipped)}: no contract in the delta band"):
        if skipped.empty:
            st.caption("Nothing waiting.")
        else:
            st.dataframe(
                skipped[["symbol", "expiry_type", "reason"]].rename(
                    columns={"symbol": "Symbol", "expiry_type": "Expiry", "reason": "Why there is no idea"}),
                hide_index=True, width="stretch",
            )


def render_calls_tab() -> None:
    st.caption(
        "Calls you BUY: you pay the premium and win if the stock rises, and the most you can lose is the premium. "
        "Selling calls against shares you own is in the Covered calls tab."
    )
    c1, c2, c3 = st.columns([4, 3, 1])
    themes = c1.multiselect("Theme", all_themes(), key="calls_themes", placeholder="All themes")
    triggers = c2.multiselect(
        "Trigger status", TRIGGER_CHOICES, key="calls_trigger", placeholder="All statuses"
    )
    top = c3.selectbox("Rows per section", [5, 10, 25, 50, "All"], index=1, key="calls_top")
    top_n = None if top == "All" else int(top)

    def buy(family: str, title: str, definition: str) -> None:
        render_buy_family(
            family, title, definition, "No buy run yet. Run the buys job in the Jobs tab.", themes, triggers, top_n
        )

    tab_dir, tab_leaps, tab_weekly, tab_lotto, tab_odte, tab_mag7, tab_monthly = st.tabs(
        ["Directional (14-60 DTE)", "LEAPS (270+ DTE)", "Weekly calls", "Weekly lottos", "0DTE", "MAG7", "Monthly calls"]
    )
    with tab_dir:
        render_call_ideas(
            "directional_call", "Directional calls (buy)",
            "Calls 14-60 days out with delta 0.60-0.70 on the bullish swing setups: real stock exposure, with the "
            "break-even weighed against the expected move.",
            themes, triggers, top_n,
        )
    with tab_leaps:
        render_call_ideas(
            "leaps_call", "LEAPS (buy, 270+ days)",
            "Calls at least 270 days out with delta 0.70-0.80 on the Long term picks and the bullish swing setups: a "
            "stock replacement that costs a fraction of the shares, with time value as the main cost.",
            themes, triggers, top_n,
        )
    with tab_weekly:
        buy(
            "weekly_call", "Weekly calls (buy, about 0.50 delta)",
            "Calls that expire this Friday with delta near 0.50 on bullish setups: a real position, not a lottery "
            "ticket, with the stock's entry, stop, and two targets.",
        )
    with tab_lotto:
        st.warning(
            "Lottos are cheap out-of-the-money weekly calls. The most you can lose is the premium, and most expire "
            "worthless. Size them small, and buy only after the trigger."
        )
        buy(
            "lotto_call", "Weekly lotto calls (buy)",
            "Cheap out-of-the-money weekly calls, picked for the best payoff if the stock reaches its first target.",
        )
    with tab_odte:
        st.error(
            "0DTE calls expire TODAY and can lose their whole premium within minutes. Quotes must be live (the buys "
            "job refreshes every 15 minutes in market hours), and the position should be closed before the bell."
        )
        buy(
            "odte_call", "0DTE calls (expire today)",
            "Same-day calls on SPY, QQQ, IWM, and the MAG7, planned from today's daily pivots: buy above the pivot, "
            "stop halfway to S1, targets at R1 and R2. If nothing expires today, the next expiry is used and noted.",
        )
    with tab_mag7:
        buy(
            "mag7_lotto_call", "MAG7 weekly lotto calls",
            "AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, always listed. Names without a bullish swing setup use the "
            "weekly pivots (reclaim P, stop S1, targets R1 and R2).",
        )
        buy(
            "mag7_swing_call", "MAG7 swing calls (buy, monthly, about 0.50 delta)",
            "Monthly calls with delta near 0.50 on the MAG7, with the stock's entry, stop, and two targets.",
        )
    with tab_monthly:
        buy(
            "long_call", "Swing calls (buy, monthly, about 0.50 delta)",
            "Monthly calls with delta near 0.50 on bullish setups, with the stock's entry, stop, and two targets.",
        )


def render_levels_tab(holdings: Holdings) -> None:
    held = sorted({sym for sym, _ in holdings.positions()}) or ["MSFT", "AMZN", "NVDA"]
    options = sorted(set(held) | {"MSFT", "AMZN", "NVDA"} | set(holdings.watchlist_scan))
    st.subheader("Levels")
    st.caption("Daily and weekly pivots, ATR(10), SMA(20), and the prior 20-day high, from live Schwab data.")
    symbols = st.multiselect("Symbols", options, default=held[:5], key="levels_symbols")
    if st.button("Fetch levels", key="levels_fetch", disabled=not symbols):
        source = get_source()
        for symbol in symbols:
            with st.expander(symbol, expanded=True):
                try:
                    bars = source.daily_bars(symbol)
                    spot = source.spot(symbol)
                    levels = compute_levels(bars, spot=spot, symbol=symbol)
                except Exception as exc:
                    st.error(f"{symbol}: {exc}")
                    continue
                render_levels(levels)
                render_zones(symbol, bars, spot, source)


def render_zones(symbol: str, bars: pd.DataFrame, spot: float, source) -> None:
    """Ranked support and resistance zones for one symbol, with the nearest monthly's open-interest walls and a chart."""
    import altair as alt

    from tool.expiries import monthly_expiry
    from tool.sr_levels import chart_data, compute_zones

    oi = None
    try:
        monthly = monthly_expiry(date.today())
        oi = {"calls": source.option_chain(symbol, expiry=monthly, right="C", strike_count=40),
              "puts": source.option_chain(symbol, expiry=monthly, right="P", strike_count=40)}
    except Exception:
        st.caption("Open-interest walls unavailable (no option chain); zones are from price history only.")
    zones = compute_zones(bars, spot, oi=oi)
    st.markdown("**Support and resistance zones** (strongest first)")
    st.caption(
        "Levels from swing highs and lows, SMA50/100/200, last month's high and low, volume nodes (60-day volume profile) and "
        "the biggest open-interest strikes of the next monthly expiry (call wall above the price, put wall below). Levels within "
        "half an ATR merge into a zone. Score ranks touches (bars that reached the zone in 120 days) and the volume traded on those days."
    )
    if not zones:
        st.info("Not enough price history to build zones.")
        return
    min_score = get_config().levels.min_zone_score
    table = pd.DataFrame({
        "Type": [z.kind for z in zones], "From": [round(z.low, 2) for z in zones], "To": [round(z.high, 2) for z in zones],
        "Away": [f"{(z.center / spot - 1):+.1%}" for z in zones], "Sources": [" + ".join(z.sources) for z in zones],
        "Touches": [z.touches for z in zones], "Volume at zone (M)": [round(z.volume / 1e6, 1) for z in zones],
        "Score": [round(z.score, 2) for z in zones], "Strong": ["yes" if z.score >= min_score else "" for z in zones],
    })
    st.dataframe(table, hide_index=True, width="stretch")
    frames = chart_data(bars, zones)
    candles = frames["candles"]
    color = alt.condition("datum.close >= datum.open", alt.value("#26a69a"), alt.value("#ef5350"))
    base = alt.Chart(candles).encode(x=alt.X("date:T", title=None))
    wick = base.mark_rule().encode(y=alt.Y("low:Q", scale=alt.Scale(zero=False), title="Price"), y2="high:Q", color=color)
    body = base.mark_bar(size=4).encode(y="open:Q", y2="close:Q", color=color)
    layers = [wick, body]
    if not frames["zones"].empty:
        layers.insert(0, alt.Chart(frames["zones"]).mark_rect().encode(
            x="start:T", x2="end:T", y="low:Q", y2="high:Q",
            color=alt.Color("kind:N", scale=alt.Scale(domain=["support", "resistance"], range=["#2e7d32", "#c62828"]), legend=None),
            opacity=alt.Opacity("score:Q", scale=alt.Scale(domain=[0, 1], range=[0.10, 0.45]), legend=None),
            tooltip=["label", "touches", "score"],
        ))
    averages = frames["averages"].melt("date", var_name="Average", value_name="price").dropna()
    layers.append(alt.Chart(averages).mark_line(strokeWidth=1).encode(
        x="date:T", y=alt.Y("price:Q", scale=alt.Scale(zero=False)), color=alt.Color("Average:N", legend=alt.Legend(orient="bottom")),
        strokeDash="Average:N"))
    st.altair_chart(alt.layer(*layers).properties(height=360), width="stretch")


def render_covered_calls_tab() -> None:
    render_cc_tiers()
    render_cc_monitor()
    render_cc_skips()


def render_longterm() -> None:
    tab_ideas, tab_mine, tab_consolidate = st.tabs(["Ideas", "My holdings", "Consolidation"])
    with tab_ideas:
        render_longterm_ideas()
    with tab_mine:
        render_holdings_review()
    with tab_consolidate:
        render_consolidation()


def _review_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    store = Store()
    try:
        return store.latest_holding_review(), store.latest_holding_pairs()
    finally:
        store.close()


def render_holdings_review() -> None:
    review, _ = _review_frames()
    cfg = get_config().longterm
    st.subheader("What you hold")
    st.caption(
        "Weights are shares of your stock portfolio at the latest price (cash and options are not included). Growth is "
        "trailing-twelve-month from Schwab; forward P/E, its 5-year average and the FCF margin come from Finnhub's annual "
        "figures; Fwd vs avg below 1.00 means cheaper than its own history. Funds and leveraged ETFs have no fundamentals, so "
        "they show price columns only. Refreshed by the daily long-term job."
    )
    if review.empty:
        st.info("No holdings review yet. Run the longterm job in the Jobs tab.")
        return
    show_freshness(review["as_of"].iloc[0], "daily", "Holdings review")
    over = review[review["concentrated"] == 1]
    if not over.empty:
        st.warning(
            f"Concentration: {', '.join(f'{r.symbol} {r.weight:.0%}' for r in over.itertuples())} "
            f"{'is' if len(over) == 1 else 'are'} above the {cfg.max_weight:.0%} limit for one position "
            "(`longterm.max_weight` in config/settings.yaml)."
        )
    num = lambda col: pd.to_numeric(review[col], errors="coerce")  # noqa: E731
    pct = lambda col, fmt="{:.0%}": num(col).map(lambda x: fmt.format(x) if pd.notna(x) else None)  # noqa: E731
    flag = review.apply(lambda r: "FUND" if r["fund"] else "", axis=1)
    flag = flag.where(review["concentrated"] != 1, "OVER LIMIT")
    view = pd.DataFrame({
        "Ticker": review["symbol"], "Company size": review["symbol"].map(size_of), "Flag": flag,
        "Weight": pct("weight", "{:.1%}"), "Value": num("value").round(0), "Price": num("price").round(2),
        "Gain on cost": pct("gain_pct"), "Revenue growth": pct("rev_growth"), "EPS growth": pct("eps_growth"),
        "Fwd P/E": num("forward_pe").round(1), "5y avg P/E": num("pe_5y").round(1), "Fwd vs avg": num("pe_vs_avg").round(2),
        "FCF margin": pct("fcf_margin"), "Below 52w high": pct("drawdown"), "vs SMA200": pct("dist_sma200", "{:+.0%}"),
        "Quality /6": num("quality"), "Trend /3": num("trend"), "Conviction": num("conviction").round(0),
    })
    st.dataframe(view.dropna(axis=1, how="all"), hide_index=True, width="stretch")


def render_consolidation() -> None:
    from tool.holding_review import HoldingReview, keep_suggestion

    review, pairs = _review_frames()
    cfg = get_config().longterm
    st.subheader("Cutting down the number of tickers")
    st.caption(
        "Conviction (0-100) blends quality 30%, trend 20%, growth 15%, valuation against its own history 15%, FCF margin 10% "
        "and resilience (shallow drawdown, above the 200-day) 10%; position size is not part of it. Lowest conviction first. "
        f"Overlap = daily returns correlated at {cfg.corr_threshold:.0%} or more over {cfg.corr_window} days: you hold the "
        "same bet twice."
    )
    if review.empty:
        st.info("No holdings review yet. Run the longterm job in the Jobs tab.")
        return
    show_freshness(review["as_of"].iloc[0], "daily", "Holdings review")
    strong = pairs[pairs["corr"] >= cfg.corr_threshold]
    scored = review[review["conviction"].notna()].sort_values("conviction")
    unscored = review[review["conviction"].isna()]

    def overlaps(symbol: str) -> str:
        hits = strong[(strong["a"] == symbol) | (strong["b"] == symbol)]
        return ", ".join(f"{r.b if r.a == symbol else r.a} ({r.corr:.2f})" for r in hits.itertuples())

    ranked = pd.DataFrame({
        "Ticker": scored["symbol"], "Conviction": scored["conviction"].round(0),
        "Weight": pd.to_numeric(scored["weight"], errors="coerce").map(lambda x: f"{x:.1%}" if pd.notna(x) else None),
        "Gain on cost": pd.to_numeric(scored["gain_pct"], errors="coerce").map(lambda x: f"{x:+.0%}" if pd.notna(x) else None),
        "Quality /6": scored["quality"], "Trend /3": scored["trend"], "Overlaps with": scored["symbol"].map(overlaps),
    })
    st.markdown(f"**Ranked by conviction, lowest first** · {len(scored)} holdings")
    st.dataframe(ranked, hide_index=True, width="stretch")
    if not unscored.empty:
        st.caption("No conviction score (funds or too little data): " + ", ".join(unscored["symbol"]))

    st.markdown(f"**Overlapping pairs** · {len(strong)} at {cfg.corr_threshold:.0%} correlation or more")
    if strong.empty:
        st.caption("No pair of holdings is that correlated over the window.")
        return
    by = {r.symbol: r for r in review.itertuples()}
    reviews = {s: HoldingReview(
        symbol=s, shares=0, price=None, value=None, weight=None, gain_pct=None, rev_growth=None, eps_growth=None,
        forward_pe=None, pe_5y=None, pe_vs_avg=None, fcf_margin=None, drawdown=None, dist_sma200=None, quality=None,
        trend=None, conviction=None if pd.isna(r.conviction) else float(r.conviction), fund=bool(r.fund), concentrated=False,
    ) for s, r in by.items()}
    table = pd.DataFrame({
        "Pair": [f"{r.a} / {r.b}" for r in strong.itertuples()],
        "Correlation": strong["corr"].round(2),
        "Weights": [f"{by[r.a].weight:.1%} / {by[r.b].weight:.1%}" if r.a in by and r.b in by else "" for r in strong.itertuples()],
        "Conviction": [
            f"{by[r.a].conviction:.0f} / {by[r.b].conviction:.0f}"
            if r.a in by and r.b in by and pd.notna(by[r.a].conviction) and pd.notna(by[r.b].conviction) else "n/a"
            for r in strong.itertuples()],
        "What to consider": [keep_suggestion(r.a, r.b, reviews) if r.a in reviews and r.b in reviews else "" for r in strong.itertuples()],
    })
    st.dataframe(table, hide_index=True, width="stretch")


def render_longterm_ideas() -> None:
    store = Store()
    try:
        recs = store.latest_longterm()
    finally:
        store.close()
    st.subheader("Long-term stock recommendations")
    st.caption(
        "Quality (growth, margins, returns, debt, valuation: at least 4 of 6) plus trend (above the 200-day, "
        "50-day above the 200-day). Entry is the 50-day average; the stop is the 200-day (never more than 12% "
        "below entry); T1 is the 52-week high; T2 is the same distance again. Minimum hold 12 months."
    )
    if recs.empty:
        st.info("No long-term run yet. Run the longterm job in the Jobs tab.")
        return
    recs = recs.assign(theme=recs["symbol"].map(theme_of))
    c1, c2 = st.columns([4, 1])
    themes = c1.multiselect("Theme", all_themes(), key="long_themes", placeholder="All themes")
    top = c2.selectbox("Rows", [10, 25, 50, "All"], index=0, key="long_top")
    if themes:
        recs = recs[recs["theme"].isin(themes)]
    show_freshness(recs["run_ts"].iloc[0] if len(recs) else None, "daily", "Long-term")
    st.caption(f"{len(recs)} stocks checked")

    def table(frame: pd.DataFrame) -> pd.DataFrame:
        view = frame.assign(
            hold=frame["hold_months"].map(lambda m: f"{m} months"), size=frame["symbol"].map(size_of)
        ).rename(
            columns={"symbol": "Ticker", "spot": "Price", "entry": "Entry", "hold": "Min hold", "stop": "SL",
                     "t1": "T1", "t2": "T2", "quality": "Quality /6", "trend": "Trend /3", "reason": "Why recommended",
                     "theme": "Theme", "size": "Company size"}
        )
        for col in ("Price", "Entry", "SL", "T1", "T2"):
            view[col] = pd.to_numeric(view[col], errors="coerce").round(2)
        return view[["Ticker", "Company size", "Why recommended", "Entry", "Min hold", "SL", "T1", "T2", "Price", "Quality /6", "Trend /3", "Theme"]]

    for status, title, note in (
        ("recommend", "Buy zone", "Quality and trend pass, and the price is within 10% of the 50-day average."),
        ("wait", "Wait for a pullback", "Quality and trend pass, but the price is extended; Entry is the pullback level."),
    ):
        part = recs[recs["status"] == status].sort_values(["quality", "trend"], ascending=False)
        shown = part if top == "All" else part.head(int(top))
        st.markdown(f"**{title}** · {len(part)}" + (f" (showing {len(shown)})" if len(shown) < len(part) else "") + f" — {note}")
        if part.empty:
            st.caption("None in this run.")
        else:
            st.dataframe(table(shown), hide_index=True, width="stretch")


ACTION_ORDER = {"CLOSE": 0, "ROLL": 1, "EXPIRES": 2, "OPEN": 3, "HOLD": 4}


TIER_ORDER = ("safe", "proven", "aggressive", "other")
CC_MODES = {"income": "Income (nearest the money)", "guardrail": "Guardrail (R1/R2, cost basis)"}


def render_roll_panel(open_calls: pd.DataFrame) -> None:
    """For each short call you hold: the rolls that pay a net credit, or the line to let it assign."""
    st.markdown("**Roll panel** (same or higher strike, same expiry or up to 3 weeks out, net credit only)")
    if open_calls.empty:
        st.caption("No open short calls on file.")
        return
    for _, call in open_calls.iterrows():
        raw = call.get("rolls")
        rolls = json.loads(raw) if isinstance(raw, str) and raw else []
        label = f"{call['symbol']} ${float(call['strike']):.2f} {call['expiry']} · {call['action']}"
        with st.expander(label, expanded=call["action"] in ("CLOSE", "ROLL")):
            if call["action"] == "EXPIRES":
                st.caption(str(call["reason"]))
            elif rolls:
                table = pd.DataFrame(rolls).rename(columns={
                    "expiry": "Roll to", "strike": "Strike", "bid": "Sell at", "net_credit": "Net credit",
                    "delta": "Delta", "weeks_out": "Weeks out"})
                for col in ("Sell at", "Net credit", "Delta"):
                    table[col] = pd.to_numeric(table[col], errors="coerce").round(2)
                st.dataframe(table[["Roll to", "Strike", "Net credit", "Sell at", "Delta", "Weeks out"]],
                             hide_index=True, width="stretch")
            else:
                st.info("No roll pays a net credit. Let it assign, then switch to selling puts (see the Puts tab).")
            if call.get("detail"):
                st.caption(str(call["detail"]))


def render_cc_tiers() -> None:
    """New covered calls to open, by tier: the income row (nearest the money at the tier's target) and the guardrail row."""
    from tool.income_calls import tier_target

    store = Store()
    try:
        saved = store.latest_covered_calls()
    finally:
        store.close()
    st.subheader("Calls to sell on your shares")
    st.caption(
        "Each holding shows two rows. Income: the strike nearest the stock price (never in the money, never under your "
        "cost basis) whose net weekly return meets the tier target. Guardrail: your CLAUDE.md rules (at or above weekly "
        "R1/R2, delta band, skips earnings). Probability ITM is approximated by delta. Choose per trade."
    )
    if saved.empty:
        st.info("No covered-call run saved yet. The monitor job saves one every 15 minutes.")
        return
    show_freshness(saved["run_ts"].iloc[0], "intraday", "Covered-call")
    recs = saved[saved["status"] == "recommend"].copy()
    if "tier" not in recs or "mode" not in recs:
        st.info("The saved run predates tiers. Run the covered-call monitor job.")
        return
    recs["tier"] = recs["tier"].fillna("other")
    recs["mode"] = recs["mode"].fillna("guardrail")
    for tier in TIER_ORDER:
        part = recs[recs["tier"] == tier]
        if part.empty:
            continue
        st.markdown(f"#### {tier.title()} · aim for {tier_target(tier):.1%} net per week")
        part = part.assign(_m=part["mode"].map({"income": 0, "guardrail": 1})).sort_values(["symbol", "account", "_m"])
        bid = pd.to_numeric(part["bid"], errors="coerce")
        view = pd.DataFrame({
            "Ticker": part["symbol"], "Account": part["account"], "Row": part["mode"].map(CC_MODES),
            "Free lots": part["free_lots"], "Expiry": part["expiry"], "Strike": part["strike"], "Bid": bid.round(2),
            "Net %": pd.to_numeric(part["net_pct"], errors="coerce").map(lambda x: f"{x:.2%}" if pd.notna(x) else None),
            "Target": pd.to_numeric(part["target_pct"], errors="coerce").map(lambda x: f"{x:.1%}" if pd.notna(x) else None),
            "Meets target": part["meets_target"].map({1: "yes", 0: "BELOW"}),
            "Delta": pd.to_numeric(part["delta"], errors="coerce").round(2),
            "Prob ITM": pd.to_numeric(part["prob_itm"], errors="coerce").map(lambda x: f"{x:.0%}" if pd.notna(x) else None),
            "Premium/lot": (bid * 100).round(0), "Total premium": (bid * 100 * part["free_lots"]).round(0),
            "Flags": part["flags"].fillna(""), "Notes": part["notes"].fillna(""),
        })
        st.dataframe(view.dropna(axis=1, how="all"), hide_index=True, width="stretch")
    held = saved[saved["reason"].fillna("").str.contains("uncapped|no_covered_calls", regex=True)]
    if not held.empty:
        names = ", ".join(sorted({f"{r.symbol} ({r.account})" for r in held.itertuples()}))
        st.caption(f"No calls (held uncapped): {names}")


def render_cc_monitor() -> None:
    store = Store()
    try:
        latest = store.latest_cc_snapshot()
        history_symbols = sorted(
            pd.read_sql_query("SELECT DISTINCT symbol FROM cc_snapshots", store._conn)["symbol"].tolist()
        )
    finally:
        store.close()
    st.subheader("Covered-call monitor")
    st.caption(
        "Every 15 minutes during market hours. CLOSE or ROLL: the stock is within 1% of your strike, or the call is "
        "in the money with under $0.50 of time value. ROLL means a same-or-higher-strike roll, within 3 weeks, pays a "
        "net credit and avoids earnings. New calls to open are in the tier view above. Snapshots are kept for 7 days."
    )
    if latest.empty:
        st.info("No monitor run yet. Run the ccmonitor job in the Jobs tab.")
        return
    show_freshness(latest["ts"].iloc[0], "intraday", "Covered-call monitor")
    open_calls = latest[latest["action"] != "OPEN"]  # new calls to open live in the tier view
    counts = open_calls["action"].value_counts()
    cols = st.columns(4)
    for col, action in zip(cols, ("CLOSE", "ROLL", "EXPIRES", "HOLD")):
        col.metric(action, int(counts.get(action, 0)))
    render_roll_panel(open_calls)
    latest = open_calls
    view = latest.assign(_o=latest["action"].map(ACTION_ORDER)).sort_values(["_o", "distance_pct"]).rename(
        columns={"symbol": "Ticker", "action": "Action", "strike": "Strike", "expiry": "Date", "contracts": "Contracts",
                 "spot": "Stock", "mid": "Option mid", "delta": "Delta", "extrinsic": "Time value",
                 "distance_pct": "To strike", "dte": "Days", "reason": "Why", "detail": "Detail"}
    )
    for col in ("Stock", "Option mid", "Time value", "Delta"):
        view[col] = pd.to_numeric(view[col], errors="coerce").round(2)
    view["To strike"] = pd.to_numeric(view["To strike"], errors="coerce").map(lambda x: f"{x:+.1%}" if pd.notna(x) else None)
    st.dataframe(
        view[["Ticker", "Action", "Strike", "Date", "Contracts", "Stock", "To strike", "Option mid", "Time value",
              "Delta", "Days", "Why", "Detail"]],
        hide_index=True, width="stretch",
    )
    with st.expander("History (last 7 days)"):
        if history_symbols:
            symbol = st.selectbox("Ticker", history_symbols, key="cc_history_symbol")
            store = Store()
            try:
                hist = store.cc_history(symbol)
            finally:
                store.close()
            st.dataframe(
                hist[["ts", "action", "strike", "expiry", "spot", "mid", "distance_pct", "reason"]],
                hide_index=True, width="stretch",
            )


def render_cc_skips() -> None:
    store = Store()
    try:
        saved = store.latest_covered_calls()
    finally:
        store.close()
    skipped = saved[saved["status"] != "recommend"] if not saved.empty else saved
    with st.expander(f"Why some positions have no new call · {len(skipped)}"):
        if saved.empty:
            st.caption("No covered-call run saved yet. The monitor job saves one every time it runs.")
            return
        show_freshness(saved["run_ts"].iloc[0], "intraday", "Covered-call")
        view = skipped.rename(columns={"symbol": "Ticker", "account": "Account", "reason": "Why", "free_lots": "Free lots"})
        st.dataframe(view[["Ticker", "Account", "Free lots", "Why"]], hide_index=True, width="stretch")


def render_swing() -> None:
    store = Store()
    try:
        setups = store.latest_setups()
    finally:
        store.close()
    holdings = load_holdings_safe()

    st.subheader("Top 10 swing recommendations")
    if setups.empty:
        st.info("No scan results yet. Start the scan job in the Jobs tab.")
    else:
        show_freshness(setups["scan_date"].max(), "daily", "Swing scan")
        st.caption(
            f"Breakout > pullback > extended, then reward:risk (each symbol shows its most recent scan). "
            "Recommend-only: confirm the setup on a chart before acting."
        )
        c_failed, c_earn = st.columns(2)
        show_failed = c_failed.checkbox(
            "Show failed breakouts", key="swing_show_failed",
            help="Closed above the 20-day high, but the live price is back under it. Hidden by default.",
        )
        allow_earnings = c_earn.checkbox(
            "Allow earnings", key="swing_allow_earnings",
            help="Keep setups whose earnings fall inside the 10-day hold, unless the expected earnings move is bigger "
                 "than the stop distance. Off by default.",
        )
        if "rechecked_at" in setups and setups["rechecked_at"].notna().any():
            st.caption(f"Prices re-checked at {pd.Timestamp(setups['rechecked_at'].max()):%a %b %d, %I:%M %p}.")
        excluded_col = setups["excluded"] if "excluded" in setups else pd.Series(None, index=setups.index, dtype=object)
        tradable = setups[(setups["kind"].isin(ACTIONABLE) | (setups["kind"] == "failed")) & setups["entry"].notna()].copy()
        reasons = excluded_col.reindex(tradable.index).copy()
        codes = tradable["excluded_code"] if "excluded_code" in tradable else pd.Series(None, index=tradable.index, dtype=object)
        if allow_earnings:
            # earnings inside the hold are fine unless the market expects a bigger move than the stop can absorb
            hold = codes.eq("earnings_hold")
            stop_pct = (tradable["entry"] - tradable["stop"]) / tradable["entry"]
            em = pd.to_numeric(tradable.get("expected_move_pct"), errors="coerce")
            too_big = hold & em.notna() & (em > stop_pct)
            reasons = reasons.mask(hold & ~too_big, None)
            for idx in tradable.index[too_big]:
                reasons[idx] = (
                    f"earnings expected move {em[idx]:.1%} is larger than the {stop_pct[idx]:.1%} stop"
                )
        tradable["excluded"] = reasons
        is_failed = reasons.fillna("").str.startswith("failed")
        keep = reasons.isna()
        actionable = tradable[keep].copy()
        actionable["rank"] = actionable["kind"].map(RANK)
        top = actionable.sort_values(["score", "rank", "rr"], ascending=[False, True, False], na_position="last").head(10)
        if top.empty:
            st.info("No actionable setups in the latest scan.")
        else:
            st.caption(
                "Ranked by score: 35% volume, 25% reward:risk, 20% trend, 20% earnings risk. Setups: breakout, bull flag, "
                "pullback to the 20/50-day average, pullback to pivot support, extended. Breakouts need a completed close "
                "above the 20-day high on 1.5x volume, price above the 200-day average, no earnings inside 10 trading days, "
                "and RSI(14) at most 75. Stop = below the recent swing low with a half-ATR buffer, never more than 2 ATR "
                "under the entry; T1 is capped at the nearest overhead resistance; R:R is measured from where you would "
                "buy (the higher of price and entry)."
            )
            st.dataframe(_setup_view(top), hide_index=True, width="stretch")
            if "shares" not in top or top["shares"].isna().all():
                st.caption("Position sizing is off: set `account: {account_size: ...}` in config/settings.yaml to see shares.")
        failed = tradable[is_failed]
        if show_failed:
            st.markdown(f"**Failed breakouts · {len(failed)}** (closed above the 20-day high, price now back under it)")
            if failed.empty:
                st.caption("None in the latest scan.")
            else:
                st.dataframe(
                    _setup_view(failed.sort_values("score", ascending=False, na_position="last")),
                    hide_index=True, width="stretch",
                )
        hide_from_excluded = ~keep & ~is_failed if show_failed else ~keep
        dropped = tradable[hide_from_excluded]
        with st.expander(f"Excluded · {len(dropped)} (setups the scanner found but dropped, and why)"):
            if dropped.empty:
                st.caption("Nothing was excluded in the latest scan.")
            else:
                view = dropped.rename(columns={"symbol": "Symbol", "kind": "Setup", "excluded": "Why excluded",
                                               "spot": "Spot", "entry": "Entry", "volume_ratio": "Volume vs avg",
                                               "rsi": "RSI"})
                view["Setup"] = view["Setup"].map(lambda k: SETUP_LABELS.get(k, k))
                view["Volume vs avg"] = pd.to_numeric(view["Volume vs avg"], errors="coerce").round(1)
                view["RSI"] = pd.to_numeric(view["RSI"], errors="coerce").round(0)
                view = view.sort_values(["Why excluded", "Symbol"])
                st.dataframe(view[["Symbol", "Setup", "Why excluded", "Spot", "Entry", "Volume vs avg", "RSI"]],
                             hide_index=True, width="stretch")
        weak = setups[setups["kind"] == "weak_breakout"].sort_values("volume_ratio", ascending=False)
        with st.expander(f"Breakouts waiting for volume · {len(weak)} (closed above the 20-day high on below-average volume)"):
            if weak.empty:
                st.caption("None in the latest scan.")
            else:
                st.dataframe(_setup_view(weak), hide_index=True, width="stretch")

    st.subheader("Watchlist")
    if setups.empty:
        st.caption("Watchlist setups appear after a scan.")
        return
    rows = []
    for symbol in holdings.watchlist_scan:
        match = setups[setups["symbol"] == symbol]
        if match.empty:
            empty = {col: None for col in SETUP_COLUMNS}
            rows.append({**empty, "symbol": symbol, "kind": "not in scan universe"})
        else:
            rows.append(match.iloc[0].to_dict())
    st.dataframe(_setup_view(pd.DataFrame(rows)), hide_index=True, width="stretch")


KIND_LABELS = {
    "breakout": "Swing: breakout",
    "pullback": "Swing: pullback",
    "extended": "Swing: extended (wait)",
    "below_weekly_s1": "Sell: below weekly S1",
    "failed_breakout": "Sell: failed breakout",
    "covered_call": "Covered call",
    "naked_call": "Naked call (UNLIMITED RISK)",
}


def render_saved(kinds: tuple[str, ...], empty_hint: str) -> None:
    store = Store()
    try:
        alerts = store.recent_alerts(limit=1000)
    finally:
        store.close()

    shown = alerts[alerts["kind"].isin(kinds)] if not alerts.empty else alerts
    if shown.empty:
        st.info(empty_hint)
        return

    shown = shown.assign(Type=shown["kind"].map(lambda k: KIND_LABELS.get(k, k)))
    shown = shown.rename(columns={"ts": "Time", "symbol": "Symbol", "detail": "Detail"})
    st.caption(f"{len(shown)} saved. Recommend-only: nothing here places an order.")
    st.dataframe(shown[["Time", "Symbol", "Type", "Detail"]], hide_index=True, width="stretch")


def render_jobs() -> None:
    jobs = JobManager()

    @st.fragment(run_every=5)
    def login_panel() -> None:
        age = token_age_days()
        if age is None:
            st.error("No Schwab login found. Press 'Sign in to Schwab' below.")
        elif age >= REFRESH_TOKEN_DAYS:
            st.error(f"Schwab login expired ({age:.1f} days old). Jobs fail until you sign in again.")
        elif age >= REFRESH_TOKEN_DAYS - 2:
            st.warning(f"Schwab login is {age:.1f} days old and expires after {REFRESH_TOKEN_DAYS}. Sign in soon.")
        else:
            st.caption(f"Schwab login: {age:.1f} days old (expires after {REFRESH_TOKEN_DAYS}).")
        valid = age is not None and age < REFRESH_TOKEN_DAYS
        b1, b2, b3 = st.columns(3)
        if b1.button("Sign in to Schwab", key="login_only", width="stretch"):
            jobs.launch(["schwab-login"], "login")
            st.info("A Schwab sign-in page is opening in your browser. Sign in once and approve; use a single tab.")
        if b2.button("Run all jobs now", key="run_all", disabled=not valid, width="stretch"):
            jobs.launch(["run-all"], "run_all")
            st.info("Running every job in order. The fast ones finish in minutes; the full scan takes about 20.")
        if b3.button("Sign in, then run all jobs", key="login_and_run", width="stretch"):
            jobs.launch(["run-all", "--login"], "run_all")
            st.info("Opening the Schwab sign-in. After you approve, every job runs in order.")
        log = jobs.dir / "run_all.log"
        if log.exists():
            with st.expander("Run-all log (latest output)"):
                lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
                st.code("\n".join(l for l in lines if "authlib" not in l.lower() and "httpx2" not in l)[-1500:] or "no output yet")

    login_panel()

    @st.fragment(run_every=5)
    def job_panel() -> None:
        for name, spec in JOB_SPECS.items():
            state = jobs.status(name)
            status = state.get("status", "never run")
            total = int(state.get("total") or 0)
            done = int(state.get("next_index") or 0) if total else 0

            st.subheader(spec["label"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Status", status)
            c2.metric("Progress", f"{min(done, total)} / {total}" if total else "-")
            c3.metric("Errors", state.get("errors", 0))
            if total:
                st.progress(min(done / total, 1.0))
            st.caption(f"Last update: {state.get('updated_at', '-')}  ·  PID {state.get('pid', '-')}")

            b1, b2, b3 = st.columns(3)
            if b1.button("Start", key=f"start-{name}", disabled=status == "running", width="stretch"):
                try:
                    jobs.start(name)
                except RuntimeError as exc:
                    st.error(str(exc))
                st.rerun(scope="fragment")
            if b2.button("Stop", key=f"stop-{name}", disabled=status != "running", width="stretch"):
                jobs.stop(name)
                st.rerun(scope="fragment")
            can_resume = status in ("stopped", "failed") and 0 < done < total
            if b3.button("Resume", key=f"resume-{name}", disabled=not can_resume, width="stretch"):
                try:
                    jobs.resume(name)
                except RuntimeError as exc:
                    st.error(str(exc))
                st.rerun(scope="fragment")

            log = jobs.log_path(name)
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:] if log.exists() else []
            with st.expander("Log (last 25 lines)"):
                st.code("\n".join(tail) or "no output yet")

    job_panel()


def render_whales_tab() -> None:
    from tool.whale_agg import burst_flags, net_flow, position_flow, position_map

    store = Store()
    try:
        latest = store.latest_whales()
        runs = store.whale_runs()
        history = store.whale_history(5)
        tracked = store.bought_options()
    finally:
        store.close()
    st.subheader("Whale activity")
    st.caption(
        "Contracts with at least $250K of premium traded today (volume x price x 100), $1M+ marked WHALE "
        "($2.5M and $10M for SPY, QQQ, IWM). Schwab gives total volume per contract, not single prints, so: "
        "Lean is inferred from where the last trade printed (at the ask a buyer, at the bid a seller); Intent compares "
        "volume with the open interest published this morning (opening = more than all of it); BURST means a large amount "
        "of premium was added since the previous 15-minute scan, which is NOT a sweep or block, whose prints are not "
        "available. Net flow totals are whale-size trades only. Deep in-the-money trades and dividend capture are "
        "filtered out as noise."
    )
    if latest.empty:
        st.info("No whale scan yet. Run the whales job in the Jobs tab.")
        return
    show_freshness(latest["ts"].iloc[0], "intraday", "Whale activity")

    flow_1d, flow_5d = net_flow(history, 1), net_flow(history, 5)
    positions = position_map(load_holdings_safe(), tracked)
    mine = position_flow(positions, flow_1d, flow_5d)
    st.markdown("#### Flow on your positions")
    if mine.empty:
        st.caption("No whale flow on names you hold shares or options in.")
    else:
        conflicts = mine[mine["status"] == "CONFLICT"]
        if not conflicts.empty:
            st.error(
                f"{len(conflicts)} of your positions run against whale flow: " + ", ".join(conflicts["symbol"]) + "."
            )
        view_mine = mine.rename(columns={
            "symbol": "Ticker", "status": "Status", "flow_today": "Flow today", "flow_5d": "Flow 5 days",
            "positions": "Your positions"})
        view_mine["Net today ($M)"] = (mine["net_today"] / 1e6).round(2)
        view_mine["Net 5 days ($M)"] = (mine["net_5d"] / 1e6).round(2)
        st.dataframe(
            view_mine[["Ticker", "Status", "Flow today", "Net today ($M)", "Flow 5 days", "Net 5 days ($M)", "Your positions"]],
            hide_index=True, width="stretch",
        )

    c1, c2, c3, c4 = st.columns(4)
    zero_dte = c1.checkbox("0DTE only", key="whales_0dte")
    opening_only = c2.checkbox("Opening only", key="whales_opening", help="Volume above, or at least half of, open interest.")
    bursts_only = c3.checkbox("Bursts only", key="whales_bursts", help="Large premium added since the previous scan.")
    show_noise = c4.checkbox("Show noise-filtered", key="whales_noise", help="Deep in the money, dividend capture.")
    c5, c6 = st.columns(2)
    leans = c5.multiselect("Lean", ["bullish", "bearish", "neutral"], key="whales_lean", placeholder="All")
    tiers = c6.multiselect("Size", ["WHALE", "Large"], key="whales_tier", placeholder="All")
    symbols = st.multiselect("Ticker", sorted(latest["symbol"].unique()), key="whales_symbols", placeholder="All")

    has_noise = latest["noise"].notna() if "noise" in latest else pd.Series(False, index=latest.index)
    view = latest if show_noise else latest[~has_noise]
    if zero_dte:
        view = view[view["dte"] == 0]
    if opening_only and "intent" in view:
        view = view[view["intent"].isin(["opening", "likely opening"])]
    if bursts_only:
        view = view[burst_flags(view)]
    if leans:
        view = view[view["lean"].isin(leans)]
    if tiers:
        view = view[view["tier"].isin(tiers)]
    if symbols:
        view = view[view["symbol"].isin(symbols)]
    if int(has_noise.sum()) and not show_noise:
        st.caption(f"{int(has_noise.sum())} contracts hidden as noise (deep in the money or dividend capture).")

    if view.empty:
        st.info("Nothing matches these filters in the latest scan.")
        return

    calls = view[view["right"] == "C"]["premium"].sum()
    puts = view[view["right"] == "P"]["premium"].sum()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Contracts", len(view))
    m2.metric("Call premium", f"${calls / 1e6:,.1f}M")
    m3.metric("Put premium", f"${puts / 1e6:,.1f}M")
    m4.metric("Put / call", f"{puts / calls:.2f}" if calls else "n/a")
    m5.metric("Bursts", int(burst_flags(view).sum()))

    st.markdown("**By ticker** (net lean = bullish premium minus bearish premium; Top = its largest contract)")
    work = view.assign(
        bull=view["premium"].where(view["lean"] == "bullish", 0.0),
        bear=view["premium"].where(view["lean"] == "bearish", 0.0),
        call_prem=view["premium"].where(view["right"] == "C", 0.0),
        put_prem=view["premium"].where(view["right"] == "P", 0.0),
    )
    by = work.groupby("symbol").agg(
        Contracts=("premium", "size"), Premium=("premium", "sum"), Bull=("bull", "sum"), Bear=("bear", "sum"),
        Calls=("call_prem", "sum"), Puts=("put_prem", "sum"), Spot=("spot", "first"),
    ).reset_index()
    big = view.sort_values("premium", ascending=False).drop_duplicates("symbol").set_index("symbol")
    by["Company size"] = by["symbol"].map(size_of)
    by["Top strike"] = by["symbol"].map(big["strike"])
    by["Top type"] = by["symbol"].map(big["right"].map({"C": "Call", "P": "Put"}))
    by["Top expiry"] = by["symbol"].map(big["expiry"])
    by["Net lean"] = (by["Bull"] - by["Bear"]).map(lambda v: f"{'+' if v >= 0 else '-'}${abs(v) / 1e6:,.2f}M")
    by["Bias"] = [
        "bullish" if b > 1.5 * r else "bearish" if r > 1.5 * b else "mixed" for b, r in zip(by["Bull"], by["Bear"])
    ]
    f1 = flow_1d.set_index("symbol")["net"] if not flow_1d.empty else pd.Series(dtype=float)
    f5 = flow_5d.set_index("symbol")["net"] if not flow_5d.empty else pd.Series(dtype=float)
    by["Net today ($M)"] = (by["symbol"].map(f1) / 1e6).round(2)
    by["Net 5 days ($M)"] = (by["symbol"].map(f5) / 1e6).round(2)
    by["Held"] = by["symbol"].map(lambda sym: "yes" if sym in positions else "")
    by = by.sort_values("Premium", ascending=False)
    for col in ("Premium", "Calls", "Puts"):
        by[col] = (by[col] / 1e6).round(2)
    by = by.rename(columns={"symbol": "Ticker", "Premium": "Premium ($M)", "Calls": "Calls ($M)", "Puts": "Puts ($M)"})
    st.dataframe(
        by[["Ticker", "Held", "Company size", "Spot", "Bias", "Net lean", "Net today ($M)", "Net 5 days ($M)", "Top strike",
            "Top type", "Top expiry", "Premium ($M)", "Calls ($M)", "Puts ($M)", "Contracts"]],
        hide_index=True, width="stretch",
    )

    st.markdown("**Largest contracts**")
    top = view.sort_values("premium", ascending=False).copy()
    top["Type"] = top["right"].map({"C": "Call", "P": "Put"})
    top["Premium ($M)"] = (top["premium"] / 1e6).round(2)
    top["From spot"] = top["otm_pct"].map(lambda x: f"{x:+.1%}")
    top["Vol / OI"] = pd.to_numeric(top["vol_oi"], errors="coerce").round(1)
    burst = pd.to_numeric(top["burst_premium"], errors="coerce") if "burst_premium" in top else pd.Series(np.nan, index=top.index)
    top["Added since last scan ($M)"] = (burst / 1e6).round(2)
    top["Burst"] = burst_flags(top).map({True: "BURST", False: ""})
    top["Delta"] = pd.to_numeric(top["delta"], errors="coerce").round(2)
    top["Company size"] = top["symbol"].map(size_of)
    top["Yours"] = top["symbol"].map(lambda sym: "; ".join(p.kind for p in positions.get(sym, [])))
    top["Noise"] = top["noise"].fillna("") if "noise" in top else ""
    top = top.rename(columns={"symbol": "Ticker", "expiry": "Expiry", "dte": "Days", "strike": "Strike",
                              "spot": "Stock", "volume": "Volume", "open_interest": "Open int", "price": "Price",
                              "side": "Traded at", "lean": "Lean", "tier": "Size", "intent": "Intent"})
    columns = ["Ticker", "Company size", "Strike", "Type", "Expiry", "Days", "Stock", "From spot", "Volume", "Open int",
               "Vol / OI", "Intent", "Price", "Premium ($M)", "Traded at", "Lean", "Size", "Burst",
               "Added since last scan ($M)", "Yours", "Delta"] + (["Noise"] if show_noise else [])
    st.dataframe(top[[c for c in columns if c in top.columns]].dropna(axis=1, how="all"), hide_index=True, width="stretch")
    with st.expander("Scan history (last 7 days)"):
        hist = runs.assign(premium=(runs["premium"] / 1e6).round(1)).rename(
            columns={"ts": "Scan time", "rows": "Contracts", "premium": "Premium ($M)"}
        )
        st.dataframe(hist, hide_index=True, width="stretch")


@st.cache_data(ttl=300, show_spinner="Scoring past recommendations...")
@st.cache_data(ttl=600, show_spinner="Scoring past ideas...")
def performance_scores(last_bar_day: str) -> dict[str, pd.DataFrame]:
    """Scored ideas: swing signals that were recommended, swing signals the filters excluded, and option buy ideas."""
    from tool.performance import load_option_ideas, load_signal_ideas, score_ideas

    store = Store()
    try:
        day = date.fromisoformat(last_bar_day)
        swing = score_ideas(load_signal_ideas(store._conn, "recommended"), store.load_daily_bars, lambda i: 1, lambda i: "limit", day)
        skipped = score_ideas(load_signal_ideas(store._conn, "excluded"), store.load_daily_bars, lambda i: 1, lambda i: "limit", day)
        options = score_ideas(
            load_option_ideas(store._conn), store.load_daily_bars,
            lambda i: -1 if str(i["group"]).endswith("put") else 1, lambda i: "stop", day,
        )
    finally:
        store.close()
    return {"swing": swing, "skipped": skipped, "options": options}


def _curve_chart(curves: dict[str, pd.DataFrame]) -> None:
    import altair as alt

    frames = [c.assign(series=name) for name, c in curves.items() if not c.empty]
    if not frames:
        return
    data = pd.concat(frames, ignore_index=True)
    st.altair_chart(
        alt.Chart(data).mark_line(point=True).encode(
            x=alt.X("date:T", title="Exit date"), y=alt.Y("equity:Q", title="Cumulative R"), color="series:N",
            tooltip=["series", "date:T", "r:Q", "equity:Q", "drawdown:Q"],
        ).properties(height=260),
        width="stretch",
    )


def _scoreboard(title: str, scored: pd.DataFrame, label: str, note: str) -> None:
    from tool.performance import equity_curve, low_sample, max_drawdown, open_mark, profit_factor, summarize

    st.markdown(f"#### {title}")
    st.caption(note)
    if scored.empty:
        st.info("Nothing to score yet.")
        return
    decided = scored[scored["result"].isin(["STOP", "T1", "T2"])]
    filled = scored[scored["result"].isin(["STOP", "T1", "T2", "OPEN"])]
    if low_sample(scored):
        st.warning(
            f"Only {len(decided)} decided ideas so far (fewer than {get_config().performance.min_sample}). "
            "Treat the statistics below as anecdotes, not evidence."
        )
    curve = equity_curve(scored)
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Ideas", len(scored), help=f"{len(filled)} filled")
    wins = decided["result"].isin(["T1", "T2"]).sum()
    m2.metric("Win rate (decided)", f"{wins / len(decided):.0%}" if len(decided) else "n/a", help="T1 or T2 reached before the stop")
    m3.metric("Expectancy", f"{decided['r'].mean():+.2f}R" if len(decided) else "n/a", help="Average R of decided ideas")
    pf = profit_factor(decided["r"]) if len(decided) else None
    m4.metric("Profit factor", f"{pf:.2f}" if pf is not None else "n/a", help="Winning R divided by losing R")
    m5.metric("Max drawdown", f"{max_drawdown(curve):.1f}R", help="Largest peak-to-trough fall of the 1R-per-trade curve")
    m6.metric("Open (marked)", f"{open_mark(scored):+.1f}R", help="Filled ideas not yet decided, at the last close; not in the curve")
    st.dataframe(summarize(scored).rename(columns={"group": label}), hide_index=True, width="stretch")
    if not curve.empty:
        st.caption("Equity curve: 1R risked on every decided idea, in exit-date order. No sizing, fees or limit on overlapping trades.")
        _curve_chart({title: curve})
    with st.expander("Every idea scored"):
        view = scored.rename(columns={"day": "Idea date", "symbol": "Ticker", "group": label, "entry": "Entry",
                                      "stop": "Stop", "t1": "T1", "t2": "T2", "result": "Result", "r": "R",
                                      "fill_date": "Filled", "exit_date": "Decided", "mfe_r": "MFE (R)", "mae_r": "MAE (R)",
                                      "days_held": "Days held"})
        view["R"] = pd.to_numeric(view["R"], errors="coerce").round(2)
        st.dataframe(view.sort_values("Idea date", ascending=False), hide_index=True, width="stretch")


def render_performance_tab() -> None:
    from tool.performance import summarize

    st.subheader("Performance")
    st.caption(
        "How the app's own ideas would have done. Each logged idea is replayed on the completed daily bars that came "
        "after it, against its own entry, stop, T1 and T2, over up to 10 trading days. A bar that touches both the "
        "stop and a target counts as a stop. R is profit in units of planned risk: a stop is -1R, T1 and T2 are "
        "positive. MFE and MAE are the best and worst excursion after the fill, in R. History starts when the app began "
        "saving, so the sample is small and grows each day."
    )
    now = datetime.now(NY)
    last_day = now.date() if now.time() >= MARKET_CLOSE else now.date() - timedelta(days=1)
    scores = performance_scores(last_day.isoformat())
    swing, skipped, options = scores["swing"], scores["skipped"], scores["options"]

    def family(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
        return frame[frame["group"].astype(str).str.endswith(suffix)] if not frame.empty else frame

    parts = [d.assign(tab=name) for name, d in (("Swing", swing), ("Calls", family(options, "call")), ("Puts", family(options, "put")))
             if not d.empty]
    if parts:
        st.markdown("#### By tab")
        st.dataframe(summarize(pd.concat(parts, ignore_index=True), by="tab"), hide_index=True, width="stretch")

    _scoreboard(
        "Swing setups", swing, "Setup",
        "Recommended swing signals from the signal log. Ideas are filled when price returns to the entry. Stock prices, no options involved.",
    )
    if not swing.empty and "verdict" in swing:
        st.markdown("#### By Claude verdict")
        if (swing["verdict"] == "unreviewed").all():
            st.caption("No Claude reviews yet (added in a later phase). Every signal is shown as unreviewed.")
        else:
            st.dataframe(summarize(swing, by="verdict"), hide_index=True, width="stretch")
    if not skipped.empty:
        with st.expander("Skipped by filters: would they have worked?"):
            st.caption(
                "Signals the filters excluded (earnings, volume, trend, overextension and so on), replayed the same way and never mixed "
                "into the numbers above. If these do as well as the recommended ones, a filter may be too strict."
            )
            by = skipped.assign(group=skipped["excluded_code"].fillna("other")) if "excluded_code" in skipped else skipped
            st.dataframe(summarize(by).rename(columns={"group": "Excluded because"}), hide_index=True, width="stretch")
    _scoreboard(
        "Option buy ideas", options, "Idea type",
        "Filled when the stock trades through the idea's entry (above for calls, below for puts), then judged on the "
        "stock reaching T1/T2 or the stop. This measures whether the direction was right, not the option's exact "
        "return, which also depends on the premium paid. Lottos are left out: they exit on the option price, not a "
        "stock stop. The same stock can appear on several days, so ideas overlap.",
    )

    st.markdown("#### Your logged trades")
    store = Store()
    try:
        trades = store.bought_options()
    finally:
        store.close()
    if trades.empty:
        st.info("No trades logged yet. Select an idea in Calls or Puts and press 'I bought this'.")
        return
    closed = trades[trades["status"] == "closed"].copy()
    openp = trades[trades["status"] == "open"].copy()
    closed["pnl"] = (closed["closed_price"] - closed["price"]) * closed["qty"] * 100
    openp["pnl"] = (pd.to_numeric(openp["last_mid"], errors="coerce") - openp["price"]) * openp["qty"] * 100
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Closed trades", len(closed))
    m2.metric("Win rate", f"{(closed['pnl'] > 0).mean():.0%}" if len(closed) else "n/a")
    m3.metric("Realized P&L", f"${closed['pnl'].sum():,.0f}")
    m4.metric("Open P&L (marked)", f"${openp['pnl'].sum():,.0f}")
    if len(closed):
        by_family = closed.groupby(closed["family"].fillna("other")).agg(
            Trades=("pnl", "size"), Wins=("pnl", lambda p: int((p > 0).sum())), PnL=("pnl", "sum")
        ).reset_index().rename(columns={"family": "Idea type", "PnL": "P&L $"})
        st.dataframe(by_family, hide_index=True, width="stretch")


def main() -> None:
    st.title("SBTrade")
    st.caption("Recommend-only. Nothing here places or stages an order.")
    auth_ok, auth_message = auth_status()
    if not auth_ok:
        st.error(
            f"Schwab re-auth required. {auth_message} Prices, chains and every job need a fresh sign-in; what is "
            "on screen is the last data saved."
        )
        if st.button("Sign in to Schwab", key="banner_login"):
            JobManager().launch(["schwab-login"], "login")
            st.info("A Schwab sign-in page is opening in your browser. Sign in once and approve; use a single tab.")

    holdings = load_holdings_safe()
    tab_long, tab_swing, tab_calls, tab_puts, tab_whales, tab_cc, tab_levels, tab_portfolio, tab_perf, tab_jobs = st.tabs(
        ["Long term", "Swing", "Calls", "Puts", "Whales", "Covered calls", "Levels", "Portfolio", "Performance", "Jobs"]
    )

    with tab_long:
        render_longterm()
    with tab_swing:
        render_swing()
        st.subheader("Sell signals on your holdings")
        render_saved(
            ("below_weekly_s1", "failed_breakout"),
            "No sell signals saved yet. Run the scan job in the Jobs tab.",
        )
    with tab_calls:
        render_calls_tab()
    with tab_puts:
        render_puts_tab()
    with tab_whales:
        render_whales_tab()
    with tab_cc:
        render_covered_calls_tab()
    with tab_levels:
        render_levels_tab(holdings)
    with tab_portfolio:
        render_portfolio()
    with tab_perf:
        render_performance_tab()
    with tab_jobs:
        render_jobs()


if __name__ == "__main__":
    main()
