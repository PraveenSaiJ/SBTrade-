"""Score the app's past recommendations against what the stock did next.

Each saved idea carries a stock-price entry, stop, T1 and T2. We replay the completed daily bars after the idea date:
- "limit" entry: filled when price comes back to the entry (swing setups, pullbacks).
- "stop" entry: filled when price trades through the entry in the idea's direction (option ideas: buy above / below X).
- After the fill the stop, T1 and T2 are checked bar by bar. A bar that touches the stop and a target counts as a
  stop (daily bars cannot show which came first, so this is the conservative reading).
Outcomes: STOP, T1 (T1 reached, T2 not), T2, OPEN (filled, nothing decided yet), NO FILL, PENDING (not yet filled, still
inside the window). R is profit in units of the planned risk (entry to stop). Stock-level only: option P&L also depends
on premium, so this measures whether the call was right, not the option's exact return.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
from tool.config import get_config

HORIZON_DAYS = get_config().performance.horizon_days  # trading days: the 1-2 week swing hold


@dataclass(frozen=True)
class Outcome:
    result: str  # STOP | T1 | T2 | OPEN | NO FILL | PENDING
    r: float | None
    fill_date: date | None = None
    exit_date: date | None = None
    mfe_r: float | None = None  # best excursion after the fill, in R (bar highs for a long)
    mae_r: float | None = None  # worst excursion after the fill, in R (a negative number)
    days_held: int | None = None  # trading days from the fill bar to the deciding bar (open: to the last bar), inclusive


def evaluate_idea(
    bars: pd.DataFrame, after: date, entry: float, stop: float, t1: float, t2: float | None,
    direction: int = 1, trigger: str = "limit", horizon: int = HORIZON_DAYS,
) -> Outcome:
    """Replay completed `bars` (index = dates) after `after`. direction: 1 long, -1 short."""
    risk = abs(entry - stop)
    if risk <= 0:
        return Outcome("NO FILL", None)
    window = bars[bars.index.date > after].head(horizon)
    # work in "long" terms: flip prices for shorts so one set of comparisons serves both
    e, s, a, b = (entry, stop, t1, t2) if direction == 1 else (-entry, -stop, -t1, None if t2 is None else -t2)
    fill_pos = hit_t1_pos = None
    fill_date = hit_t1_date = None
    mfe = mae = 0.0
    last_pos = -1

    def done(result: str, r: float, pos: int, day: date, held_to: int | None = None) -> Outcome:
        return Outcome(result, r, fill_date, day, round(mfe, 2), round(mae, 2), (pos if held_to is None else held_to) - fill_pos + 1)

    for pos, (ts, bar) in enumerate(window.iterrows()):
        high, low = float(bar["high"]), float(bar["low"])
        hi, lo = (high, low) if direction == 1 else (-low, -high)
        last_pos = pos
        if fill_pos is None:
            if not (lo <= e if trigger == "limit" else hi >= e):
                continue
            fill_pos, fill_date = pos, ts.date()
        mfe, mae = max(mfe, (hi - e) / risk), min(mae, (lo - e) / risk)
        if lo <= s:
            return done("STOP", -1.0, pos, ts.date()) if hit_t1_pos is None else done("T1", abs(t1 - entry) / risk, pos, ts.date())
        if b is not None and hi >= b:
            return done("T2", abs(t2 - entry) / risk, pos, ts.date())
        if hi >= a and hit_t1_pos is None:
            hit_t1_pos, hit_t1_date = pos, ts.date()  # T1 reached: keep looking for T2 or a stop on later bars
    if hit_t1_pos is not None:
        return done("T1", abs(t1 - entry) / risk, last_pos, hit_t1_date, held_to=hit_t1_pos)
    if fill_pos is None:
        return Outcome("PENDING" if len(window) < horizon else "NO FILL", None)
    last = float(window["close"].iloc[-1])
    return Outcome("OPEN", direction * (last - entry) / risk, fill_date, None, round(mfe, 2), round(mae, 2), last_pos - fill_pos + 1)


def score_ideas(ideas: pd.DataFrame, load_bars, direction_of, trigger_of, max_day: date) -> pd.DataFrame:
    """One row per idea with its outcome. `ideas` needs: day, symbol, group, entry, stop, t1, t2."""
    cache: dict[str, pd.DataFrame] = {}
    rows = []
    for _, i in ideas.iterrows():
        sym = i["symbol"]
        if sym not in cache:
            cache[sym] = load_bars(sym)
        bars = cache[sym]
        bars = bars[bars.index.date <= max_day]
        if bars.empty or pd.isna(i["entry"]) or pd.isna(i["stop"]) or pd.isna(i["t1"]):
            continue
        if abs(float(i["entry"]) - float(i["stop"])) < 0.001 * float(i["entry"]):
            continue  # no stock-level stop (lottos exit on the option price), so there is no risk to measure
        t2 = None if pd.isna(i["t2"]) else float(i["t2"])
        out = evaluate_idea(
            bars, date.fromisoformat(i["day"]), float(i["entry"]), float(i["stop"]), float(i["t1"]), t2,
            direction_of(i), trigger_of(i),
        )
        rows.append({**i.to_dict(), "result": out.result, "r": out.r, "fill_date": out.fill_date, "exit_date": out.exit_date,
                     "mfe_r": out.mfe_r, "mae_r": out.mae_r, "days_held": out.days_held})
    return pd.DataFrame(rows)


DECIDED = ("STOP", "T1", "T2")
FILLED = (*DECIDED, "OPEN")


def profit_factor(r: pd.Series) -> float | None:
    """Sum of winning R over the sum of losing R; None when there are no losses (or no trades)."""
    r = pd.to_numeric(r, errors="coerce").dropna()
    losses = -r[r < 0].sum()
    return None if r.empty or losses <= 0 else round(float(r[r > 0].sum() / losses), 2)


def summarize(scored: pd.DataFrame, by: str = "group") -> pd.DataFrame:
    """Per group: counts, win rate (T1 or T2 of decided), average R on filled ideas (open ones marked to the last close),
    expectancy (mean R of decided ideas), profit factor, average MFE / MAE (R) and days held."""
    if scored.empty:
        return pd.DataFrame()
    out = []
    for key, g in scored.groupby(by):
        filled = g[g["result"].isin(FILLED)]
        decided = g[g["result"].isin(DECIDED)]
        wins = decided[decided["result"].isin(["T1", "T2"])]

        def mean(frame: pd.DataFrame, col: str, digits: int = 2):
            if col not in frame:
                return None
            v = pd.to_numeric(frame[col], errors="coerce").dropna()
            return None if v.empty else round(float(v.mean()), digits)

        out.append({
            by: key, "Ideas": len(g), "Filled": len(filled), "Decided": len(decided),
            "T2": int((g["result"] == "T2").sum()), "T1": int((g["result"] == "T1").sum()),
            "Stopped": int((g["result"] == "STOP").sum()), "Open": int((g["result"] == "OPEN").sum()),
            "No fill": int((g["result"] == "NO FILL").sum()),
            "Win rate": None if decided.empty else round(len(wins) / len(decided) * 100, 0),
            "Avg R": mean(filled, "r"),
            "Expectancy": mean(decided, "r"),
            "Profit factor": profit_factor(decided["r"]),
            "Avg MFE": mean(filled, "mfe_r"), "Avg MAE": mean(filled, "mae_r"), "Avg days": mean(filled, "days_held", 1),
        })
    return pd.DataFrame(out).sort_values("Ideas", ascending=False)


def low_sample(scored: pd.DataFrame, minimum: int | None = None) -> bool:
    """True when fewer ideas than `performance.min_sample` have been decided, so the statistics are not yet meaningful."""
    minimum = get_config().performance.min_sample if minimum is None else minimum
    return scored.empty or int(scored["result"].isin(DECIDED).sum()) < minimum


def equity_curve(scored: pd.DataFrame) -> pd.DataFrame:
    """Cumulative R of a hypothetical system that risks 1R on every decided idea, in exit-date order.
    Columns: date, r, equity, drawdown (peak minus equity, in R). No sizing, fees or limit on overlapping trades."""
    cols = ["date", "r", "equity", "drawdown"]
    if scored.empty:
        return pd.DataFrame(columns=cols)
    d = scored[scored["result"].isin(DECIDED) & scored["exit_date"].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=cols)
    d["date"] = pd.to_datetime(d["exit_date"])
    d = d.sort_values(["date", "symbol"]).reset_index(drop=True)
    d["r"] = pd.to_numeric(d["r"])
    d["equity"] = d["r"].cumsum()
    d["drawdown"] = d["equity"].cummax().clip(lower=0) - d["equity"]
    return d[cols]


def max_drawdown(curve: pd.DataFrame) -> float:
    return 0.0 if curve.empty else float(curve["drawdown"].max())


def open_mark(scored: pd.DataFrame) -> float:
    """R of the filled-but-undecided ideas marked to the last close (kept out of the curve)."""
    if scored.empty:
        return 0.0
    return float(pd.to_numeric(scored[scored["result"] == "OPEN"]["r"], errors="coerce").sum())


def score_all(store, last_day: date) -> dict[str, pd.DataFrame]:
    """Scored swing signals (recommended and excluded) and option buy ideas, from completed bars up to `last_day`."""
    swing = score_ideas(load_signal_ideas(store._conn, "recommended"), store.load_daily_bars, lambda i: 1, lambda i: "limit", last_day)
    skipped = score_ideas(load_signal_ideas(store._conn, "excluded"), store.load_daily_bars, lambda i: 1, lambda i: "limit", last_day)
    options = score_ideas(
        load_option_ideas(store._conn), store.load_daily_bars,
        lambda i: -1 if str(i["group"]).endswith("put") else 1, lambda i: "stop", last_day,
    )
    return {"swing": swing, "skipped": skipped, "options": options}


BUY_FAMILIES = ("directional_call", "long_call", "long_put", "weekly_call", "weekly_put", "lotto_call", "lotto_put", "mag7_lotto_call", "mag7_swing_call", "odte_call", "odte_put")


def load_signal_ideas(conn, status: str = "recommended") -> pd.DataFrame:
    """Swing signals from the `signals` table (status recommended or excluded) as ideas: group = setup kind, plus tab and verdict."""
    return pd.read_sql_query(
        "SELECT signal_date AS day, tab, symbol, setup AS 'group', entry, stop, t1, t2, "
        "COALESCE(verdict, 'unreviewed') AS verdict, excluded_code FROM signals "
        "WHERE status = ? AND entry IS NOT NULL AND stop IS NOT NULL AND t1 IS NOT NULL ORDER BY signal_date, symbol",
        conn, params=(status,),
    )


def load_option_ideas(conn) -> pd.DataFrame:
    """Buy ideas that were recommended, first sighting per day, symbol and family (levels are stock prices)."""
    marks = ",".join("?" for _ in BUY_FAMILIES)
    df = pd.read_sql_query(
        f"SELECT substr(run_ts,1,10) AS day, run_ts, symbol, family AS 'group', entry_level AS entry, "
        f"stop_level AS stop, t1, t2 FROM option_recs WHERE status = 'recommend' AND family IN ({marks}) "
        f"AND entry_level IS NOT NULL ORDER BY run_ts",
        conn, params=BUY_FAMILIES,
    )
    return df.drop_duplicates(["day", "symbol", "group"], keep="first").drop(columns="run_ts")
