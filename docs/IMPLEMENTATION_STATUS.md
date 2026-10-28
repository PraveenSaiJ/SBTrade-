# SBTrade: implementation and status

Status as of 2026-10-28. Personal, **recommend-only** trading tool for Praveen. It never places or stages orders and calls no trading
endpoints. All execution is manual. The rules in `CLAUDE.md` are the specification; this document says what is built, how it fits together,
and what is not done.

## 1. What it does

| Engine | Output | Where |
|---|---|---|
| Swing stock signals (1-2 week holds) | entry, structural stop, T1/T2, R:R, score, expected move, position size | Swing tab |
| Covered calls on holdings | guardrail row + income row per holding, roll panel, monitor | Covered calls tab |
| Long calls / puts and cash-secured puts | directional, LEAPS, weekly, lotto, 0DTE, MAG7, monthly; support puts, weekly puts | Calls, Puts tabs |
| Whale option flow (read-only) | large-premium 0DTE/weekly contracts, intent, net flow, conflict with your positions | Whales tab |
| Long-term stock picks and holdings review | conviction score, concentration, overlap pairs | Long term tab |
| Support / resistance zones | merged zones from swings, SMAs, prior month, volume nodes, OI walls | Levels tab |
| Portfolio risk across accounts | net delta/theta, SPY beta-weighted delta, sectors, this-week risk, put collateral | Portfolio tab |
| Performance of the app's own ideas | outcomes, MFE/MAE, expectancy, profit factor, equity curve | Performance tab |
| Earnings calls | event-trade calls on names reporting within 7 days, break-even vs expected move, IV-crush note | Calls tab > Earnings |
| Options hub | weekly calls/puts, lottos, 0DTE calls/puts in one place (Calls/Puts tabs keep them too) | Options tab |
| CSV export | a CSV download in every tab plus one combined file / ZIP | each tab, Export tab |
| Job control | start/stop/resume, schedules, history, status | Jobs tab |

Dashboard: Streamlit at http://localhost:8501 (`.venv\Scripts\python.exe -m streamlit run app.py`). CLI: `uv run tool <command>`.

## 2. Architecture

```
Schwab Trader API (primary)  --\
Finnhub (earnings, metrics,     >-- src/tool/data/*  -- indicators.py / config.py -- engines -- SQLite (data/sbtrade.sqlite)
  industry, news)              /                                                            \-- app.py (Streamlit)
Robinhood / broker CSV --------/                                                             \-- CLI + scheduled jobs
```

- **Data layer** (`src/tool/data/`): `schwab.py` (batched quotes, 120/min rate limiter, option chains with delta/IV/OI/theta, price history,
  fundamentals), `barcache.py` (daily bars cached in SQLite; only bars saved after 16:15 ET are trusted), `memo.py` (per-run memo),
  `earnings.py` (Finnhub calendar cached daily, metrics cache, industry cache), `robinhood_csv.py` (activity statements), `ibkr.py` (backup).
  Token problems raise `SchwabAuthRequired`, which the dashboard shows as a re-sign-in banner.
- **Indicators and thresholds**: every indicator lives in `indicators.py` (SMA/EMA, Wilder ATR and RSI, volume ratio, swing points, IV rank, expected
  move). Every threshold lives in `config.py` (dataclass sections screener, options, whales, covered_calls, puts, calls, levels, freshness, longterm,
  performance, account, data) and can be overridden in `config/settings.yaml`. No thresholds in tab code.
- **Storage**: SQLite `data/sbtrade.sqlite` (gitignored). Tables: daily_bars, spot_snapshots, option_snapshots, setups, signals, covered_calls,
  cc_snapshots, option_recs, bought_options, whale_flow, whale_last, longterm_recs, holding_review, holding_pairs, iv_history, company_size,
  lots, short_options, long_options, settings, portfolio_positions, portfolio_info, portfolio_meta, alerts. Every saved recommendation carries
  `generated_at`, `data_as_of`, `source`.
- **Holdings**: `config/holdings.yaml` (gitignored, has cost basis) mirrored into the database; edited in the Portfolio tab; each stock lot and each option
  carries an account (`rh_taxable`, `rh_ira`, `ibkr`). Robinhood has no API, so positions come from CSV imports only.
- **Jobs**: each job is a detached `python -m tool.cli <command> --job NAME` process; state in `data/jobs/NAME.json`, log in `NAME.log`.

## 3. Tabs and the rules they apply

- **Swing**: kinds breakout, bull_flag, ma_pullback, pullback, extended, failed. Filters: breakout volume >= 1.5x, no earnings inside the 10-trading-day
  hold, close above SMA200, RSI <= 75 and not more than 2 ATR over the entry. Structural stop = max(min(entry - 0.5 ATR, swing low), entry - 2 ATR).
  T1 capped at the nearest strong resistance zone. R:R measured from the fill. Score = 0.35 volume + 0.25 R:R + 0.20 trend + 0.20 earnings risk.
  Excluded setups are logged with a reason code. Optional position size once `account.account_size` is set. Re-check job flags failed breakouts.
- **Covered calls**: guardrail row (strike >= weekly R1, R2 when breaking out, delta 0.20-0.30, never under cost basis, earnings expiries skipped) and
  income row (strike nearest the price meeting the tier return target). Flags: earnings, ex-dividend, early assignment, manual events. Roll panel lists
  net-credit rolls only. CRWV and SNOW (IRA) are uncapped. A 15-minute monitor warns about closes and rolls.
- **Calls / Puts**: directional (14-60 DTE, delta 0.60-0.70), LEAPS (270+ DTE, delta 0.70-0.80), weekly calls and lottos, 0DTE, MAG7, monthly;
  support puts (strike at or under the nearest strong support, delta 0.20-0.30, 7-45 DTE) beside the weekly-S1 guardrail puts. Break-even against
  expected move, theta share, IV-crush flag, collateral check against `puts.cash`. "I bought this" logs a trade and tracks it.
- **Whales**: contract premium of at least $250K ($1M+ is a WHALE; x10 for SPY/QQQ/IWM), scanned every 15 minutes; lean from the print location,
  intent from volume against open interest, BURST proxy for fresh premium, noise filter, 1-day and 5-day net flow, conflict with your positions.
- **Long term**: ideas, a holdings review (weights, revenue/EPS growth, valuation, FCF margin, drawdown, conviction 0-100, 15% concentration flag,
  correlated pairs at 0.80+), and a consolidation view.
- **Levels**: zones with touches, volume and a score; chart with candles, SMAs and zone bands; classic floor pivots (daily and weekly) per `CLAUDE.md`.
- **Portfolio**: Risk sub-tab (net greeks, SPY-equivalent delta, industry and theme exposure, this-week risk, held-name events, cash against put
  collateral), Stocks, Options, statement upload, positions-CSV import (replaces one account), Watchlist.
- **Performance**: replays each logged idea on completed daily bars (10 trading days) and reports outcome (STOP/T1/T2/OPEN/NO FILL/PENDING), R, MFE,
  MAE, days held, win rate, expectancy, profit factor, equity curve and drawdown, plus "skipped by filters" and a Claude-verdict table (filled when you press Review with Claude).
- **Jobs**: per-job panels with status/progress/errors/log, Schwab login age, "Run all jobs".

## 4. Scheduled work (Windows Task Scheduler, `scripts/register_tasks.ps1`)

Weekdays, local (Eastern): Schwab login 08:45 Mondays (needs you at the browser), swing scan 09:00, long-term 09:25, quick refresh 09:35,
options calls 10:00, options puts 10:10, re-check 10:15, whales / covered-call monitor / buy ideas (monthly swing + MAG7) every 15 minutes,
**0DTE calls and puts every 15 minutes** (from 09:35), **weekly calls and puts every 30 minutes** (from 09:45), covered-call and roll check 15:45,
outcomes 16:30. `run-job` skips weekends and NYSE holidays, logs every run to `data/jobs/history.sqlite`, and raises a desktop notification on
failure or when the Schwab sign-in is needed. Jobs fail with a clear message when the Schwab token is missing or older than 7 days.

### Options job speed
The options job is split into calls and puts halves; families run `data.options_workers` (4) at a time. `MemoSource` fetches each chain, price and bars
once per run (thread-safe) and serves several expiries from one Schwab range request (`hint_chains`). `CalendarEvents` answers earnings asks from one
cached Finnhub calendar. Measured: 22.6 min single job -> calls 3.3 min + puts 0.4 min; 0DTE and weekly jobs about 12 s each.

## 5. Tests and how to run

`uv run pytest -q` (498 tests at last full run; includes a headless Streamlit render of the whole dashboard against the local database).
Level math fixtures (MSFT) are in `CLAUDE.md`.

## 6. Phase status (SBTrade_improvement_plan.md)

| Phase | Subject | Status |
|---|---|---|
| 0-2 | Foundation, indicators/config, data layer | done |
| 3 | Swing: filters, structural stops, new setups, signals log | done |
| 4 | Covered calls: two rows, tiers, flags, rolls | done |
| 5 | Puts: support row, earnings gate, collateral | done |
| 6 | Calls: directional and LEAPS | done |
| 7 | Whales: intent, burst proxy, net flow, conflicts | done |
| 8 | Long term: holdings review | done |
| 9 | Support / resistance zones | done |
| 10 | Portfolio risk and positions import | done |
| 11 | Performance tab | done |
| 12 | Jobs tab: schedules, history, notifications, holidays | done (re-run `scripts/register_tasks.ps1` once to add the new tasks) |
| 13 | Claude review | API integration removed by choice; every tab offers a CSV download (plus the Export tab) for manual review |
| Extra | Earnings calls, Options tab, CSV exports, options speed-up, 0DTE/weekly jobs | done |

Repository: https://github.com/PraveenSaiJ/SBTrade- (branch main).

## 7. Known limitations

- Schwab gives day volume, not prints: no sweep or block detection (BURST is a proxy). No greeks at some times of day (a warning is shown).
- IV rank is an HV-rank stand-in until about 60 days of IV are recorded (recording began Oct 22).
- `config/holdings.yaml` is stale (expired options remain until you upload fresh CSVs); options saved before accounts existed are read as the taxable
  account. The positions importer is built from documented broker layouts and has not seen your real exports yet.
- Some Schwab betas are implausible (negative or very large), so the beta-weighted number is approximate for small speculative names.
- Performance history started Oct 22 (two days), so statistics are anecdotal until about 20 ideas are decided. Stock-level only; option premium P&L is
  not modelled.
- Task Scheduler runs only while you are logged in; a job missed while the PC is off runs when it wakes (StartWhenAvailable) but not on holidays. The Schwab refresh token needs a browser sign-in weekly.
- Offers not yet decided: a minimum R:R filter on Swing, a Schwab-login-age banner on every tab.

## 8. Next steps

Decide on the minimum R:R filter, a Schwab-login-age banner on every tab, an Excel export, and a paste-back box for the Performance verdict table.

