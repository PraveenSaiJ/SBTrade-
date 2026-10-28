# Trading Tool — Project Context

Personal tool for Praveen. It recommends trades; it NEVER places orders. All execution is manual.

## Goals (3 engines)
1. **Weekly covered-call strike recommendations** on holdings (Friday afternoon + Monday morning runs).
2. **Swing stock buy/sell signals** (1–2 week holds) with entry, stop, targets, reward:risk.
3. **Options**: cash-secured puts at swing entries + a roll monitor for open short calls/puts.

Output: Streamlit dashboard + a morning report (markdown/HTML). Runs before the open (~9:00 ET) and Friday ~3:00 PM ET.

## Accounts & data sources
| Source | Use | Notes |
|---|---|---|
| Schwab Trader API – Individual | PRIMARY: quotes, option chains (delta, IV, OI), price history, positions | Access requested Oct 2026, pending. Use `schwab-py`. Callback `https://127.0.0.1:8182`. Refresh token expires every 7 days → weekly re-login. |
| IBKR | Positions + backup price history | `ib_async` + IB Gateway (live 4001 / paper 4002). Live option bids may need market-data subscriptions (they were not returning live). |
| Robinhood (taxable + Traditional IRA) | Most holdings live here | NO official API. Never use unofficial libs (ToS risk). Import positions from CSV / manual YAML only. |
| TradingView | Charts + optional webhook alerts | A Pine script (`praveen_levels.pine`) plots the same levels. |
| Earnings calendar | Earnings guard | Needs a source (e.g., Finnhub free key). |

Secrets in `.env` (gitignored). Holdings in `config/holdings.yaml` (gitignored — contains cost basis).

## Level math (must match exactly)
Classic floor pivots from a completed bar (H, L, C):
- P = (H+L+C)/3; R1 = 2P−L; S1 = 2P−H; R2 = P+(H−L); S2 = P−(H−L); R3 = H+2(P−L); S3 = L−2(H−P)
- **Daily** levels: last completed day's bar. **Weekly** levels: last completed Mon–Fri week (H=max, L=min, C=last close).
- All dates in America/New_York. A bar is complete after 16:15 ET (or the date is before today).

Verification fixtures (MSFT):
- Daily from Oct 2 2026 bar H 522.50, L 513.66, C 517.53 → P 517.90, R1 522.13, S1 513.29, R2 526.74, S2 509.06
- Weekly from Sep 28–Oct 2 2026: H 522.85, L 502.22, C 517.53 → P 514.20, R1 526.18, S1 505.55, R2 534.83, S2 493.57

## Swing screener rules
Inputs from daily bars: ATR(10), SMA(20) of closes, prior 20-day high (excluding today), 20-day % change.
- **Other setups** (separate kinds, same filters/stop/score): `bull_flag` (≥15% pole in ≤15 bars, then 3–10 tight bars, ≤8% range, ≤50% retrace, lighter volume; entry = flag high) and `ma_pullback` (close > SMA200, SMA50 > SMA200 and rising; a low in the last 3 bars within 0.5 ATR of SMA20/SMA50 and the close back above it; ≥3% under the 20-day high; entry = that average). The older pivot-support `pullback` is unchanged.
- **Breakout**: last COMPLETED daily close > prior 20-day high → entry = that high (retest). Breakout-day volume must be ≥ 1.5× the prior 20-day average (`min_volume_ratio`); below 1.0× it is a `weak_breakout`, between 1.0× and 1.5× it is excluded as low volume. Trading above the high without a daily close is only a watch note, not a breakout. If the live spot is back under the entry the setup is `failed` and excluded (`--show-failed` keeps it).
- **Extended (wait)**: (last − SMA20)/ATR > 3 → entry = first pivot level below (last − 1 ATR).
- **Pullback**: last > SMA20 and 20d change > +2% and nearest pivot support within 1 ATR below → entry = that support.
- **Downtrend (skip)**: 20d change < −3%. Otherwise **Range (skip)**.
- **Stop (structural)** = max( min(entry − 0.5 × ATR, recent swing low), entry − 2 × ATR ): below the structure with a half-ATR buffer, never more than 2 ATR under the entry; with no swing low to use it falls back to entry − 1.5 × ATR. Position size = floor(risk_pct × account_size / (fill − stop)), only when `account.account_size` is set. T1 = first pivot ≥ entry + 1.5 ATR (else entry + 2 ATR), **capped at the nearest strong resistance zone above the fill** (the zone's bottom edge; zones from `sr_levels.py`, see the Support and resistance zones section). T2 = next pivot ≥ T1 + 0.75 ATR (else T1 + 1.5 ATR). **R:R = (T1 − fill) / (fill − stop) with fill = max(spot, entry).**
- **Filters** (config in `src/tool/config.py` / `config/settings.yaml`): earnings within `hold_days` = 10 trading days excluded (`--allow-earnings` keeps them); close must be above SMA200 (`trend_filter`: sma200 | sma50_rising | off); breakouts/pullbacks excluded when RSI(14) > 75 or (spot − entry) > 2 × ATR(14). Upcoming earnings inside 30 days are still shown on every setup; a report within the last 10 trading days shows "Reported <date>" (post-earnings flag). Every dropped symbol is logged with its reason (Excluded section).
- A setup whose price is already at or under its stop is excluded (`stopped`). Every recommendation and exclusion is logged to the `signals` table (inputs, outputs, config used; verdict columns reserved for the Claude review).
- **Rank**: score = 0.35·norm(volume ratio, cap 3×) + 0.25·norm(R:R, cap 3) + 0.20·trend score + 0.20·(1 − earnings risk), sorted descending.
- **Sell signals on holdings**: close below weekly S1, failed breakout (close back under breakout level), stop hit.

## Covered-call engine rules
- Strike ≥ weekly R1 (≥ weekly R2 when the stock is breaking out). Target delta 0.20–0.30.
- **Never strike below cost basis** (assignment would lock a loss).
- Skip expiries that include an earnings date.
- Taxable accounts: show realized gain if assigned (tax flag). IRA: no tax flag.
- Liquidity: bid-ask spread ≤ 10% of mid; recommend limit near mid.
- Show: bid, mid, return % (bid/spot), premium per lot × free lots, distance OTM.
- Prefer selling into strength (up days); premium is richer.

### Two rows per holding (Covered calls tab)
- **Guardrail row**: the rules above, unchanged.
- **Income row** (`income_calls.py`): the weekly strike nearest the stock price, never in the money and never under cost basis, whose net return `(bid − fee/100) / spot` meets the tier target (`covered_calls.tier_targets`, default 1.5%); when none does, the best is shown marked "below target". Earnings before expiry are flagged, not skipped. Probability ITM is approximated by |delta|.
- Tiers (config): safe NVDA/AMZN/MSFT, proven SNOW/ZS/SHOP, aggressive CRWV/IREN, everything else "other". `covered_calls.uncapped` (`CRWV`, `SNOW:rh_ira`) and the per-lot `no_covered_calls` flag both mean no suggestions.
- Flags before expiry: earnings, ex-dividend (Schwab `dividendDate`; `dividendAmount` is annual, so a quarterly payer is assumed), early-assignment risk when time value < the dividend, and manual events (`covered_calls.events`: investor days, product events; no free data source exists). `fee_per_contract` defaults to 0 (net = bid).

### Per-ticker overrides
- **CRWV**: NO covered calls (held uncapped).
- **SNOW (IRA)**: NO covered calls; redeploy to cash-secured puts instead.
- **ALAB (wheel)**: aggressive — sell ATM call when price touches the trigger (e.g., 370) on Mon/Tue; if not hit by Tue close, sell ~1 strike above weekly R2.
- **MSFT, AMZN, NVDA**: all lots free (no long-dated calls committed).

### Roll rule (all short calls)
- If the stock breaks out: roll at the same or a higher strike **only for a net credit**, same expiry or up to 3 weeks out (`covered_calls.roll_weeks`); the Covered calls tab lists every such roll for each open short call.
- If no credit roll exists → accept assignment, switch to selling puts.
- Roll before extrinsic value < $0.50 (early-assignment risk).
- No rolls into an earnings week unless explicitly chosen.

## Directional and LEAPS calls (buy)
- **Directional** (`directional_calls.py`): 14-60 days out (the first Friday, the monthly and the last in the window), delta 0.60-0.70 nearest 0.65, on the bullish swing setups (breakout, bull flag, MA pullback, pivot pullback) with the Swing tab's entry/stop/targets.
- **LEAPS**: the first expiry at least 270 days out, delta 0.70-0.80 nearest 0.75, on the Long term picks (their entry/stop/targets) plus the same swing setups.
- Every row shows break-even (strike + ask), the expected move to expiry (spot x contract IV x sqrt(days/365)), break-even / expected move (above 1.0 needs more than the market expects), theta per day as a share of the premium, time value paid, IV rank (HV stand-in until 60 days of IV), max loss = the whole premium, and contracts = floor(risk_pct x account_size / max loss) once `account.account_size` is set.
- **IV CRUSH RISK** flag: earnings before expiry and rank above 50 (for a LEAPS only a report within 14 days counts). Notes only; nothing is filtered out. The Monthly calls (~0.50 delta) section stays.

## Earnings calls (Calls tab, `earnings_calls.py`)
- Event trade, not a swing setup: for holdings, watchlist, tiers, MAG7 and scanned names reporting within `calls.earnings_days` (7), one call with delta 0.45-0.55 expiring the first Friday after the reaction day (report day if before the open, else the next session). Built by the options job.
- Each row: break-even vs expected move (the tab shows only ratio ≤ `calls.earnings_max_ratio` 1.0 by default, switchable), IV rank (HV stand-in until 60 days of IV), max loss = whole premium, contracts from the account risk %. Every row carries an IV CRUSH note; there is no price stop (sell the morning after the report). Recommend-only.

## Whale activity (read-only flow view)
- A whale is a 0DTE/weekly contract with at least $250K of day premium ($1M+ = WHALE; ×10 for SPY/QQQ/IWM), scanned every 15 minutes. Schwab gives total day volume, not prints, so: **lean** is inferred from where the last trade printed (ask = buyer, bid = seller); **intent** compares volume with this morning's open interest (opening = volume above all of it, likely opening ≥ 50%); **BURST** = premium added since the previous scan (never called a sweep or block).
- **Noise** (hidden by default): |delta| ≥ 0.90 (stock replacement or hedge) and dividend capture (ITM call, time value below the dividend, ex-date within 5 days).
- Net flow per symbol for 1 and 5 trading days uses each contract's last scan of the day (volume is cumulative); totals are whale-size trades only. Flow on names you hold or hold options on is labelled conflict / supports / mixed against your shares, short and long options and tracked options (expired options ignored).

## Holdings review (Long term tab)
- Per holding (`holding_review.py`, refreshed by the daily long-term job): weight of the stock portfolio (cash and options excluded), revenue and EPS growth (Schwab TTM, Finnhub fallback), forward P/E against its 5-year average (Finnhub: forward P/E over the mean of the last five positive annual P/Es), FCF margin (latest annual), drawdown from the 52-week high, distance from the SMA200, gain on cost. Funds and leveraged ETFs have no fundamentals: price columns only, no conviction score.
- **Concentration**: a position above `longterm.max_weight` (default 15%) is flagged.
- **Conviction** 0-100: quality 30%, trend 20%, growth 15%, valuation 15%, FCF margin 10%, resilience 10% (weights in config; parts without data are dropped and the rest re-weighted; under half the weight available = no score). Position size is not an input.
- **Overlap**: holdings whose daily returns (last 120 days) correlate at 0.80 or more are listed as pairs with a plain "keep the higher-conviction name" note.

## Support and resistance zones (Levels tab, `sr_levels.py`)
- Sources: fractal swing highs/lows (3 bars each side, 120 bars), SMA50/100/200, the prior calendar month's high and low, up to 3 volume nodes from a 60-day volume profile, and (Levels tab only, live) the 3 biggest open-interest strikes per side within 15% of the price for the nearest monthly expiry (call wall above the price, put wall below).
- Levels within 0.5 ATR(10) merge into a zone (padded 0.25 ATR each side). **Touches** = separate visits (runs of bars reaching the zone) in 120 days; **volume** = total volume on those bars; **score** = 0.5 touches + 0.5 volume, each relative to the symbol's strongest zone. A zone counts for targets and strikes only at score 0.30 or more.
- Swing caps T1 at the bottom edge of the nearest strong zone above the fill; the Support puts put their strike at or under the centre of the nearest strong support zone below the price. Both use price history only (no chain request per symbol); the weekly-S1 guardrail puts and the pivot table are unchanged.

## Cash-secured put rules
- **Two rows** (Puts tab): the Guardrail row below (weekly S1 cap, delta 0.15-0.25) and a **Support row** (`support_puts.py`): strike at or under the nearest support (weekly/daily S1, SMA50/100/200, last swing low) and under the stock price, |delta| 0.20-0.30 preferred, 7-45 days out (the first two Fridays, the monthly and the last in the window). Shows annualized return on collateral, break-even (strike − premium), probability OTM, distance to support, and a rank.
- Rank = IV rank once 60 days of ATM IV are recorded, otherwise an **HV rank** stand-in (20-day realized vol ranked against its year), always labelled. An expiry after earnings is gated: shown only with the "Earnings play" switch on and rank above 60.
- Collateral check: selected puts' collateral vs `puts.cash[account]` in config/settings.yaml (unset = no comparison, nothing guessed).
- Strike ≤ weekly S1. Show return %, collateral, effective cost (strike − premium).

## Tax / account rules
- Cross-account wash sale: loss sale in taxable + purchase (incl. call options) of same ticker in IRA within ±30 days = loss permanently disallowed (Rev. Rul. 2008-5). Flag this.
- IRA sales have no tax effect.

## Tech stack
Python 3.12, `uv`, pandas, schwab-py, ib_async, streamlit, python-dotenv, pytest. SQLite/DuckDB for history.

Layout:
```
src/tool/
  data/      schwab.py, ibkr.py, robinhood_csv.py, earnings.py, base.py (DataSource protocol)
  indicators.py  SMA/EMA/RSI/ATR/volume ratio/swing highs/IV rank/expected move (one implementation)
  config.py  every threshold (defaults; config/settings.yaml overrides)
  levels.py  pivots, completed-bar selection, 20D high
  screener.py
  covered_calls.py
  puts.py
  rolls.py
  report.py
app.py       streamlit
config/      holdings.yaml (gitignored), settings.yaml
tests/
```

## Build order
1. levels.py + tests (use fixtures above).
2. IBKR data adapter (works today) behind a DataSource interface.
3. covered_calls.py using holdings.yaml + chains.
4. screener.py.
5. Schwab adapter when API approval lands (becomes primary for chains/greeks).
6. rolls.py, puts.py, Streamlit app, morning report, scheduler.
7. Robinhood CSV import; optional TradingView webhook receiver.

## Conventions
- Never place or stage orders. No trading endpoints called.
- Never log secrets or account numbers.
- Every recommendation shows the data timestamp; flag stale/closing-only option quotes.
- Every saved recommendation row carries `generated_at`, `data_as_of`, `source`. Thresholds live in `config.py`, never in tab code.
- Schwab calls go through the data layer: batched quotes, a 120/min limiter, daily bars cached in SQLite (only bars saved after 16:15 ET are trusted), and a re-auth banner when the token is rejected or 7 days old.

## Portfolio risk (Phase 10)
- Every option position carries an `account` (rh_taxable, rh_ira, ibkr); older rows without one are read as `rh_taxable` and should be fixed in the Options editor.
- `tool portfolio` (Jobs tab: "Portfolio risk") saves a snapshot: one row per (account, stock) and per option, with delta, theta, beta and SPY-equivalent delta. Replaced on each run.
- Share delta = 1. Option delta/theta come from the live chain at the contract's strike and expiry, x100 x contracts; short calls and long puts are negative delta; theta is dollars a day, positive when short.
- Beta-weighted delta (SPY shares) = delta shares x price x beta / SPY price. Beta from Schwab fundamentals; missing beta counts as 1.0 and is counted as "assumed". Leveraged funds are not scaled.
- Expired options (past expiry) are shown as `expired` with a "check the account" flag, never as live risk. Options with no chain match are `no quote` and count as zero.
- Sector exposure = Finnhub industry (cached 30 days in data/cache/finnhub_profiles.json) and the app's themes, as % of live share value.
- This week (7 days): ITM short with time value < $0.50 = assignment risk; ex-dividend before expiry with time value below the dividend = assignment risk; earnings before expiry; short calls with fewer shares than 100 x contracts in the same account = NAKED; short puts show cash needed if assigned.
- Collateral per account = strike x 100 x contracts of live short puts vs `puts.cash` in settings.yaml (red when over, note when unset).
- Positions CSV import (`tool import-positions FILE --account X [--file-account NAME] [--apply]`, or Portfolio > Import positions) replaces ONE account's stocks and options; previews first; rows it cannot read are listed, never guessed; unchanged stocks keep their stored lots and flags.

## Performance tab (Phase 11)
- Swing outcomes come from the `signals` table: `recommended` rows are the scored set, `excluded` rows are replayed separately ("Skipped by filters", grouped by excluded code) and never mixed into the main numbers. Option ideas come from `option_recs` (Calls / Puts by family).
- Each idea is replayed on completed daily bars (up to `performance.horizon_days` = 10): STOP / T1 / T2 / OPEN / NO FILL / PENDING; a bar touching both stop and target counts as the stop. R = profit / planned risk; a stop is -1R.
- MFE / MAE = best / worst excursion after the fill in R; days held = trading days from the fill bar to the deciding bar (open: to the last bar).
- Win rate = T1 or T2 of decided; Expectancy = mean R of decided; Profit factor = winning R / losing R (n/a with no losses); open ideas are marked to the last close in Avg R but not in win rate, expectancy or the curve.
- Equity curve = cumulative R of a 1R-per-decided-idea system in exit-date order (no sizing, fees or overlap limit); max drawdown is peak to trough in R. Below `performance.min_sample` (20) decided ideas the tab warns that the sample is too small.
- "By Claude verdict" groups swing signals by `signals.verdict` (unreviewed until the Phase 13 review exists). Stock-level only: option premium P&L is not modelled.

## Jobs, schedule and holidays (Phase 12)
- Scheduling stays in Windows Task Scheduler (`scripts/register_tasks.ps1`, run it once to register or update; it only runs while you are logged in). Weekdays ET: Schwab login Mon 08:45, swing scan 09:00, long-term 09:25, quick refresh 09:35, options calls 10:00 and options puts 10:10 (the old single options task is removed when you re-run the script), **re-check 10:15**, whales / covered-call monitor / buy ideas (monthly swing + MAG7) / **0DTE calls and puts** every 15 minutes from the open, **weekly calls and puts** every 30 minutes, **covered-call and roll check 15:45**, **outcomes 16:30** (refreshes completed bars for open and pending ideas, scores them for the Performance tab).
- `tool run-job NAME` skips weekends and NYSE holidays (`market_calendar.py`: rules with weekend observance, Good Friday, early closes) unless `--force`; `freshness` also treats a holiday as no session.
- Every run is logged in `data/jobs/history.sqlite` (`job_runs`: job, trigger dashboard|scheduled, start, end, status done|failed|stopped|skipped, rows, errors, note), pruned after `jobs.history_days` (90). The Jobs tab shows the history, the registered schedule with next/last run, a holiday banner and the notification log.
- Desktop notification (`notify.py`: PowerShell balloon, always logged to `data/jobs/notifications.log`, no secrets) when a scheduled job fails or stops, the Schwab login is missing/expired, or it is within a day of expiry; each fires at most once per job per day. `jobs.notify` turns the balloon off.

## Exporting recommendations
- There is no Claude API call. The Swing tab ("Download recommendations (CSV)") and the Covered calls tab ("Download CSV", all tiers in one file) export the table as shown, with a "Data as of" column, as a UTF-8 CSV that opens in Excel. Upload or review it however you like; nothing is sent from the app.
- `signals.verdict` / `verdict_reason` still exist (Performance's "By Claude verdict" table reads them) but nothing fills them now.

## Options job speed
- The options job is two halves (`tool options --part calls|puts`, jobs `options_calls` / `options_puts`; plain `options` still builds everything). Families inside a half are built `data.options_workers` (4) at a time; the Schwab limiter (120/min) is shared, and a failure in one family fails the job after the others finish.
- `MemoSource` fetches each chain, price and set of bars once per run, thread-safe. Callers `hint_chains(symbol, right, expiries)`; the first chain ask for that symbol fetches all expiries within 35 days of each other in ONE request (`SchwabDataSource.option_chains`) and later single-expiry asks are served from it (same strikes-per-expiry and window, else a normal request). A symbol skipped before it needs a chain costs nothing.
- Earnings asks go through `CalendarEvents` (one cached calendar, today-14d..today+60d) instead of one Finnhub request per symbol, expiry and family; asks past that range (LEAPS) hit Finnhub once per symbol and range.
- Measured 2026-10-27: the single job took 22.6 min; calls took 3.3 min and puts 0.4 min.
- 0DTE (`tool odte`, job `odte`): index ETFs + MAG7, calls AND puts (families `odte_call`, `odte_put`), planned from the daily pivots (call: buy above P, stop halfway to S1, targets R1/R2; put: buy below P, stop halfway to R1, targets S1/S2). Weekly (`tool weekly`, job `weekly`): weekly swing and lotto calls/puts (families `weekly_*`, `lotto_*`) on the swing-scan names. The buys job now keeps only the monthly swing ideas and MAG7. All three share one memoized source and build symbols `data.options_workers` at a time.
- Options tab (`render_options_hub`): weekly calls and puts, lottos and 0DTE (calls and puts each) gathered in one place from the same saved families; the Calls and Puts tabs keep showing them too (widget keys are scoped so both can render).
