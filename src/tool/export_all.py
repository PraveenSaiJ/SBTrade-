"""One export of every current recommendation (swing, long term, option ideas, covered calls) for review outside the app.

Recommend-only data: the latest saved rows with their data timestamps. Position sizes and share counts are left out.
"""

from __future__ import annotations

import io
import zipfile

import pandas as pd

from tool.store import Store

PRIVATE_COLUMNS = ("shares", "position_value", "risk_dollars", "contracts")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in PRIVATE_COLUMNS if c in df.columns]).dropna(axis=1, how="all")


def collect(store: Store) -> dict[str, pd.DataFrame]:
    """Section name -> table of the current recommended rows (empty sections are left out)."""
    sections: dict[str, pd.DataFrame] = {}
    setups = store.latest_setups()
    if not setups.empty:
        live = setups[setups["excluded"].isna() & setups["entry"].notna()]
        sections["swing"] = live.sort_values("score", ascending=False)
    longterm = store.latest_longterm()
    if not longterm.empty:
        sections["long_term"] = longterm[longterm["status"].isin(["recommend", "wait"])]
    families = [r[0] for r in store._conn.execute("SELECT DISTINCT family FROM option_recs ORDER BY family")]
    for family in families:
        recs = store.latest_option_recs(family)
        recs = recs[recs["status"] == "recommend"] if not recs.empty else recs
        if not recs.empty:
            sections[f"options_{family}"] = recs
    cc = store.latest_covered_calls()
    if not cc.empty:
        sections["covered_calls"] = cc[cc["status"] == "recommend"]
    return {name: _clean(df) for name, df in sections.items() if not df.empty}


def combined_csv(sections: dict[str, pd.DataFrame]) -> bytes:
    """Every section stacked in one table (Section first; columns a section does not use stay blank)."""
    frames = [df.assign(**{"section": name}) for name, df in sections.items()]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if "section" in out:
        out = out[["section", *[c for c in out.columns if c != "section"]]]
    return out.to_csv(index=False).encode("utf-8-sig")


def zip_of_csvs(sections: dict[str, pd.DataFrame]) -> bytes:
    """One CSV per section (only that section's columns), zipped."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, df in sections.items():
            zf.writestr(f"{name}.csv", df.to_csv(index=False).encode("utf-8-sig"))
    return buf.getvalue()
