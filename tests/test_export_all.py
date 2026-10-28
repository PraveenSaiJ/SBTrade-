from __future__ import annotations

import io
import zipfile

import pandas as pd

from tool.export_all import combined_csv, zip_of_csvs


def sections():
    return {
        "swing": pd.DataFrame({"symbol": ["AAA"], "entry": [10.0], "score": [0.5]}),
        "options_csp": pd.DataFrame({"symbol": ["BBB"], "strike": [50.0], "bid": [1.2]}),
    }


def test_combined_csv_stacks_sections_with_section_first_and_blank_gaps():
    df = pd.read_csv(io.BytesIO(combined_csv(sections())))
    assert list(df.columns)[0] == "section" and list(df["section"]) == ["swing", "options_csp"]
    assert pd.isna(df.loc[0, "strike"]) and df.loc[1, "strike"] == 50.0


def test_zip_has_one_csv_per_section():
    with zipfile.ZipFile(io.BytesIO(zip_of_csvs(sections()))) as zf:
        assert sorted(zf.namelist()) == ["options_csp.csv", "swing.csv"]
        assert "entry" in zf.read("swing.csv").decode("utf-8-sig")
