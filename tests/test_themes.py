from __future__ import annotations

from tool.themes import THEMES, all_themes, theme_of


def test_known_and_unknown_tickers():
    assert theme_of("NVDA") == "AI infrastructure"
    assert theme_of("ionq") == "Quantum"
    assert theme_of("ZZZZ") == "Other"


def test_first_listed_theme_wins_for_duplicates():
    # a symbol listed under two themes resolves to the earlier theme in THEMES
    seen = {}
    for theme, syms in THEMES.items():
        for s in syms:
            seen.setdefault(s, theme)
    assert all(theme_of(s) == t for s, t in seen.items())


def test_other_is_last():
    assert all_themes()[-1] == "Other"
