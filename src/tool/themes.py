"""Theme tags for tickers, used to filter the dashboard. First matching theme wins; unknown tickers are 'Other'."""

from __future__ import annotations

THEMES: dict[str, list[str]] = {
    "Index ETFs": ["SPY", "QQQ", "IWM", "DIA"],
    "Mega-cap tech": ["MSFT", "AMZN", "GOOGL", "META", "AAPL", "TSLA", "NFLX"],
    "AI infrastructure": [
        "NVDA", "AVGO", "MRVL", "ARM", "TSM", "ALAB", "CRDO", "COHR", "LITE", "AAOI", "FN", "CLS",
        "SMCI", "DELL", "HPE", "VRT", "CRWV", "NBIS", "APLD", "CORZ", "WULF", "IREN", "CIFR", "ANET", "CIEN",
    ],
    "Semiconductors & equipment": [
        "AMD", "MU", "INTC", "QCOM", "AMAT", "LRCX", "KLAC", "ASML", "ONTO", "CAMT", "AMKR", "AEHR",
        "NVTS", "SNDK", "WDC", "STX", "POET",
    ],
    "Power & nuclear": ["VST", "CEG", "NRG", "TLN", "GEV", "SMR", "BE", "CCJ", "LEU", "NEE", "OKLO", "NNE"],
    "Quantum": ["IONQ", "ARQQ", "QTUM", "QUBT", "RGTI", "QBTS", "LAES"],
    "Crypto & stablecoin": [
        "COIN", "GLXY", "BLSH", "GEMI", "FIGR", "BKKT", "PYPL", "XYZ", "RIOT", "HUT", "BTBT", "BTDR", "HIVE",
        "ETHA", "FBTC", "IBIT", "MSTR", "MSTU", "MARA", "CLSK", "CRCL", "CRCA", "BMNR", "SBET", "ABTC", "HOOD",
    ],
    "Space": ["RKLB", "ASTS", "LUNR", "PL"],
    "AI software & internet": [
        "AI", "PATH", "DDOG", "MDB", "NET", "CRWD", "APP", "TTD", "ESTC", "TEM", "PLTR", "SNOW", "NOW",
        "ORCL", "CRM", "ADBE", "SOUN", "BBAI", "BABA", "BIDU",
    ],
    "Financials, consumer & industrial": ["JPM", "V", "MA", "BAC", "GS", "COST", "WMT", "UNH", "LLY", "XOM", "BA", "DIS"],
}

OTHER = "Other"
_THEME_OF = {sym: theme for theme, syms in reversed(list(THEMES.items())) for sym in syms}


def theme_of(symbol: str) -> str:
    return _THEME_OF.get(symbol.upper(), OTHER)


def all_themes() -> list[str]:
    return [*THEMES, OTHER]
