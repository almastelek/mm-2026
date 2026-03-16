"""
Team name normalization and team-year key for stable joins across data sources.
"""
import re
import pandas as pd

# Known aliases: variant -> canonical (as appears in most data)
# Build from common NCAA naming; expand if joins miss.
TEAM_ALIASES = {
    "texas ": "Texas",
    "north carolina st.": "North Carolina St.",
    "nc state": "North Carolina St.",
    "nc st.": "North Carolina St.",
    "lsu": "LSU",
    "usc": "USC",
    "uconn": "Connecticut",
    "byu": "BYU",
    "smu": "SMU",
    "tcu": "TCU",
    "unc": "North Carolina",
    "ole miss": "Mississippi",
    "texas a&m": "Texas A&M",
    "st. john's": "St. John's",
    "st john's": "St. John's",
    "st. johns": "St. John's",
    "mount st. mary's": "Mount St. Mary's",
    "mt. st. mary's": "Mount St. Mary's",
    "san diego st.": "San Diego St.",
    "michigan st.": "Michigan St.",
    "mississippi st.": "Mississippi St.",
    "colorado st.": "Colorado St.",
    "oregon st.": "Oregon St.",
    "washington st.": "Washington St.",
    "utah st.": "Utah St.",
    "boise st.": "Boise St.",
    "nevada": "Nevada",
    "iowa st.": "Iowa St.",
    "oklahoma st.": "Oklahoma St.",
    "kansas st.": "Kansas St.",
    "alabama st.": "Alabama St.",
    "grambling st.": "Grambling St.",
    "mcneese st.": "McNeese St.",
    "long beach st.": "Long Beach St.",
    "morehead st.": "Morehead St.",
    "south dakota st.": "South Dakota St.",
    "norfolk st.": "Norfolk St.",
    "siu edwardsville": "SIU Edwardsville",
    "uc san diego": "UC San Diego",
    "texas a&m corpus chris": "Texas A&M Corpus Chris",  # truncated in some data
    "texas ": "Texas",
}


def normalize_team_name(name: str) -> str:
    """Canonicalize team name for joining: strip, then apply alias map (lowercase key)."""
    if pd.isna(name) or not isinstance(name, str):
        return ""
    s = name.strip()
    if not s:
        return ""
    key = s.lower()
    return TEAM_ALIASES.get(key, s)


def team_year_key(name: str, year: int) -> str:
    """Stable join key: normalized name and year."""
    return f"{normalize_team_name(name)}@{year}"


def add_normalized_columns(df: pd.DataFrame, team_col: str = "TEAM", year_col: str = "YEAR") -> pd.DataFrame:
    """Add team_norm and team_year_key to DataFrame. In-place if possible, else copy."""
    out = df.copy()
    year = out.get(year_col)
    if year is None and "year" in out.columns:
        year = out["year"]
    elif year is None:
        year = out.get("YEAR")
    out["team_norm"] = out[team_col].astype(str).map(normalize_team_name)
    if year is not None:
        out["team_year_key"] = out.apply(
            lambda r: team_year_key(str(r[team_col]), int(r[year_col]) if pd.notna(r[year_col]) else 0),
            axis=1,
        )
    return out


def build_team_year_key_table_from_games(games: pd.DataFrame) -> pd.DataFrame:
    """From games table (team_a, team_b, year), return unique team_year_key rows for joins."""
    a = games[["year", "team_a"]].rename(columns={"team_a": "team"})
    b = games[["year", "team_b"]].rename(columns={"team_b": "team"})
    u = pd.concat([a, b]).drop_duplicates()
    u["team_norm"] = u["team"].astype(str).map(normalize_team_name)
    u["team_year_key"] = u.apply(lambda r: team_year_key(str(r["team"]), int(r["year"])), axis=1)
    return u[["year", "team", "team_norm", "team_year_key"]].drop_duplicates()
