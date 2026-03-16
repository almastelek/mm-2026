"""
Build game-level dataset with features: join team-season and contextual data, add differences/ratios.
"""
import pandas as pd
from pathlib import Path

from .config import DATA_DIR
from .games import parse_games_from_matchups, load_matchups
from .team_utils import normalize_team_name, team_year_key


def load_barttorvik_neutral(path: Path | None = None) -> pd.DataFrame:
    path = path or (DATA_DIR / "Barttorvik Neutral.csv")
    df = pd.read_csv(path)
    df["team_year_key"] = df.apply(
        lambda r: team_year_key(str(r["TEAM"]), int(r["YEAR"])), axis=1
    )
    return df


def load_resumes(path: Path | None = None) -> pd.DataFrame:
    path = path or (DATA_DIR / "Resumes.csv")
    df = pd.read_csv(path)
    df["team_year_key"] = df.apply(
        lambda r: team_year_key(str(r["TEAM"]), int(r["YEAR"])), axis=1
    )
    return df


def load_538(path: Path | None = None) -> pd.DataFrame:
    path = path or (DATA_DIR / "538 Ratings.csv")
    df = pd.read_csv(path)
    df["team_year_key"] = df.apply(
        lambda r: team_year_key(str(r["TEAM"]), int(r["YEAR"])), axis=1
    )
    return df


def load_seed_results(path: Path | None = None) -> pd.DataFrame:
    path = path or (DATA_DIR / "Seed Results.csv")
    df = pd.read_csv(path)
    if "WIN%" in df.columns:
        win = pd.to_numeric(df["WIN%"].astype(str).str.replace("%", ""), errors="coerce")
        # values may be 0.798 or 99.9
        df["seed_win_pct"] = win.where(win <= 1, win / 100)
    return df


def add_team_keys_to_games(games: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()
    g["team_a_key"] = g.apply(lambda r: team_year_key(str(r["team_a"]), int(r["year"])), axis=1)
    g["team_b_key"] = g.apply(lambda r: team_year_key(str(r["team_b"]), int(r["year"])), axis=1)
    return g


def build_game_features(
    games: pd.DataFrame,
    barttorvik: pd.DataFrame | None = None,
    resumes: pd.DataFrame | None = None,
    f538: pd.DataFrame | None = None,
    seed_results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Join team-season and seed context to games; add difference/ratio features.
    Uses left joins so games without a feature stay (with NaN); we can dropna on key features later.
    """
    g = add_team_keys_to_games(games)

    # Team-season features: keep a small set of columns to avoid redundancy
    bt_cols = ["team_year_key", "BARTHAG", "BADJ EM", "WAB"]
    res_cols = ["team_year_key", "ELO", "R SCORE", "RESUME"]
    f538_cols = ["team_year_key", "POWER RATING"]

    if barttorvik is None:
        barttorvik = load_barttorvik_neutral()
    bt = barttorvik[bt_cols].drop_duplicates(subset=["team_year_key"], keep="first")

    if resumes is None:
        resumes = load_resumes()
    res = resumes[res_cols].drop_duplicates(subset=["team_year_key"], keep="first")

    if f538 is None:
        f538 = load_538()
    f538 = f538[f538_cols].drop_duplicates(subset=["team_year_key"], keep="first")

    g = g.merge(bt, left_on="team_a_key", right_on="team_year_key", how="left", suffixes=("", "_a"))
    g = g.rename(columns={"BARTHAG": "barthag_a", "BADJ EM": "badj_em_a", "WAB": "wab_a"})
    g = g.drop(columns=[c for c in g.columns if c == "team_year_key"])

    g = g.merge(bt, left_on="team_b_key", right_on="team_year_key", how="left", suffixes=("", "_b"))
    g = g.rename(columns={"BARTHAG": "barthag_b", "BADJ EM": "badj_em_b", "WAB": "wab_b"})
    g = g.drop(columns=[c for c in g.columns if c == "team_year_key"], errors="ignore")

    g = g.merge(res, left_on="team_a_key", right_on="team_year_key", how="left")
    g = g.rename(columns={"ELO": "elo_a", "R SCORE": "r_score_a", "RESUME": "resume_a"})
    g = g.drop(columns=[c for c in g.columns if c == "team_year_key"], errors="ignore")

    g = g.merge(res, left_on="team_b_key", right_on="team_year_key", how="left")
    g = g.rename(columns={"ELO": "elo_b", "R SCORE": "r_score_b", "RESUME": "resume_b"})
    g = g.drop(columns=[c for c in g.columns if c == "team_year_key"], errors="ignore")

    g = g.merge(f538, left_on="team_a_key", right_on="team_year_key", how="left")
    g = g.rename(columns={"POWER RATING": "power_rating_a"})
    g = g.drop(columns=[c for c in g.columns if c == "team_year_key"], errors="ignore")

    g = g.merge(f538, left_on="team_b_key", right_on="team_year_key", how="left")
    g = g.rename(columns={"POWER RATING": "power_rating_b"})
    g = g.drop(columns=[c for c in g.columns if c == "team_year_key"], errors="ignore")

    # Seed context: historical win% by seed (from Seed Results)
    if seed_results is None:
        seed_results = load_seed_results()
    if "seed_win_pct" not in seed_results.columns and "WIN%" in seed_results.columns:
        seed_results = seed_results.copy()
        seed_results["seed_win_pct"] = pd.to_numeric(
            seed_results["WIN%"].astype(str).str.replace("%", ""), errors="coerce"
        ) / 100
    seed_lookup = seed_results.set_index("SEED")["seed_win_pct"].to_dict()

    g["seed_diff"] = g["seed_b"] - g["seed_a"]  # positive => B is lower seed (better)
    g["seed_win_pct_a"] = g["seed_a"].map(seed_lookup)
    g["seed_win_pct_b"] = g["seed_b"].map(seed_lookup)

    # Difference features (A - B): positive => A stronger
    g["barthag_diff"] = g["barthag_a"] - g["barthag_b"]
    g["badj_em_diff"] = g["badj_em_a"] - g["badj_em_b"]
    g["elo_diff"] = g["elo_a"] - g["elo_b"]
    g["r_score_diff"] = g["r_score_a"] - g["r_score_b"]
    g["power_rating_diff"] = g["power_rating_a"] - g["power_rating_b"]

    # Round as categorical / numeric
    g["round_num"] = g["round"]

    return g


def get_feature_columns() -> list[str]:
    """Column names used as model features (numeric)."""
    return [
        "seed_diff",
        "seed_win_pct_a",
        "seed_win_pct_b",
        "barthag_diff",
        "badj_em_diff",
        "elo_diff",
        "r_score_diff",
        "power_rating_diff",
        "round_num",
    ]


def build_merged_dataset(years: list[int] | None = None, drop_missing_core: bool = True) -> pd.DataFrame:
    """
    Full pipeline: parse games, filter years, build features.
    If drop_missing_core=True, drop rows missing barthag_diff or elo_diff (core features).
    """
    matchups = load_matchups()
    games = parse_games_from_matchups(matchups)
    if years is not None:
        games = games[games["year"].isin(years)]
    feats = build_game_features(games)
    if drop_missing_core:
        # Require at least Barttorvik and Resumes for both teams
        feats = feats.dropna(subset=["barthag_diff", "elo_diff"])
    # Fill optional 538 so model can train (use 0 = no advantage)
    if "power_rating_diff" in feats.columns:
        feats = feats.copy()
        feats["power_rating_diff"] = feats["power_rating_diff"].fillna(0)
    return feats


if __name__ == "__main__":
    df = build_merged_dataset(years=[2023, 2024, 2025])
    print("Merged shape:", df.shape)
    print("Feature cols:", get_feature_columns())
    print(df[get_feature_columns() + ["winner"]].head())
