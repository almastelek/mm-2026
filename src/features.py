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


def load_kenpom_barttorvik(path: Path | None = None) -> pd.DataFrame:
    """
    Load KenPom-style metrics from KenPom Barttorvik.csv.

    This file already includes many of the Barttorvik columns, but we only
    use the KenPom-adjusted tempo/efficiency fields to avoid duplication:
      - KADJ O, KADJ D, KADJ EM (overall, offense, defense)
      - K TEMPO, KADJ T (tempo)
    """
    path = path or (DATA_DIR / "KenPom Barttorvik.csv")
    df = pd.read_csv(path)
    df["team_year_key"] = df.apply(
        lambda r: team_year_key(str(r["TEAM"]), int(r["YEAR"])), axis=1
    )
    # Keep only the KenPom-specific columns plus join key to avoid confusion
    keep_cols = [
        "team_year_key",
        "KADJ O",
        "KADJ D",
        "KADJ EM",
        "K TEMPO",
        "KADJ T",
        # Strength-of-schedule signal from KenPom/Barttorvik export
        "ELITE SOS",
    ]
    existing = [c for c in keep_cols if c in df.columns]
    return df[existing]


def load_resumes(path: Path | None = None) -> pd.DataFrame:
    path = path or (DATA_DIR / "Resumes.csv")
    df = pd.read_csv(path)
    df["team_year_key"] = df.apply(
        lambda r: team_year_key(str(r["TEAM"]), int(r["YEAR"])), axis=1
    )
    return df


def load_teamsheet_ranks(path: Path | None = None) -> pd.DataFrame:
    """
    Load compact resume/SOS-ish metrics from Teamsheet Ranks.csv.
    Useful columns: NET, KPI, SOR, Q1/Q2/Q1&2 wins.
    """
    path = path or (DATA_DIR / "Teamsheet Ranks.csv")
    df = pd.read_csv(path)
    df["team_year_key"] = df.apply(
        lambda r: team_year_key(str(r["TEAM"]), int(r["YEAR"])), axis=1
    )
    return df


def load_upset_rates(path: Path | None = None) -> dict[tuple[int, int], float]:
    """
    Load Upset Seed Info.csv and compute a relative upset frequency per (round, seed_diff).

    Note: Upset Seed Info contains only upsets. We normalize within each round to get a
    relative \"upset likelihood\" signal by seed_diff for that round.
    """
    path = path or (DATA_DIR / "Upset Seed Info.csv")
    df = pd.read_csv(path)
    grp = (
        df.groupby(["CURRENT ROUND", "SEED DIFF"])
        .size()
        .rename("count")
        .reset_index()
    )
    grp["round_total"] = grp.groupby("CURRENT ROUND")["count"].transform("sum")
    grp["freq"] = grp["count"] / grp["round_total"]
    return {
        (int(r), int(sd)): float(f)
        for r, sd, f in grp[["CURRENT ROUND", "SEED DIFF", "freq"]].itertuples(index=False, name=None)
    }


def load_true_upset_rates_from_games(
    matchups_path: Path | None = None,
) -> dict[tuple[int, int], float]:
    """
    Compute true empirical upset probability P(upset | round, seed_diff) from ALL games.

    Definitions:
    - Favorite seed = min(seed_a, seed_b) (lower number is better seed)
    - Underdog seed = max(seed_a, seed_b)
    - seed_diff = underdog_seed - favorite_seed (positive integer)
    - upset = (winner_seed == underdog_seed)
    """
    matchups_path = matchups_path or (DATA_DIR / "Tournament Matchups.csv")
    matchups = pd.read_csv(matchups_path)
    games = parse_games_from_matchups(matchups)

    # winner_seed: seed of the team that won this game
    games = games.copy()
    games["winner_seed"] = games.apply(
        lambda r: int(r["seed_a"]) if int(r["winner"]) == 1 else int(r["seed_b"]),
        axis=1,
    )
    games["favorite_seed"] = games[["seed_a", "seed_b"]].min(axis=1).astype(int)
    games["underdog_seed"] = games[["seed_a", "seed_b"]].max(axis=1).astype(int)
    games["seed_diff"] = (games["underdog_seed"] - games["favorite_seed"]).astype(int)
    games["is_upset"] = (games["winner_seed"] == games["underdog_seed"]).astype(int)

    grp = (
        games.groupby(["round", "seed_diff"])["is_upset"]
        .agg(["mean", "count"])
        .reset_index()
    )
    # Return mean upset rate by (round, seed_diff)
    return {
        (int(r), int(sd)): float(m)
        for r, sd, m in grp[["round", "seed_diff", "mean"]].itertuples(index=False, name=None)
    }

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
    kenpom: pd.DataFrame | None = None,
    teamsheet: pd.DataFrame | None = None,
    upset_rates: dict[tuple[int, int], float] | None = None,
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
    ts_cols = ["team_year_key", "NET", "KPI", "SOR", "Q1 W", "Q2 W", "Q1&2 W"]

    if barttorvik is None:
        barttorvik = load_barttorvik_neutral()
    bt = barttorvik[bt_cols].drop_duplicates(subset=["team_year_key"], keep="first")

    if resumes is None:
        resumes = load_resumes()
    res = resumes[res_cols].drop_duplicates(subset=["team_year_key"], keep="first")

    if f538 is None:
        f538 = load_538()
    f538 = f538[f538_cols].drop_duplicates(subset=["team_year_key"], keep="first")

    if kenpom is None:
        kenpom = load_kenpom_barttorvik()
    # KenPom subset already has team_year_key + KADJ O/D/EM, K TEMPO, KADJ T
    kp = kenpom.drop_duplicates(subset=["team_year_key"], keep="first")

    if teamsheet is None:
        teamsheet = load_teamsheet_ranks()
    ts = teamsheet[ts_cols].drop_duplicates(subset=["team_year_key"], keep="first")

    # Barttorvik base
    g = g.merge(bt, left_on="team_a_key", right_on="team_year_key", how="left", suffixes=("", "_a"))
    g = g.rename(columns={"BARTHAG": "barthag_a", "BADJ EM": "badj_em_a", "WAB": "wab_a"})
    g = g.drop(columns=[c for c in g.columns if c == "team_year_key"])

    g = g.merge(bt, left_on="team_b_key", right_on="team_year_key", how="left", suffixes=("", "_b"))
    g = g.rename(columns={"BARTHAG": "barthag_b", "BADJ EM": "badj_em_b", "WAB": "wab_b"})
    g = g.drop(columns=[c for c in g.columns if c == "team_year_key"], errors="ignore")

    # Resumes
    g = g.merge(res, left_on="team_a_key", right_on="team_year_key", how="left")
    g = g.rename(columns={"ELO": "elo_a", "R SCORE": "r_score_a", "RESUME": "resume_a"})
    g = g.drop(columns=[c for c in g.columns if c == "team_year_key"], errors="ignore")

    g = g.merge(res, left_on="team_b_key", right_on="team_year_key", how="left")
    g = g.rename(columns={"ELO": "elo_b", "R SCORE": "r_score_b", "RESUME": "resume_b"})
    g = g.drop(columns=[c for c in g.columns if c == "team_year_key"], errors="ignore")

    # 538 ratings
    g = g.merge(f538, left_on="team_a_key", right_on="team_year_key", how="left")
    g = g.rename(columns={"POWER RATING": "power_rating_a"})
    g = g.drop(columns=[c for c in g.columns if c == "team_year_key"], errors="ignore")

    g = g.merge(f538, left_on="team_b_key", right_on="team_year_key", how="left")
    g = g.rename(columns={"POWER RATING": "power_rating_b"})
    g = g.drop(columns=[c for c in g.columns if c == "team_year_key"], errors="ignore")

    # KenPom metrics
    g = g.merge(kp, left_on="team_a_key", right_on="team_year_key", how="left")
    g = g.rename(
        columns={
            "KADJ O": "k_adj_o_a",
            "KADJ D": "k_adj_d_a",
            "KADJ EM": "k_adj_em_a",
            "K TEMPO": "k_tempo_a",
            "KADJ T": "k_adj_t_a",
            "ELITE SOS": "elite_sos_a",
        }
    )
    g = g.drop(columns=[c for c in g.columns if c == "team_year_key"], errors="ignore")

    g = g.merge(kp, left_on="team_b_key", right_on="team_year_key", how="left")
    g = g.rename(
        columns={
            "KADJ O": "k_adj_o_b",
            "KADJ D": "k_adj_d_b",
            "KADJ EM": "k_adj_em_b",
            "K TEMPO": "k_tempo_b",
            "KADJ T": "k_adj_t_b",
            "ELITE SOS": "elite_sos_b",
        }
    )
    g = g.drop(columns=[c for c in g.columns if c == "team_year_key"], errors="ignore")

    # Teamsheet (NET/KPI/SOR, quad wins)
    g = g.merge(ts, left_on="team_a_key", right_on="team_year_key", how="left")
    g = g.rename(
        columns={
            "NET": "net_a",
            "KPI": "kpi_a",
            "SOR": "sor_a",
            "Q1 W": "q1_w_a",
            "Q2 W": "q2_w_a",
            "Q1&2 W": "q12_w_a",
        }
    )
    g = g.drop(columns=[c for c in g.columns if c == "team_year_key"], errors="ignore")

    g = g.merge(ts, left_on="team_b_key", right_on="team_year_key", how="left")
    g = g.rename(
        columns={
            "NET": "net_b",
            "KPI": "kpi_b",
            "SOR": "sor_b",
            "Q1 W": "q1_w_b",
            "Q2 W": "q2_w_b",
            "Q1&2 W": "q12_w_b",
        }
    )
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

    # Teamsheet diffs (lower ranks are better; diffs still carry signal)
    g["net_diff"] = g["net_a"] - g["net_b"]
    g["kpi_diff"] = g["kpi_a"] - g["kpi_b"]
    g["sor_diff"] = g["sor_a"] - g["sor_b"]
    g["q1_wins_diff"] = g["q1_w_a"] - g["q1_w_b"]
    g["q12_wins_diff"] = g["q12_w_a"] - g["q12_w_b"]

    # KenPom diffs
    if "k_adj_em_a" in g.columns and "k_adj_em_b" in g.columns:
        g["k_adj_em_diff"] = g["k_adj_em_a"] - g["k_adj_em_b"]
    if "k_adj_o_a" in g.columns and "k_adj_o_b" in g.columns:
        g["k_adj_o_diff"] = g["k_adj_o_a"] - g["k_adj_o_b"]
    if "k_adj_d_a" in g.columns and "k_adj_d_b" in g.columns:
        g["k_adj_d_diff"] = g["k_adj_d_a"] - g["k_adj_d_b"]
    if "k_tempo_a" in g.columns and "k_tempo_b" in g.columns:
        g["k_tempo_diff"] = g["k_tempo_a"] - g["k_tempo_b"]
    if "k_adj_t_a" in g.columns and "k_adj_t_b" in g.columns:
        g["k_adj_t_diff"] = g["k_adj_t_a"] - g["k_adj_t_b"]
    if "elite_sos_a" in g.columns and "elite_sos_b" in g.columns:
        g["elite_sos_diff"] = g["elite_sos_a"] - g["elite_sos_b"]

    # Seed gap features and specific matchup flags
    g["abs_seed_diff"] = (g["seed_diff"]).abs()
    g["is_1_16"] = (
        ((g["seed_a"] == 1) & (g["seed_b"] == 16))
        | ((g["seed_a"] == 16) & (g["seed_b"] == 1))
    ).astype(int)
    g["is_2_15"] = (
        ((g["seed_a"] == 2) & (g["seed_b"] == 15))
        | ((g["seed_a"] == 15) & (g["seed_b"] == 2))
    ).astype(int)
    g["is_3_14"] = (
        ((g["seed_a"] == 3) & (g["seed_b"] == 14))
        | ((g["seed_a"] == 14) & (g["seed_b"] == 3))
    ).astype(int)
    g["is_4_13"] = (
        ((g["seed_a"] == 4) & (g["seed_b"] == 13))
        | ((g["seed_a"] == 13) & (g["seed_b"] == 4))
    ).astype(int)
    g["is_5_12"] = (
        ((g["seed_a"] == 5) & (g["seed_b"] == 12))
        | ((g["seed_a"] == 12) & (g["seed_b"] == 5))
    ).astype(int)

    # Historical upset signal (true empirical rate from all games)
    if upset_rates is None:
        upset_rates = load_true_upset_rates_from_games()
    g["historical_upset_prob"] = g.apply(
        lambda r: upset_rates.get((int(r["round"]), int(r["seed_diff"])), 0.0),
        axis=1,
    )

    # Round as categorical / numeric
    g["round_num"] = g["round"]

    return g


def get_feature_columns() -> list[str]:
    """Column names used as model features (numeric)."""
    return [
        "seed_diff",
        "abs_seed_diff",
        "barthag_diff",
        "badj_em_diff",
        "elo_diff",
        "r_score_diff",
        "power_rating_diff",
        "net_diff",
        "kpi_diff",
        "sor_diff",
        "q1_wins_diff",
        "q12_wins_diff",
        "k_adj_em_diff",
        "k_adj_o_diff",
        "k_adj_d_diff",
        "k_tempo_diff",
        "k_adj_t_diff",
        "elite_sos_diff",
        "round_num",
    ]


def get_core_feature_columns() -> list[str]:
    """
    Core feature set: features that are available across most years.
    Excludes Teamsheet-based features (NET/KPI/SOR/Q1) because Teamsheet Ranks is not
    historically complete in this repo.
    """
    drop = {
        "net_diff",
        "kpi_diff",
        "sor_diff",
        "q1_wins_diff",
        "q12_wins_diff",
    }
    return [c for c in get_feature_columns() if c not in drop]


def get_enhanced_feature_columns() -> list[str]:
    """Enhanced feature set: includes Teamsheet-based SOS/quad signals when available."""
    return get_feature_columns()


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
