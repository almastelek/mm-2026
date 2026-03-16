"""
Parse Tournament Matchups into game-level rows with winner.
Each game = two rows (one per team); we pair by (YEAR, CURRENT ROUND) and consecutive order (BY YEAR NO).
"""
import pandas as pd
from pathlib import Path

from .config import DATA_DIR, OUTPUT_DIR


def load_matchups(path: Path | None = None) -> pd.DataFrame:
    """Load Tournament Matchups CSV."""
    path = path or (DATA_DIR / "Tournament Matchups.csv")
    df = pd.read_csv(path)
    df["SCORE"] = pd.to_numeric(df["SCORE"], errors="coerce")
    return df


def parse_games_from_matchups(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Derive game-level table from matchups.
    For each (YEAR, CURRENT ROUND), sort by BY YEAR NO descending and pair rows 0-1, 2-3, ...
    Returns DataFrame with: year, round, team_a, team_b, seed_a, seed_b, score_a, score_b, winner (1 = team_a wins).
    """
    if df is None:
        df = load_matchups()
    df = df.dropna(subset=["SCORE"])
    df = df.sort_values(["YEAR", "CURRENT ROUND", "BY YEAR NO"], ascending=[True, True, False])

    rows = []
    for (year, current_round), group in df.groupby(["YEAR", "CURRENT ROUND"]):
        g = group.reset_index(drop=True)
        n = len(g)
        if n % 2 != 0:
            continue
        for i in range(0, n, 2):
            row_a = g.iloc[i]
            row_b = g.iloc[i + 1]
            score_a = row_a["SCORE"]
            score_b = row_b["SCORE"]
            winner_a = 1 if score_a > score_b else 0
            rows.append({
                "year": year,
                "round": int(current_round),
                "team_a": row_a["TEAM"],
                "team_b": row_b["TEAM"],
                "seed_a": row_a["SEED"],
                "seed_b": row_b["SEED"],
                "score_a": score_a,
                "score_b": score_b,
                "winner": winner_a,
            })
    return pd.DataFrame(rows)


def validate_pairing(df: pd.DataFrame) -> dict:
    """Sanity checks on parsed games. Returns dict of counts and any issues."""
    games = parse_games_from_matchups(df)
    expected_per_round = {64: 32, 32: 16, 16: 8, 8: 4, 4: 2, 2: 1}
    report = {"total_games": len(games), "years": games["year"].unique().tolist()}
    for r, exp in expected_per_round.items():
        n = (games["round"] == r).sum()
        report[f"round_{r}"] = n
        if n % exp != 0:
            report["issues"] = report.get("issues", []) + [f"round {r}: got {n} games, expected multiple of {exp}"]
    return report


if __name__ == "__main__":
    df = load_matchups()
    games = parse_games_from_matchups(df)
    print("Games shape:", games.shape)
    print("Years:", games["year"].unique())
    print("Rounds:", sorted(games["round"].unique()))
    print(validate_pairing(df))
    games.to_csv(OUTPUT_DIR / "games.csv", index=False)
    print("Wrote output/games.csv")
