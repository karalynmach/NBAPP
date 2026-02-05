import pandas as pd

KEEP_COLS = [
    "PLAYER_ID", "PLAYER_NAME", "SEASON",
    "GP", "MIN",
    "PTS", "AST", "REB", "STL", "BLK", "TOV",
    "FG_PCT", "FG3_PCT", "FT_PCT",
    "PLUS_MINUS",
]

FEATURE_COLS = [
    "GP", "MIN",
    "PTS", "AST", "REB", "STL", "BLK", "TOV",
    "FG_PCT", "FG3_PCT", "FT_PCT",
    "PLUS_MINUS",
]

def build_next_season_dataset(df: pd.DataFrame, target_col: str = "PTS") -> pd.DataFrame:
    """
    Creates a supervised dataset:
      X = player stats in season t
      y = target_col in season t+1 (same player)
    """
    df = df.copy()

    # Ensure all expected columns exist (nba_api can vary slightly)
    for c in KEEP_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    df = df[KEEP_COLS].sort_values(["PLAYER_ID", "SEASON"]).reset_index(drop=True)

    # Next-season target
    df["TARGET_NEXT"] = df.groupby("PLAYER_ID")[target_col].shift(-1)
    df["NEXT_SEASON"] = df.groupby("PLAYER_ID")["SEASON"].shift(-1)

    # Keep only rows where next season exists
    df = df.dropna(subset=["TARGET_NEXT", "NEXT_SEASON"]).reset_index(drop=True)

    return df

def split_X_y(dataset: pd.DataFrame):
    X = dataset[FEATURE_COLS].copy().fillna(0)
    y = dataset["TARGET_NEXT"].astype(float).copy()
    return X, y