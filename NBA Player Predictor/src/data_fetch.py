import time
import pandas as pd
from tqdm import tqdm
from nba_api.stats.endpoints import leaguedashplayerstats

def fetch_season_player_stats(season: str, sleep_s: float = 0.8) -> pd.DataFrame:
    """
    Fetch per-game regular season player stats for a given season.
    """
    endpoint = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="PerGame",
        season_type_all_star="Regular Season",
        timeout=100
    )
    df = endpoint.get_data_frames()[0]
    df["SEASON"] = season

    # Small delay to avoid getting rate-limited
    time.sleep(sleep_s)
    return df

def fetch_multi_season_stats(seasons: list[str]) -> pd.DataFrame:
    frames = []
    for s in tqdm(seasons, desc="Fetching seasons"):
        frames.append(fetch_season_player_stats(s))
    return pd.concat(frames, ignore_index=True)