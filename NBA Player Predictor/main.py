from src.config import SEASONS, MIN_GP, DATA_RAW, DATA_PROCESSED, MODELS_DIR
from src.data_fetch import fetch_multi_season_stats
from src.features import build_next_season_dataset
from src.train import train_and_evaluate

def main():
    raw_path = DATA_RAW / "player_stats_5_seasons.csv"
    dataset_path = DATA_PROCESSED / "next_season_dataset.csv"

    print("1) Fetching NBA player stats via nba_api...")
    df = fetch_multi_season_stats(SEASONS)
    df.to_csv(raw_path, index=False)
    print(f"Saved raw data -> {raw_path}")

    print(f"2) Filtering players with GP >= {MIN_GP}...")
    df = df[df["GP"] >= MIN_GP].copy()

    print("3) Building next-season supervised dataset (predict next-season PTS/G)...")
    dataset = build_next_season_dataset(df, target_col="PTS")
    dataset.to_csv(dataset_path, index=False)
    print(f"Saved processed dataset -> {dataset_path}")
    print(f"Rows in dataset: {len(dataset):,}")

    print("4) Training Ridge + XGBoost...")
    results = train_and_evaluate(dataset, str(MODELS_DIR))

    print("\n=== Test Results ===")
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    print(f"\nModels saved to -> {MODELS_DIR}")

if __name__ == "__main__":
    main()