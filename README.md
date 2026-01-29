# NBA Player Performance

Predicts an NBA player's **next-season points per game (PTS/G)** using the previous season’s per-game stats.

## Why this project?
Teams and analysts often need quick, data-driven baselines for scouting and roster planning. This project benchmarks a linear model against a boosted tree model to quantify predictive performance and identify key drivers.

## Data
- 5 NBA seasons of player stats (fetched via `nba_api`)
- One row per player per season
- Supervised learning setup: features from season `t`, target is performance in season `t+1`

## Models
- Ridge Regression (baseline linear model)
- XGBoost Regressor (non-linear boosted trees)

## Metrics
- MAE
- RMSE
- R²

## How to run
```bash
pip install -r requirements.txt
python main.py
