#!/usr/bin/env python3
"""Full data refresh + model refit pipeline. Safe to run via cron or by hand
with zero manual steps: re-downloads the historical dataset, recomputes Elo,
and refits the Dixon-Coles model, saving both to data/processed/ for the
CLI/app to load instantly.

Usage: python scripts/refresh_data.py [--source github|kaggle]
"""
import argparse

from wcwinner.pipeline import run_full_refresh

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["github", "kaggle"], default="github")
    args = parser.parse_args()
    run_full_refresh(source=args.source)
