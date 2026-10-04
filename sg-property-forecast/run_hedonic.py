"""Train the transaction-level model on URA private residential transactions.

    python run_hedonic.py --ura-key YOUR_URA_ACCESS_KEY      # download via URA API
    python run_hedonic.py --file data/ura_transactions.json  # or a saved dump / CSV
"""
import argparse
from pathlib import Path

import pandas as pd

from sgpf.data import fetch_ura_transactions, load_transactions
from sgpf.hedonic import train_eval

p = argparse.ArgumentParser()
g = p.add_mutually_exclusive_group(required=True)
g.add_argument("--ura-key", help="URA Data Service access key")
g.add_argument("--file", help="saved URA JSON dump or flattened CSV")
p.add_argument("--test-months", type=int, default=6)
a = p.parse_args()

df = fetch_ura_transactions(a.ura_key) if a.ura_key else load_transactions(a.file)
print(f"{len(df):,} transactions loaded")
model, metrics, te = train_eval(df, test_months=a.test_months)
print("Out-of-time test metrics:", metrics)

out = Path("outputs"); out.mkdir(exist_ok=True)
te[["project", "sale_date", "area", "floor_range", "psm", "pred_psm"]].to_csv(
    out / "hedonic_test_predictions.csv", index=False)
imp = pd.Series(model.booster_.feature_importance("gain"), index=model.feature_name_)
print("Feature importance (gain):\n", (imp / imp.sum()).sort_values(ascending=False).round(3))
