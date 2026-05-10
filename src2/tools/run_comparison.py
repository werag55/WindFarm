"""Run both comparative analyses and print high-level conclusions."""

import logging
from pathlib import Path

import pandas as pd

from src2 import config
from src2.analysis.historical_data_comparison import (
    compare_full_vs_recent,
    compare_indexation_scenarios,
)
from src2.data_preprocessing.prep import prepare_data

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)


def main() -> None:
    df = prepare_data()

    # The cleaned CSV already contains a default WB-CPI indexed budget,
    # but each scenario in compare_indexation_scenarios re-runs indexation.
    full_vs_recent = compare_full_vs_recent(df, recent_years=5)
    indexation     = compare_indexation_scenarios(df)

    print("\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)

    # --- Conclusion 1 ---
    f = full_vs_recent["full"]["best_metrics"]
    r = full_vs_recent["recent"]["best_metrics"]
    d = full_vs_recent["delta"]
    print(
        f"[Full vs Recent]\n"
        f"  Full dataset best CV-R² = {f['cv_r2_mean']:.3f}, "
        f"Recent best CV-R² = {r['cv_r2_mean']:.3f}.\n"
        f"  Predicted CAPEX for the reference 800 MW project differs by "
        f"{d['predicted_capex_pct']:+.1f}% (recent vs full).\n"
        f"  Predicted LCOE differs by {d['lcoe_pct']:+.1f}%."
    )
    if abs(d["predicted_capex_pct"]) > 25:
        print("  → Large discrepancy: older projects materially shift CAPEX estimate.")
    else:
        print("  → Estimates broadly consistent — indexation is doing its job.")

    # --- Conclusion 2 ---
    rows = indexation["dashboard_data"]["metrics_table"]
    if rows:
        df_rows = pd.DataFrame(rows)
        best = df_rows.sort_values("cv_r2_mean", ascending=False).iloc[0]
        worst = df_rows.sort_values("cv_r2_mean", ascending=True).iloc[0]
        capex_min = df_rows["predicted_capex"].min()
        capex_max = df_rows["predicted_capex"].max()
        spread_pct = (capex_max - capex_min) / capex_min * 100 if capex_min else float("nan")
        print(
            f"\n[Indexation scenarios]\n"
            f"  Best CV-R² scenario: {best['scenario']} "
            f"(model={best['best_model']}, CV-R²={best['cv_r2_mean']:.3f}).\n"
            f"  Worst CV-R² scenario: {worst['scenario']} "
            f"(CV-R²={worst['cv_r2_mean']:.3f}).\n"
            f"  Predicted CAPEX spread across scenarios: "
            f"{capex_min:.2e} – {capex_max:.2e} EUR ({spread_pct:.1f}% range).\n"
        )
        if spread_pct > 30:
            print("  → Indexation choice has a large impact — must be documented carefully.")
        else:
            print("  → Indexation choice has limited impact on the prediction.")

    print(f"\nReports written to:\n  {Path('results/full_vs_recent_comparison.json')}\n"
          f"  {Path('results/indexation_scenarios_comparison.json')}")


if __name__ == "__main__":
    main()