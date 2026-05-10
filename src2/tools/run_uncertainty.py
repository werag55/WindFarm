"""Run bootstrap uncertainty for the reference 800 MW project."""

import argparse
import logging

from src2.analysis.uncertainty import bootstrap_prediction
from src2.data_preprocessing.prep import prepare_data

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="Number of bootstrap draws")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = prepare_data()
    report = bootstrap_prediction(df, n_bootstrap=args.n, random_state=args.seed)

    print("\n" + "=" * 60)
    print("BOOTSTRAP UNCERTAINTY — REFERENCE 800 MW PROJECT")
    print("=" * 60)
    s = report["settings"]
    print(f"Draws: {s['n_successful']} successful / {s['n_failed']} failed "
          f"(seed={s['random_state']}, training_rows={s['n_training_rows']})\n")

    capex = report["statistics"]["predicted_capex_eur"]
    capex_per_mw = report["statistics"]["capex_eur_per_mw"]
    lcoe  = report["statistics"]["lcoe_eur_per_mwh"]

    print(f"CAPEX        : {capex['mean']/1e9:6.2f}  ± {capex['std']/1e9:5.2f}  bn EUR")
    print(f"               95% CI  [{capex['p2_5']/1e9:6.2f} ; {capex['p97_5']/1e9:6.2f}]  bn EUR")
    print(f"CAPEX per MW : {capex_per_mw['mean']/1e6:6.2f}  ± {capex_per_mw['std']/1e6:5.2f}  M EUR/MW")
    print(f"               95% CI  [{capex_per_mw['p2_5']/1e6:6.2f} ; {capex_per_mw['p97_5']/1e6:6.2f}]  M EUR/MW")
    print(f"LCOE         : {lcoe['mean']:6.2f}  ± {lcoe['std']:5.2f}  EUR/MWh")
    print(f"               95% CI  [{lcoe['p2_5']:6.2f} ; {lcoe['p97_5']:6.2f}]  EUR/MWh")

    rel = (capex["p97_5"] - capex["p2_5"]) / max(capex["mean"], 1e-9) * 100
    print(f"\nRelative CAPEX 95% CI width: {rel:.1f}% of mean")
    if rel > 60:
        print("→ HIGH uncertainty — predictions should be reported as a range, not a single number.")
    elif rel > 30:
        print("→ MODERATE uncertainty — typical for small-N regression on heterogeneous data.")
    else:
        print("→ LOW uncertainty — model is stable across resamples.")


if __name__ == "__main__":
    main()