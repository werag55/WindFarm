"""Run OAT sensitivity analysis for the reference 800 MW project."""

import logging

from src2.analysis.sensitivity import run_sensitivity_analysis
from src2.data_preprocessing.prep import prepare_data

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)


def main() -> None:
    df = prepare_data()
    out = run_sensitivity_analysis(df)
    summary  = out["summary"]
    rankings = summary["rankings"]
    baseline = summary["baseline"]

    print("\n" + "=" * 60)
    print("OAT SENSITIVITY — REFERENCE 800 MW PROJECT")
    print("=" * 60)
    print(f"Baseline CAPEX        : {baseline['predicted_capex_eur']/1e9:6.2f} bn EUR")
    print(f"Baseline CAPEX per MW : {baseline['capex_eur_per_mw']/1e6:6.2f} M EUR/MW")
    print(f"Baseline LCOE         : {baseline['lcoe_eur_per_mwh']:6.2f} EUR/MWh")

    for metric_key, label in [("predicted_capex_eur", "CAPEX"),
                              ("lcoe_eur_per_mwh", "LCOE")]:
        print(f"\nTop 5 features by {label} swing:")
        for i, rec in enumerate(rankings[metric_key][:5], start=1):
            print(
                f"  {i}. {rec['feature']:<36} "
                f"{rec['swing_pct']:+6.2f}%  "
                f"({rec['min']:.3e} ↔ {rec['max']:.3e})"
            )

    print("\nFull table → results/sensitivity_analysis.csv")
    print("Summary    → results/sensitivity_summary.json")


if __name__ == "__main__":
    main()