"""Main execution script for the offshore wind project profitability analysis pipeline"""

from .utils.logging import configure_logging
from .model.linear_regression import train_and_evaluate
from .visualisations.profitability_map import profitability_map
from .calculations import calculate
from .data_preparation.prep import prepare_data

#TODO: fill missing data (marked in red in ../data/european_offshore_wind_capex.xlsx)
#TODO: add more prediction models
#TODO: perform sensitivity analysis

def main():
    """Full pipeline execution"""
    #TODO: Enhence logging throughout the pipeline
    configure_logging()
    df = prepare_data()
    df = calculate(df)
    profitability_map(df)
    train_and_evaluate(df)


if __name__ == "__main__":
    main()
