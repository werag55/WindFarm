"""Main entry point for the offshore wind analysis project."""

from .data_preprocessing.prep import prepare_data
from .utils.logging import configure_logging
from .model.model import train_model
from .ui.app import start_ui
from .ui.dashboard import create_dashboard
from .calculations.calculations import build_calculated_dataset


def main():
    """
    Main function to run the data analysis pipeline.
    """
    configure_logging()
    cleaned_data = prepare_data()
    calculated_data = build_calculated_dataset(cleaned_data)
    model, comparison_metrics, feature_importance, best_y_data, variants = train_model(calculated_data)
    create_dashboard(comparison_metrics, feature_importance, best_y_data, variants)
    start_ui(calculated_data, model)


if __name__ == "__main__":
    main()
