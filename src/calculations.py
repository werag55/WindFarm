"""Calculation functions for offshore wind project profitability analysis."""

import os

import numpy as np
import pandas as pd

from . import config

def unit_capex_eur_per_mw(budget_eur: float, capacity_mw: float) -> float:
    """Calculate EUR/MW CAPEX."""
    return budget_eur / capacity_mw


def capacity_factor_from_speed(speed_mps: float) -> float:
    """Map wind speed to the nearest empirical capacity factor."""
    if pd.isna(speed_mps):
        return np.nan
    nearest_speed = min(config.CAPACITY_FACTOR_BY_WIND_SPEED, \
        key=lambda value: abs(value - speed_mps))
    return config.CAPACITY_FACTOR_BY_WIND_SPEED[nearest_speed]


def productivity_gross_mwh_per_mw(capacity_factor: float) -> float:
    """Calculate gross MWh/MW/year productivity."""
    return config.HOURS_PER_YEAR * capacity_factor


def productivity_net_mwh_per_mw(gross_productivity: float) -> float:
    """Calculate net MWh/MW/year productivity after losses."""
    return gross_productivity * (1 - config.LOSS_FACTOR)


def annual_energy_mwh(net_productivity: float, capacity_mw: float) -> float:
    """Calculate annual energy production in MWh."""
    return net_productivity * capacity_mw


def opex_eur_per_mw(distance_km: float) -> float:
    """Compute EUR/MW/year OPEX based on distance brackets."""
    if pd.isna(distance_km):
        return np.nan
    base_eur = config.OPEX_BASE_KEUR_PER_MW * 1000.0
    for threshold, correction in config.OPEX_DISTANCE_CORRECTIONS:
        if distance_km <= threshold:
            return base_eur + correction * 1000.0
    return base_eur


def annual_opex_eur(opex_per_mw: float, capacity_mw: float) -> float:
    """Calculate annual OPEX in EUR."""
    return opex_per_mw * capacity_mw


def capital_recovery_factor(rate: float, years: int) -> float:
    """Standard CRF formula."""
    return rate / (1 - (1 + rate) ** (-years))


def annual_capex_eur(budget_eur: float, crf: float) -> float:
    """Calculate annualized CAPEX in EUR."""
    return budget_eur * crf

def lcoe_eur_per_mwh(annual_capex: float, annual_opex: float, annual_energy_mwh: float) -> float:
    """Calculate LCOE in EUR/MWh."""
    return (annual_capex + annual_opex) / annual_energy_mwh

def calculate(df: pd.DataFrame) -> pd.DataFrame:
    """Build the profitability output table."""
    
    if not os.path.isfile(config.CALCULATED_DATA_PATH):
        
        df = df.copy()
        
        df["unit_capex_eur_per_mw"] = unit_capex_eur_per_mw(df["total_project_budget_eur_indexed"], df["installed_capacity_MW"])
        df["unit_capex_eur_per_mw_indexed"] = unit_capex_eur_per_mw(df["total_project_budget_eur_indexed"], df["installed_capacity_MW"])

        df["capacity_factor"] = df["mean_wind_speed_mps"].apply(capacity_factor_from_speed)
        df["productivity_gross_mwh_per_mw"] = productivity_gross_mwh_per_mw(df["capacity_factor"])
        df["productivity_net_mwh_per_mw"] = productivity_net_mwh_per_mw(df["productivity_gross_mwh_per_mw"])
        df["annual_energy_mwh"] = annual_energy_mwh(df["productivity_net_mwh_per_mw"], df["installed_capacity_MW"])

        df["opex_eur_per_mw"] = df["distance_from_shore_km"].apply(opex_eur_per_mw)
        df["annual_opex_eur"] = annual_opex_eur(df["opex_eur_per_mw"], df["installed_capacity_MW"])

        df["crf"] = capital_recovery_factor(config.DISCOUNT_RATE, df["project_lifetime_years"])
        df["annual_capex_eur"] = annual_capex_eur(df["total_project_budget_eur"], df["crf"])
        df["annual_capex_eur_indexed"] = annual_capex_eur(df["total_project_budget_eur_indexed"], df["crf"])
        df["lcoe_eur_per_mwh"] = lcoe_eur_per_mwh(df["annual_capex_eur"], df["annual_opex_eur"], df["annual_energy_mwh"])
        df["lcoe_eur_per_mwh_indexed"] = lcoe_eur_per_mwh(df["annual_capex_eur_indexed"], df["annual_opex_eur"], df["annual_energy_mwh"])

        df.to_csv(config.CALCULATED_DATA_PATH, index=False)
        
    else:
        df = pd.read_csv(config.CALCULATED_DATA_PATH)
        
    return df
