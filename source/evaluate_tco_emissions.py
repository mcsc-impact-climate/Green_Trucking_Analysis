"""
Date: Feb 28, 2024
Purpose: Evaluate lifecycle costs and emissions as a function of:
    - payload
    - electricity price
    - demand charge
    - grid emission intensity
    - VMT
This code was originally written by Sayandeep Biswas, with modifications by Danika MacDonell
"""

# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import truck_model_tools
import costing_tools
import emissions_tools
from datetime import datetime

MONTHS_PER_YEAR = 12

"""
Function: Calculate the monthly charging energy requirements given a truck's annual miles traveled (VMT) and fuel economy
Inputs:
    - VMT: Annual miles traveled (miles / year)
    - fuel_economy: Fuel economy of the truck (kWh / mile)
"""
def calculate_charging_energy_per_month(VMT, fuel_economy):
    battery_energy_per_year = VMT * fuel_economy  # kWh / year
    return battery_energy_per_year / MONTHS_PER_YEAR    # kWh / month


def read_charger_cost_info(filename, scenario='Baseline'):
    charger_cost_df = pd.read_csv(filename, index_col='Scenario')
    installation_cost = float(charger_cost_df['Installation cost'].loc[scenario])
    hardware_cost = float(charger_cost_df['Hardware cost'].loc[scenario])
    fixed_monthly_cost = float(charger_cost_df['Fixed monthly cost'].loc[scenario])
    
    return installation_cost, hardware_cost, fixed_monthly_cost

"""
Function: Calculate the electricity price per kWh
Inputs:
    - VMT: Annual miles traveled (miles / year)
    - fuel_economy: Fuel economy of the truck (kWh / mile)
    - demand charge: Monthly charge for peak power used ($/kW)
    - electricity_charge: Retail electricity price ($/kWh)
    - charging_power: Average charging power (kW)
    - charging_efficiency: Efficiency of charging the battery (relative to energy from the power source)
"""
def calculate_electricity_price(VMT, fuel_economy, demand_charge, electricity_charge, charging_power, charging_efficiency=0.92, charger_cost_filename='data/charger_cost_data.csv', charger_cost_scenario='Baseline'):
    lifetime = 15       # Truck lifetime
    discount_rate = 7        # Discount rate (%)
    
    # Read in the charger cost info
    installation_cost, hardware_cost, fixed_monthly_cost = read_charger_cost_info(charger_cost_filename, charger_cost_scenario)
    
    # Convert charging energy per month to kWh
    charging_energy_per_month = calculate_charging_energy_per_month(VMT, fuel_economy)
        
    lifetime_energy_sold = (charging_energy_per_month * MONTHS_PER_YEAR * lifetime)
    capital_cost = (hardware_cost + installation_cost) * (1 + discount_rate / 100.)**lifetime
    norm_cap_cost = capital_cost / lifetime_energy_sold
    norm_demand_charge = charging_power * demand_charge / charging_energy_per_month
    norm_energy_charge = electricity_charge / charging_efficiency
    norm_fixed_monthly = fixed_monthly_cost * MONTHS_PER_YEAR * lifetime / lifetime_energy_sold
    total_charge = norm_cap_cost + (norm_demand_charge + norm_energy_charge + norm_fixed_monthly)
    
    return np.array([total_charge, norm_cap_cost, norm_fixed_monthly, norm_energy_charge, norm_demand_charge])
    
def 

print(calculate_electricity_price(180000, 2, 10, 0.15, 300, charger_cost_scenario='Pessimistic'))
