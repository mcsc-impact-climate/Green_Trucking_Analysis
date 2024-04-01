"""
Date: Feb 28, 2024
Purpose: Evaluate lifecycle costs and emissions as a function of:
    - payload
    - electricity price
    - demand charge
    - grid emission intensity
    - VMT
This code was originally written by Kariana Moreno Sader and Sayandeep Biswas, with modifications by Danika MacDonell
"""

# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import truck_model_tools
import costing_tools
import emissions_tools
import data_collection_tools
from datetime import datetime

MONTHS_PER_YEAR = 12
KG_PER_TON = 1000
KG_PER_LB = 0.453592

"""
Function: Calculate the monthly charging energy requirements given a truck's annual miles traveled (VMT) and fuel economy
Inputs:
    - VMT (float): Annual miles traveled (miles / year)
    - mileage (float): Fuel economy of the truck (kWh / mile)
"""
def calculate_charging_energy_per_month(VMT, mileage):
    battery_energy_per_year = VMT * mileage  # kWh / year
    return battery_energy_per_year / MONTHS_PER_YEAR    # kWh / month

"""
Function: Reads in cost info for chargers for a given scenario
Inputs:
    - filename (string): Path to the csv file containing the charger cost info
    - scenario (string): Name of the scenario to consider in the csv file
"""
def read_charger_cost_info(filename, scenario='Baseline'):
    charger_cost_df = pd.read_csv(filename, index_col='Scenario')
    installation_cost = float(charger_cost_df['Installation cost'].loc[scenario])
    hardware_cost = float(charger_cost_df['Hardware cost'].loc[scenario])
    fixed_monthly_cost = float(charger_cost_df['Fixed monthly cost'].loc[scenario])
    
    return installation_cost, hardware_cost, fixed_monthly_cost

"""
Function: Calculate the electricity price per kWh
Inputs:
    - VMT (float): Annual miles traveled (miles / year)
    - mileage (float): Fuel economy of the truck (kWh / mile)
    - demand charge (float): Monthly charge for peak power used ($/kW)
    - electricity_charge (float): Retail electricity price ($/kWh)
    - charging_power (float): Average charging power (kW)
    - charging_efficiency (float): Efficiency of charging the battery (relative to energy from the power source)
"""
def calculate_electricity_unit(VMT, mileage, demand_charge, electricity_charge, charging_power, charging_efficiency=0.92, charger_cost_filename='data/charger_cost_data.csv', charger_cost_scenario='Baseline'):
    lifetime = 15       # Truck lifetime
    discount_rate = 7        # Discount rate (%)
    
    # Read in the charger cost info
    installation_cost, hardware_cost, fixed_monthly_cost = read_charger_cost_info(charger_cost_filename, charger_cost_scenario)
    
    # Convert charging energy per month to kWh
    charging_energy_per_month = calculate_charging_energy_per_month(VMT, mileage)
        
    lifetime_energy_sold = (charging_energy_per_month * MONTHS_PER_YEAR * lifetime)
    capital_cost = (hardware_cost + installation_cost) * (1 + discount_rate / 100.)**lifetime
    norm_cap_cost = capital_cost / lifetime_energy_sold
    norm_demand_charge = charging_power * demand_charge / charging_energy_per_month
    norm_energy_charge = electricity_charge / charging_efficiency
    norm_fixed_monthly = fixed_monthly_cost * MONTHS_PER_YEAR * lifetime / lifetime_energy_sold
    total_charge = norm_cap_cost + (norm_demand_charge + norm_energy_charge + norm_fixed_monthly)
    
    return total_charge, norm_cap_cost, norm_fixed_monthly, norm_energy_charge, norm_demand_charge
    
def calculate_electricity_unit_by_row(row, mileage, demand_charge, electricity_charge, charging_power):
    total_charge, norm_cap_cost, norm_fixed_monthly, norm_energy_charge, norm_demand_charge = calculate_electricity_unit(
        VMT = row['VMT (miles)'],
        mileage = mileage,
        demand_charge = demand_charge,
        electricity_charge = electricity_charge,
        charging_power = charging_power)
    return pd.Series([total_charge, norm_cap_cost, norm_fixed_monthly, norm_energy_charge, norm_demand_charge])
        
    
"""
Function: Given an average lifetime VMT, obtains the distribution of VMT over a 7-year period, assuming it follows the shape defined in Burnham, A et al. (2021)
Inputs:
    - average_VMT (float): Average annual miles traveled over the truck's lifetime
"""
def get_VMT_distribution(nominal_VMT, average_VMT):
    return average_VMT * nominal_VMT / np.mean(nominal_VMT)

"""
Function: Gets the mileage given an input payload
Inputs:
    - payload (float): Typical payload that the truck carries, in lb
    - f_linear_params (string): Path to a csv file containing the best-fit linear fit slope and y-intersect (along with uncertainties) for mileage vs. payload
"""
def get_mileage(m_payload, f_linear_params = 'tables/payload_vs_mileage_best_fit_params.csv'):
    payload_vs_mileage_params_df = pd.read_csv(f_linear_params)
    slope = payload_vs_mileage_params_df['slope (kWh/lb-mi)'].iloc[0]
    slope_unc = payload_vs_mileage_params_df['slope unc (kWh/lb-mi)'].iloc[0]
    b = payload_vs_mileage_params_df['b (kWh/mi)'].iloc[0]
    b_unc = payload_vs_mileage_params_df['b unc (kWh/mi)'].iloc[0]
    
    mileage = slope * m_payload + b
    mileage_unc = slope_unc * m_payload + b_unc
    
    return mileage, mileage_unc
    
"""
Function: Collects the VIUS payload distribution for class 8 semis and scales it to have the given average payload
Inputs:
    - payload (float): Desired average payload, in lb
"""
def get_payload_distribution(m_payload_lb):
    nominal_payload_distribution = pd.read_excel('data/payloaddistribution.xlsx')
    payload_distribution = nominal_payload_distribution.copy()
    payload_distribution['Payload (lb)'] = m_payload_lb * payload_distribution['Payload (lb)'] / np.mean(payload_distribution['Payload (lb)'])
    payload_distribution['Payload (kg)'] = payload_distribution['Payload (lb)']*KG_PER_LB #payload distribution in kgs
    return payload_distribution

"""
Function: Evaluates the payload penalty, which quantifies the relative increase in number of trucks needed to carry the given payload distribution given the reduced payload incurred by the battery weight.
Inputs:
    - payload_distribution (pd.DataFrame): Dataframe containing the payload distribution in both lb and kg
    - m_bat_kg (float): Mass of the battery, in kg
    - m_truck_no_bat_kg (float): Mass of the truck without payload or battery, in kg
    - m_truck_max_kg (float): Maximum GVW of the truck, including payload and battery, in kg
    - alpha (float): Used for payload penalty factor calculations (alpha = 1 for base case, alpha = 2: complete dependency in payload measurements)
"""
def get_payload_penalty(payload_distribution, m_bat_kg, m_truck_no_bat_kg, m_truck_max_kg, alpha=1):
    payload_max_kg = m_truck_max_kg - m_bat_kg - m_truck_no_bat_kg # payload + trailer
    payload_distribution['Payload loss (kg)'] = payload_distribution['Payload (kg)'].apply(lambda x: np.maximum(x - payload_max_kg, 0))
    payload_penalty = 1 + (alpha*payload_distribution['Payload loss (kg)'].mean()) / payload_max_kg
    return payload_penalty

"""
Function: Evaluates the total electricity cost for each year given the varying VMT
Inputs:
    - parameters (read_parameters class instance): Instance of the read_parameters class defined in truck_model_tools, containing truck parameters
    - mileage (float): Fuel economy of the truck (kWh / mile)
    - demand charge (float): Monthly charge for peak power used ($/kW)
    - electricity_charge (float): Retail electricity price ($/kWh)
    - charging_power (float): Average charging power (kW)
"""
def get_electricity_cost_by_year(parameters, mileage, demand_charge, electricity_charge, charging_power):
    electricity_cost_df = parameters.VMT.copy()
    electricity_cost_df[['Total', 'Normalized capital', 'Normalized fixed', 'Normalized energy charge', 'Normalized demand charge']] = electricity_cost_df.apply(calculate_electricity_unit_by_row, axis=1, mileage=mileage, demand_charge=demand_charge, electricity_charge=electricity_charge, charging_power=charging_power)     # $/kWh
    return electricity_cost_df
    
"""
Function: Reads in and evaluates specs and performance parameters for the truck
Inputs:
    - m_payload_lb (float): Payload carried by the truck, in lb
    - truck_type (string): String identifier for truck specs
    - battery_chemistry (string): Battery chemistry (either NMC or LFP)
    - e_bat (float): Energy capacity of the truck battery, in kWh
    - m_truck_max_lb (float): Maximum allowable GVW of the truck (82000lb for EVs in California)
    - scenario (string): Time scenario (Present, Mid term or Long term)
"""
def get_vehicle_model_results(m_payload_lb, average_VMT, truck_type='semi', battery_chemistry='NMC', e_bat=825, m_truck_max_lb=82000, scenario='Present'):

    # Read in parameters for the given truck type
    parameters = data_collection_tools.read_parameters(truck_params = truck_type)
    
    parameters.VMT['VMT (miles)'] = get_VMT_distribution(parameters.VMT['VMT (miles)'], average_VMT)
    
    # Read in data for the chosen scenario
    scenario_data = data_collection_tools.read_scenario_data(scenario=scenario, chemistry=battery_chemistry)

    # Get the mileage and uncertainty for the given payload
    mileage, mileage_unc = get_mileage(m_payload_lb, f_linear_params = 'tables/payload_vs_mileage_best_fit_params.csv')
    
    # Calculate the masses of the battery and truck, given the input battery capacity and energy density
    e_density = scenario_data['Energy Density (kWh/ton)']
    m_truck_max_kg = m_truck_max_lb * KG_PER_LB
    m_bat_kg = e_bat / e_density * KG_PER_TON          # Battery mass, in kg
    m_bat_lb = m_bat_kg / KG_PER_LB
    m_truck_no_bat_kg = parameters.m_truck_no_bat
    m_truck_no_bat_lb = m_truck_no_bat_kg / KG_PER_LB
    m_truck_lb = m_bat_lb + m_truck_no_bat_lb + m_payload_lb
    
    # Scale the VIUS payload distribution to one with the same shape whose average is the given payload
    payload_distribution = get_payload_distribution(m_payload_lb)
    
    # Calculate the payload penalty factor
    payload_penalty_factor = get_payload_penalty(payload_distribution, m_bat_kg, parameters.m_truck_no_bat, m_truck_max_kg)
    
    vehicle_model_results_dict = {
        'Battery capacity (kWh)': e_bat,
        'Battery mass (lbs)': m_bat_lb,
        'Fuel economy (kWh/mi)': mileage,
        'Payload penalty factor': payload_penalty_factor,
        'Total vehicle mass (lbs)': m_truck_lb
    }
    
    return parameters, vehicle_model_results_dict
    
"""
Function: Calculates lifecycle GHG emissions of the truck per mile driven, accounting for battery manufacturing and grid electricity production
Inputs:
    - m_payload_lb (float): Payload carried by the truck, in lb
    - grid_emission_intensity (float): Emission intensity of the power grid (g CO2 / kWh)
    - battery_chemistry (string): Battery chemistry (either NMC or LFP)
    - e_bat (float): Energy capacity of the truck battery, in kWh
    - m_truck_max_lb (float): Maximum allowable GVW of the truck (82000lb for EVs in California)
    - scenario (string): Time scenario (Present, Mid term or Long term)
"""
def evaluate_emissions(m_payload_lb, grid_emission_intensity, average_VMT=85000, grid_emission_intensity_year=2020, e_bat=825, battery_chemistry='NMC', m_truck_max_lb=82000, scenario='Present'):
    
    # Evaluate parameters and vehicle model results for the given payload
    parameters, vehicle_model_results_dict = get_vehicle_model_results(m_payload_lb, average_VMT)
    
    calculate_replacements(parameters.VMT['VMT (miles)'], vehicle_model_results_dict['Fuel economy (kWh/mi)'], e_bat=825, max_battery_cycles=1000)
    
    # Read in data for the chosen scenario
    scenario_data = data_collection_tools.read_scenario_data(scenario=scenario, chemistry=battery_chemistry)
    
    # Read in battery parameters
    battery_params_dict = data_collection_tools.read_battery_params(chemistry=battery_chemistry)
    
    # Calculate the number of battery replacements needed
    battery_params_dict['Replacements'] = calculate_replacements(parameters.VMT['VMT (miles)'], vehicle_model_results_dict['Fuel economy (kWh/mi)'])
    
    # Calculate GHG emissions per mile
    GHG_emissions = emissions_tools.emission(parameters).get_WTW(vehicle_model_results_dict, battery_params_dict['Manufacturing emissions (CO2/kWh)'],  battery_params_dict['Replacements'], grid_intensity_start=grid_emission_intensity, start_year=grid_emission_intensity_year)
    
    return GHG_emissions

"""
Function: Calculates the total number of battery replacements needed for the truck over its lifetime
Inputs:
    - VMT_df (pd.DataFrame): Dataframe containing the annual miles traveled (VMT) for each year of the truck's life
    - mileage (float): Fuel economy of the truck (kWh / mile)
    - e_bat (float): Energy capacity of the truck battery, in kWh
    - max_battery_cycles (int): Maximum number of full battery charge-discharge cycles before it needs to be replaced
"""
def calculate_replacements(VMT_df, mileage, e_bat=825, max_battery_cycles=1500):
    lifetime_miles_traveled = VMT_df.sum()
    lifetime_kWh_charged = lifetime_miles_traveled * mileage
    lifetime_cycles = lifetime_kWh_charged / e_bat
    n_replacements = np.floor(lifetime_cycles / max_battery_cycles)
    return n_replacements
    
"""
Function: Calculates lifecycle costs of purchasing and operating per mile driven. Costs account for:
    - Truck purchase (capital)
    - Operating costs (maintenance & repair, insurance, misc)
    - Labor
    - Electricity
Inputs:
    - m_payload_lb (float): Payload carried by the truck, in lb
    - grid_emission_intensity (float): Emission intensity of the power grid (g CO2 / kWh)
    - battery_chemistry (string): Battery chemistry (either NMC or LFP)
    - e_bat (float): Energy capacity of the truck battery, in kWh
    - m_truck_max_lb (float): Maximum allowable GVW of the truck (82000lb for EVs in California)
    - vehicle_purchase_price (float): Purchase price of the vehicle. Defaults to the inferred estimated price of $250,000 for the Tesla Semi, based on reports that PepsiCo purchased 18 Semis with $4.5 million in grants (https://www.sacbee.com/news/business/article274186280.html)
    - scenario (string): Time scenario (Present, Mid term or Long term)
"""
def evaluate_costs(m_payload_lb, electricity_charge, demand_charge, average_VMT=85000, charging_power=750, e_bat=825, battery_chemistry='NMC', m_truck_max_lb=82000, vehicle_purchase_price=250000, scenario='Present'):
    
    # Evaluate parameters and vehicle model results for the given payload
    parameters, vehicle_model_results_dict = get_vehicle_model_results(m_payload_lb, average_VMT)
    
    # Read in data for the chosen scenario
    scenario_data = data_collection_tools.read_scenario_data(scenario=scenario, chemistry=battery_chemistry)
    
    # Read in battery parameters
    battery_params_dict = data_collection_tools.read_battery_params(chemistry=battery_chemistry)
    
    # Calculate the number of battery replacements needed
    battery_params_dict['Replacements'] = calculate_replacements(parameters.VMT['VMT (miles)'], vehicle_model_results_dict['Fuel economy (kWh/mi)'])
    
    # Calculate the electricity price breakdown for each year
    electricity_cost_df = get_electricity_cost_by_year(parameters, vehicle_model_results_dict['Fuel economy (kWh/mi)'], demand_charge, electricity_charge, charging_power)

    # Calculate TCO per mile
    # ToDo: implement a function to calculate the number of replacements given the VMT distribution
    TCO = costing_tools.cost(parameters).get_TCO(vehicle_model_results_dict, scenario_data['Capital Costs ($/kW)'], scenario_data['Battery Unit Cost ($/kWh)'], scenario_data['Operating Costs ($/mi)'], electricity_cost_df['Total'], battery_params_dict['Replacements'], vehicle_purchase_price = vehicle_purchase_price)
    
    return TCO

## Uncomment the main function to test the functions defined above
#def main():
#    # Set default values for variable parameters
#    m_payload_lb = 50000                        # lb
#    demand_charge = 10                          # $/kW
#    grid_emission_intensity = 200               # Present grid emission intensity, in g CO2 / kWh
#    electricity_charge = 0.15                   # cents/kW
#
#    emissions = evaluate_emissions(m_payload_lb, grid_emission_intensity)
#    costs = evaluate_costs(m_payload_lb, electricity_charge, demand_charge)
#
#    print(emissions)
#    print(costs)
#
#if __name__ == '__main__':
#    main()
