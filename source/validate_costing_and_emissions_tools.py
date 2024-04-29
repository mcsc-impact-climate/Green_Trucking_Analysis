"""
Date: April 1, 2024
Purpose: Code to validate functionality encoded in costing_and_emissions_tools and emissions_tools
"""

import costing_and_emissions_tools
import emissions_tools
import data_collection_tools
import numpy as np
import matplotlib.pyplot as plt

G_PER_LB = 453.592
KWH_PER_MWH = 1000

"""
Function: Plots the given VMT distribution
Inputs:
    - VMT_distribution (pd.DataFrame): Distribution of VMT by year
"""
def plot_VMT_distribution(VMT_distribution):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlabel('Year', fontsize=18)
    ax.set_ylabel('VMT (miles/year)', fontsize=18)
    ax.set_xticks(range(1,11))
    ax.bar(VMT_distribution['Year'], VMT_distribution['VMT (miles)'])
    average_VMT = np.mean(VMT_distribution['VMT (miles)'])
    ax.axhline(average_VMT, label=f'Average VMT (miles/year): {average_VMT:.0f}', linestyle='--', color='red')
    ax.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'plots/VMT_distribution_average_{average_VMT:.0f}.png')
    
"""
Function: Plots the given payload distribution
Inputs:
    - payload_distribution (pd.DataFrame)
"""
def plot_payload_distribution(payload_distribution):
    payload_distribution_lb = np.asarray(payload_distribution['Payload (lb)'])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlabel('Payload (lb)', fontsize=18)
    ax.hist(payload_distribution_lb, bins=20)
    average_payload = np.mean(payload_distribution_lb)
    ax.axvline(average_payload, label=f'Average payload (lb): {average_payload:.0f}', linestyle='--', color='red')
    ax.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'plots/payload_distribution_average_{average_payload:.0f}lb.png')
    
"""
Function: Plots the electricity unit cost for each year, broken down into its components
Inputs:
    - electricity_cost_df (pd.DataFrame): Distribution of electricity unit costs by year
    - identifier_str (string): If not None, adds a string identifier to the name of the saved plot
"""
def plot_electricity_cost_by_year(electricity_cost_df, identifier_str=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlabel('Year', fontsize=18)
    ax.set_ylabel('Electricity unit cost ($/kWh)', fontsize=18)
    
    # Positions of the bars on the x-axis
    ind = electricity_cost_df['Year']
    
    # Stack each component of the electricity unit price
    p1 = ax.bar(ind, electricity_cost_df['Normalized capital'], label='Normalized capital')
    p2 = ax.bar(ind, electricity_cost_df['Normalized fixed'], bottom=electricity_cost_df['Normalized capital'], label='Normalized fixed')
    p3 = ax.bar(ind, electricity_cost_df['Normalized energy charge'], bottom=electricity_cost_df['Normalized capital'] + electricity_cost_df['Normalized fixed'], label='Normalized energy charge')
    p4 = ax.bar(ind, electricity_cost_df['Normalized demand charge'], bottom=electricity_cost_df['Normalized capital'] + electricity_cost_df['Normalized fixed'] + electricity_cost_df['Normalized energy charge'], label='Normalized demand charge')
    
    ax.set_xticks(ind)
    ax.legend(fontsize=16)
    plt.tight_layout()
    if identifier_str:
        plt.savefig(f'plots/electricity_unit_price_{identifier_str}.png')
    else:
        plt.savefig('plots/electricity_unit_price.png')

def main():
    # Set default values for variable parameters
    m_payload_lb = 50000                        # lb
    demand_charge = 10                          # $/kW
    grid_emission_intensity = 200               # Present grid emission intensity, in g CO2 / kWh
    electricity_charge = 0.15                   # cents/kW
    average_VMT = 85000                         # miles/year
    charging_power = 750                        # Max charging power, in kW
    
    # Test getting the payload distribution
    payload_distribution = costing_and_emissions_tools.get_payload_distribution(m_payload_lb)
    plot_payload_distribution(payload_distribution)
    
    # Test getting the VMT distribution
    parameters = data_collection_tools.read_parameters(truck_params = 'Semi', vmt_params = 'daycab_vmt_vius_2021')
    parameters.VMT['VMT (miles)'] = costing_and_emissions_tools.get_VMT_distribution(parameters.VMT['VMT (miles)'], average_VMT)
    plot_VMT_distribution(parameters.VMT)
    
    # Test obtaining and plotting the electricity unit cost for California electricity rates
    demand_charge_CA = 13.4       # $/kW
    electricity_rate_CA = 0.1918   # $/kWh
    parameters, vehicle_model_results_dict = costing_and_emissions_tools.get_vehicle_model_results(m_payload_lb, average_VMT)
    electricity_cost_df = costing_and_emissions_tools.get_electricity_cost_by_year(parameters, vehicle_model_results_dict['Fuel economy (kWh/mi)'], demand_charge_CA, electricity_rate_CA, charging_power)
    plot_electricity_cost_by_year(electricity_cost_df, 'CA')

    # Plot the projected grid emission intensity for the WECC California balancing authority
    emission_intensity_lb_per_MWh = 515.483      # lb CO2e / MWh
    emission_intensity_g_per_kWh = emission_intensity_lb_per_MWh * G_PER_LB / KWH_PER_MWH
    emissions_tools.emission(parameters).plot_CI_grid_projection(scenario='Present', grid_intensity_start=emission_intensity_g_per_kWh, start_year=2020, label='WECC California', label_save='CAMX')

    # Make a quick validation plot for lifecycle emissions
    emissions = costing_and_emissions_tools.evaluate_emissions(m_payload_lb, grid_emission_intensity)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_ylabel('Lifecycle emissions (CO2e / mile)', fontsize=18)
    ax.bar(range(len(emissions)), emissions.values(), tick_label=['Manufacturing', 'Grid', 'Total'])
    plt.savefig('plots/lifecycle_emissions_validation.png')
    
    # Make a quick validation plot for lifecycle costs
    costs = costing_and_emissions_tools.evaluate_costs(m_payload_lb, electricity_charge, demand_charge)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_ylabel('Lifecycle costs ($ / mile)', fontsize=18)
    ax.bar(range(len(costs)), costs.values(), tick_label=['Capital', 'Operating', 'Electricity', 'Labor', 'Other OPEX', 'Total'])
    plt.savefig('plots/lifecycle_costs_validation.png')

if __name__ == '__main__':
    main()
