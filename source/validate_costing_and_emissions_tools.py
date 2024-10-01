"""
Date: May 5, 2024
Purpose: Code to validate functionality encoded in costing_and_emissions_tools and emissions_tools
"""

import costing_and_emissions_tools
import emissions_tools
import data_collection_tools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

G_PER_LB = 453.592
KWH_PER_MWH = 1000

"""
Function: Plots the given VMT distribution
Inputs:
    - VMT_distribution (pd.DataFrame): Distribution of VMT by year
"""
def plot_VMT_distribution(VMT_distribution_default, VMT_distribution_custom=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xlabel('Year', fontsize=18)
    ax.set_ylabel('VMT (miles/year)', fontsize=22)
    ax.set_xticks(range(1, 11))
    
    bar_width = 0.4  # Set width for each bar
    years = np.array(VMT_distribution_default['Year'])
    
    # Adjust positions for side-by-side bars
    ax.bar(years - bar_width / 2, VMT_distribution_default['VMT (miles)'],
           width=bar_width, color="indianred", label="Default VMT distribution (miles)")
    average_VMT_default = np.mean(VMT_distribution_default['VMT (miles)'])
    ax.axhline(average_VMT_default, label=f'Default average VMT (miles/year): {average_VMT_default:.0f}',
               linestyle='--', color='red')
    
    if VMT_distribution_custom is not None:
        ax.bar(years + bar_width / 2, VMT_distribution_custom['VMT (miles)'],
               width=bar_width, color="cornflowerblue", label="Custom VMT distribution (miles)")
        average_VMT_custom = np.mean(VMT_distribution_custom['VMT (miles)'])
        ax.axhline(average_VMT_custom, label=f'Custom average VMT (miles/year): {average_VMT_custom:.0f}',
                   linestyle='--', color='blue')

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.55)
    ax.legend(fontsize=16)
    plt.tight_layout()
    
    # Save the plot with appropriate file names
    if VMT_distribution_custom is not None:
        plt.savefig(f'plots/VMT_distribution_average_{average_VMT_custom:.0f}.png')
        plt.savefig(f'plots/VMT_distribution_average_{average_VMT_custom:.0f}.pdf')
    else:
        plt.savefig(f'plots/VMT_distribution_default.png')
        plt.savefig(f'plots/VMT_distribution_default.pdf')
    
"""
Function: Plots the given payload distribution
Inputs:
    - payload_distribution (pd.DataFrame)
"""
def plot_payload_distribution(payload_distribution_default, payload_distribution_custom=None):
    payload_distribution_default_lb = np.asarray(payload_distribution_default['Payload (lb)'])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xlabel('Payload (lb)', fontsize=22)
    n, bins, patches = ax.hist(payload_distribution_default_lb, bins=20, histtype=
    'step', color='indianred', label="Default payload distribution")
    average_payload_default = np.mean(payload_distribution_default_lb)
    ax.axvline(average_payload_default, label=f'Default average payload (lb): {average_payload_default:.0f}', linestyle='--', color='red')
    
    if payload_distribution_custom is not None:
        payload_distribution_custom_lb = np.asarray(payload_distribution_custom['Payload (lb)'])
        ax.hist(payload_distribution_custom_lb, bins=20, histtype=
    'step', color='cornflowerblue', label="Custom payload distribution")
        average_payload_custom = np.mean(payload_distribution_custom_lb)
        ax.axvline(average_payload_custom, label=f'Custom average payload (lb): {average_payload_custom:.0f}', linestyle='--', color='blue')
    
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax*1.7)
    ax.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(f'plots/payload_distribution_average_{average_payload_custom:.0f}lb.png')
    plt.savefig(f'plots/payload_distribution_average_{average_payload_custom:.0f}lb.pdf')
    
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
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax*1.2)
    ax.legend(fontsize=16)
    plt.tight_layout()
    if identifier_str:
        plt.savefig(f'plots/electricity_unit_price_{identifier_str}.png')
        plt.savefig(f'plots/electricity_unit_price_{identifier_str}.pdf')
    else:
        plt.savefig('plots/electricity_unit_price.png')
        plt.savefig('plots/electricity_unit_price.pdf')

def main():
    # Set default values for variable parameters
    m_payload_lb = 40000                        # lb
    average_VMT = 150000                         # miles/year
    charging_power = 400                        # Max charging power, in kW
    
    """
    ####### Plot the payload distribution with default and custom mean #######
    m_payload_lb = 40000                        # lb
    average_VMT = 150000                         # miles/year
    charging_power = 400                        # Max charging power, in kW
    # Collect and plot the payload distribution with the default average value or a custom input average
    payload_distribution_default = costing_and_emissions_tools.get_payload_distribution(None)
    payload_distribution_custom = costing_and_emissions_tools.get_payload_distribution(m_payload_lb)
    plot_payload_distribution(payload_distribution_default, payload_distribution_custom)
    ##########################################################################

    ####### Plot the VMT distribution with default and custom mean #######
    m_payload_lb = 40000                        # lb
    average_VMT = 150000                         # miles/year
    charging_power = 400                        # Max charging power, in kW
    parameters = data_collection_tools.read_parameters(truck_params = 'Semi', vmt_params = 'daycab_vmt_vius_2021')
    VMT_default = parameters.VMT.copy()
    VMT_custom = parameters.VMT.copy()
    VMT_default['VMT (miles)'] = costing_and_emissions_tools.get_VMT_distribution(parameters.VMT['VMT (miles)'], None)
    VMT_custom['VMT (miles)'] = costing_and_emissions_tools.get_VMT_distribution(parameters.VMT['VMT (miles)'], average_VMT)
    plot_VMT_distribution(VMT_default, VMT_custom)
    ######################################################################
    """
    
    """
    # Test obtaining and plotting the electricity unit cost for California electricity rates
    demand_charge_CA = 13.4       # $/kW
    electricity_rate_CA = 0.1918   # $/kWh
    m_payload_lb = 40000                        # lb
    average_VMT = 100000                         # miles/year
    charging_power = 400                        # Max charging power, in kW
    parameters, vehicle_model_results_dict = costing_and_emissions_tools.get_vehicle_model_results(m_payload_lb, average_VMT)
    electricity_cost_df = costing_and_emissions_tools.get_electricity_cost_by_year(parameters, vehicle_model_results_dict['Fuel economy (kWh/mi)'], demand_charge_CA, electricity_rate_CA, charging_power)
    plot_electricity_cost_by_year(electricity_cost_df, 'CA')
    """
    
    """
    # Plot the number of battery replacements as a function of VMT for a truck carrying a payload of 50,000 lb
    m_payload_lb = 40000                        # lb
    average_VMT = 100000                         # miles/year
    charging_power = 400                        # Max charging power, in kW
    average_VMTs = np.linspace(40000, 190000, 100)
    N_battery_replacements = np.zeros(0)
    for average_VMT in average_VMTs:
        parameters, vehicle_model_results_dict = costing_and_emissions_tools.get_vehicle_model_results(m_payload_lb, average_VMT)
        N_battery_replacements = np.append(N_battery_replacements, costing_and_emissions_tools.calculate_replacements(parameters.VMT['VMT (miles)'], vehicle_model_results_dict['Fuel economy (kWh/mi)']))
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xlabel('Average VMT (1000 miles/year)', fontsize=20)
    ax.set_ylabel('Lifetime battery replacements', fontsize=20)
    ax.grid(axis='y', ls='--')
    plt.plot(average_VMTs/1000, N_battery_replacements, linestyle='-', color='red', linewidth=3)
    plt.tight_layout()
    plt.savefig(f"plots/battery_replacements_vs_VMT_payload{m_payload_lb}.png")
    plt.savefig(f"plots/battery_replacements_vs_VMT_payload{m_payload_lb}.pdf")
    """
    
    
    # Plot the projected grid emission intensity for the WECC California balancing authority
    parameters, vehicle_model_results_dict = costing_and_emissions_tools.get_vehicle_model_results(m_payload_lb, average_VMT)
    emission_intensity_lb_per_MWh = 515.483      # lb CO2e / MWh
    emission_intensity_g_per_kWh = emission_intensity_lb_per_MWh * G_PER_LB / KWH_PER_MWH
    emissions_tools.emission(parameters).plot_CI_grid_projection(scenario='Present', grid_intensity_start=emission_intensity_g_per_kWh, grid_intensity_start_year=2022, start_year=2024, label='WECC California', label_save='CAMX')
    

    """
    # Make a quick validation plot for lifecycle emissions
    m_payload_lb = 40000                        # lb
    average_VMT = 100000                         # miles/year
    charging_power = 400                        # Max charging power, in kW
    emission_intensity_lb_per_MWh = 515.483      # lb CO2e / MWh
    demand_charge_CA = 13.4       # $/kW
    electricity_rate_CA = 0.1918   # $/kWh
    emissions = costing_and_emissions_tools.evaluate_emissions(m_payload_lb, emission_intensity_lb_per_MWh)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_ylabel('Lifecycle emissions (CO2e / mile)', fontsize=18)
    ax.bar(range(len(emissions)), emissions.values(), tick_label=['Manufacturing', 'Grid', 'Total'])
    plt.savefig('plots/lifecycle_emissions_validation.png')
    plt.savefig('plots/lifecycle_emissions_validation.pdf')
    
    # Make a quick validation plot for lifecycle costs of EV trucking
    costs = costing_and_emissions_tools.evaluate_costs(m_payload_lb, electricity_rate_CA, demand_charge_CA, charging_power = charging_power)
    """
    
    """
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_ylabel('Lifecycle costs ($ / mile)', fontsize=18)
    ax.bar(range(len(costs)), costs.values(), tick_label=['Capital', 'Operating', 'Electricity', 'Labor', 'Other OPEX', 'Total'])
    plt.savefig('plots/lifecycle_costs_validation.png')
    plt.savefig('plots/lifecycle_costs_validation.pdf')
    
    # Make a quick validation plot for lifecycle costs of diesel trucking in California
    diesel_price_df = pd.read_csv('tables/average_diesel_price_by_state.csv').set_index('State')
    diesel_price_CA = diesel_price_df['Average Price ($/gal)'].loc['CA']
    
    costs = costing_and_emissions_tools.evaluate_costs_diesel(m_payload_lb, diesel_price = diesel_price_CA)
        
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_ylabel('Lifecycle costs ($ / mile)', fontsize=18)
    ax.bar(range(len(costs)), costs.values(), tick_label=['Capital', 'Operating', 'Fuel', 'Labor', 'Other OPEX', 'Total'])
    plt.savefig('plots/lifecycle_costs_validation_diesel.png')
    plt.savefig('plots/lifecycle_costs_validation_diesel.pdf')
    """
    
if __name__ == '__main__':
    main()
