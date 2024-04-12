import costing_and_emissions_tools
import json
import copy
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

G_PER_LB = 453.592
DEFAULT_PAYLOAD = 50000     # Default payload, in lb
DEFAULT_AVG_VMT = 85000     # Default average VMT
KWH_PER_MWH = 1000
CENTS_PER_DOLLAR = 100

################################### Emissions ###################################
"""
Function: Collect the geojson containing grid emission intensity for different balancing authorities used to calculate emissions per mile
Inputs: None
"""
def collect_grid_intensity_geo():
    with open('geojsons/egrid2020_subregions_merged.geojson', mode='r') as geojson_file:
        grid_intensity_geojson = json.load(geojson_file)
    return grid_intensity_geojson

"""
Function: Evaluates the lifecycle emission rate for a given grid carbon intensity
Inputs:
    - lbCO2e_per_kWh (float): Grid emission intensity for the given region (lb CO2e / kWh)
    - average_payload (float): Average payload that the truck carries, in lb
"""
def get_gCO2e_per_mile(lbCO2e_per_kWh, average_payload = DEFAULT_PAYLOAD, average_VMT = DEFAULT_AVG_VMT):
    gCO2e_per_kWh = lbCO2e_per_kWh * G_PER_LB / KWH_PER_MWH
    gCO2e_per_mi_df = costing_and_emissions_tools.evaluate_emissions(average_payload, gCO2e_per_kWh, average_VMT=average_VMT)
    return gCO2e_per_mi_df
    
"""
Function: Plots lifecycle emissions per mile, broken down into components
Inputs:
    - emissions_per_mi_geojson (geojson): Geojson object containing emission per mile components and boundaries for each eGRIDs region
    - BAs (list): List containing the balancing authorities for which to plot emissions per mile breakdowns
    - identifier_str (string): If not None, adds a string identifier to the name of the saved plot
"""
def plot_emissions_per_mile_breakdown(emissions_per_mi_geojson, BAs=['ERCT', 'CAMX', 'NEWE', 'NYUP'], identifier_str=None):
    emissions_to_plot_df = pd.DataFrame(columns=['BA', 'GHGs manufacturing (gCO2/mi)', 'GHGs grid (gCO2/mi)'])
    for feature in emissions_per_mi_geojson['features']:
        for BA in BAs:
            if 'ZipSubregi' in feature['properties'] and feature['properties']['ZipSubregi'] == BA:
                emissions_per_mile_dict = {
                    'BA': [BA],
                    'GHGs manufacturing (gCO2/mi)': [feature['properties']['C_mi_man']],
                    'GHGs grid (gCO2/mi)': [feature['properties']['C_mi_grid']],
                }
                emissions_to_plot_df = pd.concat([emissions_to_plot_df, pd.DataFrame(emissions_per_mile_dict)], ignore_index=True)
    
    # Plot as a bar plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlabel('Balancing Authority', fontsize=18)
    ax.set_ylabel('Lifecycle emissions per mile (g CO2e/mile)', fontsize=17)
    ind = emissions_to_plot_df['BA']
    
    # Stack each component of the costs / mile
    p1 = ax.bar(ind, emissions_to_plot_df['GHGs manufacturing (gCO2/mi)'], label='Manufacturing')
    p2 = ax.bar(ind, emissions_to_plot_df['GHGs grid (gCO2/mi)'], bottom=emissions_to_plot_df['GHGs manufacturing (gCO2/mi)'], label='Grid')
    ax.set_xticks(ind)
    
    # Adjust the y range to make space for the legend
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax*1.1)
    
    ax.legend(fontsize=16)
    plt.tight_layout()
    if identifier_str:
        plt.savefig(f'plots/emissions_per_mile_{identifier_str}.png')
    else:
        plt.savefig(f'plots/emissions_per_mile.png')

"""
Function: Loop through all the grid balancing authorities to evaluate the emissions per mile using the regional grid carbon intensity
Inputs:
    - average_payload (float): Average payload of shipments carried by the truck
    - average_VMT (float): Average annual miles traveled over the truck's lifetime
"""
def make_emissions_per_mi_geo(average_payload, average_VMT, grid_intensity_geojson):
    for feature in grid_intensity_geojson['features']:
        # Check if the 'SRC2ERTA' field exists in the properties
        if 'SRC2ERTA' in feature['properties']:
            gCO2e_per_mi_df = get_gCO2e_per_mile(feature['properties']['SRC2ERTA'], average_payload, average_VMT)
            feature['properties']['C_mi_man'] = gCO2e_per_mi_df['GHGs manufacturing (gCO2/mi)']
            feature['properties']['C_mi_grid'] = gCO2e_per_mi_df['GHGs grid (gCO2/mi)']
            feature['properties']['C_mi_tot'] = gCO2e_per_mi_df['GHGs total (gCO2/mi)']
            del feature['properties']['SRC2ERTA']

    with open(f'geojsons/emissions_per_mile_payload{average_payload}_avVMT{average_VMT}.geojson', mode='w') as emissions_geojson:
        json.dump(grid_intensity_geojson, emissions_geojson, indent=4)
    
    ## Plot emissions/mile breakdown for a few sample balancing authorities
    #plot_emissions_per_mile_breakdown(grid_intensity_geojson)

#################################################################################


##################################### Costs #####################################
"""
Function: Evaluates the lifecycle emission rate for a given grid carbon intensity
Inputs:
    - lbCO2e_per_kWh (float): Grid emission intensity for the given region (lb CO2e / kWh)
    - average_payload (float): Average payload that the truck carries, in lb
"""
def get_costs_per_mile(electricity_rate_cents, demand_charge, average_payload = DEFAULT_PAYLOAD, average_VMT=DEFAULT_AVG_VMT):
    electricity_rate_dollars = electricity_rate_cents / CENTS_PER_DOLLAR
    cost_per_mi_df = costing_and_emissions_tools.evaluate_costs(average_payload, electricity_rate_dollars, demand_charge, average_VMT=average_VMT)
    return cost_per_mi_df

"""
Function: Plots lifecycle costs per mile, broken down into components
Inputs:
    - costs_per_mi_geojson (geojson): Geojson object containing cost per mile components and boundaries for each state
    - states (list): List containing the states for which to plot cost per mile breakdowns
    - identifier_str (string): If not None, adds a string identifier to the name of the saved plot
"""
def plot_costs_per_mile_breakdown(costs_per_mi_geojson, states=['CA', 'TX', 'MA', 'IA'], identifier_str=None):
    rates_to_plot_df = pd.DataFrame(columns=['State', 'Total capital ($/mi)', 'Total electricity ($/mi)', 'Total labor ($/mi)', 'Other OPEXs ($/mi)'])
    for feature in costs_per_mi_geojson['features']:
        for state in states:
            if 'STUSPS' in feature['properties'] and feature['properties']['STUSPS'] == state:
                costs_per_mile_dict = {
                    'State': [state],
                    'Total capital ($/mi)': [feature['properties']['$_mi_cap']],
                    'Total electricity ($/mi)': [feature['properties']['$_mi_el']],
                    'Total labor ($/mi)': [feature['properties']['$_mi_lab']],
                    'Other OPEXs ($/mi)': [feature['properties']['$_mi_op']],
                }
                rates_to_plot_df = pd.concat([rates_to_plot_df, pd.DataFrame(costs_per_mile_dict)], ignore_index=True)

    # Plot as a bar plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlabel('State', fontsize=18)
    ax.set_ylabel('Lifecycle cost per mile ($/mile)', fontsize=18)
    ind = rates_to_plot_df['State']

    # Stack each component of the costs / mile
    p1 = ax.bar(ind, rates_to_plot_df['Total capital ($/mi)'], label='Capital')
    p2 = ax.bar(ind, rates_to_plot_df['Total labor ($/mi)'], bottom=rates_to_plot_df['Total capital ($/mi)'], label='Labor')
    p3 = ax.bar(ind, rates_to_plot_df['Other OPEXs ($/mi)'], bottom=rates_to_plot_df['Total capital ($/mi)'] + rates_to_plot_df['Total labor ($/mi)'], label='Other OPEX')
    p4 = ax.bar(ind, rates_to_plot_df['Total electricity ($/mi)'], bottom=rates_to_plot_df['Total capital ($/mi)'] + rates_to_plot_df['Total labor ($/mi)'] + rates_to_plot_df['Other OPEXs ($/mi)'], label='Electricity')
    ax.set_xticks(ind)

    # Adjust the y range to make space for the legend
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax*1.5)

    ax.legend(fontsize=15)
    plt.tight_layout()
    if identifier_str:
        plt.savefig(f'plots/costs_per_mile_{identifier_str}.png')
    else:
        plt.savefig(f'plots/costs_per_mile.png')

"""
Function: Collect geojsons containing state-level electriity price and demand charge geojsons used to calculate cost per mile
Inputs: None
"""
def collect_cost_geos():
    with open('geojsons/electricity_rates_by_state_merged.geojson', mode='r') as geojson_file:
        electricity_rates_geojson = json.load(geojson_file)

    with open('geojsons/demand_charges_by_state.geojson', mode='r') as geojson_file:
        demand_charges_geojson = json.load(geojson_file)
    
    # Ensure both GeoJSON files have the same number of features
    assert len(electricity_rates_geojson['features']) == len(demand_charges_geojson['features']), 'GeoJSON files have different numbers of features.'
    
    return electricity_rates_geojson, demand_charges_geojson

"""
Function: Loop through all states to evaluate the costs per mile using the state-level electriity price and demand charge
Inputs:
    - average_payload (float): Average payload of shipments carried by the truck
    - average_VMT (float): Average annual miles traveled over the truck's lifetime
Note: The state features in the electricity rate and demand charge geojsons are in the same order because they're both derived from the same base shapefile
"""
def make_costs_per_mi_geo(average_payload, average_VMT, electricity_rates_geojson, demand_charges_geojson):
    costs_per_mi_geojson = copy.deepcopy(electricity_rates_geojson)
    for electricity_rate_feature, demand_charge_feature, cost_per_mi_feature in zip(electricity_rates_geojson['features'], demand_charges_geojson['features'], costs_per_mi_geojson['features']):
        # Check if the 'STUSPS' field (state abbreviation) exists in the properties
        if 'Cents_kWh' in electricity_rate_feature['properties'] and 'Average Ma' in demand_charge_feature['properties']:

            del cost_per_mi_feature['properties']['Cents_kWh']

            if electricity_rate_feature['properties']['Cents_kWh'] is None or demand_charge_feature['properties']['Average Ma'] is None:
                cost_per_mi_feature['properties']['$_mi_tot'] = None
                cost_per_mi_feature['properties']['$_mi_cap'] = None
                cost_per_mi_feature['properties']['$_mi_el'] = None
                cost_per_mi_feature['properties']['$_mi_lab'] = None
                cost_per_mi_feature['properties']['$_mi_op'] = None
            else:
                costs_per_mile = get_costs_per_mile(electricity_rate_feature['properties']['Cents_kWh'], demand_charge_feature['properties']['Average Ma'], average_payload, average_VMT)
                cost_per_mi_feature['properties']['$_mi_tot'] = costs_per_mile['TCO ($/mi)']
                cost_per_mi_feature['properties']['$_mi_cap'] = costs_per_mile['Total capital ($/mi)']
                cost_per_mi_feature['properties']['$_mi_el'] = costs_per_mile['Total electricity ($/mi)']
                cost_per_mi_feature['properties']['$_mi_lab'] = costs_per_mile['Total labor ($/mi)']
                cost_per_mi_feature['properties']['$_mi_op'] = costs_per_mile['Other OPEXs ($/mi)']

    with open(f'geojsons/costs_per_mile_payload{average_payload}_avVMT{average_VMT}.geojson', mode='w') as cost_geojson:
        json.dump(costs_per_mi_geojson, cost_geojson, indent=4)
        
    ## Plot cost/mile breakdown for a few sample states
    #plot_costs_per_mile_breakdown(costs_per_mi_geojson)
    
def parallel_make_emissions_and_costs(average_payload, average_VMT, grid_intensity_geojson, electricity_rates_geojson, demand_charges_geojson):
    # Function to execute both tasks sequentially for a given set of arguments
    make_emissions_per_mi_geo(average_payload, average_VMT, grid_intensity_geojson)
    make_costs_per_mi_geo(average_payload, average_VMT, electricity_rates_geojson, demand_charges_geojson)
    
def main():
    grid_intensity_geojson = collect_grid_intensity_geo()
    electricity_rates_geojson, demand_charges_geojson = collect_cost_geos()
    
    average_payloads = [0, 10000, 20000, 30000, 40000, 50000]
    average_VMTs = [40000, 70000, 100000, 130000, 160000, 190000]
    
    # Setup the executor
    with ProcessPoolExecutor() as executor:
        # Dictionary to keep track of futures
        futures = {}

        for average_payload in average_payloads:
            for average_VMT in average_VMTs:
                # Submit each combination of tasks to be executed in parallel
                future = executor.submit(parallel_make_emissions_and_costs, average_payload, average_VMT, grid_intensity_geojson, electricity_rates_geojson, demand_charges_geojson)
                futures[future] = (average_payload, average_VMT)

        # Wait for the futures to complete and handle them if necessary
        for future in as_completed(futures):
            # You can add error handling or results processing here
            average_payload, average_VMT = futures[future]
            try:
                result = future.result()
                # Process result if needed
            except Exception as exc:
                print(f'Generated an exception: {exc} for payload: {average_payload}, VMT: {average_VMT}')


if __name__ == '__main__':
    main()

#################################################################################
