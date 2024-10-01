import costing_and_emissions_tools
import json
import copy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

G_PER_LB = 453.592
DEFAULT_PAYLOAD = 50000     # Default payload, in lb
DEFAULT_AVG_VMT = 85000     # Default average VMT
DEFAULT_CHARGING_POWER = 750     # Default average VMT
KWH_PER_MWH = 1000
CENTS_PER_DOLLAR = 100

################################### Emissions ###################################
"""
Function: Collect the geojson containing grid emission intensity for different balancing authorities used to calculate emissions per mile
Inputs:
    - region_type (string): 'ba' means read in emission intensity by balancing authority region, and 'state' means to read in state-level emission intensity.
"""
def collect_grid_intensity_geo(region_type='ba'):
    filename = ''
    if region_type == 'ba':
        filename = 'egrid2022_subregions_merged.geojson'
    elif region_type == 'state':
        filename = 'eia2022_state_merged.geojson'
    else:
        print(f'Error: Cannot read in region type {region_type}')
    with open(f'geojsons/{filename}', mode='r') as geojson_file:
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
def plot_emissions_per_mile_breakdown(emissions_per_mi_geojson, states=['CA', 'TX', 'MA', 'IA'], identifier_str=None):
    emissions_to_plot_df = pd.DataFrame(columns=['State', 'GHGs manufacturing (gCO2/mi)', 'GHGs grid (gCO2/mi)'])
    for feature in emissions_per_mi_geojson['features']:
        for state in states:
            if 'STUSPS' in feature['properties'] and feature['properties']['STUSPS'] == state:
                emissions_per_mile_dict = {
                    'State': [state],
                    'GHGs manufacturing (gCO2/mi)': [feature['properties']['C_mi_man']],
                    'GHGs grid (gCO2/mi)': [feature['properties']['C_mi_grid']],
                }
                emissions_to_plot_df = pd.concat([emissions_to_plot_df, pd.DataFrame(emissions_per_mile_dict)], ignore_index=True)
    
    # Plot as a bar plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xlabel('State', fontsize=22)
    ax.set_ylabel('Lifecycle emissions (g CO2e / mile)', fontsize=22)
    ind = emissions_to_plot_df['State']
    
    # Stack each component of the costs / mile
    p1 = ax.bar(ind, emissions_to_plot_df['GHGs manufacturing (gCO2/mi)'], label='Manufacturing')
    p2 = ax.bar(ind, emissions_to_plot_df['GHGs grid (gCO2/mi)'], bottom=emissions_to_plot_df['GHGs manufacturing (gCO2/mi)'], label='Grid')
    ax.set_xticks(ind)
    
    # Adjust the y range to make space for the legend
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax*1.3)
    
    ax.legend(fontsize=18)
    plt.tight_layout()
    if identifier_str:
        plt.savefig(f'plots/emissions_per_mile_{identifier_str}.png')
        plt.savefig(f'plots/emissions_per_mile_{identifier_str}.pdf')
    else:
        plt.savefig(f'plots/emissions_per_mile.png')
        plt.savefig(f'plots/emissions_per_mile.pdf')

"""
Function: Loop through all the grid balancing authorities to evaluate the emissions per mile using the regional grid carbon intensity
Inputs:
    - average_payload (float): Average payload of shipments carried by the truck
    - average_VMT (float): Average annual miles traveled over the truck's lifetime
"""
def make_emissions_per_mi_geo(average_payload, average_VMT, grid_intensity_geojson, filename_prefix='', plot_validation=False):
    for feature in grid_intensity_geojson['features']:
        # Check if the 'CO2_rate' field exists in the properties
        if 'CO2_rate' in feature['properties'] and not (feature['properties']['CO2_rate'] is None):
            gCO2e_per_mi_df = get_gCO2e_per_mile(feature['properties']['CO2_rate'], average_payload, average_VMT)
            feature['properties']['C_mi_man'] = gCO2e_per_mi_df['GHGs manufacturing (gCO2/mi)']
            feature['properties']['C_mi_grid'] = gCO2e_per_mi_df['GHGs grid (gCO2/mi)']
            feature['properties']['C_mi_tot'] = gCO2e_per_mi_df['GHGs total (gCO2/mi)']
            del feature['properties']['CO2_rate']

    with open(f'geojsons/{filename_prefix}emissions_per_mile_payload{average_payload}_avVMT{average_VMT}.geojson', mode='w') as emissions_geojson:
        json.dump(grid_intensity_geojson, emissions_geojson, indent=4)
        
    # Plot emissions/mile breakdown for a few sample balancing authorities
    if plot_validation:
        plot_emissions_per_mile_breakdown(grid_intensity_geojson)

#################################################################################


##################################### Costs #####################################
"""
Function: Evaluates the lifecycle cost of EV trucking per mile, for a given set of costing inputs
Inputs:
    - electricity_rate_cents (float): Electricity rate by state, in cents/kWh
    - demand_charge (float): Average demand charge by state, in $/kW
    - average_payload (float): Average payload that the truck carries, in lb
    - average_VMT (float): Average annual miles traveled over the truck's lifetime
    - max_charging_power (float): Max power used by the truck's charger, in kW
"""
def get_costs_per_mile(electricity_rate_cents, demand_charge, average_payload = DEFAULT_PAYLOAD, average_VMT=DEFAULT_AVG_VMT, max_charging_power=DEFAULT_CHARGING_POWER):
    electricity_rate_dollars = electricity_rate_cents / CENTS_PER_DOLLAR
    cost_per_mi_df = costing_and_emissions_tools.evaluate_costs(average_payload, electricity_rate_dollars, demand_charge=demand_charge, average_VMT=average_VMT, charging_power=max_charging_power)
    return cost_per_mi_df
    
"""
Function: Evaluates the lifecycle cost of diesel trucking per mile, for a given set of costing inputs
Inputs:
    - diesel_price (float): Average diesel price by state, in $/gal
    - average_payload (float): Average payload that the truck carries, in lb
    - average_VMT (float): Average annual miles traveled over the truck's lifetime
"""
def get_costs_per_mile_diesel(diesel_price, average_payload = DEFAULT_PAYLOAD, average_VMT=DEFAULT_AVG_VMT):
    cost_per_mi_df = costing_and_emissions_tools.evaluate_costs_diesel(average_payload, diesel_price, average_VMT=average_VMT)
    return cost_per_mi_df

"""
Function: Plots lifecycle costs per mile for EV trucking, broken down into components
Inputs:
    - costs_per_mi_geojson (geojson): Geojson object containing cost per mile components and boundaries for each state
    - states (list): List containing the states for which to plot cost per mile breakdowns
    - identifier_str (string): If not None, adds a string identifier to the name of the saved plot
"""
def plot_costs_per_mile_breakdown(costs_per_mi_geojson, states=['CA', 'TX', 'MA', 'IA'], identifier_str=None):
    rates_to_plot_ev_df = pd.DataFrame(columns=['State', 'Total capital ($/mi)', 'Total electricity or fuel ($/mi)', 'Total labor ($/mi)', 'Other OPEXs ($/mi)'])
    rates_to_plot_diesel_df = pd.DataFrame(columns=['State', 'Total capital ($/mi)', 'Total electricity or fuel ($/mi)', 'Total labor ($/mi)', 'Other OPEXs ($/mi)'])
    
    for feature in costs_per_mi_geojson['features']:
        for state in states:
            if 'STUSPS' in feature['properties'] and feature['properties']['STUSPS'] == state:
                costs_per_mile_ev_dict = {
                    'State': state,
                    'Total capital ($/mi)': feature['properties']['$_mi_cap'],
                    'Total electricity or fuel ($/mi)': feature['properties']['$_mi_el'],
                    'Total labor ($/mi)': feature['properties']['$_mi_lab'],
                    'Other OPEXs ($/mi)': feature['properties']['$_mi_op'],
                }
                costs_per_mile_diesel_dict = {
                    'State': state,
                    'Total capital ($/mi)': feature['properties']['dies_cap'],
                    'Total electricity or fuel ($/mi)': feature['properties']['dies_fu'],
                    'Total labor ($/mi)': feature['properties']['dies_lab'],
                    'Other OPEXs ($/mi)': feature['properties']['dies_op'],
                }
                rates_to_plot_ev_df = pd.concat([rates_to_plot_ev_df, pd.DataFrame([costs_per_mile_ev_dict])], ignore_index=True)
                rates_to_plot_diesel_df = pd.concat([rates_to_plot_diesel_df, pd.DataFrame([costs_per_mile_diesel_dict])], ignore_index=True)

    # Plot side-by-side bar plot with solid and x fill
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xlabel('State', fontsize=22)
    ax.set_ylabel('Lifecycle cost ($/mile)', fontsize=22)

    ind = np.arange(len(states))  # the x locations for the states
    width = 0.35  # the width of the bars

    # Colors for both EV and diesel bars
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # EV bars (solid fill)
    p1 = ax.bar(ind - width/2, rates_to_plot_ev_df['Total capital ($/mi)'], width, color=colors[0], label='Capital')
    p2 = ax.bar(ind - width/2, rates_to_plot_ev_df['Total labor ($/mi)'], width,
                bottom=rates_to_plot_ev_df['Total capital ($/mi)'], color=colors[1], label='Labor')
    p3 = ax.bar(ind - width/2, rates_to_plot_ev_df['Other OPEXs ($/mi)'], width,
                bottom=rates_to_plot_ev_df['Total capital ($/mi)'] + rates_to_plot_ev_df['Total labor ($/mi)'],
                color=colors[2], label='Other OPEX')
    p4 = ax.bar(ind - width/2, rates_to_plot_ev_df['Total electricity or fuel ($/mi)'], width,
                bottom=rates_to_plot_ev_df['Total capital ($/mi)'] + rates_to_plot_ev_df['Total labor ($/mi)'] +
                       rates_to_plot_ev_df['Other OPEXs ($/mi)'], color=colors[3], label='Electricity')

    # Diesel bars (x fill)
    p5 = ax.bar(ind + width/2, rates_to_plot_diesel_df['Total capital ($/mi)'], width, color=colors[0], hatch='xx')
    p6 = ax.bar(ind + width/2, rates_to_plot_diesel_df['Total labor ($/mi)'], width,
                bottom=rates_to_plot_diesel_df['Total capital ($/mi)'], color=colors[1], hatch='xx')
    p7 = ax.bar(ind + width/2, rates_to_plot_diesel_df['Other OPEXs ($/mi)'], width,
                bottom=rates_to_plot_diesel_df['Total capital ($/mi)'] + rates_to_plot_diesel_df['Total labor ($/mi)'],
                color=colors[2], hatch='xx')
    p8 = ax.bar(ind + width/2, rates_to_plot_diesel_df['Total electricity or fuel ($/mi)'], width,
                bottom=rates_to_plot_diesel_df['Total capital ($/mi)'] + rates_to_plot_diesel_df['Total labor ($/mi)'] +
                       rates_to_plot_diesel_df['Other OPEXs ($/mi)'], color=colors[3], hatch='xx')

    ax.set_xticks(ind)
    ax.set_xticklabels(states)

    # Adjust the y range to make space for the legend
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.3)

    # Create a legend for the colors (Capital, Labor, Other OPEX, Electricity)
    color_legend = ax.legend(fontsize=18, loc='upper left')

    # Add an additional legend for solid and hashed fill (EV vs Diesel)
    solid_patch = plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', label='EV')
    hashed_patch = plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', hatch='xx', label='Diesel')
    
    fill_legend = plt.legend(handles=[solid_patch, hashed_patch], loc='upper right', fontsize=18, title='Powertrain Type', title_fontsize=22)

    # Add the first legend back
    ax.add_artist(color_legend)

    plt.tight_layout()

    if identifier_str:
        plt.savefig(f'plots/costs_per_mile_{identifier_str}.png')
        plt.savefig(f'plots/costs_per_mile_{identifier_str}.pdf')
    else:
        plt.savefig(f'plots/costs_per_mile.png')
        plt.savefig(f'plots/costs_per_mile.pdf')

"""
Function: Collect geojsons containing state-level electriity price and demand charge geojsons used to calculate cost per mile
Inputs: None
"""
def collect_cost_geos():
    with open('geojsons/electricity_rates_by_state_merged.geojson', mode='r') as geojson_file:
        electricity_rates_geojson = json.load(geojson_file)

    with open('geojsons/demand_charges_by_state.geojson', mode='r') as geojson_file:
        demand_charges_geojson = json.load(geojson_file)
        
    with open('geojsons/diesel_price_by_state.geojson', mode='r') as geojson_file:
        diesel_prices_geojson = json.load(geojson_file)
    
    # Ensure both GeoJSON files have the same number of features
    assert len(electricity_rates_geojson['features']) == len(demand_charges_geojson['features']), 'GeoJSON files have different numbers of features.'
    
    return electricity_rates_geojson, demand_charges_geojson, diesel_prices_geojson

"""
Function: Loop through all states to evaluate the costs per mile for EV trucking using the state-level electricity price and demand charge
Inputs:
    - average_payload (float): Average payload of shipments carried by the truck
    - average_VMT (float): Average annual miles traveled over the truck's lifetime
Note: The state features in the electricity rate and demand charge geojsons are in the same order because they're both derived from the same base shapefile
"""
def make_costs_per_mi_geo(average_payload, average_VMT, max_charging_power, electricity_rates_geojson, demand_charges_geojson, diesel_prices_geojson, plot_validation=False):
    costs_per_mi_geojson = copy.deepcopy(electricity_rates_geojson)
    for electricity_rate_feature, demand_charge_feature, diesel_price_feature, cost_per_mi_feature in zip(electricity_rates_geojson['features'], demand_charges_geojson['features'], diesel_prices_geojson['features'], costs_per_mi_geojson['features'], ):
        # Check if the 'STUSPS' field (state abbreviation) exists in the properties
        if 'Cents_kWh' in electricity_rate_feature['properties'] and 'Average Ma' in demand_charge_feature['properties'] and 'dies_price' in diesel_price_feature['properties']:

            del cost_per_mi_feature['properties']['Cents_kWh']

            if electricity_rate_feature['properties']['Cents_kWh'] is None or demand_charge_feature['properties']['Average Ma'] is None or diesel_price_feature['properties']['dies_price'] is None:
                cost_per_mi_feature['properties']['$_mi_tot'] = None
                cost_per_mi_feature['properties']['$_mi_cap'] = None
                cost_per_mi_feature['properties']['$_mi_el'] = None
                cost_per_mi_feature['properties']['$_mi_lab'] = None
                cost_per_mi_feature['properties']['$_mi_op'] = None
                
                cost_per_mi_feature['properties']['dies_tot'] = None
                cost_per_mi_feature['properties']['dies_cap'] = None
                cost_per_mi_feature['properties']['dies_fu'] = None
                cost_per_mi_feature['properties']['dies_lab'] = None
                cost_per_mi_feature['properties']['dies_op'] = None
                
                cost_per_mi_feature['properties']['diff_tot'] = None
                cost_per_mi_feature['properties']['diff_cap'] = None
                cost_per_mi_feature['properties']['diff_fu'] = None
                cost_per_mi_feature['properties']['diff_lab'] = None
                cost_per_mi_feature['properties']['diff_op'] = None
                
                cost_per_mi_feature['properties']['perc_tot'] = None
                cost_per_mi_feature['properties']['perc_cap'] = None
                cost_per_mi_feature['properties']['perc_fu'] = None
                cost_per_mi_feature['properties']['perc_lab'] = None
                cost_per_mi_feature['properties']['perc_op'] = None
            else:
                costs_per_mile = get_costs_per_mile(electricity_rate_feature['properties']['Cents_kWh'], demand_charge_feature['properties']['Average Ma'], average_payload, average_VMT, max_charging_power)
                costs_per_mile_diesel = get_costs_per_mile_diesel(diesel_price_feature['properties']['dies_price'], average_payload, average_VMT)
                cost_per_mi_feature['properties']['$_mi_tot'] = costs_per_mile['TCO ($/mi)']
                cost_per_mi_feature['properties']['$_mi_cap'] = costs_per_mile['Total capital ($/mi)']
                cost_per_mi_feature['properties']['$_mi_el'] = costs_per_mile['Total electricity ($/mi)']
                cost_per_mi_feature['properties']['$_mi_lab'] = costs_per_mile['Total labor ($/mi)']
                cost_per_mi_feature['properties']['$_mi_op'] = costs_per_mile['Other OPEXs ($/mi)']
                
                cost_per_mi_feature['properties']['dies_tot'] = costs_per_mile_diesel['TCO ($/mi)']
                cost_per_mi_feature['properties']['dies_cap'] = costs_per_mile_diesel['Total capital ($/mi)']
                cost_per_mi_feature['properties']['dies_fu'] = costs_per_mile_diesel['Total fuel ($/mi)']
                cost_per_mi_feature['properties']['dies_lab'] = costs_per_mile_diesel['Total labor ($/mi)']
                cost_per_mi_feature['properties']['dies_op'] = costs_per_mile_diesel['Other OPEXs ($/mi)']

                cost_per_mi_feature['properties']['diff_tot'] = costs_per_mile['TCO ($/mi)'] - costs_per_mile_diesel['TCO ($/mi)']
                cost_per_mi_feature['properties']['diff_cap'] = costs_per_mile['Total capital ($/mi)'] - costs_per_mile_diesel['Total capital ($/mi)']
                cost_per_mi_feature['properties']['diff_fu'] = costs_per_mile['Total electricity ($/mi)'] - costs_per_mile_diesel['Total fuel ($/mi)']
                cost_per_mi_feature['properties']['diff_lab'] = costs_per_mile['Total labor ($/mi)'] - costs_per_mile_diesel['Total labor ($/mi)']
                cost_per_mi_feature['properties']['diff_op'] = costs_per_mile['Other OPEXs ($/mi)'] - costs_per_mile_diesel['Other OPEXs ($/mi)']
                
                cost_per_mi_feature['properties']['perc_tot'] = 100*(costs_per_mile['TCO ($/mi)'] - costs_per_mile_diesel['TCO ($/mi)']) / costs_per_mile_diesel['TCO ($/mi)']
                cost_per_mi_feature['properties']['perc_cap'] = 100*(costs_per_mile['Total capital ($/mi)'] - costs_per_mile_diesel['Total capital ($/mi)']) / costs_per_mile_diesel['Total capital ($/mi)']
                cost_per_mi_feature['properties']['perc_fu'] = 100*(costs_per_mile['Total electricity ($/mi)'] - costs_per_mile_diesel['Total fuel ($/mi)']) / costs_per_mile_diesel['Total fuel ($/mi)']
                cost_per_mi_feature['properties']['perc_lab'] = 100*(costs_per_mile['Total labor ($/mi)'] - costs_per_mile_diesel['Total labor ($/mi)']) / costs_per_mile_diesel['Total labor ($/mi)']
                cost_per_mi_feature['properties']['perc_op'] = 100*(costs_per_mile['Other OPEXs ($/mi)'] - costs_per_mile_diesel['Other OPEXs ($/mi)']) / costs_per_mile_diesel['Other OPEXs ($/mi)']

    with open(f'geojsons/costs_per_mile_payload{average_payload}_avVMT{average_VMT}_maxChP{max_charging_power}.geojson', mode='w') as cost_geojson:
        json.dump(costs_per_mi_geojson, cost_geojson, indent=4)
        
    # Plot cost/mile breakdown for a few sample states
    if plot_validation:
        plot_costs_per_mile_breakdown(costs_per_mi_geojson)
    
def parallel_make_emissions(average_payload, average_VMT, grid_intensity_geojson_ba, grid_intensity_geojson_state):
    # Function to execute both tasks sequentially for a given set of arguments
    make_emissions_per_mi_geo(average_payload, average_VMT, grid_intensity_geojson_ba, 'ba_')
    make_emissions_per_mi_geo(average_payload, average_VMT, grid_intensity_geojson_state, 'state_')

def parallel_make_costs(average_payload, average_VMT, max_charging_power, electricity_rates_geojson, demand_charges_geojson, diesel_prices_geojson):
    make_costs_per_mi_geo(average_payload, average_VMT, max_charging_power, electricity_rates_geojson, demand_charges_geojson, diesel_prices_geojson)

    
def main():

    grid_intensity_geojson_ba = collect_grid_intensity_geo('ba')
    grid_intensity_geojson_state = collect_grid_intensity_geo('state')
    
    electricity_rates_geojson, demand_charges_geojson, diesel_prices_geojson = collect_cost_geos()
    
    ############################# Make validation plots #############################
    average_payload_default = 40000
    average_VMT_default = 100000
    max_charging_power_default = 400
    make_costs_per_mi_geo(average_payload_default, average_VMT_default, max_charging_power_default, electricity_rates_geojson, demand_charges_geojson, diesel_prices_geojson, plot_validation=True)
    make_emissions_per_mi_geo(average_payload_default, average_VMT_default, grid_intensity_geojson_state, filename_prefix='state_', plot_validation=True)
    #################################################################################
    
    average_payloads = [0, 10000, 20000, 30000, 40000, 50000]
    average_VMTs = [40000, 70000, 100000, 130000, 160000, 190000]
    max_charging_powers = [100, 200, 400, 800]
    
    # Setup the executor
    with ProcessPoolExecutor() as executor:
        # Dictionary to keep track of futures
        futures_emissions = {}
        futures_costs = {}
        
        
        # Evaluate emissions in parallel
        for average_payload in average_payloads:
            for average_VMT in average_VMTs:
                # Submit each combination of tasks to be executed in parallel
                future_emissions = executor.submit(parallel_make_emissions, average_payload, average_VMT, grid_intensity_geojson_ba, grid_intensity_geojson_state)
                futures_emissions[future_emissions] = (average_payload, average_VMT)
                
        # Wait for the emissions futures to complete and handle them if necessary
        for future_emissions in as_completed(futures_emissions):
            # You can add error handling or results processing here
            average_payload, average_VMT = futures_emissions[future_emissions]
            try:
                result = future_emissions.result()
                # Process result if needed
            except Exception as exc:
                print(f'Generated an exception: {exc} for payload: {average_payload}, VMT: {average_VMT}')
        
        
        # Evaluate EV trucking costs in parallel
        for average_payload in average_payloads:
            for average_VMT in average_VMTs:
                for max_charging_power in max_charging_powers:
                    # Submit each combination of tasks to be executed in parallel
                    future_costs = executor.submit(parallel_make_costs, average_payload, average_VMT, max_charging_power, electricity_rates_geojson, demand_charges_geojson, diesel_prices_geojson)
                    futures_costs[future_costs] = (average_payload, average_VMT, max_charging_power)
                
        # Wait for the costs futures to complete and handle them if necessary
        for future_costs in as_completed(futures_costs):
            # You can add error handling or results processing here
            average_payload, average_VMT, max_charging_power = futures_costs[future_costs]
            try:
                result = future_costs.result()
                # Process result if needed
            except Exception as exc:
                print(f'Generated an exception: {exc} for payload: {average_payload}, VMT: {average_VMT}, max charging power: {max_charging_power}')

if __name__ == '__main__':
    main()

#################################################################################
