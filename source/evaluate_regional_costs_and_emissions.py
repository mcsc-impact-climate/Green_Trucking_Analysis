import tco_emissions_tools
import json
import copy
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import shape, mapping
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon

G_PER_LB = 453.592
DEFAULT_PAYLOAD = 50000     # Default payload, in lb
KWH_PER_MWH = 1000
CENTS_PER_DOLLAR = 100

################################### Emissions ###################################
with open('geojsons/egrid2020_subregions_merged.geojson', mode='r') as geojson_file:
    geojson_data = json.load(geojson_file)

"""
Function: Evaluates the lifecycle emission rate for a given grid carbon intensity
Inputs:
    - lbCO2e_per_kWh (float): Grid emission intensity for the given region (lb CO2e / kWh)
    - average_payload (float): Average payload that the truck carries, in lb
"""
def get_gCO2e_per_mile(lbCO2e_per_kWh, average_payload = DEFAULT_PAYLOAD):
    gCO2e_per_kWh = lbCO2e_per_kWh * G_PER_LB / KWH_PER_MWH
    gCO2e_per_mi_df = tco_emissions_tools.evaluate_emissions(average_payload, gCO2e_per_kWh)
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

for feature in geojson_data['features']:
    # Check if the 'SRC2ERTA' field exists in the properties
    if 'SRC2ERTA' in feature['properties']:
        gCO2e_per_mi_df = get_gCO2e_per_mile(feature['properties']['SRC2ERTA'])
        feature['properties']['C_mi_man'] = gCO2e_per_mi_df['GHGs manufacturing (gCO2/mi)']
        feature['properties']['C_mi_grid'] = gCO2e_per_mi_df['GHGs grid (gCO2/mi)']
        feature['properties']['C_mi_tot'] = gCO2e_per_mi_df['GHGs total (gCO2/mi)']
        del feature['properties']['SRC2ERTA']

with open('geojsons/emissions_per_mile.geojson', mode='w') as emissions_geojson:
    json.dump(geojson_data, emissions_geojson, indent=4)
    
# Plot emissions/mile breakdown for a few sample balancing authorities
plot_emissions_per_mile_breakdown(geojson_data)

#################################################################################


###################################### Costs #####################################
#"""
#Function: Evaluates the lifecycle emission rate for a given grid carbon intensity
#Inputs:
#    - lbCO2e_per_kWh (float): Grid emission intensity for the given region (lb CO2e / kWh)
#    - average_payload (float): Average payload that the truck carries, in lb
#"""
#def get_costs_per_mile(electricity_rate_cents, demand_charge, average_payload = DEFAULT_PAYLOAD):
#    electricity_rate_dollars = electricity_rate_cents / CENTS_PER_DOLLAR
#    cost_per_mi_df = tco_emissions_tools.evaluate_costs(average_payload, electricity_rate_dollars, demand_charge)
#    return cost_per_mi_df
#
#"""
#Function: Plots lifecycle costs per mile, broken down into components
#Inputs:
#    - costs_per_mi_geojson (geojson): Geojson object containing cost per mile components and boundaries for each state
#    - states (list): List containing the states for which to plot cost per mile breakdowns
#    - identifier_str (string): If not None, adds a string identifier to the name of the saved plot
#"""
#def plot_costs_per_mile_breakdown(costs_per_mi_geojson, states=['CA', 'TX', 'MA', 'IA'], identifier_str=None):
#    rates_to_plot_df = pd.DataFrame(columns=['State', 'Total capital ($/mi)', 'Total electricity ($/mi)', 'Total labor ($/mi)', 'Other OPEXs ($/mi)'])
#    for feature in costs_per_mi_geojson['features']:
#        for state in states:
#            if 'STUSPS' in feature['properties'] and feature['properties']['STUSPS'] == state:
#                costs_per_mile_dict = {
#                    'State': [state],
#                    'Total capital ($/mi)': [feature['properties']['$_mi_cap']],
#                    'Total electricity ($/mi)': [feature['properties']['$_mi_el']],
#                    'Total labor ($/mi)': [feature['properties']['$_mi_lab']],
#                    'Other OPEXs ($/mi)': [feature['properties']['$_mi_op']],
#                }
#                rates_to_plot_df = pd.concat([rates_to_plot_df, pd.DataFrame(costs_per_mile_dict)], ignore_index=True)
#
#    # Plot as a bar plot
#    fig, ax = plt.subplots(figsize=(8, 5))
#    ax.tick_params(axis='both', which='major', labelsize=15)
#    ax.set_xlabel('State', fontsize=18)
#    ax.set_ylabel('Lifecycle cost per mile ($/mile)', fontsize=18)
#    ind = rates_to_plot_df['State']
#
#    # Stack each component of the costs / mile
#    p1 = ax.bar(ind, rates_to_plot_df['Total capital ($/mi)'], label='Capital')
#    p2 = ax.bar(ind, rates_to_plot_df['Total labor ($/mi)'], bottom=rates_to_plot_df['Total capital ($/mi)'], label='Labor')
#    p3 = ax.bar(ind, rates_to_plot_df['Other OPEXs ($/mi)'], bottom=rates_to_plot_df['Total capital ($/mi)'] + rates_to_plot_df['Total labor ($/mi)'], label='Other OPEX')
#    p4 = ax.bar(ind, rates_to_plot_df['Total electricity ($/mi)'], bottom=rates_to_plot_df['Total capital ($/mi)'] + rates_to_plot_df['Total labor ($/mi)'] + rates_to_plot_df['Other OPEXs ($/mi)'], label='Electricity')
#    ax.set_xticks(ind)
#
#    # Adjust the y range to make space for the legend
#    ymin, ymax = ax.get_ylim()
#    ax.set_ylim(ymin, ymax*1.5)
#
#    ax.legend(fontsize=15)
#    plt.tight_layout()
#    if identifier_str:
#        plt.savefig(f'plots/costs_per_mile_{identifier_str}.png')
#    else:
#        plt.savefig(f'plots/costs_per_mile.png')
#
#with open('geojsons/electricity_rates_by_state_merged.geojson', mode='r') as geojson_file:
#    electricity_rates_geojson = json.load(geojson_file)
#
#with open('geojsons/demand_charges_by_state.geojson', mode='r') as geojson_file:
#    demand_charges_geojson = json.load(geojson_file)
#
#costs_per_mi_geojson = copy.deepcopy(electricity_rates_geojson)
#
## Ensure both GeoJSON files have the same number of features
#assert len(electricity_rates_geojson['features']) == len(demand_charges_geojson['features']), 'GeoJSON files have different numbers of features.'
#
## Loop through all states (note: the state features in the electricity rate and demand charge geojsons are in the same order because they're both derived from the same base shapefile)
#
#for electricity_rate_feature, demand_charge_feature, cost_per_mi_feature in zip(electricity_rates_geojson['features'], demand_charges_geojson['features'], costs_per_mi_geojson['features']):
#    # Check if the 'STUSPS' field (state abbreviation) exists in the properties
#    if 'Cents_kWh' in electricity_rate_feature['properties'] and 'Average Ma' in demand_charge_feature['properties']:
#
#        del cost_per_mi_feature['properties']['Cents_kWh']
#
#        if electricity_rate_feature['properties']['Cents_kWh'] is None or demand_charge_feature['properties']['Average Ma'] is None:
#            cost_per_mi_feature['properties']['$_mi_tot'] = None
#            cost_per_mi_feature['properties']['$_mi_cap'] = None
#            cost_per_mi_feature['properties']['$_mi_el'] = None
#            cost_per_mi_feature['properties']['$_mi_lab'] = None
#            cost_per_mi_feature['properties']['$_mi_op'] = None
#        else:
#            costs_per_mile = get_costs_per_mile(electricity_rate_feature['properties']['Cents_kWh'], demand_charge_feature['properties']['Average Ma'])
#            cost_per_mi_feature['properties']['$_mi_tot'] = costs_per_mile['TCO ($/mi)']
#            cost_per_mi_feature['properties']['$_mi_cap'] = costs_per_mile['Total capital ($/mi)']
#            cost_per_mi_feature['properties']['$_mi_el'] = costs_per_mile['Total electricity ($/mi)']
#            cost_per_mi_feature['properties']['$_mi_lab'] = costs_per_mile['Total labor ($/mi)']
#            cost_per_mi_feature['properties']['$_mi_op'] = costs_per_mile['Other OPEXs ($/mi)']
##        print(electricity_rate_feature['properties']['STUSPS'])
##        print(cost_per_mi_feature['properties']['$_mi_tot'])
#
#with open('geojsons/costs_per_mile.geojson', mode='w') as cost_geojson:
#    json.dump(costs_per_mi_geojson, cost_geojson, indent=4)
#
## Plot cost/mile breakdown for a few sample states
#plot_costs_per_mile_breakdown(costs_per_mi_geojson)
##################################################################################
