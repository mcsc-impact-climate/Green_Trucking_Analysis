"""
Date: Feb 19, 2024
Purpose: Perform a step by step validation of the truck model against the Tesla Semi NACFE data as suggested by Kariana Moreno Sader
"""

# Import packages
import pandas as pd
import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import truck_model_tools
import costing_tools
import emissions_tools
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
plt.rcParams.update(new_rc_params)

KG_PER_TON = 1000
KG_PER_LB = 0.453592
SECONDS_PER_HOUR = 3600

###################################### Select drivecycles to consider #####################################
drivecycles = {
    'pepsi_1': [2, 9, 13, 15, 33],
    'pepsi_2': [7, 10, 14, 22, 25, 31],
    'pepsi_3': [8, 10, 13, 16, 21, 24, 28, 32, 33]
}
###########################################################################################################

######################################### Obtain model parameters #########################################
# Read in default truck model parameters
parameters = truck_model_tools.read_parameters('data/default_truck_params.csv', 'data/default_economy_params.csv', 'data/constants.csv', 'data/default_vmt.csv')

# Read in default battery parameters
df_battery_data = pd.read_csv('data/default_battery_params.csv', index_col=0)

# Read in present NMC battery parameters
df_scenarios = pd.read_csv('data/scenario_data.csv', index_col=0)
e_present_density_NMC = float(df_scenarios['NMC battery energy density'].iloc[0])
eta_battery_NMC = df_battery_data['Value'].loc['NMC roundtrip efficiency']
alpha = 1 #for payload penalty factor calculations (alpha = 1 for base case, alpha = 2: complete dependency in payload measurements)

# Read in GHG emissions parameters
GHG_bat_unit_NMC = df_battery_data['Value'].loc['NMC manufacturing emissions'] #g CO2/kWh
replacements_NMC = df_battery_data['Value'].loc['NMC replacements']

# Read in costing parameters for present day scenario

# Motor and inverter cost is given per unit of drivetrain power rating (Motor peak power)
# DC-DC converter cost is given per unit of auxiliary power rating  (Auxiliary loads)
# Insurance cost is per unit of capital cost of a single BET (no payload penalty included). We computed from reported insurance cost (0.1969 $/mi) for a BET vehicle cost (0.9933 $/mi). Source: https://publications.anl.gov/anlpubs/2021/05/167399.pdf
# Glider cost from Jones, R et al. (2023). Developing and Benchmarking a US Long-haul Drive Cycle forVehicle Simulations, Costing and Emissions Analysis
capital_cost_unit = pd.DataFrame({'glider ($)': [float(df_scenarios['Cost of glider'].iloc[0])], 'motor and inverter ($/kW)': [float(df_scenarios['Cost of motor and inverter'].iloc[0])], 'DC-DC converter ($/kW)': [float(df_scenarios['Cost of DC-DC converter'].iloc[0])]})

operating_cost_unit = pd.DataFrame({'maintenance & repair ($/mi)': [float(df_scenarios['Maintenance and repair cost'].iloc[0])], 'labor ($/mi)': [float(df_scenarios['Labor cost'].iloc[0])], 'insurance ($/mi)': [float(df_scenarios['Insurance cost'].iloc[0])], 'misc ($/mi)': [float(df_scenarios[' Miscellaneous costs'].iloc[0])]})

electricity_unit = [float(df_scenarios['Electricity price'].iloc[0])]

SCC = [float(df_scenarios['Social cost of carbon'].iloc[0])] #social cost of carbon in $/ton CO2. Source: https://www.whitehouse.gov/wp-content/uploads/2021/02/TechnicalSupportDocument_SocialCostofCarbonMethaneNitrousOxide.pdf

battery_unit_cost_NMC = [float(df_scenarios['NMC battery unit cost'].iloc[0])] #NMC unit cost in $/kWh
###########################################################################################################

############################# Evaluate model parameters for Tesla drivecycles #############################
# Set the drag coefficient to the reported value for the Tesla semi
parameters.cd = 0.22   # Source: https://eightify.app/summary/technology-and-innovation/elon-musk-unveils-tesla-semi-impressive-aerodynamic-design-long-range-efficient-charging
parameters.a_cabin = 10.7  # Source: https://www.motormatchup.com/catalog/Tesla/Semi-Truck/2022/Empty
###########################################################################################################

# Function to get NACFE results for the given truck and driving event
def get_nacfe_results(truck_name, driving_event):
    # Collect the battery info extracted from the NACFE data for each truck
    battery_capacity_df = pd.read_csv('data/pepsi_semi_battery_capacities.csv')
    
    # Collect the info extracted from the drivecycle
    drivecycle_data_df = pd.read_csv(f'data/{truck_name}_drivecycle_data.csv', index_col='Driving event')
    
    # Collect NACFE results
    NACFE_results = {
        'Battery capacity (kWh)': battery_capacity_df[truck_name].iloc[0],
        'Battery capacity unc (kWh)': battery_capacity_df[truck_name].iloc[1],
        'Fuel economy (kWh/mi)': drivecycle_data_df['Fuel economy (kWh/mile)'].loc[driving_event],
        'Fuel economy unc (kWh/mi)': drivecycle_data_df['Fuel economy unc (kWh/mile)'].loc[driving_event],
    }
    
    return NACFE_results

# Function to get the relative depth of discharge evaluated for the given truck and driving event and update the parameters with this dod
def update_event_dod(parameters, truck_name, driving_event):

    # Collect the info extracted from the drivecycle
    drivecycle_data_df = pd.read_csv(f'data/{truck_name}_drivecycle_data.csv', index_col='Driving event')
    parameters.DoD = drivecycle_data_df['Depth of Discharge (%)'].loc[driving_event]/100.
    

def compare_cumulative_distance(truck_name, driving_event):
    # Collect the drivecycle (no cumulative distance)
    df_drivecycle = truck_model_tools.extract_drivecycle_data(f'data/{truck_name}_drive_cycle_{driving_event}.csv')
    
    # Collect the drivecycle (with cumulative distance)
    df_drivecycle_with_distance = pd.read_csv(f'data/{truck_name}_drive_cycle_{driving_event}_with_distance.csv')
    
    # Compare the cumulative distance between the two
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})  # 2 rows, 1 column
    name_title = truck_name.replace('_', ' ').capitalize()
    axs[0].tick_params(axis='both', which='major', labelsize=16)
    axs[1].tick_params(axis='both', which='major', labelsize=16)
    
    axs[0].set_title(f'{name_title} Event {driving_event}: Cumulative Distance Comparison with NACFE', fontsize=20)
    axs[0].set_ylabel('Cumulative Distance (m)', fontsize=20)
    axs[1].set_ylabel('Difference (%)', fontsize=20)
    axs[1].set_xlabel('Time (s)', fontsize=20)
    axs[1].axhline(0, ls='--')
    axs[0].plot(df_drivecycle['Time (s)'], df_drivecycle['Cumulative distance (m)'], label='From trapz integration')
    axs[0].plot(df_drivecycle_with_distance['Time (s)'], df_drivecycle_with_distance['Cumulative distance (m)'], label='From NACFE data')
    axs[1].plot(df_drivecycle['Time (s)'], 100*(df_drivecycle['Cumulative distance (m)'] - df_drivecycle_with_distance['Cumulative distance (m)']) / df_drivecycle_with_distance['Cumulative distance (m)'])
    axs[0].legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'plots/cumulative_distance_comparison_{truck_name}_{driving_event}.png')
    plt.close()
    
    total_cum_distance_model = df_drivecycle['Cumulative distance (m)'].iloc[-1]
    total_cum_distance_nacfe = df_drivecycle_with_distance['Cumulative distance (m)'].iloc[-1]
    
    perc_diff = 100*(total_cum_distance_model - total_cum_distance_nacfe) / total_cum_distance_nacfe
    
    return total_cum_distance_model, total_cum_distance_nacfe, perc_diff
    
def compare_all_cumulative_distances():
    evaluated_perc_diffs_df = pd.DataFrame(columns=['Truck Name', 'Driving event', 'Cumulative distance in model (m)', 'Cumulative distance in NACFE (m)' '% Diff in Cumulative Distances'])
    for truck_name in drivecycles:
        drivecycle_events_list = drivecycles[truck_name]
        for driving_event in drivecycle_events_list:
            total_cum_distance_model, total_cum_distance_nacfe, perc_diff = compare_cumulative_distance(truck_name, driving_event)
            
            new_row = pd.DataFrame({
                'Truck Name': [truck_name],
                'Driving event': [driving_event],
                'Cumulative distance in model (m)': [total_cum_distance_model],
                'Cumulative distance in NACFE (m)': [total_cum_distance_nacfe],
                '% Diff in Cumulative Distances': [perc_diff],
                })
            evaluated_perc_diffs_df = pd.concat([evaluated_perc_diffs_df, new_row], ignore_index=True)
    evaluated_perc_diffs_df.to_csv('tables/cumulative_distance_comparisons.csv')
    
def compare_drivecycle_speed(NACFE_results, truck_name, driving_event, m_total_lb):

    m_total = m_total_lb * KG_PER_LB        # Convert mass from lb to kg
    
    # Collect the drivecycle
    df_drivecycle = truck_model_tools.extract_drivecycle_data(f'data/{truck_name}_drive_cycle_{driving_event}.csv')
    
    df_drivecycle, simulated_vehicle_speed, power_request_motor = truck_model_tools.truck_model(parameters).get_simulated_vehicle_power(df_drivecycle, m_total)
    
#    # Compare the cumulative distance between the two
#    fig, axs = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})  # 2 rows, 1 column
#    name_title = truck_name.replace('_', ' ').capitalize()
#    axs[0].tick_params(axis='both', which='major', labelsize=16)
#    axs[1].tick_params(axis='both', which='major', labelsize=16)
#
#    axs[0].set_title(f'{name_title} Event {driving_event}: Vehicle Speed Comparison (Truck weight: {m_total_lb} lb)', fontsize=20)
#    axs[0].set_ylabel('Vehicle speed (m/s)', fontsize=20)
#    axs[1].set_ylabel('Difference (m/s)', fontsize=20)
#    axs[1].set_xlabel('Time (s)', fontsize=20)
#    axs[1].axhline(0, ls='--')
#    axs[0].plot(df_drivecycle['Time (s)'], df_drivecycle['Vehicle speed (m/s)'], label='From drivecycle')
#    axs[0].plot(df_drivecycle['Time (s)'], simulated_vehicle_speed, label='From model')
#    diff = simulated_vehicle_speed - df_drivecycle['Vehicle speed (m/s)']
#    axs[1].plot(df_drivecycle['Time (s)'], diff)
#    axs[0].legend(fontsize=16)
#    axs[0].set_ylim(-1,30)
#    plt.tight_layout()
#    plt.savefig(f'plots/vehicle_speed_comparison_{truck_name}_{driving_event}_mass_{m_total_lb}.png')
#    plt.close()
    
    total_absolute_diff = np.sum(np.absolute(df_drivecycle['Vehicle speed (m/s)'] - simulated_vehicle_speed))
    return total_absolute_diff
    

def compare_all_drivecycle_speeds(NACFE_results):
    m_totals_lb = 10000*np.arange(1,9)
    for truck_name in drivecycles:
        drivecycle_events_list = drivecycles[truck_name]
        for driving_event in drivecycle_events_list:
            total_abs_diffs_df = pd.DataFrame(columns=['Vehicle mass (lb)', 'Total absolute diff (m/s)'])
            for m_total_lb in m_totals_lb:
                total_absolute_diff = compare_drivecycle_speed(NACFE_results, truck_name, driving_event, m_total_lb)
                total_absolute_diff = compare_drivecycle_speed(NACFE_results, truck_name, driving_event, m_total_lb)
                new_row = pd.DataFrame({
                    'Vehicle mass (lb)': [m_total_lb],
                    'Total absolute diff (m/s)': [total_absolute_diff],
                    })
                total_abs_diffs_df = pd.concat([total_abs_diffs_df, new_row], ignore_index=True)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            name_title = truck_name.replace('_', ' ').capitalize()
            ax.set_title(f'{name_title}: Vehicle Speed Model Comparison', fontsize=20)
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.set_ylabel('Total absolute difference (m/s)', fontsize=15)
            ax.set_xlabel('Vehicle mass (lb)', fontsize=15)
            ax.plot(total_abs_diffs_df['Vehicle mass (lb)'], total_abs_diffs_df['Total absolute diff (m/s)'])
            plt.savefig(f'plots/vehicle_speed_comparison_vs_truck_mass_{truck_name}_{driving_event}.png')
            total_abs_diffs_df.to_csv(f'tables/vehicle_speed_comparison_vs_truck_mass_{truck_name}_{driving_event}.csv')
    
    
def main():
    truck_name = 'pepsi_1'
    driving_event = 2
    
    # Read in the NACFE results
    NACFE_results = get_nacfe_results(truck_name, driving_event)
    
    # Update the depth of discharge for the driving event based on the NACFE data
    update_event_dod(parameters, truck_name, driving_event)
    
    # Compare all cumulative distances
    #compare_all_cumulative_distances()
    
    # Compare speeds for the given parameters
    #m_total_lb = 80000      # Vehicle mass in lb
    #compare_drivecycle_speed(NACFE_results, truck_name, driving_event, m_total_lb)
    
    compare_all_drivecycle_speeds(NACFE_results)
    
if __name__ == '__main__':
    main()
    
