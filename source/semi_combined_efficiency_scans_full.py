"""
Date: Feb 14, 2024
Purpose: Evaluate truck model output parameters for different Tesla semi drivecycles and compare with parameters derived independently from the PepsiCo Tesla Semi NACFE data.
"""

# Import packages
import pandas as pd
import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import truck_model_tools
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
import pickle
import concurrent.futures
from datetime import datetime

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
e_present_density_NMC = df_battery_data['Value'].loc['NMC battery energy density']
eta_battery_NMC = df_battery_data['Value'].loc['NMC roundtrip efficiency']   # https://www.statista.com/statistics/1423012/efficiency-of-battery-energy-systems/
###########################################################################################################

############################# Evaluate model parameters for Tesla drivecycles #############################
# Set the drag coefficient to the reported value for the Tesla semi
parameters.cd = 0.36   # Source: https://www.notateslaapp.com/tesla-reference/963/everything-we-know-about-the-tesla-semi
parameters.a_cabin = 10.7  # Source: https://www.motormatchup.com/catalog/Tesla/Semi-Truck/2022/Empty
parameters.p_motor_max = 942900   # Source: https://www.motormatchup.com/catalog/Tesla/Semi-Truck/2022/Empty
parameters.cr = 0.0044
parameters.m_max = 120000

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

# Function to get truck model results over a range of payload sizes
def get_model_results_vs_payload(truck_name, driving_event, payload_min=0, payload_max=100000, n_payloads=10):

    # Collect the drivecycle
    df_drivecycle = truck_model_tools.extract_drivecycle_data(f'data/{truck_name}_drive_cycle_{driving_event}.csv')

    vehicle_model_results = pd.DataFrame(columns = ['Average Payload (lb)', 'Battery capacity (kWh)', 'Fuel economy (kWh/mi)', 'Total vehicle mass (lbs)'])
    for m_ave_payload in np.linspace(payload_min, payload_max, n_payloads):
        parameters.m_ave_payload = m_ave_payload * KG_PER_LB
        m_bat, e_bat, mileage, m = truck_model_tools.truck_model(parameters).get_battery_size(df_drivecycle, eta_battery_NMC, e_present_density_NMC)
        new_row = new_row = pd.DataFrame({
            'Average Payload (lb)': [m_ave_payload],
            'Battery capacity (kWh)': [e_bat],
            'Fuel economy (kWh/mi)': [mileage],
            'Total vehicle mass (lbs)': [m/KG_PER_LB]
        })
        vehicle_model_results = pd.concat([vehicle_model_results, new_row], ignore_index=True)
    return vehicle_model_results
    
# Function to evaluate the payload and GVW for which the fuel economy and battery capacity best match the values evaluated from the NACFE data
def evaluate_matching_payloads(vehicle_model_results, NACFE_results, payload_min=0, payload_max=100000):
    cs_e_bat = interp1d(vehicle_model_results['Average Payload (lb)'], vehicle_model_results['Battery capacity (kWh)'])
    cs_mileage = interp1d(vehicle_model_results['Average Payload (lb)'], vehicle_model_results['Fuel economy (kWh/mi)'])
    cs_m = interp1d(vehicle_model_results['Average Payload (lb)'], vehicle_model_results['Total vehicle mass (lbs)'])

    def root_func(x, cs, y_target):
        return cs(x) - y_target
    
    payload_e_bat = root_scalar(lambda x: root_func(x, cs_e_bat, NACFE_results['Battery capacity (kWh)']), bracket=[payload_min, payload_max]).root
    payload_mileage = root_scalar(lambda x: root_func(x, cs_mileage, NACFE_results['Fuel economy (kWh/mi)']), bracket=[payload_min, payload_max]).root
    payload_average = (payload_e_bat + payload_mileage) / 2.
    
    gvw_payload_average = cs_m(payload_average)
    
    return payload_e_bat, payload_mileage, payload_average, gvw_payload_average, cs_e_bat, cs_mileage, cs_m

# Function to visualize fit results
def visualize_results(truck_name, driving_event, vehicle_model_results, NACFE_results, payload_e_bat, payload_mileage, payload_average, gvw_payload_average, cs_e_bat, cs_mileage, cs_m, combined_eff=None):
    fig, axs = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 1, 1]})  # 3 rows, 1 column
    name_title = truck_name.replace('_', ' ').capitalize()
    if not combined_eff is None:
        axs[0].set_title(f'{name_title}: Payload Estimation for Driving Event {driving_event} (Combined Eff: {combined_eff:.2f})', fontsize=20)
    else:
        axs[0].set_title(f'{name_title}: Payload Estimation for Driving Event {driving_event}', fontsize=20)
    axs[0].tick_params(axis='both', which='major', labelsize=14)
    axs[1].tick_params(axis='both', which='major', labelsize=14)
    axs[2].tick_params(axis='both', which='major', labelsize=14)

    axs[0].set_ylabel('Battery capacity (kWh)', fontsize=16)
    axs[1].set_ylabel('Fuel economy (kWh/mile)', fontsize=16)
    axs[2].set_ylabel('GVW (lbs)', fontsize=16)
    axs[2].set_xlabel('Payload (lb)', fontsize=16)

    axs[0].plot(vehicle_model_results['Average Payload (lb)'], vehicle_model_results['Battery capacity (kWh)'], 'o')
    xmin = min(vehicle_model_results['Average Payload (lb)'])
    xmax = max(vehicle_model_results['Average Payload (lb)'])
    xs=np.linspace(xmin, xmax, 100)
    axs[0].plot(xs, cs_e_bat(xs), color='purple', label='cubic spline')
    axs[0].axvline(payload_e_bat, color='green', label='Payload to match NACFE battery capacity')
    axs[0].axhline(NACFE_results['Battery capacity (kWh)'], label='NACFE Analysis Result', color='red')
    axs[0].fill_between(np.linspace(xmin, xmax, 5), NACFE_results['Battery capacity (kWh)']-NACFE_results['Battery capacity unc (kWh)'], NACFE_results['Battery capacity (kWh)']+NACFE_results['Battery capacity unc (kWh)'], label='NACFE Analysis Result', color='red', alpha=0.5, edgecolor=None)
    xmin_plot, xmax_plot = axs[0].get_xlim()
    ymin, ymax = axs[0].get_ylim()
    axs[0].set_ylim(ymin - 0.5*(ymax-ymin), ymax)

    axs[1].plot(vehicle_model_results['Average Payload (lb)'], vehicle_model_results['Fuel economy (kWh/mi)'], 'o')
    xs=np.linspace(xmin, xmax, 100)
    axs[1].plot(xs, cs_mileage(xs), color='purple')
    axs[1].axvline(payload_mileage, color='green', label='Payload to match NACFE fuel economy')
    axs[1].axhline(NACFE_results['Fuel economy (kWh/mi)'], color='red')
    axs[1].fill_between(np.linspace(xmin, xmax, 5), NACFE_results['Fuel economy (kWh/mi)']-NACFE_results['Fuel economy unc (kWh/mi)'], NACFE_results['Fuel economy (kWh/mi)']+NACFE_results['Fuel economy unc (kWh/mi)'], color='red', alpha=0.5, edgecolor=None)
    axs[1].set_xlim(xmin_plot, xmax_plot)

    axs[2].plot(vehicle_model_results['Average Payload (lb)'], vehicle_model_results['Total vehicle mass (lbs)'], 'o')
    xs=np.linspace(xmin, xmax, 100)
    axs[2].plot(xs, cs_m(xs), color='purple')
    axs[2].axvline(payload_average, color='green', ls='--', label=f'Average payload to match NACFE: {payload_average:.0f}')
    
    axs[2].set_xlim(xmin_plot, xmax_plot)
    battery_weight_payload_average = gvw_payload_average - payload_average - parameters.m_truck_no_bat / KG_PER_LB
    tractor_weight_payload_average = gvw_payload_average - payload_average
    axs[2].axhline(gvw_payload_average, color='red', ls='--', label=f'GVW for matching payload: {gvw_payload_average:.0f} lb\nBattery weight: {battery_weight_payload_average:.0f} lb\nUnloaded weight: {tractor_weight_payload_average:.0f} lb')
    axs[0].legend(fontsize=16)
    axs[1].legend(fontsize=16)
    axs[2].legend(fontsize=16)
    plt.tight_layout()
    if not combined_eff is None:
        combined_eff_save = str(combined_eff).replace('.', '')
        plt.savefig(f'plots/truck_model_results_vs_payload_{truck_name}_drivecycle_{driving_event}_combinedeff_{combined_eff_save}.png')
    else:
        plt.savefig(f'plots/truck_model_results_vs_payload_{truck_name}_drivecycle_{driving_event}.png')
    plt.close()

####################### Plot the best-fitting GVW as a function of various parameters #####################
# Allow the max motor power to vary between 300,000W and 1,000,000W
truck_names = ['pepsi_1', 'pepsi_2', 'pepsi_3']
driving_events = {
    'pepsi_1': [2, 9, 13, 15, 33],
    'pepsi_2': [7, 10, 14, 22, 25, 31],
    'pepsi_3': [8, 10, 13, 16, 21, 24, 28, 32, 33]
}

def evaluate_matching_gvw(truck_name, driving_event, combined_eff):
    parameters.eta_i = 1.
    parameters.eta_m = 1.
    parameters.eta_gs = combined_eff
    
#    print(f'Processing {truck_name} event {driving_event} with combined efficiency {combined_eff:.2f}W')
        
    # Read in the NACFE results
    NACFE_results = get_nacfe_results(truck_name, driving_event)
    
    # Update the depth of discharge for the driving event based on the NACFE data
    update_event_dod(parameters, truck_name, driving_event)
    
    # Get the vehicle model results (as a dataframe) as a function of payload
    vehicle_model_results = get_model_results_vs_payload(truck_name, driving_event)
    
    # Get the payloads and resulting GVW for which the truck model results best match the NACFE data. Also collect the cubic splines used for this evaluation (for the purpose of visualization)
    payload_e_bat, payload_mileage, payload_average, gvw_payload_average, cs_e_bat, cs_mileage, cs_m = evaluate_matching_payloads(vehicle_model_results, NACFE_results)
    battery_weight_payload_average = gvw_payload_average - payload_average - parameters.m_truck_no_bat / KG_PER_LB
    tractor_weight_payload_average = gvw_payload_average - payload_average
    
#    print(f'Evaluated GVW: {gvw_payload_average}')
    
    # Visualize the results
    visualize_results(truck_name, driving_event, vehicle_model_results, NACFE_results, payload_e_bat, payload_mileage, payload_average, gvw_payload_average, cs_e_bat, cs_mileage, cs_m, combined_eff=combined_eff)
    
    # Document the evaluated GVW
    #evaluated_gvws_df = pd.concat([evaluated_gvws_df, pd.DataFrame({'Combined efficiency': [combined_eff], 'Max GVW (lb)': [gvw_payload_average]})], ignore_index=True)
    
    return combined_eff, gvw_payload_average, battery_weight_payload_average, tractor_weight_payload_average


def parallel_evaluate_matching_gvw(args):

    # Wrapper function to call evaluate_matching_gvw with a specific combined_eff.
    # This is necessary because ProcessPoolExecutor.map only works with functions that take a single argument.
    
    # Unpack arguments
    truck_name, driving_event, combined_eff = args
    
    return evaluate_matching_gvw(truck_name, driving_event, combined_eff)

def main():
    # Define parameters
    for truck_name in drivecycles:
        drivecycle_events_list = drivecycles[truck_name]
        for driving_event in drivecycle_events_list:
    
            startTime = datetime.now()
            print(f'Processing {truck_name} event {driving_event}')
            
            combined_effs = np.linspace(0.829, 1., 10)
            # Prepare a list of tuples where each tuple contains all arguments for a single call to parallel_evaluate_matching_gvw
            args_list = [(truck_name, driving_event, combined_eff) for combined_eff in combined_effs]

            # Use ProcessPoolExecutor to parallelize the evaluation
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = list(executor.map(parallel_evaluate_matching_gvw, args_list))

            # Assuming results are in the format expected, create a DataFrame
            evaluated_gvws_df = pd.DataFrame(results, columns=['Combined efficiency', 'Fitted GVW (lb)', 'Battery weight (lb)', 'Tractor weight (lb)'])

            # Save the evaluated GVWs as a csv file
            filename = f'tables/fitted_gvws_{truck_name}_{driving_event}_vs_combined_eff.csv'
            
            evaluated_gvws_df.to_csv(filename, index=False)
            
            run_time = datetime.now() - startTime
            run_time = run_time.total_seconds()
            print(f'Processing time for event: {run_time}s')
        
if __name__ == '__main__':
    main()

###################################################################
