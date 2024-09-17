"""
Date: March 25, 2024
Purpose: General tools for payload matching
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

new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
plt.rcParams.update(new_rc_params)

KG_PER_TON = 1000
KG_PER_LB = 0.453592
SECONDS_PER_HOUR = 3600

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
def get_model_results_vs_payload(truck_name, driving_event, parameters, battery_params, payload_min=0, payload_max=100000, n_payloads=10):

    # Collect the drivecycle
    df_drivecycle = truck_model_tools.extract_drivecycle_data(f'data/{truck_name}_drive_cycle_{driving_event}.csv')

    vehicle_model_results = pd.DataFrame(columns = ['Average Payload (lb)', 'Battery capacity (kWh)', 'Fuel economy (kWh/mi)', 'Total vehicle mass (lbs)'])
    for m_ave_payload in np.linspace(payload_min, payload_max, n_payloads):
        parameters.m_ave_payload = m_ave_payload * KG_PER_LB
        m_bat, e_bat, mileage, m = truck_model_tools.truck_model(parameters).get_battery_size(df_drivecycle, battery_params['Roundtrip efficiency'], battery_params['Energy density (kWh/ton)'])
        new_row = new_row = pd.DataFrame({
            'Average Payload (lb)': [m_ave_payload],
            'Battery capacity (kWh)': [e_bat],
            'Fuel economy (kWh/mi)': [mileage],
            'Total vehicle mass (lbs)': [m/KG_PER_LB]
        })
        vehicle_model_results = pd.concat([vehicle_model_results, new_row], ignore_index=True)
    return vehicle_model_results
    
# Function to evaluate the payload and GVW for which the fuel economy and battery capacity best match the values evaluated from the NACFE data
def fit_payload(vehicle_model_results, NACFE_results, payload_min=0, payload_max=100000):
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
def visualize_results(truck_name, driving_event, vehicle_model_results, NACFE_results, payload_e_bat, payload_mileage, payload_average, gvw_payload_average, cs_e_bat, cs_mileage, cs_m):
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [1, 1, 1]})  # 3 rows, 1 column
    name_title = truck_name.replace('_', ' ').capitalize()
    #axs[0].set_title(f'{name_title}: Payload Estimation for Driving Event {driving_event}', fontsize=20)
    axs[0].tick_params(axis='both', which='major', labelsize=18)
    axs[1].tick_params(axis='both', which='major', labelsize=18)
    axs[2].tick_params(axis='both', which='major', labelsize=18)

    axs[0].set_ylabel('Battery capacity (kWh)', fontsize=20)
    axs[1].set_ylabel('Energy economy (kWh/mi)', fontsize=20)
    axs[2].set_ylabel('GVW (lbs)', fontsize=20)
    axs[2].set_xlabel('Payload (lb)', fontsize=20)

    axs[0].plot(vehicle_model_results['Average Payload (lb)'], vehicle_model_results['Battery capacity (kWh)'], 'o')
    xmin = min(vehicle_model_results['Average Payload (lb)'])
    xmax = max(vehicle_model_results['Average Payload (lb)'])
    xs=np.linspace(xmin, xmax, 100)
    axs[0].plot(xs, cs_e_bat(xs), color='purple', label='cubic spline')
    axs[0].axvline(payload_e_bat, color='green', label='Payload matching NACFE battery capacity')
    axs[0].axhline(NACFE_results['Battery capacity (kWh)'], label='NACFE Analysis Result', color='red')
    #axs[0].fill_between(np.linspace(xmin, xmax, 5), NACFE_results['Battery capacity (kWh)']-NACFE_results['Battery capacity unc (kWh)'], NACFE_results['Battery capacity (kWh)']+NACFE_results['Battery capacity unc (kWh)'], label='NACFE Analysis Result', color='red', alpha=0.5, edgecolor=None)
    xmin_plot, xmax_plot = axs[0].get_xlim()
    ymin, ymax = axs[0].get_ylim()
    axs[0].set_ylim(ymin - 0.5*(ymax-ymin), ymax)

    axs[1].plot(vehicle_model_results['Average Payload (lb)'], vehicle_model_results['Fuel economy (kWh/mi)'], 'o')
    xs=np.linspace(xmin, xmax, 100)
    axs[1].plot(xs, cs_mileage(xs), color='purple')
    axs[1].axvline(payload_mileage, color='green', label='Payload matching NACFE energy economy')
    axs[1].axhline(NACFE_results['Fuel economy (kWh/mi)'], color='red')
    axs[1].fill_between(np.linspace(xmin, xmax, 5), NACFE_results['Fuel economy (kWh/mi)']-NACFE_results['Fuel economy unc (kWh/mi)'], NACFE_results['Fuel economy (kWh/mi)']+NACFE_results['Fuel economy unc (kWh/mi)'], color='red', alpha=0.5, edgecolor=None)
    axs[1].set_xlim(xmin_plot, xmax_plot)

    axs[2].plot(vehicle_model_results['Average Payload (lb)'], vehicle_model_results['Total vehicle mass (lbs)'], 'o')
    xs=np.linspace(xmin, xmax, 100)
    axs[2].plot(xs, cs_m(xs), color='purple')
    axs[2].axvline(payload_average, color='green', ls='--', label=f'Average payload to match NACFE: {payload_average:.0f}')
    axs[2].set_xlim(xmin_plot, xmax_plot)
    axs[2].axhline(gvw_payload_average, color='red', ls='--', label=f'GVW for matching payload: {gvw_payload_average:.0f} lb')
    axs[0].legend(fontsize=20)
    axs[1].legend(fontsize=20)
    axs[2].legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(f'plots/truck_model_results_vs_payload_{truck_name}_drivecycle_{driving_event}.png')
    plt.close()
    
# Function to evaluate the payload that best reproduces the battery size and fuel economy for the given drivecycle
def evaluate_matching_payload(truck_name, driving_event, parameters, battery_params_dict):
        
    # Read in the NACFE results
    NACFE_results = get_nacfe_results(truck_name, driving_event)
    
    # Update the depth of discharge for the driving event based on the NACFE data
    update_event_dod(parameters, truck_name, driving_event)
    
    # Get the vehicle model results (as a dataframe) as a function of payload
    vehicle_model_results = get_model_results_vs_payload(truck_name, driving_event, parameters, battery_params_dict)
    
    # Get the payloads and resulting GVW for which the truck model results best match the NACFE data. Also collect the cubic splines used for this evaluation (for the purpose of visualization)
    payload_e_bat, payload_mileage, payload_fit, gvw_fit, cs_e_bat, cs_mileage, cs_m = fit_payload(vehicle_model_results, NACFE_results)
        
    # Visualize the results
    visualize_results(truck_name, driving_event, vehicle_model_results, NACFE_results, payload_e_bat, payload_mileage, payload_fit, gvw_fit, cs_e_bat, cs_mileage, cs_m)

    return payload_fit, gvw_fit, NACFE_results['Fuel economy (kWh/mi)']
    
#parameters = read_parameters(truck_params='semi')
#parameters.m_max = 120000
#battery_params_dict = read_battery_params()
#payload_fit, gvw_fit, mileage = evaluate_matching_payload('pepsi_1', 2, parameters, battery_params_dict)
#print(payload_fit, gvw_fit, mileage)
