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
# Annual VMT from VIUS 2002
VMT = np.array(pd.read_csv('data/default_vmt.csv')['VMT (miles)'])

# Default drivecycle used for emission and costing analysis
# Source: Jones, R et al. (2023).Developing and Benchmarking a US Long-haul Drive Cycle forVehicle Simulations, Costing and Emissions Analysis
# https://docs.google.com/spreadsheets/d/1Q2uO-JHfwvGxir_PU8IO5zmo0vs4ooC_/edit?usp=sharing&ouid=102742490305620802920&rtpof=true&sd=true
df_drivecycle = truck_model_tools.extract_drivecycle_data('data/drivecycle.xlsx') #drive cycle as a dataframe
df_drivecycle_flat = truck_model_tools.extract_drivecycle_data('data/drivecycle_nograde.xlsx') #drive cycle with zero road grade everywhere
#print(df_drivecycle.head())

# Payload distribution from VIUS 2002
# Note: Dataset from VIUS 2002, filtered and cleaned by authors for this analysis. Source: 2002 Economic Census: Vehicle Inventory and Use Survey
# https://docs.google.com/spreadsheets/d/1Oe_jBIUb-kJ5yy9vkwaPgldVe4cloAtG/edit?usp=sharing&ouid=102742490305620802920&rtpof=true&sd=true
df_payload_distribution = pd.read_excel('data/payloaddistribution.xlsx')
df_payload_distribution['Payload (kg)'] = df_payload_distribution['Payload (lb)']*KG_PER_LB #payload distribution in kgs

# Read in default truck model parameters
parameters = truck_model_tools.read_parameters('data/default_truck_params.csv', 'data/default_economy_params.csv', 'data/constants.csv', 'data/default_vmt.csv')

# Read in default battery parameters
df_battery_data = pd.read_csv('data/default_battery_params.csv', index_col=0)

# Read in present NMC battery parameters
df_scenarios = pd.read_csv('data/scenario_data.csv', index_col=0)
e_present_density_NMC = float(df_scenarios['NMC battery energy density'].iloc[0])
eta_battery_NMC = df_battery_data['Value'].loc['NMC roundtrip efficiency']

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


# Function to get truck model results over a range of payload sizes
def get_model_results_vs_payload(truck_name, driving_event, payload_min=0, payload_max=100000, n_payloads=10, e_density_battery = e_present_density_NMC, battery_roundtrip_efficiency = eta_battery_NMC):

    # Collect the drivecycle
    df_drivecycle = truck_model_tools.extract_drivecycle_data(f'data/{truck_name}_drive_cycle_{driving_event}.csv')

    vehicle_model_results = pd.DataFrame(columns = ['Average Payload (lb)', 'Battery capacity (kWh)', 'Fuel economy (kWh/mi)', 'Total vehicle mass (lbs)'])
    for m_ave_payload in np.linspace(payload_min, payload_max, n_payloads):
        parameters.m_ave_payload = m_ave_payload * KG_PER_LB
        m_bat, e_bat, mileage, m = truck_model_tools.truck_model(parameters).get_battery_size(df_drivecycle, battery_roundtrip_efficiency, e_density_battery)
        new_row = new_row = pd.DataFrame({
            'Average Payload (lb)': [m_ave_payload],
            'Battery capacity (kWh)': [e_bat],
            'Fuel economy (kWh/mi)': [mileage],
            'Total vehicle mass (lbs)': [m/KG_PER_LB]
        })
        vehicle_model_results = pd.concat([vehicle_model_results, new_row], ignore_index=True)
    return vehicle_model_results
    
# Function to evaluate the payload and GVW for which the fuel economy and battery capacity best match the values extrapolated from the NACFE data
def evaluate_matching_payloads(vehicle_model_results, payload_min=0, payload_max=100000):
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
def visualize_results(truck_name, driving_event, vehicle_model_results, NACFE_results, payload_e_bat, payload_mileage, payload_average, gvw_payload_average, cs_e_bat, cs_mileage, cs_m, combined_eff=None, max_power=None, battery_energy_density=None, battery_roundtrip_efficiency=None, resistance_coef=None):
    fig, axs = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 1, 1]})  # 3 rows, 1 column
    name_title = truck_name.replace('_', ' ').capitalize()
    if not combined_eff is None:
        axs[0].set_title(f'{name_title}: Payload Estimation for Driving Event {driving_event} (Combined Eff: {combined_eff:.2f})', fontsize=20)
    elif not max_power is None:
        axs[0].set_title(f'{name_title}: Payload Estimation for Driving Event {driving_event} (Max Power: {max_power:.0f})', fontsize=20)
    elif not battery_energy_density is None:
        axs[0].set_title(f'{name_title}: Payload Estimation for Driving Event {driving_event} (Battery Energy Density: {battery_energy_density:.0f} kWh/ton)', fontsize=20)
    elif not battery_roundtrip_efficiency is None:
        axs[0].set_title(f'{name_title}: Payload Estimation for Driving Event {driving_event} (Battery Roundtrip Efficiency: {battery_roundtrip_efficiency:.2f})', fontsize=20)
    elif not resistance_coef is None:
        axs[0].set_title(f'{name_title}: Payload Estimation for Driving Event {driving_event} (Rolling Resistance: {resistance_coef:.4f})', fontsize=20)
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
    elif not max_power is None:
        max_power_save = str(int(max_power))
        plt.savefig(f'plots/truck_model_results_vs_payload_{truck_name}_drivecycle_{driving_event}_maxpower_{max_power_save}.png')
    elif not battery_energy_density is None:
        battery_energy_density_save = str(int(battery_energy_density))
        plt.savefig(f'plots/truck_model_results_vs_payload_{truck_name}_drivecycle_{driving_event}_battery_density_{battery_energy_density_save}.png')
    elif not battery_roundtrip_efficiency is None:
        battery_roundtrip_efficiency_save = str(int(battery_roundtrip_efficiency))
        plt.savefig(f'plots/truck_model_results_vs_payload_{truck_name}_drivecycle_{driving_event}_battery_roundtrip_efficiency_{battery_roundtrip_efficiency_save}.png')
    elif not resistance_coef is None:
        resistance_coef_save = str(int(resistance_coef))
        plt.savefig(f'plots/truck_model_results_vs_payload_{truck_name}_drivecycle_{driving_event}_battery_resistance_coef_{resistance_coef}.png')
    else:
        plt.savefig(f'plots/truck_model_results_vs_payload_{truck_name}_drivecycle_{driving_event}.png')
    plt.close()


# Evaluate GVW for each truck and drivecycle event
evaluated_gvws = {}
for truck_name in drivecycles:
    evaluated_gvws[truck_name] = []
    drivecycle_events_list = drivecycles[truck_name]
    for driving_event in drivecycle_events_list:
        print(f'Processing {truck_name} event {driving_event}')
        
        # Read in the NACFE results
        NACFE_results = get_nacfe_results(truck_name, driving_event)
        
        # Update the depth of discharge for the driving event based on the NACFE data
        update_event_dod(parameters, truck_name, driving_event)
        
        # Get the vehicle model results (as a dataframe) as a function of payload
        vehicle_model_results = get_model_results_vs_payload(truck_name, driving_event)
        
        # Get the payloads and resulting GVW for which the truck model results best match the NACFE data. Also collect the cubic splines used for this evaluation (for the purpose of visualization)
        payload_e_bat, payload_mileage, payload_average, gvw_payload_average, cs_e_bat, cs_mileage, cs_m = evaluate_matching_payloads(vehicle_model_results)
        
        # Visualize the results
        visualize_results(truck_name, driving_event, vehicle_model_results, NACFE_results, payload_e_bat, payload_mileage, payload_average, gvw_payload_average, cs_e_bat, cs_mileage, cs_m)
        
        # Document the evaluated GVW
        evaluated_gvws[truck_name].append(gvw_payload_average)

# Save the evaluated GVWs as a pickle file
#with open('pickle/fitted_gvws.pkl', 'wb') as f:
#    pickle.dump(evaluated_gvws, f)

###########################################################################################################


######################### Analyze the distribution of GVWs evaluated by the model #########################
#with open('pickle/fitted_gvws.pkl', 'rb') as f:
#    evaluated_gvws = pickle.load(f)

all_evaluated_gvws = np.zeros(0)
data_boxplot = []
labels_boxplot = []
for truck_name in evaluated_gvws:
    evaluated_gvws_truck = np.array([float(i) for i in evaluated_gvws[truck_name]])
    evaluated_gvws[truck_name] = evaluated_gvws_truck
    all_evaluated_gvws = np.append(all_evaluated_gvws, evaluated_gvws_truck)
    
    data_boxplot.append(evaluated_gvws_truck)
    labels_boxplot.append(truck_name.replace('_', ' ').capitalize())

data_boxplot.append(all_evaluated_gvws)
labels_boxplot.append('Combined')

#print(all_evaluated_gvws)
#print(evaluated_gvws)

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_ylabel('GVW best matching NACFE Results (lbs)', fontsize=15)
ax.axhline(70000, color='red', ls='--')
ax.tick_params(axis='both', which='major', labelsize=14)
box = plt.boxplot(data_boxplot)
plt.xticks([1, 2, 3, 4], labels_boxplot)

for i in range(len(data_boxplot)):

    # Get the x position for the current box plot
    x_position = i+1
    
    # Get the y position for the text annotation. This can be slightly above the box plot.
    # You may need to adjust this depending on your specific data range and desired appearance.
    data_max = max(data_boxplot[i])
    data_min = min(data_boxplot[i])
    y_position = data_max + 0.2*(data_max-data_min)  # Just above the upper whisker
    
    # Place the text annotation
    n_drivecycles = len(data_boxplot[i])
    ax.text(x_position, y_position, f'{n_drivecycles} drivecycles', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig('plots/Evaluated_GVW_Distribution.png')
plt.close()
###########################################################################################################



###################### Plot the best-fitting GVW as a function of various parameters ######################

# Allow the max motor power to vary between 300,000W and 1,000,000W
truck_name = 'pepsi_1'
name_title = truck_name.replace('_', ' ').capitalize()
driving_event = 2
motor_powers = np.linspace(300000, 1000000, 10)


########## Evaluate best-fitting GVW vs. max motor power ##########
evaluated_gvws_df = pd.DataFrame(columns=['Max Motor Power (W)', 'Max GVW (lb)'])
for power in motor_powers:
    parameters.p_motor_max = power
    
    print(f'Processing {truck_name} event {driving_event} with motor power {power:.0f}W')
        
    # Read in the NACFE results
    NACFE_results = get_nacfe_results(truck_name, driving_event)
    
    # Update the depth of discharge for the driving event based on the NACFE data
    update_event_dod(parameters, truck_name, driving_event)
    
    # Get the vehicle model results (as a dataframe) as a function of payload
    vehicle_model_results = get_model_results_vs_payload(truck_name, driving_event)
    
    # Get the payloads and resulting GVW for which the truck model results best match the NACFE data. Also collect the cubic splines used for this evaluation (for the purpose of visualization)
    payload_e_bat, payload_mileage, payload_average, gvw_payload_average, cs_e_bat, cs_mileage, cs_m = evaluate_matching_payloads(vehicle_model_results)
    
    # Visualize the results
    visualize_results(truck_name, driving_event, vehicle_model_results, NACFE_results, payload_e_bat, payload_mileage, payload_average, gvw_payload_average, cs_e_bat, cs_mileage, cs_m, max_power=power)
    
    # Document the evaluated GVW
    evaluated_gvws_df = pd.concat([evaluated_gvws_df, pd.DataFrame({'Max Motor Power (W)': [power], 'Max GVW (lb)': [gvw_payload_average]})], ignore_index=True)
        
## Save the evaluated GVWs as a pickle file
#with open(f'pickle/fitted_gvws_{truck_name}_{driving_event}_vs_motor_power.pkl', 'wb') as f:
#    pickle.dump(evaluated_gvws_df, f)
###################################################################


############ Plot best-fitting GVW vs. max motor power ############
#with open(f'pickle/fitted_gvws_{truck_name}_{driving_event}_vs_motor_power.pkl', 'rb') as f:
#    evaluated_gvws_df = pickle.load(f)

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_title(f'{name_title} Event {driving_event}', fontsize=20)
ax.set_ylabel('GVW best matching NACFE Results (lbs)', fontsize=15)
ax.set_xlabel('Max Motor Power (W)', fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.plot(evaluated_gvws_df['Max Motor Power (W)'], evaluated_gvws_df['Max GVW (lb)'], 'o')
ax.axvline(942900, color='red', ls='--', label='Tesla Semi Motor Power')
ax.legend(fontsize=16)
plt.savefig('plots/matching_gvw_vs_max_motor_power.png')

###################################################################


############# Evaluate best-fitting GVW vs. efficiency ############
evaluated_gvws_df = pd.DataFrame(columns=['Max GVW (lb)', 'Combined efficiency'])
combined_effs = np.linspace(0.83, 1., 10)
parameters.p_motor_max = 942900
for combined_eff in combined_effs:
    parameters.eta_i = 1.
    parameters.eta_m = 1.
    parameters.eta_gs = combined_eff
    
    print(f'Processing {truck_name} event {driving_event} with combined efficiency {combined_eff:.2f}W')
        
    # Read in the NACFE results
    NACFE_results = get_nacfe_results(truck_name, driving_event)
    
    # Update the depth of discharge for the driving event based on the NACFE data
    update_event_dod(parameters, truck_name, driving_event)
    
    # Get the vehicle model results (as a dataframe) as a function of payload
    vehicle_model_results = get_model_results_vs_payload(truck_name, driving_event)
    
    # Get the payloads and resulting GVW for which the truck model results best match the NACFE data. Also collect the cubic splines used for this evaluation (for the purpose of visualization)
    payload_e_bat, payload_mileage, payload_average, gvw_payload_average, cs_e_bat, cs_mileage, cs_m = evaluate_matching_payloads(vehicle_model_results)
    
    print(f'Evaluated GVW: {gvw_payload_average}')
    
    # Visualize the results
    visualize_results(truck_name, driving_event, vehicle_model_results, NACFE_results, payload_e_bat, payload_mileage, payload_average, gvw_payload_average, cs_e_bat, cs_mileage, cs_m)
    
    # Document the evaluated GVW
    evaluated_gvws_df = pd.concat([evaluated_gvws_df, pd.DataFrame({'Combined efficiency': [combined_eff], 'Max GVW (lb)': [gvw_payload_average]})], ignore_index=True)
        
# Save the evaluated GVWs as a pickle file
#with open(f'pickle/fitted_gvws_{truck_name}_{driving_event}_vs_combined_eff.pkl', 'wb') as f:
#    pickle.dump(evaluated_gvws_df, f)
###################################################################


############ Plot best-fitting GVW vs. combined efficiency ############
#with open(f'pickle/fitted_gvws_{truck_name}_{driving_event}_vs_combined_eff.pkl', 'rb') as f:
#    evaluated_gvws_df = pickle.load(f)

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_title(f'{name_title} Event {driving_event}', fontsize=20)
ax.set_ylabel('GVW best matching NACFE Results (lbs)', fontsize=15)
ax.set_xlabel('Combined powertrain efficiency (%)', fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.plot(evaluated_gvws_df['Combined efficiency'], evaluated_gvws_df['Max GVW (lb)'], 'o')
plt.savefig('plots/matching_gvw_vs_combined_eff.png')

###################################################################
    
###########################################################################################################



############# Evaluate best-fitting GVW vs. energy density ############
truck_name = 'pepsi_1'
name_title = truck_name.replace('_', ' ').capitalize()
driving_event = 2
evaluated_gvws_df = pd.DataFrame(columns=['Max GVW (lb)', 'Battery Energy Density (kWh/ton)'])
battery_energy_densities = np.linspace(150, 500, 10)

for e_density_battery in battery_energy_densities:
    
    print(f'Processing {truck_name} event {driving_event} with energy denstiy {e_density_battery:.0f}kWh/ton')
        
    # Read in the NACFE results
    NACFE_results = get_nacfe_results(truck_name, driving_event)
    
    # Update the depth of discharge for the driving event based on the NACFE data
    update_event_dod(parameters, truck_name, driving_event)
    
    # Get the vehicle model results (as a dataframe) as a function of payload
    vehicle_model_results = get_model_results_vs_payload(truck_name, driving_event, e_density_battery = e_density_battery)
    
    # Get the payloads and resulting GVW for which the truck model results best match the NACFE data. Also collect the cubic splines used for this evaluation (for the purpose of visualization)
    payload_e_bat, payload_mileage, payload_average, gvw_payload_average, cs_e_bat, cs_mileage, cs_m = evaluate_matching_payloads(vehicle_model_results)
    
    print(f'Evaluated GVW: {gvw_payload_average}')
    
    # Visualize the results
    visualize_results(truck_name, driving_event, vehicle_model_results, NACFE_results, payload_e_bat, payload_mileage, payload_average, gvw_payload_average, cs_e_bat, cs_mileage, cs_m, battery_energy_density = e_density_battery)
    
    # Document the evaluated GVW
    evaluated_gvws_df = pd.concat([evaluated_gvws_df, pd.DataFrame({'Battery Energy Density (kWh/ton)': [e_density_battery], 'Max GVW (lb)': [gvw_payload_average]})], ignore_index=True)
        
# Save the evaluated GVWs as a pickle file
#with open(f'pickle/fitted_gvws_{truck_name}_{driving_event}_vs_battery_energy_density.pkl', 'wb') as f:
#    pickle.dump(evaluated_gvws_df, f)
###################################################################


############ Plot best-fitting GVW vs. combined efficiency ############
#with open(f'pickle/fitted_gvws_{truck_name}_{driving_event}_vs_battery_energy_density.pkl', 'rb') as f:
#    evaluated_gvws_df = pickle.load(f)

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_title(f'{name_title} Event {driving_event}', fontsize=20)
ax.set_ylabel('GVW best matching NACFE Results (lbs)', fontsize=15)
ax.set_xlabel('Battery Energy Density (kWh/ton)', fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.plot(evaluated_gvws_df['Battery Energy Density (kWh/ton)'], evaluated_gvws_df['Max GVW (lb)'], 'o')
plt.savefig('plots/matching_gvw_vs_battery_energy_density.png')

###################################################################
    
###########################################################################################################



############# Evaluate best-fitting GVW vs. battery roundtrip efficiency ############
truck_name = 'pepsi_1'
name_title = truck_name.replace('_', ' ').capitalize()
driving_event = 2
evaluated_gvws_df = pd.DataFrame(columns=['Max GVW (lb)', 'Battery Roundtrip Efficiency'])
battery_roundtrip_efficiencies = np.linspace(0.9, 1, 10)
parameters.m_max = 100000

for roundtrip_efficiency in battery_roundtrip_efficiencies:
    
    print(f'Processing {truck_name} event {driving_event} with roundtrip effiency {roundtrip_efficiency:.2f}')
        
    # Read in the NACFE results
    NACFE_results = get_nacfe_results(truck_name, driving_event)
    
    # Update the depth of discharge for the driving event based on the NACFE data
    update_event_dod(parameters, truck_name, driving_event)
    
    # Get the vehicle model results (as a dataframe) as a function of payload
    vehicle_model_results = get_model_results_vs_payload(truck_name, driving_event, battery_roundtrip_efficiency = roundtrip_efficiency)
    
    # Get the payloads and resulting GVW for which the truck model results best match the NACFE data. Also collect the cubic splines used for this evaluation (for the purpose of visualization)
    payload_e_bat, payload_mileage, payload_average, gvw_payload_average, cs_e_bat, cs_mileage, cs_m = evaluate_matching_payloads(vehicle_model_results)
    
    print(f'Evaluated GVW: {gvw_payload_average}')
    
    # Visualize the results
    visualize_results(truck_name, driving_event, vehicle_model_results, NACFE_results, payload_e_bat, payload_mileage, payload_average, gvw_payload_average, cs_e_bat, cs_mileage, cs_m, battery_roundtrip_efficiency = roundtrip_efficiency)
    
    # Document the evaluated GVW
    evaluated_gvws_df = pd.concat([evaluated_gvws_df, pd.DataFrame({'Battery Roundtrip Efficiency': [roundtrip_efficiency], 'Max GVW (lb)': [gvw_payload_average]})], ignore_index=True)
        
# Save the evaluated GVWs as a pickle file
#with open(f'pickle/fitted_gvws_{truck_name}_{driving_event}_vs_battery_roundtrip_efficiency.pkl', 'wb') as f:
#    pickle.dump(evaluated_gvws_df, f)
###################################################################


############ Plot best-fitting GVW vs. combined efficiency ############
#with open(f'pickle/fitted_gvws_{truck_name}_{driving_event}_vs_battery_roundtrip_efficiency.pkl', 'rb') as f:
#    evaluated_gvws_df = pickle.load(f)
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_title(f'{name_title} Event {driving_event}', fontsize=20)
ax.set_ylabel('GVW best matching NACFE Results (lbs)', fontsize=15)
ax.set_xlabel('Battery Roundtrip Efficiency', fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.plot(evaluated_gvws_df['Battery Roundtrip Efficiency'], evaluated_gvws_df['Max GVW (lb)'], 'o')
plt.savefig('plots/matching_gvw_vs_battery_roundtrip_efficiency.png')

###################################################################
    
###########################################################################################################


############# Evaluate best-fitting GVW vs. rolling resistance coefficient ############
truck_name = 'pepsi_1'
name_title = truck_name.replace('_', ' ').capitalize()
driving_event = 2
evaluated_gvws_df = pd.DataFrame(columns=['Max GVW (lb)', 'Resistance Coefficient'])
resistance_coefs = np.linspace(0.004, 0.008, 10)
parameters.m_max = 100000

for resistance_coef in resistance_coefs:
    
    print(f'Processing {truck_name} event {driving_event} with resistance coef {resistance_coef:.4f}')
    parameters.cr = resistance_coef
        
    # Read in the NACFE results
    NACFE_results = get_nacfe_results(truck_name, driving_event)
    
    # Update the depth of discharge for the driving event based on the NACFE data
    update_event_dod(parameters, truck_name, driving_event)
    
    # Get the vehicle model results (as a dataframe) as a function of payload
    
    vehicle_model_results = get_model_results_vs_payload(truck_name, driving_event)
    
    # Get the payloads and resulting GVW for which the truck model results best match the NACFE data. Also collect the cubic splines used for this evaluation (for the purpose of visualization)
    payload_e_bat, payload_mileage, payload_average, gvw_payload_average, cs_e_bat, cs_mileage, cs_m = evaluate_matching_payloads(vehicle_model_results)
    
    print(f'Evaluated GVW: {gvw_payload_average}')
    
    # Visualize the results
    visualize_results(truck_name, driving_event, vehicle_model_results, NACFE_results, payload_e_bat, payload_mileage, payload_average, gvw_payload_average, cs_e_bat, cs_mileage, cs_m, resistance_coef = resistance_coef)
    
    # Document the evaluated GVW
    evaluated_gvws_df = pd.concat([evaluated_gvws_df, pd.DataFrame({'Resistance Coefficient': [resistance_coef], 'Max GVW (lb)': [gvw_payload_average]})], ignore_index=True)
        
# Save the evaluated GVWs as a pickle file
#with open(f'pickle/fitted_gvws_{truck_name}_{driving_event}_vs_resistance_coef.pkl', 'wb') as f:
#    pickle.dump(evaluated_gvws_df, f)
###################################################################


############ Plot best-fitting GVW vs. combined efficiency ############
#with open(f'pickle/fitted_gvws_{truck_name}_{driving_event}_vs_resistance_coef.pkl', 'rb') as f:
#    evaluated_gvws_df = pickle.load(f)
fig, ax = plt.subplots(figsize=(9, 5))
ax.set_title(f'{name_title} Event {driving_event}', fontsize=20)
ax.set_ylabel('GVW best matching NACFE Results (lbs)', fontsize=15)
ax.set_xlabel('Rolling Resistance Coefficient', fontsize=15)
ax.axvline(0.0044, ls='--', color='red', label='Best value in literature')
ax.tick_params(axis='both', which='major', labelsize=14)
ax.plot(evaluated_gvws_df['Resistance Coefficient'], evaluated_gvws_df['Max GVW (lb)'], 'o')
ax.legend(fontsize=16)
plt.tight_layout()
plt.savefig('plots/matching_gvw_vs_resistance_coef.png')

###################################################################
    
###########################################################################################################

