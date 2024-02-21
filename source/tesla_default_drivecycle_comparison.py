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
import costing_tools
import emissions_tools
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
import pickle

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

# Read in present LFP battery parameters
df_scenarios = pd.read_csv('data/scenario_data.csv', index_col=0)
e_present_density_LFP = float(df_scenarios['NMC battery energy density'].iloc[0])
eta_battery_LFP = df_battery_data['Value'].loc['NMC roundtrip efficiency']
alpha = 1 #for payload penalty factor calculations (alpha = 1 for base case, alpha = 2: complete dependency in payload measurements)

# Read in GHG emissions parameters
GHG_bat_unit_LFP = df_battery_data['Value'].loc['NMC manufacturing emissions'] #g CO2/kWh
replacements_LFP = df_battery_data['Value'].loc['NMC replacements']

# Read in costing parameters for present day scenario

# Motor and inverter cost is given per unit of drivetrain power rating (Motor peak power)
# DC-DC converter cost is given per unit of auxiliary power rating  (Auxiliary loads)
# Insurance cost is per unit of capital cost of a single BET (no payload penalty included). We computed from reported insurance cost (0.1969 $/mi) for a BET vehicle cost (0.9933 $/mi). Source: https://publications.anl.gov/anlpubs/2021/05/167399.pdf
# Glider cost from Jones, R et al. (2023). Developing and Benchmarking a US Long-haul Drive Cycle forVehicle Simulations, Costing and Emissions Analysis
capital_cost_unit = pd.DataFrame({'glider ($)': [float(df_scenarios['Cost of glider'].iloc[0])], 'motor and inverter ($/kW)': [float(df_scenarios['Cost of motor and inverter'].iloc[0])], 'DC-DC converter ($/kW)': [float(df_scenarios['Cost of DC-DC converter'].iloc[0])]})

operating_cost_unit = pd.DataFrame({'maintenance & repair ($/mi)': [float(df_scenarios['Maintenance and repair cost'].iloc[0])], 'labor ($/mi)': [float(df_scenarios['Labor cost'].iloc[0])], 'insurance ($/mi)': [float(df_scenarios['Insurance cost'].iloc[0])], 'misc ($/mi)': [float(df_scenarios[' Miscellaneous costs'].iloc[0])]})

electricity_unit = [float(df_scenarios['Electricity price'].iloc[0])]

SCC = [float(df_scenarios['Social cost of carbon'].iloc[0])] #social cost of carbon in $/ton CO2. Source: https://www.whitehouse.gov/wp-content/uploads/2021/02/TechnicalSupportDocument_SocialCostofCarbonMethaneNitrousOxide.pdf

battery_unit_cost_LFP = [float(df_scenarios['NMC battery unit cost'].iloc[0])] #LFP unit cost in $/kWh
###########################################################################################################

"""
################################# Analyze the VIUS payload distribution ###################################
fig, ax = plt.subplots(figsize=(10, 6))
ax.tick_params(axis='both', which='major', labelsize=14)
n, bins, patches = ax.hist(df_payload_distribution['Payload (lb)'], bins=100)
bin_width = bins[1] - bins[0]
ax.set_xlabel('Payload (lb)', fontsize=16)
ax.set_ylabel(f'Events / {bin_width:.0f}lb', fontsize=16)
plt.show()
###########################################################################################################
"""


############################# Evaluate model parameters for Tesla drivecycles #############################
# Set the drag coefficient to the reported value for the Tesla semi
parameters.cd = 0.22

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
def get_model_results_vs_payload(truck_name, driving_event, payload_min=0, payload_max=30000, n_payloads=10):

    # Collect the drivecycle
    df_drivecycle = truck_model_tools.extract_drivecycle_data(f'data/{truck_name}_drive_cycle_{driving_event}.csv')

    vehicle_model_results = pd.DataFrame(columns = ['Average Payload (lb)', 'Battery capacity (kWh)', 'Fuel economy (kWh/mi)', 'Total vehicle mass (lbs)'])
    for m_ave_payload in np.linspace(payload_min, 30000, n_payloads):
        parameters.m_ave_payload = m_ave_payload
        m_bat, e_bat, mileage, m = truck_model_tools.truck_model(parameters).get_battery_size(df_drivecycle, eta_battery_LFP, e_present_density_LFP)
        new_row = new_row = pd.DataFrame({
            'Average Payload (lb)': [m_ave_payload],
            'Battery capacity (kWh)': [e_bat],
            'Fuel economy (kWh/mi)': [mileage],
            'Total vehicle mass (lbs)': [m/KG_PER_LB]
        })
        vehicle_model_results = pd.concat([vehicle_model_results, new_row], ignore_index=True)
    return vehicle_model_results
    
# Function to evaluate the payload and GVW for which the fuel economy and battery capacity best match the values extrapolated from the NACFE data
def evaluate_matching_payloads(vehicle_model_results, payload_min=0, payload_max=30000):
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
def visualize_results(truck_name, driving_event, vehicle_model_results, NACFE_results, payload_e_bat, payload_mileage, payload_average, gvw_payload_average, cs_e_bat, cs_mileage):
    fig, axs = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 1, 1]})  # 3 rows, 1 column
    name_title = truck_name.replace('_', ' ').capitalize()
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
    axs[2].axhline(gvw_payload_average, color='red', ls='--', label=f'GVW for matching payload: {gvw_payload_average:.0f}')
    axs[0].legend(fontsize=16)
    axs[1].legend(fontsize=16)
    axs[2].legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'plots/truck_model_results_vs_payload_{truck_name}_drivecycle_{driving_event}.png')

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
        visualize_results(truck_name, driving_event, vehicle_model_results, NACFE_results, payload_e_bat, payload_mileage, payload_average, gvw_payload_average, cs_e_bat, cs_mileage)
        
        # Document the evaluated GVW
        evaluated_gvws[truck_name].append(gvw_payload_average)
        
# Save the evaluated GVWs as a pickle file
with open('pickle/fitted_gvws.pkl', 'wb') as f:
    pickle.dump(evaluated_gvws, f)
###########################################################################################################


######################### Analyze the distribution of GVWs evaluated by the model #########################
with open('pickle/fitted_gvws.pkl', 'rb') as f:
    evaluated_gvws = pickle.load(f)

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
###########################################################################################################

