
"""
Date: Feb 13, 2024
Purpose: Compare outputs for the truck model with vs. without road grades
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

new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
plt.rcParams.update(new_rc_params)

KG_PER_TON = 1000
KG_PER_LB = 0.453592
SECONDS_PER_HOUR = 3600

######################################### Obtain model parameters #########################################
# Annual VMT from VIUS 2002
VMT = np.array(pd.read_csv('data/default_vmt.csv')['VMT (miles)'])

# Default drivecycle used for emission and costing analysis
df_drivecycle = truck_model_tools.extract_drivecycle_data('data/drivecycle.xlsx') #drive cycle as a dataframe
print(df_drivecycle.head())

# Payload distribution from VIUS 2002
df_payload_distribution = pd.read_excel('data/payloaddistribution.xlsx')
df_payload_distribution['Payload (kg)'] = df_payload_distribution['Payload (lb)']*KG_PER_LB #payload distribution in kgs


# Read in default truck model parameters
parameters = truck_model_tools.read_parameters('data/default_truck_params.csv', 'data/default_economy_params.csv', 'data/constants.csv', 'data/default_vmt.csv')

# Read in default battery parameters
df_battery_data = pd.read_csv('data/default_battery_params.csv', index_col=0)

# Read in present LFP battery parameters
df_scenarios = pd.read_csv('data/scenario_data.csv', index_col=0)
e_present_density_LFP = float(df_scenarios['LFP battery energy density'].iloc[0])
eta_battery_LFP = df_battery_data['Value'].loc['LFP roundtrip efficiency']
alpha = 1 #for payload penalty factor calculations (alpha = 1 for base case, alpha = 2: complete dependency in payload measurements)

# Read in GHG emissions data
GHG_bat_unit_LFP = df_battery_data['Value'].loc['LFP manufacturing emissions'] #g CO2/kWh
replacements_LFP = df_battery_data['Value'].loc['LFP replacements']
###########################################################################################################

"""
###################################### Analyze road grade distribution ####################################

########### Road grade over time ###########
fig, axs = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [2, 1]})  # 2 rows, 1 column
axs[0].set_title('Long-haul Drivecycle', fontsize=18)
axs[0].set_ylabel('Speed (miles/hour)', fontsize=16)
axs[1].set_ylabel('Road grade (%)', fontsize=16)
axs[1].set_xlabel('Drive time (h)', fontsize=16)
axs[0].tick_params(axis='both', which='major', labelsize=14)
axs[1].tick_params(axis='both', which='major', labelsize=14)

# Add major/minor ticks and gridlines
axs[0].xaxis.set_major_locator(MultipleLocator(2))
axs[0].xaxis.set_minor_locator(MultipleLocator(0.5))
axs[0].grid(which='minor', linestyle='--', linewidth=0.5, color='gray')
axs[0].grid(which='major', linestyle='-', linewidth=0.5, color='black')
axs[1].xaxis.set_major_locator(MultipleLocator(2))
axs[1].xaxis.set_minor_locator(MultipleLocator(0.5))
axs[1].grid(which='minor', linestyle='--', linewidth=0.5, color='gray')
axs[1].grid(which='major', linestyle='-', linewidth=0.5, color='black')

axs[0].plot(df_drivecycle['Time (s)']/SECONDS_PER_HOUR, df_drivecycle['Vehicle speed (m/s)'], color='blue')
axs[1].plot(df_drivecycle['Time (s)']/SECONDS_PER_HOUR, df_drivecycle['Road grade (%)'], color='green')
plt.savefig('plots/long_haul_drivecycle.png')
plt.close()
############################################

########### Road grade over time ###########
fig, ax = plt.subplots(figsize=(10, 6))
ax.tick_params(axis='both', which='major', labelsize=14)
n, bins, patches = ax.hist(df_drivecycle['Road grade (%)'], bins=100)
bin_width = bins[1] - bins[0]
ax.set_xlabel('Road grade (%)', fontsize=16)
ax.set_ylabel(f'Events / {bin_width:.2f}%', fontsize=16)

avg_grade = np.mean(df_drivecycle['Road grade (%)'])
std_grade = np.std(df_drivecycle['Road grade (%)'])
max_grade = np.max(df_drivecycle['Road grade (%)'])
min_grade = np.min(df_drivecycle['Road grade (%)'])

plt.text(0.6, 0.7, f'Average Grade: {avg_grade:.1f}%\nStandard deviation: {std_grade:.1f}%\n(Min, Max): ({min_grade:.1f}, {max_grade:.1f})%', transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.7), fontsize=16)
plt.savefig('plots/long_haul_roadgrade_distribution.png')
############################################
"""

###########################################################################################################

######################### Compare model results with vs. without road grade data ##########################

df_drivecycle_flat = df_drivecycle.copy()
df_drivecycle_flat['Road grade (%)'] = np.absolute(df_drivecycle_flat['Road grade (%)'])*0.

drivecycles = {
'With road grade': df_drivecycle,
'No road grade': df_drivecycle_flat
}

results = {}
for drivecycle in drivecycles:
    results[drivecycle] = {}
    
    # Truck model results
    vehicle_model_results_LFP = pd.DataFrame(columns = ['Energy battery (kWh)', 'Battery mass (lbs)', 'Fuel economy (kWh/mi)', 'Payload penalty factor', 'Total vehicle mass (lbs)'])
    m_bat, e_bat, mileage, m = truck_model_tools.truck_model(parameters).get_battery_size(drivecycles[drivecycle], eta_battery_LFP, e_present_density_LFP)
    payload_penalty_factor=truck_model_tools.payload(parameters).get_penalty(df_payload_distribution, m_bat, alpha)
    vehicle_model_results_LFP.loc[len(vehicle_model_results_LFP)] = [e_bat, m_bat/KG_PER_LB, mileage, payload_penalty_factor, m/KG_PER_LB]

    results[drivecycle]['Battery mass (lbs)'] = m_bat/KG_PER_LB
    results[drivecycle]['Battery capacity (kWh)'] = m_bat
    results[drivecycle]['Fuel economy (kWh/mi)'] = mileage
    results[drivecycle]['Total vehicle mass (lbs)'] = m
    results[drivecycle]['Payload penalty factor'] = payload_penalty_factor
    
    # Emissions model results
    GHG_emissions_LFP = emissions_tools.emission(parameters).get_WTW(vehicle_model_results_LFP, GHG_bat_unit_LFP,  replacements_LFP)
    print(GHG_emissions_LFP)
    results[drivecycle]['GHGs manufacturing (gCO2/mi)'] = GHG_emissions_LFP['GHGs manufacturing (gCO2/mi)']
    results[drivecycle]['GHGs grid (gCO2/mi)'] = GHG_emissions_LFP['GHGs grid (gCO2/mi)']
    
    # Costing model results
    
    
###########################################################################################################
