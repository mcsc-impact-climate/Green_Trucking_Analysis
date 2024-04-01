
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
import costing_tools_orig as costing_tools
import emissions_tools_orig as emissions_tools

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
e_present_density_LFP = float(df_scenarios['LFP battery energy density'].iloc[0])
eta_battery_LFP = df_battery_data['Value'].loc['LFP roundtrip efficiency']
alpha = 1 #for payload penalty factor calculations (alpha = 1 for base case, alpha = 2: complete dependency in payload measurements)

# Read in GHG emissions parameters
GHG_bat_unit_LFP = df_battery_data['Value'].loc['LFP manufacturing emissions'] #g CO2/kWh
replacements_LFP = df_battery_data['Value'].loc['LFP replacements']

# Read in costing parameters for present day scenario

# Motor and inverter cost is given per unit of drivetrain power rating (Motor peak power)
# DC-DC converter cost is given per unit of auxiliary power rating  (Auxiliary loads)
# Insurance cost is per unit of capital cost of a single BET (no payload penalty included). We computed from reported insurance cost (0.1969 $/mi) for a BET vehicle cost (0.9933 $/mi). Source: https://publications.anl.gov/anlpubs/2021/05/167399.pdf
# Glider cost from Jones, R et al. (2023). Developing and Benchmarking a US Long-haul Drive Cycle forVehicle Simulations, Costing and Emissions Analysis
capital_cost_unit = pd.DataFrame({'glider ($)': [float(df_scenarios['Cost of glider'].iloc[0])], 'motor and inverter ($/kW)': [float(df_scenarios['Cost of motor and inverter'].iloc[0])], 'DC-DC converter ($/kW)': [float(df_scenarios['Cost of DC-DC converter'].iloc[0])]})

operating_cost_unit = pd.DataFrame({'maintenance & repair ($/mi)': [float(df_scenarios['Maintenance and repair cost'].iloc[0])], 'labor ($/mi)': [float(df_scenarios['Labor cost'].iloc[0])], 'insurance ($/mi)': [float(df_scenarios['Insurance cost'].iloc[0])], 'misc ($/mi)': [float(df_scenarios[' Miscellaneous costs'].iloc[0])]})

electricity_unit = [float(df_scenarios['Electricity price'].iloc[0])]

SCC = [float(df_scenarios['Social cost of carbon'].iloc[0])] #social cost of carbon in $/ton CO2. Source: https://www.whitehouse.gov/wp-content/uploads/2021/02/TechnicalSupportDocument_SocialCostofCarbonMethaneNitrousOxide.pdf

battery_unit_cost_LFP = [float(df_scenarios['LFP battery unit cost'].iloc[0])] #LFP unit cost in $/kWh
###########################################################################################################


###################################### Analyze road grade distribution ####################################

########### Road grade over time ###########
fig, axs = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [2, 1]})  # 2 rows, 1 column
axs[0].set_title('Long-haul Drivecycle', fontsize=18)
axs[0].set_ylabel('Speed (m/s)', fontsize=16)
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


###########################################################################################################

######################### Compare model results with vs. without road grade data ##########################

drivecycles = {
'With road grade': df_drivecycle,
'Without road grade': df_drivecycle_flat
}

results = {}
for drivecycle in drivecycles:
    results[drivecycle] = {}
    
    # Truck model results
    results[drivecycle]['Truck Model'] = {}
    
    vehicle_model_results_LFP = pd.DataFrame(columns = ['Energy battery (kWh)', 'Battery mass (lbs)', 'Fuel economy (kWh/mi)', 'Payload penalty factor', 'Total vehicle mass (lbs)'])
    m_bat, e_bat, mileage, m = truck_model_tools.truck_model(parameters).get_battery_size(drivecycles[drivecycle], eta_battery_LFP, e_present_density_LFP)
    payload_penalty_factor=truck_model_tools.payload(parameters).get_penalty(df_payload_distribution, m_bat, alpha)
    vehicle_model_results_LFP.loc[len(vehicle_model_results_LFP)] = [e_bat, m_bat/KG_PER_LB, mileage, payload_penalty_factor, m/KG_PER_LB]
    
    for model_component in vehicle_model_results_LFP:
        results[drivecycle]['Truck Model'][model_component] = vehicle_model_results_LFP[model_component].iloc[0]

    results[drivecycle]['Battery mass (lbs)'] = m_bat/KG_PER_LB
    results[drivecycle]['Battery capacity (kWh)'] = e_bat
    results[drivecycle]['Fuel economy (kWh/mi)'] = mileage
    results[drivecycle]['Total vehicle mass (lbs)'] = m
    results[drivecycle]['Payload penalty factor'] = payload_penalty_factor
    
    # Emissions model results
    results[drivecycle]['Emissions Model'] = {}
    
    GHG_emissions_LFP = emissions_tools.emission(parameters).get_WTW(vehicle_model_results_LFP, GHG_bat_unit_LFP,  replacements_LFP)
    
    for emissions_component in GHG_emissions_LFP:
        emissions_component_short = emissions_component.replace('GHGs ', '').split('(')[0].capitalize()
        results[drivecycle]['Emissions Model'][emissions_component_short] = GHG_emissions_LFP[emissions_component].iloc[0]

    # Costing model results
    results[drivecycle]['Costing Model'] = {}
    TCS_LFP = costing_tools.cost(parameters).get_TCS(vehicle_model_results_LFP, capital_cost_unit, battery_unit_cost_LFP, operating_cost_unit, electricity_unit, replacements_LFP, GHG_emissions_LFP, SCC)
    
    for cost_component in TCS_LFP:
        cost_component_short = cost_component.split('(')[0].replace('Total ', '').replace('GHGs ', '').replace(' ', '\n')
        cost_component_short = cost_component_short[0].upper() + cost_component_short[1:]
        results[drivecycle]['Costing Model'][cost_component_short] = TCS_LFP[cost_component].iloc[0]
    
# Evaluate percent differences between results with vs. without road grade
results['Percent Difference'] = {}
for category in ['Truck Model', 'Emissions Model', 'Costing Model']:
    results['Percent Difference'][category] = {}
    for result in results['With road grade'][category]:
        results['Percent Difference'][category][result] = 100*(results['Without road grade'][category][result] - results['With road grade'][category][result]) / results['With road grade'][category][result]

# Make plots for the emissions and costing categories to visualize the differences
for category in ['Emissions Model', 'Costing Model']:
    if category == 'Emissions Model':
        fig, axs = plt.subplots(2, 1, figsize=(7, 7), gridspec_kw={'height_ratios': [2, 1]})  # 2 rows, 1 column
    else:
        fig, axs = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [2, 1]})  # 2 rows, 1 column
    title_label = category.split(' ')[0]
    axs[0].set_title(f'Impact of Road Grade on Present Day {title_label}', fontsize=18)
    if category == 'Costing Model':
        axs[0].set_ylabel('Cost ($/mile)', fontsize=16)
    else:
        axs[0].set_ylabel('Emissions (gCO2e/mile)', fontsize=16)
    axs[1].set_ylabel('% Change Without Grade', fontsize=16)
    axs[0].tick_params(axis='both', which='major', labelsize=14)
    axs[1].tick_params(axis='both', which='major', labelsize=14)

    # Add major/minor ticks and gridlines
    axs[1].xaxis.set_major_locator(MultipleLocator(1))
    #axs[1].xaxis.set_minor_locator(MultipleLocator(0.25))
    axs[1].grid(which='major', linestyle='-', linewidth=0.5, color='black', axis='y')
    
    indices = np.arange(len(results['With road grade'][category].keys()))
    axs[0].bar(indices+0.875, results['With road grade'][category].values(), width=0.25, label = 'With road grade')
    axs[0].bar(indices+1.125, results['Without road grade'][category].values(), width=0.25, label = 'Without road grade')
    xmin, xmax = axs[0].get_xlim()
    axs[1].set_xlim(xmin, xmax)
    axs[0].set_xticks([])
    axs[1].set_xticks(indices + 1)
    axs[1].set_xticklabels(results['With road grade'][category].keys())
    axs[1].errorbar(indices+1, results['Percent Difference'][category].values(), xerr=0.25, fmt='o', color='green', linewidth=2)
    save_label = category.split(' ')[0].lower()
    axs[0].legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'plots/results_comparison_{save_label}.png')
    plt.close()
    
###########################################################################################################
