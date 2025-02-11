"""
Date: March 4, 2024
Purpose: Compare distributions of evaluated GVWs for different combined powertrain efficiencies
"""

# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

###################################### Select drivecycles to consider #####################################
drivecycles = {
    'pepsi_1': [2, 9, 13, 15, 33],
    'pepsi_2': [7, 10, 14, 22, 25, 31],
    'pepsi_3': [8, 10, 13, 16, 21, 24, 28, 32, 33]
}
###########################################################################################################

# Collect all GVWs in a dataframe
all_evaluated_gvws_df = pd.DataFrame(columns = ['Truck', 'Drivecycle', 'Combined efficiency', 'Fitted GVW (lb)'])
for truck_name in drivecycles:

    drivecycle_events_list = drivecycles[truck_name]
    for driving_event in drivecycle_events_list:
        fitted_gvws_df = pd.read_csv(f'tables/fitted_gvws_{truck_name}_{driving_event}_vs_combined_eff.csv')
        combined_effs_save = fitted_gvws_df['Combined efficiency']
        fitted_gvws_save = fitted_gvws_df['Fitted GVW (lb)']
        n_effs = len(combined_effs_save)
        trucks_save = [truck_name]*n_effs
        drivecycles_save = [driving_event]*n_effs
        
        all_evaluated_gvws_df = pd.concat([all_evaluated_gvws_df, pd.DataFrame({'Truck': trucks_save, 'Drivecycle': drivecycles_save, 'Combined efficiency': combined_effs_save, 'Fitted GVW (lb)': fitted_gvws_save})], ignore_index=True)

# Plot the GVW distribution for each efficiency
data_boxplot = []
data_top_65perc = []
labels_boxplot = []

combined_effs = np.unique(all_evaluated_gvws_df['Combined efficiency'])
for combined_eff in combined_effs:
    labels_boxplot.append(f'{combined_eff:.3f}')
    fitted_gvws_array = (all_evaluated_gvws_df['Fitted GVW (lb)'][all_evaluated_gvws_df['Combined efficiency'] == combined_eff]).astype(float).to_numpy()
    top_65_thres = np.percentile(fitted_gvws_array, 35)
    data_top_65perc.append(fitted_gvws_array[fitted_gvws_array >= top_65_thres])
    data_boxplot.append(fitted_gvws_array)
    
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_ylabel('GVW best matching NACFE Results (lbs)', fontsize=15)
ax.set_xlabel('Combined powertrain efficiency', fontsize=15)
ax.axhline(70000, color='red', ls='--')
n_effs = len(combined_effs)
ax.axvline(1 + (n_effs-1)*(.931-min(combined_effs))/(1-min(combined_effs)), color='green', ls='--', label='Max combined eff found in lit')

ax.tick_params(axis='both', which='major', labelsize=13)
box = plt.boxplot(data_boxplot)
plt.xticks(np.arange(n_effs)+1, labels_boxplot)

i=0
for combined_eff in combined_effs:
    top_65perc_min = np.min(data_top_65perc[i])
    top_65perc_max = np.max(data_top_65perc[i])
    if i==0:
        ax.plot([i+1, i+1], [top_65perc_min, top_65perc_max], color='blue', linewidth=22, alpha=0.3, label='Top 65% of GVWs')
    else:
        ax.plot([i+1, i+1], [top_65perc_min, top_65perc_max], color='blue', linewidth=22, alpha=0.3)
    i += 1

ax.legend(fontsize=15, edgecolor='white', loc='lower right', facecolor='white')
plt.tight_layout()
plt.savefig(f'plots/gvw_dist_vs_combined_eff.png')
