"""
Date: March 26, 2024
Purpose: Using the evaluated payload and energy economy for each drivecycle, maek a function to describe the relationship between these two quantities
"""

import data_collection_tools
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np
KG_PER_TON = 1000
KG_PER_LB = 0.453592

new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
plt.rcParams.update(new_rc_params)

GVW_max = 82000   # Max vehicle weight for EV class 8 trucks in California
battery_capacity = 825  # Evaluated battery capacity for Tesla Semi, in kWh

def main():
    # Read in the evaluated payload vs. mileage for the Tesla Semi
    payload_vs_mileage_df = pd.read_csv('tables/payload_vs_mileage_semi.csv')
    
    
    # Read in battery parameters
    battery_params_dict = data_collection_tools.read_battery_params()
    
    # Read in truck parameters
    parameters = data_collection_tools.read_parameters(truck_params='semi')

    battery_weight = battery_capacity / battery_params_dict['Energy density (kWh/ton)'] * KG_PER_TON / KG_PER_LB   # Battery weight, in lb
    
    payload_GVW_max = GVW_max - battery_weight - parameters.m_truck_no_bat / KG_PER_LB    # Maximum payload before maxing out GVW, in lb

    # Plot the data
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel('Best-fitting Payload (lb)', fontsize=20)
    ax.set_ylabel('Fuel Economy (kWh/mile)', fontsize=20)
    ax.scatter(payload_vs_mileage_df['Payload (lb)'], payload_vs_mileage_df['Mileage (kWh/mi)'], color='black')
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    
    ###### Linear Fit ######
    x_values = payload_vs_mileage_df['Payload (lb)']
    y_values = payload_vs_mileage_df['Mileage (kWh/mi)']
    
    coefficients, covariance = np.polyfit(x_values, y_values, 1, cov=True)
    
    # Get the linear coefficients and uncertainty
    slope = coefficients[0]
    b = coefficients[1]
    slope_unc = np.sqrt(covariance[0, 0])
    b_unc = np.sqrt(covariance[1, 1])
    
    best_fit_line = f"Best-fit Line \ny = mx + b \nm={slope*1000:.3f}$\pm${slope_unc*1000:.3f} Wh/lb$\cdot$mile\nb: {b:.1f}$\pm${b_unc:.1f} kWh/mile"
    
    x_plot = np.linspace(0, xmax, 1000)
    plt.plot(x_plot, slope * x_plot + b, color='blue', label=best_fit_line, linewidth=2)
    ax.fill_between(x_plot, (slope+slope_unc)*x_plot + b + b_unc, (slope-slope_unc)*x_plot + b - b_unc, color='blue', alpha=0.3)
        
    ax.set_xlim(0, xmax)
    ax.set_ylim(b-b_unc-0.01, ymax)
    ax.fill_betweenx(np.linspace(b-b_unc-0.01, ymax, 10), payload_GVW_max, xmax, color='red', alpha=0.3, label='GVW > 82,000 lb')
    
    ax.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/payload_vs_mileage_function.png')
    
    # Save the coefficients to a csv file
    best_fit_line_results = {
        'slope (kWh/lb-mi)': [slope],
        'slope unc (kWh/lb-mi)': [slope_unc],
        'b (kWh/mi)': [b],
        'b unc (kWh/mi)': [b_unc]
    }
    coefs_save_df = pd.DataFrame(best_fit_line_results)
    coefs_save_df.to_csv('tables/payload_vs_mileage_best_fit_params.csv')

if __name__ == '__main__':
    main()
