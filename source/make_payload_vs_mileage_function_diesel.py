"""
Date: May 2, 2024
Purpose: Evaluate the diesel mpg as a function of payload, with uncertainty, using parameters derived from the Tesla Semi drivecycles.
"""

import data_collection_tools
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np
KG_PER_TON = 1000
KG_PER_LB = 0.453592
LB_PER_KILOTON = 2204622.6218488

new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
plt.rcParams.update(new_rc_params)

GVW_max = 80000   # Max vehicle weight for class 8 diesel trucks

def main():
    # Read in the linear coefficients of gal/mile evaluated for each drivecycle
    linear_coefficients_df = pd.read_csv('tables/linear_mpg_params_diesel_daycab.csv')
    
    # Evaluate the average slope and its standard deviation, and convert to gal/mile/lb
    slope_av = linear_coefficients_df['Linear Slope (gal/mile/kiloton)'].mean() / LB_PER_KILOTON
    slope_std = linear_coefficients_df['Linear Slope (gal/mile/kiloton)'].std() / LB_PER_KILOTON
    
    # Evaluate the average y intercept and its standard deviation
    y_intercept_av = linear_coefficients_df['Y Intercept (gal/mile)'].mean()
    y_intercept_std = linear_coefficients_df['Y Intercept (gal/mile)'].std()
    
    # Read in truck parameters
    parameters = data_collection_tools.read_parameters(truck_params='diesel_daycab', vmt_params = 'daycab_vmt_vius_2021', truck_type = 'diesel')
    
    # Maximum payload before maxing out GVW, in lb
    payload_GVW_max = GVW_max - parameters.m_truck / KG_PER_LB
        
    # Evaluate and plot the mpg as a function of payload
    payloads = np.linspace(0, payload_GVW_max, 1000)
    
    #best_fit_line = f"Best-fit Line \ny = mx + b \nm={slope*1000:.3f}$\pm${slope_unc*1000:.3f} Wh/lb$\cdot$mile\nb: {b:.1f}$\pm${b_unc:.1f} kWh/mile"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel('Payload (lb)', fontsize=20)
    ax.set_ylabel('Fuel Economy (miles/gallon)', fontsize=20)
    plt.plot(payloads, 1/(slope_av * payloads + y_intercept_av), color='blue', linewidth=2)
    ax.fill_between(payloads, 1 / ((slope_av+slope_std)*payloads + y_intercept_av + y_intercept_std), 1 / ((slope_av-slope_std)*payloads + y_intercept_av - y_intercept_std), color='blue', alpha=0.3)
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    ax.fill_betweenx(np.linspace(ymin, ymax, 10), payload_GVW_max, payload_GVW_max*1.1, color='red', alpha=0.3, label='GVW > 80,000 lb')
    ax.set_xlim(xmin, payload_GVW_max*1.1)
    ax.set_ylim(ymin, ymax)
    ax.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/payload_vs_mileage_function_diesel.png')
    
    linear_coef_results = {
        'slope (gal/mile/kiloton)': [slope_av],
        'slope unc (gal/mile/kiloton)': [slope_std],
        'b (gal/mile)': [y_intercept_av],
        'b unc (gal/mile)': [y_intercept_std]
    }
    
    coefs_save_df = pd.DataFrame(linear_coef_results)
    coefs_save_df.to_csv('tables/payload_vs_mileage_linear_coefs_diesel.csv')
    
if __name__ == '__main__':
    main()
