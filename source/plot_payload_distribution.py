"""
Date: 240917
Author: danikam
Purpose: Plot the distribution of payloads evaluated by the truck simulation model for the NACFE data from the PepsiCo Tesla Semi pilot.
"""

import pandas as pd
from common_tools import make_title_string
import matplotlib.pyplot as plt

def plot_payload_distribution(payload_variable = "Predicted Payload (lb)", powertrain="ev"):
    """
    Produces box plots of the payload or GVW evaluated by the truck simulation model.
    
    Parameters
    ----------
    payload_variable (string): Variable to plot (currently must be one of 'Predicted Payload (lb)' or 'Predicted GVW (lb)', 'Linear Slope (kWh/mi/kiloton)', or 'Y Intercept (kWh/mi)'.

    Returns
    -------
    top_dir (string): Path to the top level of the git repo
    """

    allowed_variables_ev = ["Predicted Payload (lb)", "Predicted GVW (lb)", "Linear Slope (kWh/mi/kiloton)", "Y Intercept (kWh/mi)"]
    allowed_variables_diesel = ["Linear Slope (gal/mile/kiloton)", "Y Intercept (gal/mile)"]

    if powertrain == "ev":
        if payload_variable not in allowed_variables_ev:
            raise Exception(f"payload_variable '{payload_variable}' cannot be passed to function plot_payload_distribution. Acceptable values are {', '.join(allowed_variables_ev)}")
    elif powertrain == "diesel":
        if payload_variable not in allowed_variables_diesel:
            raise Exception(f"payload_variable '{payload_variable}' cannot be passed to function plot_payload_distribution. Acceptable values are {', '.join(allowed_variables_diesel)}")
        
    payload_variable_save = ""
    if payload_variable == "Predicted Payload (lb)":
        payload_variable_save = "payload"
    elif payload_variable == "Predicted GVW (lb)":
        payload_variable_save = "gvw"
    elif payload_variable == "Linear Slope (kWh/mi/kiloton)":
        payload_variable_save = "slope"
    elif payload_variable == "Y Intercept (kWh/mi)":
        payload_variable_save = "intercept"
    elif payload_variable == "Linear Slope (gal/mile/kiloton)":
        payload_variable_save = "slope"
    elif payload_variable == "Y Intercept (gal/mile)":
        payload_variable_save = "intercept"
    
    if powertrain == "ev":
        payload_fit_results = pd.read_csv("tables/linear_energy_economy_params_semi.csv")
    elif powertrain == "diesel":
        payload_fit_results = pd.read_csv("tables/linear_mpg_params_diesel_daycab.csv")

    all_evaluated_gvws = payload_fit_results[payload_variable]
    truck_names = sorted(payload_fit_results["Truck"].unique())
    
    data_boxplot = []
    labels_boxplot = []
    for truck_name in truck_names:
        evaluated_gvws_truck = payload_fit_results[payload_variable][payload_fit_results["Truck"] ==  truck_name]
        
        data_boxplot.append(evaluated_gvws_truck)
        labels_boxplot.append(make_title_string(truck_name))

    data_boxplot.append(all_evaluated_gvws)
    labels_boxplot.append('Combined')

    #print(all_evaluated_gvws)
    #print(evaluated_gvws)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_ylabel(payload_variable, fontsize=18)
    ax.set_xlabel("Truck", fontsize=18)
    #ax.axhline(70000, color='red', ls='--')
    ax.tick_params(axis='both', which='major', labelsize=17)
    box = plt.boxplot(data_boxplot)
    plt.xticks([1, 2, 3, 4], labels_boxplot)
    
    # Add some space above the box plots to include labels
    ymin, ymax = ax.get_ylim()
    ymax = ymax*1.05
    ax.set_ylim(ymin, ymax)

    for i in range(len(data_boxplot)):

        # Get the x position for the current box plot
        x_position = i+1
        
        # Get the y-position to plot
        y_position = ymax*0.96  # Just above the upper whisker
        
        # Place the text annotation
        n_drivecycles = len(data_boxplot[i])
        ax.text(x_position, y_position, f'{n_drivecycles} drivecycles', ha='center', va='bottom', fontsize=14)

    plt.tight_layout()
    plt.savefig(f'plots/{payload_variable_save}_distribution_{powertrain}.png', dpi=300)
    plt.savefig(f'plots/{payload_variable_save}_distribution_{powertrain}.pdf')
    plt.close()

def main():
    plot_payload_distribution("Predicted Payload (lb)", "ev")
    plot_payload_distribution("Predicted GVW (lb)", "ev")
    plot_payload_distribution("Linear Slope (kWh/mi/kiloton)", "ev")
    plot_payload_distribution("Y Intercept (kWh/mi)", "ev")
    plot_payload_distribution("Linear Slope (gal/mile/kiloton)", "diesel")
    plot_payload_distribution("Y Intercept (gal/mile)", "diesel")

if __name__ == '__main__':
    main()
