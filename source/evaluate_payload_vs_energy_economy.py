"""
Date: May 2, 2024
Purpose: Evaluate linear coefficients to calculate energy economy as a function of payload for each drivecycle
"""

import data_collection_tools
import truck_model_tools_diesel
import truck_model_tools
import pandas as pd
from common_tools import get_linear_drivecycles
from common_tools import make_title_string
from common_tools import get_nacfe_results

KG_PER_KILOTON = 1e6
KG_PER_TON = 1000
KG_PER_LB = 0.453592

###################################### Select drivecycles to consider #####################################
drivecycles = get_linear_drivecycles()
###########################################################################################################

def evaluate_linear_params_diesel():
    """
    Evaluates linear parameters of miles per gallon as a function of payload for diesel trucks
    
    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    parameters = data_collection_tools.read_parameters(truck_params='diesel_daycab', vmt_params = 'daycab_vmt_vius_2021', truck_type = 'diesel')

    evaluated_linear_params_df = pd.DataFrame(columns=['Linear Slope (gal/mile/kiloton)', 'Y Intercept (gal/mile)'])
    for truck_name in drivecycles:
        for drivecycle in drivecycles[truck_name]:
            df_drivecycle = truck_model_tools_diesel.extract_drivecycle_data(f'data/{truck_name}_drive_cycle_{drivecycle}.csv')
            slope_gpm, y_intercept_payload_gpm, miles_per_gallon_ave_payload = truck_model_tools_diesel.truck_model(parameters).get_linear_gpm_coefs(df_drivecycle)
            
            result = {
                'Truck': [make_title_string(truck_name)],
                'Linear Slope (gal/mile/kiloton)': [slope_gpm * KG_PER_KILOTON],
                'Y Intercept (gal/mile)': [y_intercept_payload_gpm]
            }
            
        
            # Assuming results are in the format expected, create a DataFrame
            evaluated_linear_params_df = pd.concat([evaluated_linear_params_df, pd.DataFrame(result)])

    # Save the evaluated linear parameters as a csv file
    filename = f'tables/linear_mpg_params_diesel_daycab.csv'

    evaluated_linear_params_df.to_csv(filename, index=False)
    
def evaluate_linear_params_ev():
    """
    Evaluates linear parameters of miles per gallon as a function of payload for diesel trucks
    
    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    parameters = data_collection_tools.read_parameters(truck_params='semi', vmt_params = 'daycab_vmt_vius_2021')
    parameters.m_max = 120000
    parameters.m_ave_payload = 10000 * KG_PER_LB
    battery_params_dict = data_collection_tools.read_battery_params()

    evaluated_linear_params_df = pd.DataFrame(columns=['Truck', 'Linear Slope (kWh/mi/kiloton)', 'Y Intercept (kWh/mi)', 'Predicted Payload (lb)', 'Predicted GVW (lb)'])
    battery_capacity_info = pd.read_csv('data/pepsi_semi_battery_capacities.csv', index_col='Value')
    for truck_name in drivecycles:
        for drivecycle in drivecycles[truck_name]:
            # Read in the NACFE results
            NACFE_results = get_nacfe_results(truck_name, drivecycle)
            
            df_drivecycle = truck_model_tools.extract_drivecycle_data(f'data/{truck_name}_drive_cycle_{drivecycle}.csv')
            slope_per_kg, y_intercept_payload, kwh_per_mile_ave_payload = truck_model_tools.truck_model(parameters).get_linear_energy_economy_coefs(df_drivecycle, battery_params_dict['Roundtrip efficiency'], battery_params_dict['Energy density (kWh/ton)'], battery_capacity_info[truck_name]['Mean'])
            
            slope_per_lb = slope_per_kg * KG_PER_LB
            slope_per_kiloton = slope_per_kg * KG_PER_KILOTON
            
            predicted_payload = ( NACFE_results['Fuel economy (kWh/mi)'] - y_intercept_payload ) / slope_per_lb
            m_bat = battery_capacity_info[truck_name]['Mean'] * KG_PER_TON / battery_params_dict['Energy density (kWh/ton)']
            predicted_gvw = predicted_payload + (parameters.m_truck_no_bat + m_bat) / KG_PER_LB
            
            result = {
                'Truck': [make_title_string(truck_name)],
                'Energy Economy (kWh/mi)': [NACFE_results['Fuel economy (kWh/mi)']],
                'Range (miles)': [NACFE_results['Range (miles)']],
                'Linear Slope (kWh/mi/kiloton)': [slope_per_kiloton],
                'Y Intercept (kWh/mi)': [y_intercept_payload],
                'Predicted Payload (lb)': [predicted_payload],
                'Predicted GVW (lb)': [predicted_gvw]
            }
            
            # Assuming results are in the format expected, create a DataFrame
            evaluated_linear_params_df = pd.concat([evaluated_linear_params_df, pd.DataFrame(result)])

    # Save the evaluated linear parameters as a csv file
    filename = f'tables/linear_energy_economy_params_semi.csv'

    evaluated_linear_params_df.to_csv(filename, index=False)
    

def main():
    evaluate_linear_params_diesel()
    evaluate_linear_params_ev()
    
if __name__ == '__main__':
    main()
