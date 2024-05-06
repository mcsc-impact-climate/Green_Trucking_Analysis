"""
Date: May 2, 2024
Purpose: Evaluate linear coefficients to calculate mpg as a function of payload for each drivecycle
"""

import data_collection_tools
import truck_model_tools_diesel
import pandas as pd

KG_PER_KILOTON = 1e6

###################################### Select drivecycles to consider #####################################
drivecycles = {
    'pepsi_1': [2, 9, 13, 15, 33],
    'pepsi_2': [7, 10, 14, 22, 25, 31],
    'pepsi_3': [8, 10, 13, 16, 21, 24, 28, 32, 33]
}
###########################################################################################################

def main():
    parameters = data_collection_tools.read_parameters(truck_params='diesel_daycab', vmt_params = 'daycab_vmt_vius_2021', truck_type = 'diesel')

    evaluated_linear_params_df = pd.DataFrame(columns=['Linear Slope (gal/mile/kiloton)', 'Y Intercept (gal/mile)'])
    for truck_name in drivecycles:
        for drivecycle in drivecycles[truck_name]:
            df_drivecycle = truck_model_tools_diesel.extract_drivecycle_data(f'data/{truck_name}_drive_cycle_{drivecycle}.csv')
            slope_gpm, y_intercept_payload_gpm, miles_per_gallon_ave_payload = truck_model_tools_diesel.truck_model(parameters).get_power_requirement(df_drivecycle)
            
            result = {
                'Linear Slope (gal/mile/kiloton)': [slope_gpm * KG_PER_KILOTON],
                'Y Intercept (gal/mile)': [y_intercept_payload_gpm]
            }
            
        
            # Assuming results are in the format expected, create a DataFrame
            evaluated_linear_params_df = pd.concat([evaluated_linear_params_df, pd.DataFrame(result)])

    # Save the evaluated linear parameters as a csv file
    filename = f'tables/linear_mpg_params_diesel_daycab.csv'

    evaluated_linear_params_df.to_csv(filename, index=False)
    
    
if __name__ == '__main__':
    main()
