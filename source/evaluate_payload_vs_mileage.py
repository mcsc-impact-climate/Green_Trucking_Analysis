"""
Date: March 25, 2024
Purpose: Evaluate the payload and energy economy for each drivecycle
"""

import payload_matching_tools
import data_collection_tools
import concurrent.futures
from datetime import datetime
import pandas as pd
from common_tools import get_linear_drivecycles

KG_PER_TON = 1000
KG_PER_LB = 0.453592
SECONDS_PER_HOUR = 3600

###################################### Select drivecycles to consider #####################################
drivecycles = get_linear_drivecycles()
###########################################################################################################

# Function to evaluate matching payloads in parallel
def parallel_evaluate_matching_payload(args):

    # Wrapper function to call evaluate_matching_gvw with a specific combined_eff.
    # This is necessary because ProcessPoolExecutor.map only works with functions that take a single argument.
    
    # Unpack arguments
    truck_name, driving_event, parameters, battery_params_dict = args
    
    return payload_matching_tools.evaluate_matching_payload(truck_name, driving_event, parameters, battery_params_dict)

def main():
    parameters = data_collection_tools.read_parameters(truck_params='semi', vmt_params = 'daycab_vmt_vius_2021')
    parameters.m_max = 120000
    battery_params_dict = data_collection_tools.read_battery_params()

    evaluated_gvws_df = pd.DataFrame(columns=['Payload (lb)', 'Gross Vehicle Weight (lb)', 'Mileage (kWh/mi)'])
    for truck_name in drivecycles:
        drivecycle_events_list = drivecycles[truck_name]
        args_list = [(truck_name, driving_event, parameters, battery_params_dict) for driving_event in drivecycle_events_list]
        startTime = datetime.now()
        print(f'Processing {truck_name}')
        # Use ProcessPoolExecutor to parallelize the evaluation
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(parallel_evaluate_matching_payload, args_list))
        
        # Assuming results are in the format expected, create a DataFrame
        evaluated_gvws_df = pd.concat([evaluated_gvws_df, pd.DataFrame(results, columns=['Payload (lb)', 'Gross Vehicle Weight (lb)', 'Mileage (kWh/mi)'])])
        run_time = datetime.now() - startTime
        run_time = run_time.total_seconds()
        print(f'Processing time for {truck_name}: {run_time}s')

    # Save the evaluated GVWs as a csv file
    filename = f'tables/payload_vs_mileage_semi.csv'
    
    evaluated_gvws_df.to_csv(filename, index=False)
    
    
if __name__ == '__main__':
    main()
    
