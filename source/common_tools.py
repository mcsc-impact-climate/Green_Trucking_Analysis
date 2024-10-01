import os
import pandas as pd
from pathlib import Path
import glob

def get_top_dir():
    """
    Gets the path to the top level of the git repo (one level up from the source directory)

    Parameters
    ----------
    None

    Returns
    -------
    top_dir (string): Path to the top level of the git repo
    """
    source_path = Path(__file__).resolve()
    source_dir = source_path.parent
    top_dir = os.path.dirname(source_dir)
    return top_dir

def get_linear_drivecycles(RMSE_cutoff=10):
    """
    Creates a dictionary of drivecycles where RMSE < RMSE_cutoff.

    Parameters
    ----------
    RMSE_cutoff : float, optional
        The RMSE threshold for filtering drive cycles (default is 10).

    Returns
    -------
    drivecycles : dict
        A dictionary where the keys are file identifiers ('pepsi_1', 'pepsi_2', etc.)
        and the values are lists of Driving Event numbers where RMSE < RMSE_cutoff.
    """
    top_dir = get_top_dir()
    data_path_pattern = os.path.join(top_dir, 'data', 'pepsi_*_drivecycle_data.csv')
    
    # Initialize an empty dictionary to store the results
    drivecycles = {}

    # Loop through all matching files
    for file_path in glob.glob(data_path_pattern):
        # Extract the file identifier (e.g., 'pepsi_1', 'pepsi_2', etc.) from the filename
        file_name = os.path.basename(file_path)
        file_identifier = file_name.split('_')[1]  # Extracts the number part (e.g., '1', '2', etc.)

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Filter rows where RMSE is less than the RMSE_cutoff
        filtered_df = df[df['RMSE'] < RMSE_cutoff]
        
        # Get the Driving Event numbers from the filtered rows
        driving_events = filtered_df['Driving event'].tolist()
        
        # Add the results to the dictionary
        drivecycles[f'pepsi_{file_identifier}'] = driving_events

    return drivecycles
    
def make_title_string(info_string):
    """
    Converts an info string (assumed to be lower case and underscore-separated) into a title string (space-separated with first letter of each word capitalized).
    
    Parameters
    ----------
    info_string (string): Input info string (lower case, underscore-separated)

    Returns
    -------
    title_string (string): Output title string (space-separated, first letter of each word capitalized).
    """
    
    space_separated = info_string.replace("_", " ")
    title_string = space_separated.title()
    return title_string
    
def get_nacfe_results(truck_name, drivecycle):
    """
    Get NACFE results for the given truck and drivecycle
    """
    # Collect the battery info extracted from the NACFE data for each truck
    battery_capacity_df = pd.read_csv('data/pepsi_semi_battery_capacities.csv')
    
    # Collect the info extracted from the drivecycle
    drivecycle_data_df = pd.read_csv(f'data/{truck_name}_drivecycle_data.csv', index_col='Driving event')
    
    # Collect NACFE results
    NACFE_results = {
        'Battery capacity (kWh)': battery_capacity_df[truck_name].iloc[0],
        'Battery capacity unc (kWh)': battery_capacity_df[truck_name].iloc[1],
        'Fuel economy (kWh/mi)': drivecycle_data_df['Fuel economy (kWh/mile)'].loc[drivecycle],
        'Fuel economy unc (kWh/mi)': drivecycle_data_df['Fuel economy unc (kWh/mile)'].loc[drivecycle],
        'Range (miles)': drivecycle_data_df['Range (miles)'].loc[drivecycle],
        'Range unc (miles)': drivecycle_data_df['Range unc (miles)'].loc[drivecycle],
    }
    
    return NACFE_results
    
