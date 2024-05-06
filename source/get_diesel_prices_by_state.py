#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 3 17:46:00 2024

@author: danikam
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import os

N_YEARS_TO_AVERAGE = 5      # Number of years to average historical diesel prices over
MONTHS_PER_YEAR = 12
DAYS_PER_MONTH = 30.437
MONTHS_PER_YEAR = 12

padd_name_dict = {
    'New England (PADD 1A) No 2 Diesel Ultra Low Sulfur (0-15 ppm) Retail Prices (Dollars per Gallon)': 'New England',
    'Central Atlantic (PADD 1B) No 2 Diesel Ultra Low Sulfur (0-15 ppm) Retail Prices (Dollars per Gallon)': 'Central Atlantic',
    'Lower Atlantic (PADD 1C) No 2 Diesel Ultra Low Sulfur (0-15 ppm) Retail Prices (Dollars per Gallon)': 'Lower Atlantic',
    'Midwest No 2 Diesel Ultra Low Sulfur (0-15 ppm) Retail Prices (Dollars per Gallon)': 'Midwest',
    'Gulf Coast No 2 Diesel Ultra Low Sulfur (0-15 ppm) Retail Prices (Dollars per Gallon)': 'Gulf Coast',
    'Rocky Mountain No 2 Diesel Ultra Low Sulfur (0-15 ppm) Retail Prices (Dollars per Gallon)': 'Rocky Mountain',
    'California No 2 Diesel Ultra Low Sulfur (0-15 ppm) Retail Prices (Dollars per Gallon)': 'California',
    'West Coast (PADD 5) Except California No 2 Diesel Ultra Low Sulfur (0-15 ppm) Retail Prices (Dollars per Gallon)': 'Rest of West Coast'
}

state_padd_dict = {
    'CT': 'New England',
    'ME': 'New England',
    'MA': 'New England',
    'NH': 'New England',
    'RI': 'New England',
    'VT': 'New England',
    'DE': 'Central Atlantic',
    'DC': 'Central Atlantic',
    'MD': 'Central Atlantic',
    'NJ': 'Central Atlantic',
    'NY': 'Central Atlantic',
    'PA': 'Central Atlantic',
    'FL': 'Lower Atlantic',
    'GA': 'Lower Atlantic',
    'NC': 'Lower Atlantic',
    'SC': 'Lower Atlantic',
    'VA': 'Lower Atlantic',
    'WV': 'Lower Atlantic',
    'IL': 'Midwest',
    'IN': 'Midwest',
    'IA': 'Midwest',
    'KS': 'Midwest',
    'KY': 'Midwest',
    'MI': 'Midwest',
    'MN': 'Midwest',
    'MO': 'Midwest',
    'NE': 'Midwest',
    'ND': 'Midwest',
    'OH': 'Midwest',
    'OK': 'Midwest',
    'SD': 'Midwest',
    'TN': 'Midwest',
    'WI': 'Midwest',
    'AL': 'Gulf Coast',
    'AR': 'Gulf Coast',
    'LA': 'Gulf Coast',
    'MS': 'Gulf Coast',
    'NM': 'Gulf Coast',
    'TX': 'Gulf Coast',
    'CO': 'Rocky Mountain',
    'ID': 'Rocky Mountain',
    'MT': 'Rocky Mountain',
    'UT': 'Rocky Mountain',
    'WY': 'Rocky Mountain',
    'AK': 'Rest of West Coast',
    'AZ': 'Rest of West Coast',
    'CA': 'California',
    'HI': 'Rest of West Coast',
    'NV': 'Rest of West Coast',
    'OR': 'Rest of West Coast',
    'WA': 'Rest of West Coast'
}

def read_state_data(discountrate):
    '''
    Reads in historical monthly diesel prices for each PADD region.
    
    Parameters
    ----------
    discountrate (float): Annual discount rate to adjust historical and future prices to present day

    Returns
    -------
    padd_data (pd.DataFrame): A pandas dataframe containing the 2021 electricity rate data for each state
    '''
    
    # Evaluate the discount factor over the last N_YEARS_TO_AVERAGE years
    #discountfactor = 1 / np.power(1 + self.parameters.discountrate, np.arange(1, N_YEARS_TO_AVERAGE+1)) #life time of trucks is 10 years
    
    # Read in the monthly data for each PADD region
    dataPath = f'data/psw18vwall.xls'
    data = pd.ExcelFile(dataPath)
    data_df = pd.read_excel(data, 'Data 6', skiprows=[0,1])
    
    # Simplify the column names for the diesel prices
    data_df = data_df.rename(columns=padd_name_dict)
    
    # Read in the historical US urban consumer price index (seasonally adjusted) to adjust historical diesel prices for inflation
    cpi_df = pd.read_csv('data/CPIAUCSL.csv')
    cpi_df = cpi_df.rename(columns = {'DATE': 'Date'})
    
    # Convert the date to datetime format
    cpi_df['Date'] = pd.to_datetime(cpi_df['Date'])
    
    # Adjust dates to the 15th of each month to match the monthly diesel price data
    cpi_df['Date'] = cpi_df['Date'] + pd.Timedelta(days=14)
    
    # Merge the CPI and diesel data dfs according to the date
    data_df = pd.merge(cpi_df, data_df, on='Date', how='left')
    
    # Get the last 60 rows (representing the last 5 years)
    data_df = data_df.tail(N_YEARS_TO_AVERAGE * MONTHS_PER_YEAR)
    
    # Skim columns down to diesel prices for the PADD regions and CPI
    data_df = data_df[list(padd_name_dict.values()) + ['CPIAUCSL']]
    
    # Adjust the diesel prices based on the CPI
    inflation_scale_factor = data_df['CPIAUCSL'].iloc[-1]/data_df['CPIAUCSL']
    data_df[list(padd_name_dict.values())] = data_df[list(padd_name_dict.values())].mul(inflation_scale_factor, axis=0)
    
    # Drop the CPI column since we no longer need it
    data_df = data_df.drop(['CPIAUCSL'], axis=1)
            
    # Evaluate the mean and standard deviation over the last 10 years
    price_stats_df = pd.DataFrame({
        'Average Price ($/gal)': data_df.mean(),
        'Standard Deviation ($/gal)': data_df.std()
        })
        
    # Create a DataFrame from the state to PADD mapping
    state_padd_df = pd.DataFrame(list(state_padd_dict.items()), columns=['State', 'PADD Region'])
    
    # Map the mean and standard deviation values from stats_df to the state DataFrame
    state_padd_df = state_padd_df.merge(price_stats_df, left_on='PADD Region', right_index=True)
    
    return state_padd_df
    

def main():

    # Get the annual discount rate for the study
    df_economy_params = pd.read_csv('data/default_economy_params.csv', index_col=0)
    discountrate = float(df_economy_params['Value'].loc['Discount rate'])

    # Collect the average historical diesel price over the last 5 years
    state_diesel_prices_df = read_state_data(discountrate)
    
    # Save to a csv file
    state_diesel_prices_df.to_csv('tables/average_diesel_price_by_state.csv')

if __name__ == '__main__':
    main()
