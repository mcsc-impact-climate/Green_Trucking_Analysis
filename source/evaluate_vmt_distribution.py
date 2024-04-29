"""
Date: April 25, 2024
Purpose: Evaluate the average VMT distribution for day cab trucks in the US using the 2021 VIUS data.
"""

# Import packages
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# Read in the 2021 VIUS data
data_df = pd.read_csv('data/vius_2021_puf.csv')

# Select data for class 8 day cabs
class8_daycab_df = data_df[(data_df['CABDAY']=='1') & (data_df['GVWR_CLASS']=='8')]

# Add a column with the truck's age, based on the model year and that the data was collected for 2021
class8_daycab_df = class8_daycab_df[(class8_daycab_df['MODELYEAR'] != 'P99') & ((class8_daycab_df['MODELYEAR'] != '99'))]
class8_daycab_df['MODELYEAR'][class8_daycab_df['MODELYEAR'] == '21P'] = '21'
class8_daycab_df['MODELYEAR_INT'] = class8_daycab_df['MODELYEAR'].astype(int)
class8_daycab_df['YEAR'] = 22 - class8_daycab_df['MODELYEAR'].astype(int)


data_boxplot = []
average_vmts = []
median_vmts = []
years = np.unique(class8_daycab_df['YEAR'])
for year in np.unique(class8_daycab_df['YEAR']):
    vmts_year = class8_daycab_df['MILESANNL'][class8_daycab_df['YEAR']==year].astype(float).to_numpy()
    data_boxplot.append(vmts_year)
    average_vmts.append(np.mean(vmts_year))
    median_vmts.append(np.median(vmts_year))

# Plot the annual miles traveled for each year
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlabel('Year', fontsize=18)
ax.set_ylabel('VMT (miles)', fontsize=18)
box = plt.boxplot(data_boxplot, showfliers=False)
ax.plot(years, average_vmts, 'o', label='Average')
ax.legend(fontsize=15)
plt.xticks(years)
ax.tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.savefig('plots/vmt_distribution_vius_2021.png')

# Save the median annual miles traveled to a csv file
df_vmt_save = pd.DataFrame({'Year': range(1,11), 'VMT (miles)': median_vmts[:10]})
df_vmt_save = df_vmt_save.set_index('Year')
df_vmt_save['Source'] = ''
df_vmt_save['Source'].iloc[0]='VIUS 2021'
print(df_vmt_save)
df_vmt_save.to_csv('data/daycab_vmt_vius_2021.csv')
