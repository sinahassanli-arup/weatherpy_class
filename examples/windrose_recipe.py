# %% [markdown]
# <center>
#     <img src="https://weatherpy.s3.ap-southeast-2.amazonaws.com/weatherpylogo_final.png" width="400" alt="weatherpy logo"  />
# </center>

# %% [markdown]
# # Windrose Recipe

# %% [markdown]
# 1. [Objectives](#objectives)
# 2. [Initialization](#initialization)
# 3. [Input Section](#input-section)
#     - [3.1 Input | Station](#input-station)
#     - [3.2 Input | Cleaning](#input-cleaning)
#     - [3.3 Input | Terrain and other corrections](#input-terrain-and-other-corrections)
#     - [3.4 Input | Plot](#input-plot)
# 4. [Analysis Section](#analysis-section)

# %% [markdown]
# 
# ## Objectives
# This recipe generates different windroses and relevant plots for historic weather data from BOM and NOAA. This notebook is prepared to show the appropriate steps for importing, unifying, cleaning and processing the data and, to make it easier for the user to extract required windroses and relevant plots.
# 
# The notewbook is devided into two main sections: **input** and **analysis** each having their own sub-sections. Before that, we have to import weatherpy library:

# %% [markdown]
# ### Import Weatherpy
# Lets import required libraries first

# %%
import os
import sys
import numpy as np
import pandas as pd

# %% [markdown]
# It is assumed weatherpy is installed. In case you have not installed it or if you want to use a local repo you can add it to your to your system path.
# 
# It is recommended to use the installed version for proper implementation of the library.

# %%
# Uncomment the line below to import weatherpy locally
# sys.path.insert(0, r'C:\Users\sina.hassanli\OneDrive - Arup\Arup\Github\weatherpy')
# Uncomment the line below to import weatherpy locally
sys.path.insert(0, r'C:\Users\sina.hassanli\OneDrive - Arup\Arup\Github\weatherpy')

# %% [markdown]
# We now import weatherpy

# %%
import weatherpy as wp

# %% [markdown]
# ## Input Section

# %% [markdown]
# ### Input | Station
# 
# To begin the analysis, we need to define the required inputs for our weather data. These inputs include:
# 
# - Station Number: This is the unique identifier for the weather station.
# - Start Year: The year from which the data analysis should begin.
# - End Year: The year at which the data analysis should end.
# - Weather Data Type: Currently, weatherpy supports two data types: "BOM" and "NOAA".
# - Interval (for BOM data only): If you are using BOM data, you need to specify the interval of your observations. You can choose between 1, 10, or 60 minutes.
# - Station Name: This is an optional parameter used for annotation purposes.
# - Time Zone: By default, the time zone is set to local time for BOM data and UTC for NOAA data. However, you can specify a different time zone using the "timeZone" parameter.
# 
# It is important to provide accurate and valid inputs to ensure the analysis is performed correctly.
# 

# %%
# full_path = r'C:\Users\sina.hassanli\OneDrive - Arup\Arup\Github\weatherpy\weatherpy\data\src\BOM_stations_clean.csv'
# statation_db = pd.read_csv(full_path,converters={'Station Code':str,
#     'Station Name':str,'Country':str,'State':str,'Latitude':float,
#     'Longitude':float,'Elevation':str,'Start':str,'End':str,
#     'Timezone':str,'Source':str,'Wind Direction':str,
#     'Wind Speed':str,'Wind Gust':str,'Sea Level Pressure':str,
#     'Dry Bulb Temperature':str,'Wet Bulb Temperature':str,
#     'Relative Humidity':str,'Rain':str,'Rain Intensity':str,
#     'Cloud Oktas':str},index_col=False)

# no_st = statation_db.shape[0]
# for i, st_id in enumerate(statation_db['Station Code']):
    
# print(f'{i}/{no_st}: {st_id}')
# %%
# Station number or ID. This entry is used to import the data
stationID = '066037' #st_id[-6:]

# Type of the weather data. choose between 'BOM', 'NOAA', or 'custom'
dataType = 'BOM'

# Weather station name. This is simply used for figure notation
stationName = wp.station_info(stationID, printed=False)['Station Name']
# stationName = 'Sydney Airport'

# First year of imported data. Data will start 1-Jan of selected year 
yearStart = 2003

# Last year of imported data. Data will end 31-Dec of selected year 
yearEnd = 2023

# If it is a BOM weather station define the interval. You can choose between 1, 10, 60 minute data:
# use 60 if you don't know what to choose, and use other intervals with care as not all 
# functionailities are compatible with other intervals
interval = 60

# Timezone: "Local Time" or "UTC"
timeZone = 'LocalTime'

# %% [markdown]
# Below is an example for importing NOAA weather station. You should use either of above (for BOM) or below (for NOAA) cell to import your data

# %%
# # Station number or ID. This entry is used to import the data
# stationID = '3969099999'

# # Type of the weather data. choose between 'BOM', 'NOAA', or 'custom'
# dataType = 'NOAA'

# # Weather station name. This is simply used for figure notation
# stationName = 'Dublin Internaional Airport'

# # First year of imported data. Data will start 1-Jan of selected year 
# yearStart = 2003

# # Last year of imported data. Data will end 31-Dec of selected year 
# yearEnd = 2022

# # Timezone: "Local Time" or "UTC"
# timeZone = 'UTC'


# %% [markdown]
# ### Input | Cleaning 
# There are different cleaning methods to remove invalid/misreading/outlier observations, all coming under one cleaning function called "clean_data". clean_data has different input arguments to clean; some methods are shared between BOM and NOAA and some methods specific to each.
# 
# Since cleaning could have significant impact on the bias and quality of the results here we discuss the arguments:
# 
# #### Generic cleaning options
# - `clean_invalid`: An optional True/False flag True/False flag to indicate whether certain columns should be cleaned for invalid/NaN values. If set to True, the columns to check are defined as a list in the `col2valid` variable. Default is False.
# 
# - `col2valid`: An optional list of columnes/attributes to consider for invalid/NaN check if `clean_invalid` is set to True. A list of  Default is ['WindSpeed', 'WindDirection', 'WindType'].
# 
# - `clean_threshold` : An optional True/False flag to indicate whether certain columns should be cleaned based on user defined valid thresholds. If set to True, the columns to check are defined as a dictionary in the thresholds. It is recommended  that if a column is being threshold cleaned, it should also be invalid cleaned (clean_invalids = True). Default is False. 
# 
# - `thresholds` : An optional dictionary required when clean_threshold is set to True. if `clean_threshold` is True, then data will be cleaned in accordance with acceptable ranges. The dict is in the form: {'column':(lower,upper)}.
# 
# - `clean_calms` : Remove calm winds. Default is False.
# 
# #### BOM-specific cleaning options
# - `clean_off_clock`: An optional True/False flag to indicate whether timestamps should be removed if they are not on 30-minute intervals. This option is specific to BOM data. Default is False.
# 
# #### NOAA-specific cleaning options
# - `clean_ranked_rows`: An optional True/False flag to indicate whether duplicate timestamps should be removed. This option is specific to NOAA data. Default is False.
# - `clean_VC_filter`:  An optional True/False flag to indicate whether timestamps should be removed based on the VC filter. This option is specific to NOAA data. Default is False.
# - `clean_direction`:  An optional True/False flag to indicate whether timestamps should be removed if the wind direction is not a multiple of 10. This option is specific to NOAA data. Default is False.
# - `clean_storms`:  An optional True/False flag to indicate whether timestamps should be removed during thunderstorms. This option is specific to NOAA data. Default is False.

# %%
# Flags if timestamps should be removed if they contain invalid readings
clean_invalid = True

# If clean_invalid = True, this list of columns will be checked for invalids
col2valid = ['WindGust', 'WindDirection']

# Flags if timestamps should be removed if they are outside of certain thresholds
clean_threshold = True

# If clean_threshold = True, this dictionary of columns will be checked
thresholds = {'WindSpeed': (0, 50),
              'PrePostRatio': (5, 30)}

# clean the calm values. by default it is False. If you want to run EVA/Weibull consider it to set it to Ture
clean_calms = False

# %% [markdown]
# Cleaning options specific to NOAA. If you are using BOM you can skip this part

# %%
# Cleans duplicate timestamps
# clean_ranked_rows = True

# # Apply VC filter to data (Overrides clean_ranked_rows filter)
# clean_VC_filter = False

# # Remove timestamps with wind direction not a multiple of 10
# clean_direction = True

# # Clean timestamps during hunderstorms (NOAA data only)
# clean_storms = True

# %% [markdown]
# ### Input | Terrain and other corrections 
# 
# Here we define if we want to use correction factor and how we want to define it. 
# - Setting "source" as database will check if we have established corretion factors for the station otherwise it would not apply corrections. 
# - Setting "source" as manual will use the "correctionFactors" parameters as input.

# %%
# Correcting data to TC10,2 if correction factors exists
terrainCorrection = True

# Type of terrain correction to be applied. Could be 'database' or 'manual'
source = 'database'
    
# a list of correction factors if source = 'manual'. Otherwise can be left as None
correctionFactors = None #[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# terrain label to be used in the figures, if not defined here 'Corrected to open terrain' will be used
terrainLabel = 'Corrected to open terrain'

# %% [markdown]
# There has been some issues in terms of caliberation for low wind speed for most of BOM weather stations.
# A correction flag is introduced to take that into account.

# %%
spdCorrection_bomOffset = True

# %% [markdown]
# ### Input | Plot
# Last stop is to set parameters for plotting. If save is True, savePath should be specified.

# %%
# Number of wind directions to divide the wind data
numDirs = 16

# List of wind speed intervals fo be analysed
windSpdRange = [0, 2, 4, 6, 8, 10, 15]
windSpdRange = np.arange(41)

# Indicates whether the user wishes to have multiple charts subplotted
subplot = True

# To scale all wind roses to the same radial axis
max_radius = True

# Flag to decide whether visualisations should be saved as PNGs
save = True

# If save = True, then the images will be saved to the directory specified here
state = wp.station_info(stationID)['State']
savePath = rf"C:\Users\sina.hassanli\OneDrive - Arup\Arup\Github\weatherpy_cache"

# %% [markdown]
# ## Initialization Section

# %% [markdown]
# #### Initialization | Import
# We are going to import the data from BOM or NOAA servers (or local temporary files if available)

# %%
# defining a site description label
site_description = f'{stationName} ({stationID})'

# set the interval to None if it has not been defined for NOAA
interval = None if "interval" not in locals().keys() else interval

# try:
    # import the data from BOM or NOAA servers or use local files if available
data_raw, _, _ = wp.import_data(
    stationID,
    dataType,
    timeZone,
    yearStart,
    yearEnd,
    interval,
    save_raw=True,
    local_filepath=None)
# except:
    # continue
# data_raw = pd.read_pickle(r"C:\Users\sina.hassanli\OneDrive - Arup\Arup\Github\GPTUnitTest\BOM_066037_2001-2021_60minute.zip")

# %% [markdown]
# #### Initialization | Unify
# The purpose of unify_datatype is to ensure consitent weather attribute or column names and number for NOAA and BOM data

# %%
# Extract wind fields and make consistent columns for BOM and NOAA
data_unified = wp.unify_datatype(
    data_raw,
    dataType)

# %% [markdown]
# #### Initialization | Clean
# clean_data grabs all the cleaning flags defined in the input section and clean the data in one go. 
# 
# It rquires the unified data so it can work with column names.

# %%
# Clean data
data_cleaned, data_removed, data_calm = wp.clean_data(
    data_unified,
    dataType = dataType,
    # clean_ranked_rows = clean_ranked_rows,
    # clean_VC_filter = clean_VC_filter,
    # clean_calms = clean_calms,
    # clean_direction = clean_direction,
    # clean_storms = clean_storms,
    clean_invalid = clean_invalid,
    clean_threshold = clean_threshold, 
    col2valid = col2valid, 
    thresholds = thresholds)

# %% [markdown]
# #### Initialization | Correction
# The following lines will:
# - apply terrain correction if the terrain correction is requested
# - apply wind speed correction for low wind speeds of BOM weather observations (by default it reduces mean wind speed by 0.4m/s for speeds lower than 2 m/s)
# - and finally assign the last corrected dataset to data_final which will be used as input for analysis in the remainder of the script

# %%
# Correct data
if terrainCorrection:
    data_corrected, isTerrainCorrected = wp.correct_terrain(
        data_cleaned,
        stationID = stationID,
        dataType = dataType,
        source = source,
        correctionFactors = correctionFactors)   
else:
    data_corrected = data_cleaned.copy(deep = True)
    isTerrainCorrected = False

if isTerrainCorrected:
    TC_label = terrainLabel if 'terrainLabel' in locals() else 'Corrected to open terrain'
else:
    TC_label = 'No terrain correction applied'

# Correct data
data_corrected = wp.spdCorrection_bomOffset(data_corrected) if (dataType == 'BOM' and spdCorrection_bomOffset) else data_corrected
data_corrected = wp.spdCorrection_10minTo1h(data_corrected) if (interval == 10) else data_corrected

# Final data
data_final = data_corrected

# %% [markdown]
# ## Analysis Section

# %% [markdown]
# ### Analysis | Export Data Weather data
# You can export the dataset by providing a csv file path for export. (thtese two lines are commented out by default)
# 
# Remember to use "r" at the begining of your path string if you are using "\" method (e.g. r'c:\Users\Sina.Hassanli\Documents\Sydney Airprot_2000-2022.csv')
# 
# You can use this method (e.g., variableName.to_csv(filepath)) to export any dataframe into csv file

# %%
# filepath = os.path.join(os.path.expanduser('~'),'Documents',f'{site_description}_{yearStart}_{yearEnd}.csv')
# data_final.to_csv(filepath)

# %% [markdown]
# #### Analysis | Weather data Overview

# %% [markdown]
# 
# We can review the overal weather data by:
# - completness map which graphically shows the missing values. It looks at all the missing values for all the attributes
# - plotting mean/median/qunatiles for any weather attribute for the range of years. There is an optional argument (show_min,show_max) to show the minimum and maximum for each year. Enabling this option will make reading mean values difficult so it is turned off by default.
# - Table Description of min/mean/max, count and 25-50-75 percentiles for all weather attributes

# %%
# completeness analysis
completeness_df = wp.completeness(data_final, site_description=site_description,  save=True, save_dirpath=savePath)


# plot annual mean
data_desc = wp.plot_annual_mean(data_final,  site_description, column='WindSpeed', show_min=False, show_max=True, save=save, save_dirpath=savePath)
    
# You can have a better insight using "describe" method on a dataframe that shows no. of recordings, mean, standard deviation, min, 25%, 50%, 75%, and max for each column.
print(data_final.describe())


# %%
# data_desc

# %% [markdown]
# #### Analysis | PDF, CPDF, Speed Exceedance
# You can get the probability distribution function (pdf) and cumulative pdf, speed exceedance per direction and in total

# %%
# Set the display format for floating-point numbers globally
pd.set_option('display.float_format', '{:.3f}'.format)

#%% pdf, cpdf and exceedances of wind speed
pdf, cpdf, spdExceed_perDirection, spdExceed_total  = wp.windspeed_statistics(
    data_final['WindDirection'],
    data_final['WindSpeed'],
    numDirs=16,
    wind_spd_ranges=windSpdRange,
    )

# filapath = os.path.join(savePath, site_description, 'cpdf_6am-11pm.xlsx')
# cpdf.to_excel(filapath)
# print(filapath)

# %% [markdown]
# #### Analysis | Windrose for all hours
# You can define modify the windrose by:
# - changing the number of wind directions to be plotted using numDirs
# - changing ranges (colors) of wind speed by changing windSpdRange
# - changing the label for site
# 

# %%
#%% WINDROSE ALL HOURS
windrose_by_allhours = wp.windrose_by_attribute(data_final, 
                                                numDirs = numDirs,
                                                wind_spd_ranges = windSpdRange,
                                                site_description = site_description,
                                                terrainCorrection_label=TC_label,
                                                save = save,
                                                save_dirpath = savePath)

# %% [markdown]
# #### Analysis | Windrose for business hours
# You can also specify which attribute you want to use to clip/filter or split your windrose. 

# For this purpose attribute type (attrType) and attribute values (attrValues) should be specified.

# attribute types could be one of the weather attributes/columns or could be temporal values including "Hour", "Month".

# In the example below we plot the windrose only for a business hours from 6am-10pm. "Business Hours" is used for labeling purposes and any other definition can be used.



# %%
# %% WINDROSE BUSINESS HOURS

attrType = 'Hour'
attrValues = {'Business hours': [6,22]}

windrose_businessHours = wp.windrose_by_attribute(data_final,
                                                attrType = attrType,
                                                attrValues = attrValues,
                                                numDirs = numDirs,
                                                wind_spd_ranges = windSpdRange,
                                                site_description = site_description,
                                                terrainCorrection_label=TC_label,
                                                useLabel = True,
                                                max_radius = max_radius,
                                                save = save,
                                                save_dirpath = savePath)

# %% [markdown]
# #### Analysis |  Windrose by one attribute

# %% [markdown]
# ##### Windrose by Season
# You can also have multiple ranges for the selected attribute. In the example below we use "Month" as attribute type and seasons with their corresponding month number (Jan=1, Dec=12) for Australian Season. Another example is commented out; so you can uncomment and play with it.
# 
# When you are using Month, you should specify all the corresponding months as a list
# 
# Here we are using more optional arguments from windrose_by_attribute:
# - `subplot`: When using multiple values, weatherpy automatically detects that there are a number of plots, If subplot flag is True, all plots will be inside a one figure. otherwise each plot has a separate figure.
# 
# - `useLabel`: this is an optional flag. If it sets to True, it uses the labels from attrValues as headers for plots. If it sets to False, it tries to get the labels from the type of the data and selected ranges. In winrose by temperature you will see how setting to false infer the label from the data.
# 
# - `probPer`: This is used to indicate how the occurrence probability should be calculated. for any plot that does not use the entire dataset such as windrose by season or by temperature, the probPer argument can be set to the following:
#     * "figure" : The probability of each windrose is the percent of the data used, compared to total data used for all windroses (default)
#     * "dataset" : The probability of each windrose is the percent of the data used, compared to the total data in the dataset
# - `max_radius`: if True, max_radius will use consistent radius scale for all the subplots by calculating the maximum probabiloty occur across all plots. 

# %%
# %% WINDROSE BY SEASON

attrType = 'Month'
attrValues = {'Summer (Dec, Jan, Feb)': [1, 2, 12], 'Autumn (Mar, Apr, May)': [3, 4, 5], 'Winter (Jun, Jul, Aug)': [6, 7, 8], 'Spring (Sep, Oct, Nov)': [9, 10, 11]}

# attrType = 'Month'
# attrValues = {'Warm Season (Nov - Apr)': [1, 2, 3, 4, 11, 12], 'Cool Season (Mar - Oct)': [5, 6, 7, 8, 9, 10]}

windrose_by_season = wp.windrose_by_attribute(data_final,
                                            attrType = attrType,
                                            attrValues = attrValues,
                                            numDirs = numDirs,
                                            wind_spd_ranges = windSpdRange,
                                            site_description = site_description,
                                            terrainCorrection_label=TC_label,
                                            useLabel=True,
                                            probPer='figure', # can be figure or dataset, default is figure
                                            max_radius = max_radius,
                                            save = save,
                                            save_dirpath = savePath,
                                            subplot = subplot)

# %% [markdown]
# ##### Windrose by Time of Day

# %%
attrType = 'Hour'
attrValues = {'Morning (7-11)': [7, 10], 'Noon(11-14)': [11, 13], 'Afternoon (14-18)': [14, 17], 'Evening (18-22)': [18, 21]}

windrose_by_season = wp.windrose_by_attribute(data_final,
                                            attrType = attrType,
                                            attrValues = attrValues,
                                            numDirs = numDirs,
                                            wind_spd_ranges = windSpdRange,
                                            site_description = site_description,
                                            terrainCorrection_label=TC_label,
                                            useLabel=True,
                                            probPer='figure', # can be figure or dataset, default is figure
                                            max_radius = max_radius,
                                            save = save,
                                            save_dirpath = savePath,
                                            subplot = subplot)

# %% [markdown]
# ##### Windrose by Dry-bulb Temperature

# This is similar to the previous plot only attrType is changed to DruBylbTmeperature and attrValues to a few selected ranges.

# Here we try to infer the labels from the data by setting `useLabel` to False and we get something like below:

# ![image.png](attachment:image.png)

# %%
# %% WINDROSE BY DRY BULB TEMPERATURE

attrType = 'DryBulbTemperature'
# attrValues = {'below 15': [-np.inf, 15], '15C-25C': [15,25], '25C-30C': [25,30], 'above 30C': [30, np.inf]}
attrValues = {'below 18': [-np.inf, 18], '18C-28C': [18,28], 'above 28C': [28, np.inf]}
# attrValues = {
#     'Cold (below 12C)': [-np.inf, 12],
#     'Cool (12-18C)': [12, 18],
#     'Moderate (18-25C)': [18, 25],
#     'Warm (25-30C)': [25, 30],
#     'Hot (above 30C)': [30, np.inf]
# }
windrose_by_DBT = wp.windrose_by_attribute(data_final,
                                            attrType = attrType,
                                            attrValues = attrValues,
                                            numDirs = numDirs,
                                            wind_spd_ranges = windSpdRange,
                                            site_description = site_description,
                                            terrainCorrection_label=TC_label,
                                            useLabel=False,
                                            probPer='figure', # can be figure or dataset, default is figure
                                            max_radius = max_radius,
                                            save = save,
                                            save_dirpath = savePath,
                                            subplot = subplot)

# %% [markdown]
# ##### Windrose by Rain Intensity
# 
# Still the same function and the same format, only attrType and attrValues are changed

# %% WINDROSE BY RAIN INTENSITY

attrType = 'RainIntensity'
attrValues = {'any rain': [0.0016, np.inf], '0.0016-0.025 mm/h': [0.0016, 0.025], '0.025-0.1 mm/h': [0.025, 0.1], '0.1-0.34 mm/h': [0.1, 0.34], '0.34-1.3 mm/h': [0.34, 1.3], 'above 1.3': [1.3, np.inf]}

windrose_by_rain = wp.windrose_by_attribute(data_final,
                                            attrType = attrType,
                                            attrValues = attrValues,
                                            numDirs = numDirs,
                                            wind_spd_ranges = windSpdRange,
                                            site_description = site_description,
                                            terrainCorrection_label=TC_label,
                                            useLabel=True,
                                            probPer='figure', # can be figure or dataset, default is figure
                                            max_radius = max_radius,
                                            save = save,
                                            save_dirpath = savePath,
                                            subplot = subplot)

# %% [markdown]
# #### Analysis | Windrose by two attributes

# %% [markdown]
# ##### Windrose by season and time of day
# Here we are producing a windrose by splitting the data based on two different attributes. As before attribute types `attrType` can be temporal like Hour and Month or one of the weather attributes like DryBulbTemperature.
# 
# The resulting figure is created as a grid plots where attrType1 forms the rows and attrType2 the columns of the grid.

# %%
# %% WINDROSE BY SEASON AND TIME OF DAY

attrType2 = 'Month'
attrValues2 = {'Summer (Dec, Jan, Feb)': [1, 2, 12], 'Autumn (Mar, Apr, May)': [3, 4, 5], 'Winter (Jun, Jul, Aug)': [6, 7, 8], 'Spring (Sep, Oct, Nov)': [9, 10, 11]}

attrType1 = 'Hour'
attrValues1 = {'Morning 7-11': [7, 10], 'Noon 11-14': [11, 13], 'Afternoon 14-18': [14, 17], 'Evening 18-22': [18, 21]}

windrose_by_season_and_TOD = wp.windrose_by_attributes(data_final,
                                                        attrType1 = attrType1,
                                                        attrValues1 = attrValues1,
                                                        attrType2 = attrType2,
                                                        attrValues2 = attrValues2,
                                                        numDirs = numDirs,
                                                        wind_spd_ranges = windSpdRange,
                                                        site_description = site_description,
                                                        terrainCorrection_label=TC_label,
                                                        useLabel=True,
                                                        probPer='figure', # can be row, column, figure or dataset, default is figure
                                                        max_radius = max_radius,
                                                        save = save,
                                                        save_dirpath = savePath,
                                                        subplot = subplot)

# %% [markdown]
# ##### Windrose by time of day and dry-bulb temperature
# The same concept as above but for time of day and temperature

# %%
# %% WINDROSE BY TIME OF DAY AND DRY BULB TEMPERATURE

attrType1 = 'Hour'
attrValues1 = {'Morning 7-11': [7, 10], 'Noon 11-14': [11, 13], 'Afternoon 14-18': [14, 17], 'Evening 18-22': [18, 21]}

attrType2 = 'DryBulbTemperature'
# attrValues2 = {'below 20°C': [-np.inf, 20], '20-30°C': [20,30], 'above 30°C': [30, np.inf]}
# attrValues2 = {'below 15': [-np.inf, 15], '15C-25C': [15,25], '25C-30C': [25,30], 'above 30C': [30, np.inf]}
attrValues2 = {'below 18': [-np.inf, 18], '18C-28C': [18,28], 'above 28C': [28, np.inf]}

windrose_by_TOD_and_DBT = wp.windrose_by_attributes(data_final,
                                        attrType1 = attrType1,
                                        attrValues1 = attrValues1,
                                        attrType2 = attrType2,
                                        attrValues2 = attrValues2,
                                        numDirs = numDirs,
                                        wind_spd_ranges = windSpdRange,
                                        site_description = site_description,
                                        terrainCorrection_label=TC_label,
                                        useLabel=True,
                                        probPer='figure', # can be row, column, figure or dataset, default is figure
                                        max_radius = max_radius,
                                        save = save,
                                        save_dirpath = savePath,
                                        subplot = subplot)

# %% [markdown]
# #### Percentage Usability
# Calculate the % usability for different wind speed based on the calculated windroses

# %%
windrose_list = [windrose_by_allhours,
windrose_businessHours,
windrose_by_season,
windrose_by_DBT,
# windrose_by_rain,
windrose_by_season_and_TOD,
windrose_by_TOD_and_DBT]

usability = pd.DataFrame()
for wr in windrose_list:
    for (i, df), (k, wr_props) in zip(wr[0].items(), wr[1].items()):
        prob = (df.shape[0]/data_final.shape[0])
        usability[k] =  [prob] + (1-wr_props['cpdf'].sum(axis=1)).values.tolist()
usability.index = ['probability'] + wr_props['cpdf'].index.tolist()

filename = 'Percentage usability.xlsx'
filapath = os.path.join(savePath, site_description, filename)
usability.to_excel(filapath)

print(usability)

# %% [markdown]
# #### Temperature PDF
# 
# This plot shows the probability distribution function for drybulb or wetbult temnperature for 8 cardinal directions. It also tabulated statistics of temperature for each direction which will be very useful in application where directional dry-bulb temperature is important such as data center projects.

# %%
# %% PLOT PROBABILITY OF TEMPERATURE FOR 8 CARDINAL DIRECTIONS

windRange = list(range(11))
data_queried, _, _ = wp.query_range(data_final, {'Hour': [6,22]}, closed = 'lower')
Tpdf, Tcpdf, Texceed = wp.pdf_temperature(data_queried,
                                          site_description,
                                          wind_speed_ranges=windRange,
                                          tempType='DB',
                                          orientation='horizontal',
                                          save=save,
                                          save_dirpath=savePath)


