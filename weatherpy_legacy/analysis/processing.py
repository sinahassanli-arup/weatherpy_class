import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import circmean
from ..analysis import psychrolib
import math

import inspect
import sys


DEFAULT_WIND_SPEED_RANGES = [0, 2, 4, 6, 8, 10, 15]

def _generate_windspeed_ranges(spd_range):
    return list(zip(spd_range, spd_range[1:]+[np.inf]))

def _generate_label(spd_range):
    """Converts wind speed range to a label (str)
    """
    lower_bound, upper_bound = spd_range
    if np.isinf(upper_bound):
        return ">"+str(lower_bound)+" m/s"
    else:
        return ">"+str(lower_bound)+"-"+str(upper_bound)+" m/s"
    
def _generate_label_T(temp_range):
    """Converts wind speed range to a label (str)
    """
    lower_bound, upper_bound = temp_range
    if np.isinf(upper_bound):
        return ">"+str(lower_bound)+" C"
    else:
        return ">"+str(lower_bound)+"-"+str(upper_bound)+" C"

def _wind_speed_colour(spd_range, wind_spd_ranges=DEFAULT_WIND_SPEED_RANGES):
    """Convert wind speed range to colour for plotting
    """
    wind_spd_ranges = _generate_windspeed_ranges(wind_spd_ranges)
    for i, wind_speed_range in enumerate(wind_spd_ranges):
        if _generate_label(wind_speed_range) == spd_range:
            return plt.cm.jet((i+1)/len(wind_spd_ranges))

def _sector_windDirections(no_sectors):
    return np.linspace(360/no_sectors, 360, no_sectors, endpoint=True)

def convertGust(fromDuration, toDuration):
    """
    Covert wind speed for different gust durations

    Parameters
    ----------
    fromDuration : float
        from gust duration
    
    toDuration : float
        to gust duration

    Returns
    -------
    conversionFactor : float
        the conversion factor to convert fromDuration to toDuration

    """
    convertGustMatrix = pd.DataFrame(data=[[1, 1/1.08],[1.08, 1]], index=[0.5, 3.0], columns=[0.5, 3.0])
    
    conversionFactor = convertGustMatrix.loc[float(fromDuration),float(toDuration)]
    
    return conversionFactor

#%%       
def generate_label(spd_range):
    """
    Converts list of speeds to a list of wind speed ranges as labels (str)
    format: [<2m/s, 2-4m/s, 6-8m/2, <8m/s]
    Parameters
    ----------
    spd_range : List
        List of speeds.

    Returns
    -------
    spd_labels : List
        Generate a list of wind speed ranges as labels.

    """
    for i, s in enumerate(spd_range):
        if s==spd_range[0]:
            spd_labels = ['<{}m/s'.format(spd_range[0])]
        else:
            spd_labels.append('{}-{}m/s'.format(spd_range[i-1],spd_range[i]))
    return spd_labels

# %%
def generate_label2(spd_range):
    """
    Converts list of speeds to a list of wind speed ranges as labels (str)
    format: [<2m/s, <4m/s, <6m/2, <8m/s]
    Parameters
    ----------
    spd_range : List
        List of speeds.

    Returns
    -------
    spd_labels : List
        Generate a list of wind speed ranges as labels.

    """
    spd_labels=[]
    for i, s in enumerate(spd_range):
        spd_labels.append('<{}m/s'.format(spd_range[i]))
    return spd_labels


   
def generate_ranges(lst, closed=True):
    """
    Generate list of ranges and its label 

    Parameters
    ----------
    lst : list
        List of values for range.
    closed : boolean, optional
        True if lower and upper bounds are included in the list. 
        Otherwise, it will -np.inf and +np.inf for lower nad upper bounds.
        The default is True.

    Returns
    -------
    range_list : list
        List of tuples for lower and upper bounds.
    range_label : List
        List of range labels.

    """
    
    if closed:
        range_list = list(zip(lst[:-1], lst[1:]))
        range_label=['{}-{}'.format(lower,upper) for lower,upper in zip(lst[:-1], lst[1:])]
    else:
        range_list = list(zip([-np.inf]+lst, lst+[np.inf]))
        range_label=[]
        for rng in range_list:
            lower_bound, upper_bound = rng
            
            if np.isinf(lower_bound):
                label = "<"+str(upper_bound)
            elif np.isinf(upper_bound):
                label = ">"+str(lower_bound)
            else:
                label = str(lower_bound)+"-"+str(upper_bound)
            range_label.append(label)
        
    return range_list, range_label
  
def generate_ranges_wind_direction(numDirs, label_midvalue=False):
    """
    Generate list of ranges and its label for wind directions

    Parameters
    ----------
    numDirs : int
        Number of wind directions.
    
    label_midvalue : Boolean, Optional
        True/False for whether the mid value should be labeled. Default is False.

    Returns
    -------
    windDirs_range : list
        List of tuples for lower and upper bounds for wind directions.
    
    windDirs_range_label : list
        List of range labels for wind directions.

    """
    
    interval = 360/numDirs
    windDirs = list(np.arange(0,360,interval))
    windDirs_range=[]
    for i in windDirs:
        if i==0:
            # add calm wind direction range as (0,0)
            windDirs_range=[(0,0)]
            # add winds from north
            windDirs_range.append((360-interval/2, i+interval/2))
        else:
            windDirs_range.append((i-interval/2, i+interval/2))
    if label_midvalue:
        windDirs_range_label = ['Calm']+[ '%g' %i for i in np.arange(0,360,interval)]
    else:
        if numDirs==8: 
            windDirs_range_label=['Calm','N','NE','E','SE','S','SW','W','NW']
        elif numDirs==16: 
            windDirs_range_label=['Calm','N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']
        else: 
            windDirs_range_label = ['Calm']+[str(i) for i in windDirs]
    
    return windDirs_range, windDirs_range_label

def wind_direction_label(windDirs, numDirs=16, label_midvalue=False):
    """
    Generates labels for wind directions.

    Parameters
    ----------
    windDirs : pd.series
        pd.series containing wind directions
    
    numDirs : int, optional
        Number of wind directions. The default is 16.
    
    label_midvalue : Boolean, Optional
        True/False for whether the mid value should be labeled. Default is False.

    Returns
    -------
    WindDir_label: pd.series
        A series containing the label of wind directions

    """
    
    bins_wind = [-1]

    windDirs_range, windDirs_label = generate_ranges_wind_direction(numDirs, label_midvalue)

    for i in windDirs_range:
        bins_wind.append(i[1])

    windDirs_label_df = pd.cut(windDirs, bins_wind, labels=windDirs_label, right=True)
    
    if label_midvalue:
        windDirs_label_df = windDirs_label_df.fillna('0')
    else:
        windDirs_label_df = windDirs_label_df.fillna('N')
    
    return windDirs_label_df

# TODO: Convert lables to list of mid values
def dir_label_to_midvalue(labels):
    
    lst = list(labels)
    # if len(lst)==16:

        
#%% Interval processing
def intervals_dataframe(intervals, column, detail = False):
    '''
    A function for processing custom interval dataframes.
    This function will raise errors to the user for the following reasons:
    - There are any missing values in the input data lists
    - If the interval list are not in ascending order
    - If there is an overlap between to interval ranges. E.g: (5, 10) & (9, 15)
    - If there is a gap between two consecutive intervals. E.g (5, 10) & (11, 15)
    - There is a missing entry in one of the dictionaries
    - If an RGB colour code is incorrect. Must be a tuple of 3 numbers between 0 and 1
    - If a colour name is not recognised by matplotlib

    The correct form looks like: 
    intervals = {'Extreme cold stress': (np.NINF, -27), ...]
    or intervals = {'Extreme cold stress': [(np.NINF, -27), 'red'], ...]

    Parameters
    ----------
    intervals : list 
        A list of dictionaries with interval names, value ranges, and colours (optional)
        
    column : String
        Specifies the column to which the categorisation is applied to
    
    detail : Boolean, Optional
        Specifies whether the category string should include range. E.g:
        detail = False : "No Thermal Stress"
        detail = True  : "No Thermal Stress, 18°C - 26°C"

    Raises
    ------
    ValueError
        Raised for multiple different invalid inputs. See main description of possible reasons
    
    Returns
    -------
    intervals_df : pandas.DataFrame
        A dataframe wil columns "Name", "Range", and "Colour"
    '''
    # Converts list of interval dictionaries to a dataframe
    intervals_df = pd.DataFrame.from_dict(intervals, orient='index')
    intervals_df.reset_index(inplace=True)
    intervals_df.rename(columns={'index': 'Name'}, inplace=True)

    if type(intervals_df[0][0]) == type((1, 2)):
        intervals_df.rename(columns={0: 'Range', 1: 'Colour'}, inplace=True)
    else:
        intervals_df['Range'] = intervals_df.apply(lambda row: (row[0],row[1]), axis=1)
        intervals_df.drop(columns = [0, 1], inplace = True)
  
    # Creates a list of all value range tuples
    tuples = list(intervals_df['Range'])

    # Check if there is any missing values
    if intervals_df.isna().any().any():
        raise ValueError('All interval keys must have corresponding ranges tuples and colours')

    # Checks if the intervals are not in ascending order 
    if tuples != sorted(tuples, key=lambda x: x[0]):
        raise ValueError('The intervals must be in ascending order from lowest to highest interval')

    # Checks of gaps or overlaps in the degree intervals 
    for i in range(len(tuples) - 1):
        if tuples[i][1] > tuples[i+1][0]:
            raise ValueError('There is an overlap in the temperature ranges of ' +
                             'the intervals: \"{}\" and \"{}\"'.format(
                             intervals_df['Name'][i],intervals_df['Name'][i+1]))
        
        elif tuples[i][1] < tuples[i+1][0]:
            raise ValueError('There is an gap in the temperature ranges of ' +
                             'the intervals: \"{}\" and \"{}\"'.format(
                             intervals_df['Name'][i],intervals_df['Name'][i+1]))
    
    # If Default colour settings are needed
    if 'Colour' not in set(intervals_df.columns):
        # Initialise colour gradient 
        colors = ['black', 'navy', 'royalblue', 'mediumturquoise', 'limegreen', 'yellow', 'orange', 'crimson']
        cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', colors)

        # Generate equally spaced values from 0 to 1
        positions = np.linspace(0, 1, len(intervals))

        # Generate RGB values for each position in the gradient
        rgb_values = cmap(positions)[:,:-1]
        rgb_values = [tuple(row) for row in rgb_values]
        intervals_df['Colour'] = pd.Series(rgb_values)

    # Checks the "Colour" columns
    else:
        for i, r in intervals_df.iterrows():
            if type(r['Colour']) == type((1,2)):
                # Raise error if there is not exactly three entries in RGB tuple
                if len(r['Colour']) != 3: raise ValueError('For custom colour input in RGB form, there must be three values e.g. (0.2, 0.8, 0.2). You have input: {}'.format((r['Colour'])))
                
                for item in r['Colour']:
                    # Raise error if RGB tuple has invalid entries
                    if type(item) not in {type(1), type(1.1)}: raise ValueError('For custom colour input in RGB form, all values must be integers or floats. You have input: {}'.format(r['Colour']))
                    if item < 0 or item > 1: raise ValueError('For custom colour input in RGB form, all values must be between 0 and 1. You have input: {}'.format(r['Colour']))
            
            # Raise error if color name is not recognised by matplotlib.
            # List available from either: print(mcolors.CSS4_COLORS.keys()) or https://matplotlib.org/stable/gallery/color/named_colors.html
            elif type(r['Colour']) == type('str'):
                intervals_df.at[i, 'Colour'] = r['Colour'].lower()
                if intervals_df.at[i, 'Colour'] not in set(mcolors.CSS4_COLORS.keys()): raise ValueError('The colour \"{}\" is not a recognised colour in matplotlib'.format(intervals_df.at[i, 'Colour']))
    
    # If detailed interval names are requesed
    if detail == True:
        # Standard units for various data types
        units_dict = {'WindDirection': u'\N{DEGREE SIGN}',
                      'WindSpeed': 'm/s',
                      'WindGust': 'm/s',
                      'SeaLevelPressure': 'hPa', 
                      'DryBulbTemperature': u'\N{DEGREE SIGN}C',
                      'WetBulbTemperature': u'\N{DEGREE SIGN}C',
                      'DewPointTemperature': u'\N{DEGREE SIGN}C', 
                      'RelativeHumidity': '%',
                      'Rain': 'mm',
                      'RainIntensity': 'mm/min',
                      'RainCumulative': 'mm',
                      'CloudHeight': 'm',
                      'Visibility': 'm',
                      'WindType': '',
                      'CloudOktas': '',
                      'UTCI': u'\N{DEGREE SIGN}C',
                      'MRT': u'\N{DEGREE SIGN}C'}
     
        # initialise the variable based o nthe data column
        try:
            unit = units_dict[column]
        except:
            unit = ''
        
        # Loops through the intervals dataframe and turn the names to datailed names
        for i, r in intervals_df.iterrows():
            name_1 = '{}, '.format(r['Name'])
            
            # For when the lower bound is negative infiniy
            if r['Range'][0] == np.NINF:
                name_2 = 'Below {}{}'.format(r['Range'][1],unit)
            
            # For when the upper bound is positive infinity
            elif r['Range'][1] == np.inf:
                name_2 = 'Above {}{}'.format(r['Range'][0],unit)
            
            # For all other intervals
            else:
                name_2 = '{}{} to {}{}'.format(r['Range'][0],unit,r['Range'][1],unit)
            name_detailed = name_1 + name_2
            intervals_df.at[i,'Name'] = name_detailed
    
    return intervals_df


#%% INTERVAL BINS
def interval_bins(intervals_df):
    '''
    Looks at the ranges in a interval dataframe and converts to a bins list

    Parameters
    ----------
    intervals_df : pandas.Dataframe or Dict
        A dataframe of intervals. If a unprocessed interval dictionary is passed
        then it will be converted to an interval dataframe.
        
    Returns
    -------
    Returns the original dataframe with the new category column 
    '''
    # Converts interval dicionary to dataframe if needed
    if not isinstance(intervals_df, pd.DataFrame):
        intervals_df  = intervals_dataframe(intervals_df)

    # Create a list of all interval range values
    tuples = list(intervals_df['Range'])
    bins = []
    for i in tuples:
        bins.append(i[0])
    bins.append(tuples[-1][1])

    return bins


#%% INTERVAL CATEGORISER 
def categorise_data(data, intervals_df, column):
    '''
    Applies the ranges defined in an intevals dataframe to an imported dataframe

    Parameters
    ----------
    data : pandas.Dataframe
        An imported weather dataframe
    
    intervals_df : pandas.Dataframe or Dict
        A dataframe of intervals. If a unprocessed interval dictionary is passed
        then it will be converted to an interval dataframe.
        
    column : String
        Specifies the column to which the categorisation should be applied 

    Returns
    -------
    df : pandas.DataFrame
        Returns the original dataframe with the new category column 
    '''
    df = data.copy(deep = True)
    
    # Converts interval dicionary to dataframe if needed
    if not isinstance(intervals_df, pd.DataFrame):
        intervals_df  = intervals_dataframe(intervals_df, column, detail=True)
    
    # Creates the bins list 
    bins = interval_bins(intervals_df)
    
    df[f'{column} Cat'] = pd.cut(df[column], bins=bins, labels=intervals_df['Name'], right=True)
    
    return df


#%% CALCULATE WET BULB TEMPERATURE FROM DEW POINT TEMPERATURE

def safe_GetWBFromRelHumAndDewPoint(tdb, rh, tdp, stnLvlPres):
    try:
        return psychrolib.GetTWetBulbFromHumRatioAndDewPoint(tdb, rh, tdp, stnLvlPres)
    except:
        return np.nan
    
def data_WBFromRelHumAndDewPoint(data, station_altitude=0):
    """
    calculate wet-bulb based on drybulb, relative humidity and dew point
    this uses psychrolib library implemented in weatherpy. 
    **psychrolib documentation**: 
    https://psychrometrics.github.io/psychrolib/api_docs.html

    **psychrolib function list**:
    https://github.com/psychrometrics/psychrolib/blob/master/docs/overview.md

    Parameters
    ----------
    data : pandas.dataframe
        unified data with 'DryBulbTemperature' and 'RelativeHumidity','DewPointTemperature' and 'SeaLevelPressure' columns.

    Returns
    -------
    data_revised : pandas.dataframe
        revised unified data with 'WetBulbTemperature' column added.

    """
    
    df = data.copy()
    
    tdb, tdp, rh, slp = df['DryBulbTemperature'], df['DewPointTemperature'], df['RelativeHumidity'], df['SeaLevelPressure']
    
    psychrolib.SetUnitSystem(psychrolib.SI)
    
    # Get the station level pressure
    Altitude = station_altitude*np.ones_like(tdb)
    GetStationPressure_array = np.vectorize(psychrolib.GetStationPressure)
    stnLvlPres = GetStationPressure_array(slp*100, Altitude, tdb) 
    
    
    safe_GetWBFromRelHumAndDewPoint_array = np.vectorize(safe_GetWBFromRelHumAndDewPoint)
    
    twb = safe_GetWBFromRelHumAndDewPoint_array(tdb, rh, tdp, stnLvlPres)
    
    data['WetBulbTemperature'] = twb
    not_nan = (data['WetBulbTemperature'].notnull())
    data_revised = data[not_nan]
    
    return data_revised


def safe_GetWBFromDP(Tdb, Tdp, P):
    try:
        return psychrolib.GetTWetBulbFromTDewPoint(Tdb, Tdp, P)
    except:
        return np.nan


def data_WB_from_DewPoint(data):
    """
    calculate wet-bulb based on drybulb and dew point
    this uses psychrolib library implemented in weatherpy. 
    **psychrolib documentation**: 
    https://psychrometrics.github.io/psychrolib/api_docs.html

    **psychrolib function list**:
    https://github.com/psychrometrics/psychrolib/blob/master/docs/overview.md

    Parameters
    ----------
    data : pandas.dataframe
        unified data with 'DryBulbTemperature' and 'DewPointTemperature' columns.

    Returns
    -------
    data_revised : pandas.dataframe
        revised unified data with 'WetBulbTemperature' column added.

    """
    # start_time = time.time()
    
    print ("\nUsing psychrolib library to calculate WB temperature based on DB and Dew point")
    
    psychrolib.SetUnitSystem(psychrolib.SI)
    
    twb = np.zeros(len(data['DryBulbTemperature']))
    
    # GetWBFromDP_array = np.vectorize(safe_GetWBFromDP) 
    # data['WetBulbTemperature']  = GetWBFromDP_array(data['DryBulbTemperature'],data['DewPointTemperature'],data['SeaLevelPressure']*100)
    
    for i in range(len(data['DryBulbTemperature'])):
        # print(data.index[i])
        tdb = data['DryBulbTemperature'][i]
        tdp = data['DewPointTemperature'][i]
        if not np.isnan(data['SeaLevelPressure'][i]):
            pressure = data['SeaLevelPressure'][i]*100
        else:
            pressure = 100000
            
        if tdb < tdp:
            twb[i] = np.nan
        else:
            try:
                # GetWBFromDP_array = np.vectorize(psychrolib.GetTWetBulbFromTDewPoint) 
                # twb[i] = psychrolib.GetTWetBulbFromTDewPoint(tdb,tdp,pressure)
                twb[i] = psychrolib.GetTWetBulbFromTDewPoint(tdb,tdp,pressure)
            except:
                twb[i]= 1
   
    data['WetBulbTemperature'] = twb
    not_nan = (data['WetBulbTemperature'].notnull())
    data_revised = data[not_nan]
    
    # print('{} sec to calculate WB temperature'.format(time.time() - start_time))
    return data_revised

def data_WB_from_RelHum(data):
    """
    calculate wet-bulb based on drybulb and dew point
    this uses psychrolib library implemented in weatherpy. 
    psychrolib documentation: 
    https://psychrometrics.github.io/psychrolib/api_docs.html

    psychrolib function list:
    https://github.com/psychrometrics/psychrolib/blob/master/docs/overview.md

    Parameters
    ----------
    data : pandas.dataframe
        unified data with 'DryBulbTemperature' and 'RelativeHumidity' columns.

    Returns
    -------
    data_revised :  pandas.dataframe
        revised unified data with 'WetBulbTemperature' column added.

    """
    # start_time = time.time()

    from ..libs import psychrolib
    
    print ("\nUsing psychrolib library to calculate WB temperature based on DB and RH")
    
    psychrolib.SetUnitSystem(psychrolib.SI)
    
    twb = np.zeros(len(data['DryBulbTemperature']))
    
    #Correct if it is presented in % format
    if data['RelativeHumidity'].mean()>1:
        RelHum = data['RelativeHumidity']/100
    else:
        RelHum = data['RelativeHumidity']
        
    for i in range(len(data['DryBulbTemperature'])):
        print(data.index[i])
        tdb = data['DryBulbTemperature'][i]
        rh = RelHum[i]

        if not np.isnan(data['SeaLevelPressure'][i]):
            pressure = data['SeaLevelPressure'][i]*100
        else:
            pressure = 100000
        
        twb[i] = psychrolib.GetTWetBulbFromRelHum(tdb, rh, pressure)
   
    data['WetBulbTemperature'] = twb
    not_nan = (data['WetBulbTemperature'].notnull())
    data_revised = data[not_nan]
    
    # print('{} sec to calculate WB temperature'.format(time.time() - start_time))
    return data_revised

def data_RelHum_from_DewPoint(data, drybulb_col, dewpoint_col):
    """
    calculate relative humidity based on drybulb and dew point
    this uses psychrolib library implemented in weatherpy. 
    psychrolib documentation: 
    https://psychrometrics.github.io/psychrolib/api_docs.html

    psychrolib function list:
    https://github.com/psychrometrics/psychrolib/blob/master/docs/overview.md

    Parameters
    ----------
    data : pandas.dataframe
        unified data with 'DryBulbTemperature' and 'DewPointTemperature' columns.
    drybulb_col : pandas.Series
        The DryBulbTemperature column from a data import dataframe
    dewpoint_col : pandas.Series
        The DewPointTemperature column from a data import dataframe

    Returns
    -------
    data_revised :  pandas.dataframe
        revised unified data with 'RelativeHumidity' column added.

    """
    # start_time = time.time()
    # import psychrolib
    # from weatherpy.libs import psychrolib
    
    print ("Using psychrolib library to calculate RH temperature based on DB and DP")
    
    psychrolib.SetUnitSystem(psychrolib.SI)
    
    rh = np.zeros(len(data[drybulb_col]))
    
    #Correct if it is presented in % format
    
        
    for i in range(len(data[drybulb_col])):
        tdb = data[drybulb_col][i]
        tdp = data[dewpoint_col][i]
        
        rh[i] = psychrolib.GetRelHumFromTDewPoint(tdb, tdp)
   
    data['RelativeHumidity'] = rh*100
    # not_nan = (data['WetBulbTemperature'].notnull())
    # data_revised = data[not_nan]
    
    # print('{} sec to calculate WB temperature'.format(time.time() - start_time))
    return data

# %% INTERVAL CONVERTER

def interval_converter(
        input_data,
        timeframe = 'hourly',
        aggregate_function = 'mean',
        months_in_quarter = {1:[12,1,2],2:[3,4,5],3:[6,7,8],4:[9,10,11]}
        ):
    
    """
    
    Groups numerical column data of input data according to timeframe.
    
    ----------
    Parameters
    ----------
    
    input_data : dataframe
        Cleaned BOM weather data
    
    timeframe : string
        Determines what timestamps to aggregate data as. Includes first timestamp in the period,
        e.g if timestamps are 1600, 1630, 1700, the first two timestamps are included in 1600 group.
        Default is 'hourly'. Alternatively can choose 'daily', 'monthly', or 'yearly'.
    
    aggregate_function : string, numpy or scipy function
        Determines the numerical operation carried out on dataframe columns to group data.
        Default is 'mean'. Alternatively can use numpy functions such as np.std.  
        Can accept custom defined functions such as 'max - mean' and 'max - 75'
        
    months_in_quarter : dictionary
        Default is {1:[12,1,2],
                    2:[3,4,5],
                    3:[6,7,8],
                    4:[9,10,11]}
        Keys of the dictionary determine each quarter,
        Values in list determine month of year that belong to the quarter.
        1:[12,1,2] means that the 12th, 1st, and 2nd month of the year belong
        to the first quarter of the year.
    
    ----------
    Returns
    ----------
    
    grouped : dataframe
        pandas dataframe grouped according to selected timeframe and aggregate function.
        Wind Direction column is always grouped according to mean.
    
    """
    
    df = input_data.copy()

    ## Add columns to dataframe to allow timeframe filters

    df['year, month'] = df.index.strftime("%Y, %b")
    df['year'] = df.index.strftime("%Y")
    df['year, week'] = df.index.strftime("%Y, %W")
    df['date, hour'] = df.index.strftime("%Y-%m-%d, %H")
    df['date'] = df.index.date
    df['year, week'] = df['year, week'].replace('1995, 00', '1995, 01')
    
    if timeframe == 'quarterly':
        years = df.index.year - df.iloc[0].name.year
        months = df.index.month
        years_mon = years.astype(str) + ', ' + months.astype(str)
        df['year, month'] = years_mon
    
        months_in_quarter = months_in_quarter
        conds = []
        choice = []
    
        year_ints = years.value_counts().index.sort_values()
        period_list = []
        for year in year_ints:
            for i,v in enumerate(months_in_quarter.values()):
                new_list = []
                for j in v:
                    new_list.append(f'{year}, {j}')
                period_list.append(new_list)
    
        for i,v in enumerate(period_list):
            conds.append(df['year, month'].isin(v))
            choice.append(i)
    
        quarters = np.select(conds,choice)
        df['quarter'] = quarters
    
    timeframe_dict = {
        'hourly': ('date, hour','1H'),
        'daily': ('date','D'),
        'monthly': ('year, month', 'MS'),
        'quarterly':('quarter','Q'),
        'yearly':('year','A'),
        'weekly': ('year, week', 'W'),
        }
    
    ## Define circular mean function to calculate mean wind direction
    
    def circular_mean(x, dirs):
        return round(np.rad2deg(circmean(np.deg2rad(x[f'{dirs}'].values))),2)
    
    ## Currently weekly timeframe does not work
    ## Calculate mean wind direction using circular mean
    
    if timeframe == 'weekly':
        mean_winddir = df.resample(rule = timeframe_dict[f'{timeframe}'][1]).apply(circular_mean,'WindDirection')
        mean_winddir = mean_winddir.dropna()
    else:
        mean_winddir = df.resample(rule = timeframe_dict[f'{timeframe}'][1]).apply(circular_mean,'WindDirection')
        mean_winddir = mean_winddir.dropna()
    
    
    ## Define a percentile function that for use in custom function
    
    def percentile(n):
        def percentile_(x):
            return np.percentile(x, n)
        percentile_.__name__ = 'percentile_%s' % n
        return percentile_
        
    ## groupby according to aggregate function
    
    if aggregate_function == 'max - mean':
        df_1 = df.groupby(timeframe_dict[f'{timeframe}'][0]).agg('max')
        df_2 = df.groupby(timeframe_dict[f'{timeframe}'][0]).agg('mean')
        grouped_data = df_1 - df_2
    elif aggregate_function == 'max - 75':
        df_1 = df.groupby(timeframe_dict[f'{timeframe}'][0]).agg('max')
        df_2 = df.groupby(timeframe_dict[f'{timeframe}'][0]).agg(percentile(75))
        grouped_data = df_1 - df_2
    else:
        grouped_data = df.groupby(timeframe_dict[f'{timeframe}'][0]).agg(aggregate_function, numeric_only=True)
       
    ## Organize index of grouped data and add mean wind direction column    
    
    grouped_data.index = pd.to_datetime(grouped_data.index)
    grouped_data= grouped_data.sort_index()
    
    grouped_data['WindDirection'] = mean_winddir
        
    return grouped_data


# %% MODULE DOCUMENTATION

def module_documentation(path=None):
    '''
    Searches weatherpy directory and returns documentation of all functions written.
    
    Parameters
    ----------
    
    path : string
        Default is None.
        Determines where weatherpy is installed.
    
    Returns
    ----------
    wp_doc_df : pandas dataframe
        dataframe of documentation, with function names as index
    
    '''
    
    
    if path == None:
        import weatherpy as wp
        print('Installed weatherpy is used')
    else:
        sys.path.insert(0, path)
        import weatherpy as wp
        print('Local weatherpy found')

    funcs_dict = {}
    for func in dir(wp):
        is_func = inspect.isfunction(getattr(wp,func))
        if not func.startswith('__'):
            if is_func:
                path = os.path.abspath(inspect.getfile(getattr(wp,func)))
                if 'weatherpy' not in path:
                    pass
                split_path = path.split('weatherpy')[-1].split('.py')[0].split('\\')[1:3]
                docu = getattr(wp, func).__doc__
                if docu is None:
                    docu = 'No Documentation written'
                else:
                    funcs_dict.update({f'{split_path},{func}':docu})
                
    wp_doc_df = pd.DataFrame(data={'Documentation':funcs_dict.values()},index = funcs_dict.keys())
    wp_doc_df['index_drop'] = wp_doc_df.index
    wp_doc_df[['path','function name']] = wp_doc_df['index_drop'].str.split(']',expand=True)
    wp_doc_df[['folder','file']] = wp_doc_df['path'].str.split(',',expand=True)
    wp_doc_df[['comma','function name']] = wp_doc_df['function name'].str.split(',',expand=True)
    wp_doc_df = wp_doc_df.set_index('function name')
    wp_doc_df[['[','folder']] = wp_doc_df['folder'].str.split("[",expand=True)
    wp_doc_df[["'",'folder',"'"]] = wp_doc_df['folder'].str.split("'",expand=True)
    wp_doc_df[["'",'file',"'"]] = wp_doc_df['file'].str.split("'",expand=True)
    wp_doc_df = wp_doc_df[['folder','file','Documentation']]

    return wp_doc_df

def documentation_to_console(function = None,path=None):
    '''
    Print to console and save a string of weatherpy documentation
    
    Parameters
    ----------
    
    function : string
        Default is None. 
        Weatherpy function documentation sought
        
    path : string
        Default is None.
        Determines where weatherpy is installed.
        
    Returns
    ----------
    docu_string : str, 
        default NoneType, if no function is entered.
        String print of documentation.
    
    '''
    
    if path == None:
        import weatherpy as wp
        # print('Installed weatherpy is used')
    else:
        sys.path.insert(0, path)
        import weatherpy as wp
        # print('Local weatherpy found')
    
    if function is None:
        print('No function is chosen')
        # function = wp_functions.loc[wp_functions.sample().index]
        docu_string = None
    else:
        try:
            print(getattr(wp,function).__doc__)
            docu_string = getattr(wp,function).__doc__
        except AttributeError:
            print("Either the function does not exist, or the name was mispelt.\nType exactly the function name, e.g 'query_range', not 'query range'")
            docu_string = None
        
    return docu_string

#%% Find the closest point and its index to a target location
def find_closest_index(coordinates, target_point):
    """
    This function find the index and coordinates of the closest point in a array
    of points to a target point

    Parameters
    ----------
    coordinates : np.array
        The array of points with rows as points number and columns as x, y , z coordinates
    target_point : tuple of three numbers for x, y, z direction.
        target point.

    Returns
    -------
    idx : int
        index of the closest point in the array to the target point.
    pt : tuple of three numbers for x, y, z direction.
        The closest point in the array to the target point.

    """
    # Calculate the Euclidean distance between each coordinate and the target point
    distances = np.linalg.norm(coordinates - target_point, axis=1)

    # Find the index of the minimum distance
    idx = np.argmin(distances)
    pt = coordinates[idx]
    return idx, pt


def filter_points(coordinates, axis, lower=None, upper=None):
    """
    This function filters points based on lower or upper values of the x, y, or z coordinate.

    Parameters
    ----------
    coordinates : np.array
        The array of points with rows as points number and columns as x, y, z coordinates.
    axis : str
        The coordinate type to filter on. Valid values are 'x', 'y', or 'z'.
    lower : float or None, optional
        The lower value for the coordinate filter. If None, no lower bound is applied.
    upper : float or None, optional
        The upper value for the coordinate filter. If None, no upper bound is applied.

    Returns
    -------
    filtered_indices : np.array
        The indices of the filtered points in the original array.
    filtered_coordinates : np.array
        The filtered array of points based on the coordinate filter.

    """
    # Get the column index based on the coordinate type
    if axis == 'x':
        col_idx = 0
    elif axis == 'y':
        col_idx = 1
    elif axis == 'z':
        col_idx = 2
    else:
        raise ValueError("Invalid coordinate type. Valid values are 'x', 'y', or 'z'.")

    # Apply the coordinate filter
    if lower is not None and upper is not None:
        mask = (coordinates[:, col_idx] >= lower) & (coordinates[:, col_idx] <= upper)
    elif lower is not None:
        mask = coordinates[:, col_idx] >= lower
    elif upper is not None:
        mask = coordinates[:, col_idx] <= upper
    else:
        mask = np.full(len(coordinates), True)  # No filter, select all points

    filtered_indices = np.where(mask)[0]
    filtered_coordinates = coordinates[filtered_indices]

    return filtered_coordinates, filtered_indices

def rotate_point(point, rotation):
    
    x, y, z = point
    rotation_x, rotation_y, rotation_z = rotation

    # Convert rotation angles to radians
    rotation_x = math.radians(rotation_x)
    rotation_y = math.radians(rotation_y)
    rotation_z = math.radians(rotation_z)

    # Rotate around x-axis
    cos_x = math.cos(rotation_x)
    sin_x = math.sin(rotation_x)
    y_rotated = y * cos_x - z * sin_x
    z_rotated = y * sin_x + z * cos_x

    # Rotate around y-axis
    cos_y = math.cos(rotation_y)
    sin_y = math.sin(rotation_y)
    x_rotated = x * cos_y + z_rotated * sin_y
    z_rotated = -x * sin_y + z_rotated * cos_y

    # Rotate around z-axis
    cos_z = math.cos(rotation_z)
    sin_z = math.sin(rotation_z)
    x_rotated_final = x_rotated * cos_z - y_rotated * sin_z
    y_rotated_final = x_rotated * sin_z + y_rotated * cos_z

    pt_rotated = (x_rotated_final, y_rotated_final, z_rotated)
    return pt_rotated


def spdLocal_to_10tc2(spd_local_over_ref, Mzcat_ref_over_10tc2, numDirs, wind_spd_ranges):
    """
    Calculate wind speed at 10m TC2 which causes certain local wind speed for all directions.

    Parameters
    ----------
    spd_local_over_ref : list
        list of ratios of maximum local wind speed over reference wind speed 
        from wind tunnel testing or CFD modelling for all direction. 
        each element of the list corresponds to certain wind directions
    
    Mzcat_ref_over_10tc2 : int or 1D array
        Mzcat of reference to 10m, TC2. Length should corresponds to number of wind directions.
        If integer is used, the same value will be used for all directions.
    
    numDirs : int
        number of wind directions.
    
    wind_spd_ranges : list
        list of local wind speeds to consider. 
        It is best to usually set this to cover all wind speeds (e.g. list(np.r_[0:45]))

    Returns
    -------
    spd10tc2 : pandas.dataframe
        Wind speeds that occur at 10m TC2 resulting in 
        certain local wind speed (columns) for each wind direction (rows).

    """
    
    # check if single value is passed for all directions or 
    # an array like list is passed containing correction factors for all directions
    if type(Mzcat_ref_over_10tc2)!=list or type(Mzcat_ref_over_10tc2)!=np.ndarray:
       Mzcat_ref_over_10tc2_dirs  = Mzcat_ref_over_10tc2*np.ones(numDirs)
    else:
       Mzcat_ref_over_10tc2_dirs = Mzcat_ref_over_10tc2
       
    spd_local_over_10tc2 = spd_local_over_ref*Mzcat_ref_over_10tc2_dirs
    
    windDirs = np.linspace(0, 360, numDirs+1)[0:-1]
    spd10tc2 = pd.DataFrame(index=wind_spd_ranges, columns=windDirs)
    spd10tc2.index.name = 'local windSpd'
    for s in range(len(wind_spd_ranges)):
        spd10tc2.iloc[s,:] =  wind_spd_ranges[s]/spd_local_over_10tc2
        
    return spd10tc2
