import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from ..analysis.statistics import data_quantile
import pytz
from datetime import datetime, timedelta

def clean_data(
    data,
    dataType = 'BOM',
    clean_ranked_rows = False,
    clean_VC_filter = False,
    clean_calms = False,
    clean_direction = True,
    clean_off_clock = False,
    clean_storms = True,
    clean_invalid = True,
    clean_threshold = True, 
    col2valid = ['WindSpeed', 'WindDirection', 'WindType'], 
    thresholds = {
        'WindSpeed': (0,50),
        'DryBulbTemperature': (-25,55),
        'PrePostRatio': (5, 30)}):
    
    """
    Clean data. remove nan data and outliers.
    Performs adjustments to wind data for timestamps with calm winds.
    Rounding wind direction to nearest 10 degrees.

    Parameters
    ----------
    data : pandas.dataframe
        Imported dataframe to be cleaned. The data should have unified columns
        (i.e you should use the unify_datatype() function before cleaning it).
    
    dataType : str, optional
        Data type to use: BOM, NOAA, or custom. Note that there are some
        cleaning methods that work differently depending on the data type. These
        will be noted in the following parameter descriptions. Default is "BOM".
    
    clean_VC_filter : Boolean, Optional
        Applies a special cleaning method to NOAA dataframes for removing
        duplicate timestamps and converting to data with a frequency of one
        hour. The default is False.
    
    clean_calms : Boolean, optional
        Remove calm winds. Default is False.
         
    clean_direction : Boolean, optional
        Remove NOAA timestamps if wind directions are not a multiple of 10°.
        NOAA stations only report in 10° increments and all other readings are
        considered to be misreadings. The default is True. 
    
    clean_off_clock : Boolean, optional
        Only applied to BOM data. Removes timestamps from before 01/Sep/2017 if
        they do not fall on an hour or half-hour since they are associated with
        extreme weather events. The default is False.
    
    clean_thunderstorms : Boolean, optional
        Only applied to NOAA data. Removes timestamps if a thunderstorm is
        indicated by the MW1_0 column. The default is False
    
    clean_invalid : Boolean, optional
        A True/False flag to indicate whether certain columns should be cleaned
        for invalid values. If set to True, the columns to check are defined as
        a list in the col2valid variable. Default is False.
    
    clean_threshold : Boolean, optional
        A True/False flag to indicate whether certain columns should be cleaned
        based on user defined valid thresholds. If set to True, the columns to
        check are defined as a dictionary in the thresholds. It is recommended 
        that if a column is being threshold cleaned, it should also be invalid
        cleaned (clean_invalids = True). Default is False. 
    
    col2valid : list, optional
        A list of columns that will be checked for invalid entries. When an
        invalid entry is found, that row is deleted. Note if a selected column
        contains only invalid values, it will not be cleaned as this would
        result no data remaining. Default is ['WindSpeed', 'WindDirection'].
    
    thresholds : dictionary, optional
        if clean_threshold = True, then data will be cleaned in accordance with
        acceptable ranges. The dict is in the form: {'column':(lower,upper)}.
        
    Returns
    -------
    data_cleaned : pandas.DataFrame
        The data that passed though all the cleaning methods.
        
    data_removed : dictionary
        A dictionary with reasons for removal and removed timestamps.

    data_calm : pandas.DatFrame
        A dataframe for just calm wind speeds.
    """
    
    print('Cleaning Data')
    
    # Initilise important variables
    data = data.copy(deep = True)
    
    l0 = data.shape[0]
    data_columns = list(data.columns)
    valid_ranges = {'WindDirection': (0, 360),
                    'WindSpeed': (0, 150),
                    'WindGust': (0, 150),
                    'SeaLevelPressure': (0, 1200),
                    'DryBulbTemperature': (-90, 60),
                    'WetBulbTemperature': (-90, 60),
                    'DewPointTemperature': (-90, 60),
                    'RelativeHumidity': (0,100),
                    'Rain': (0, 900),
                    'RainIntensity': (0, 50),
                    'CloudOktas': (0, 8),
                    'WindType': ['V', 9]}

    # Create a list of all columns of validation columns
    if clean_invalid == True and col2valid == 'all':
        col2valid = data_columns
    elif clean_invalid == False:
        col2valid = None
    
    # Creates a list of threshsold check columns
    if clean_threshold == True:
        threshold_cols = list(thresholds.keys())
        threshold_cols = [x for x in threshold_cols if x in data_columns]
    else:
        threshold_cols = None

    # Combines the two lists of columns of interest
    if col2valid is not None and threshold_cols is not None:
        cols_of_interest = list(set(col2valid) | set(threshold_cols))
    elif col2valid is not None:
        cols_of_interest = col2valid
    elif threshold_cols is not None:
        cols_of_interest = threshold_cols
    else:
        cols_of_interest = []

    # Check the columns of interest for empty columns
    if clean_invalid == True or clean_threshold == True:
        empty_cols = []
        empty_col_str = '\tThe following column(s) contain no data and won\'t be cleaned:\n'
        for i in cols_of_interest:
            if set(pd.isnull(data[i])) == {True}:
                empty_cols.append(i)
                print(empty_col_str + f'\t  - \"{i}\"')
                empty_col_str = ''


    # Initialise removed datarame dictionary
    data_removed = {}
    special_clean = {}


    # VC Filter
    if clean_VC_filter == True and dataType == 'NOAA':
        data = VC_filter(data)
        if clean_ranked_rows == True:
            print('\tVC method supersedes remove ranked rows')
    
    # Removed Ranked Rows
    elif clean_ranked_rows == True and dataType == 'NOAA':
        l1 = data.shape[0]
        data, ranked_removed = remove_ranked_rows(data)
        l2 = data.shape[0]
        print(f'\tClean duplicate rows: {l1-l2} timestamps ({round((l1-l2)/l0*100,1)}%) are removed')
        
        special_clean['Ranked Duplicates'] = ranked_removed


    # Clean Calms 
    if dataType == 'BOM':
        # Rounding directions less than 5 to 360 rather than 0/calm
        filt = (data['WindDirection']<=5) & (data['WindSpeed']>0)
        data.loc[filt,'WindDirection'] = 360
        print('\tWindDirections between 0° & 5° (inclusive) are set to 360°')
        # Rounding wind direction to nearest 10 degree
        data['WindDirection'] = data['WindDirection'].round(-1)
        print('\tWindDirection is rounded to nearest 10°')
        
        filt = (data['WindSpeed'] == 0)
        data.loc[filt,'WindDirection'] = 0
        print('\tWindDirection when wind speed is calm is set to 0°')

    elif dataType == 'NOAA':
        # Correct wind direction = 999 to 0
        data.loc[data['WindType'] == 'C', 'WindDirection'] = 0
        print('\tConverted WindDirection to 0° when WindType is \"C\"')
        
        # Clean wind directions 
        # https://forecast.weather.gov/glossary.php?word=wind%20direction
        # wind direction = 0 is calm
        if clean_direction == True:
            l1 = data.shape[0]
            data, data_non_mult, _ = remove_WD_notmult(data, multiple=10)
            l2 = data.shape[0]
            print(f"\tClean wind directions not multiples of 10°: {l1-l2} timestamps ({round((l1-l2)/l0*100,1)}%)")
            
            special_clean['WD not multiple of 10'] = data_non_mult
    
    
    # Clean Thunderstorms
    if clean_storms == True and dataType == 'NOAA':
        l1 = data.shape[0]
        data, data_out = clean_thunderstorms(data)
        l2 = data.shape[0]
        special_clean['Thunderstorms'] = data_out
        print(f'\tClean thunderstorms NOAA: {l1-l2} timestamps ({round((l1-l2)/l0*100,1)}%) are removed')
    
    
    # Clean off clock
    if clean_off_clock == True and dataType == 'BOM':
        l1 = data.shape[0]
        data, storm_data_removed = clean_bom_historic_offClock(data)
        l2 = data.shape[0]
        special_clean['Off Clock'] = storm_data_removed
        print(f'\tClean off-clock BOM: {l1-l2} timestamps ({round((l1-l2)/l0*100,1)}%) are removed')
    
    data_removed['Special Cleaned Timestamps'] = special_clean
    
    
    # Clean columns of invlaid values
    if clean_invalid == True:
        # Initialise shape and mask series
        l1 = data.shape[0]
        valids_global = pd.Series(index=data.index, dtype=bool)
        invalid_removed = {}
        
        for i in col2valid:  
            # Does not validate columns with no data
            if i not in empty_cols and i in valid_ranges.keys():

                # Initialise local mask
                valids_local = pd.Series(index=data.index, dtype=bool)

                # Remove NaN readings
                valids_local &= data[i].notnull()
                valid_ranges_value = valid_ranges[i]
                
                # If the valid ranges value is a tuple, this is an allowable range
                if isinstance(valid_ranges_value, tuple):
                    valids_local &= (data[i] >= valid_ranges_value[0]) & (data[i] <= valid_ranges_value[1])
    
                # If the valid ranges value is a list, this is a list of invalid values
                elif isinstance(valid_ranges_value, list):
                    for n in valid_ranges_value:
                        valids_local &= data[i] != n
    
                invalid_removed[i] = data[-valids_local]
                
                valids_global &= valids_local
    
        # Applies validity mask to dataframe
        data = data[valids_global]
        data_removed['Invalid Timestamps'] = invalid_removed

        # Calculates new data size and notifies user 
        l2 = data.shape[0]
        print(f'\tClean invalid: {l1-l2} timestamps ({round((l1-l2)/l0*100,1)}%) are removed')
    

    # Clean columns based on threshold values 
    if clean_threshold == True:
        # Initialise shape and mask series
        l1 = data.shape[0]
        valids_global = pd.Series(index=data.index, dtype=bool)
        threshold_removed = {}

        # iterrates through each of the threshold keys
        for i in thresholds.keys():
            valids_local = pd.Series(index=data.index, dtype=bool)
            
            # Performs a special clean for 'Hours' threshold (NOAA only)
            if i == 'Hours':
                # I think this should be a query
                pass
            
            elif i == 'WindSpeed' and dataType == 'NOAA':
                valids_local = clean_threshold_NOAA(data,max(thresholds['WindSpeed']))
                threshold_removed['WindSpeed'] = data[-valids_local]
                
            # Performs a special clean for 'PrePostRatio' threshold
            elif i == 'PrePostRatio':
                valids_local = clean_ratio(data, 
                                           ratio = thresholds['PrePostRatio'][0],
                                           threshold = thresholds['PrePostRatio'][1])
                
                threshold_removed['Ratio'] = data[-valids_local]
                
            # Performs a threshold clean on data columns
            elif i not in empty_cols:
                valids_local &= (data[i] >= thresholds[i][0]) & (data[i] <= thresholds[i][1])
                threshold_removed[i] = data[-valids_local]
            
            valids_global &= valids_local
    
        # Applies validity mask to dataframe
        data = data[valids_global]
        data_removed['Outside Threshold Timestamps'] = threshold_removed
        
        # Calculates new data size and notifies user 
        l2 = data.shape[0]
        print(f'\tClean threshold: {l1-l2} timestamps ({round((l1-l2)/l0*100,1)}%) are removed')
    
    
    # Create a dataframe of just calms without removing them 
    if clean_calms:
        l1 = data.shape[0]
        data, data_calm = isolate_calms(data, dataType, remove = True)
        l2 = data.shape[0]
        print(f'\tClean calm wind: {l1-l2} timestamps ({round((l1-l2)/l0*100,1)}%) are removed')
        
        data_removed['Special Cleaned Timestamps']['Calm Wind Conditions'] = data_calm
    
    else:
        data_calm = isolate_calms(data, dataType, remove = False)

    
    # Final size calculation
    l_final = data.shape[0]
    print(f'\n\tTotal timestamps removed: {l0-l_final} ({round((l0-l_final)/l0*100,1)}%)')
    print('\t--Cleaning Complete--\n')
    

    # Return cleaned data, all removed data, and the calm timestamps
    return data, data_removed, data_calm
  

#%% VC FILTER

def choose_rank(x):
    idx = np.argmax(x['Rank'])
    return x.iloc[idx]

def adjust_hour_past30min(x):
    if (30 < x.minute < 60):
        x += pd.DateOffset(minutes=60-(x.minute % 30)*2)
    return x

def rank_closest_to_hour(x):
    minute = x.minute
    if (0 <= minute <= 10) or (50 <= minute <= 59):
        rank_quality = 1
    elif (10 < minute <= 20) or (40 < minute < 50):
        rank_quality = 0.5
    elif (20 < minute <= 40):
        rank_quality = 0
    else:
        rank_quality = 0
    return rank_quality

def rank_quality(x):
    if x not in [2, 3, 6, 7]:
        rank_quality = 1
    else:
        rank_quality = 0
    return rank_quality

def rank_reporttype(x):
    if x in ['S-S-A','SA-AU','SY-AE','SY-AU','SY-MT']:
        rank_quality = 1
    else:
        rank_quality = 0
    return rank_quality

def VC_filter(input_data):
    """
    Visual Crossing filter to keep only a single data entry per provided hour.
    Details of the filter are provided here:
    https://www.visualcrossing.com/resources/documentation/weather-data/how-we-process-integrated-surface-database-historical-weather-data/

    Parameters
    ----------
    input_data : pandas dataframe
        data to be filtered. Please ensure outliers have already been removed

    Returns
    -------
    data_cleaned_VC : pandas dataframe
        filtered dataframe using VC method. Only one data entry per provided hour

    """    
    index_type = input_data.index.name
    
    try:
        tz = input_data.index[0].tzinfo

    except:
        print('\tIndex of raw data is not timezone aware! Assuming UTC')
        tz = 'UTC'
        
        
    try:
        # if (data['WindSpeed'] >= 100).any():
        #     raise Exception("Please remove all outlier wind speeds (greater than 100 m/s) before applying the VC filter!")
        # elif (data['WindDir'] > 360).any():
        #     raise Exception("Please remove all outlier wind directions (greater than 360 degrees) before applying the VC filter!")
            
        data = input_data.copy()
        
        start_year = data.index.year.min()
        end_year = data.index.year.max()
    
    
        #"""
        #Move information beyond the 30 minute mark to the next hour
        #"""
        data.index = data.index.map(adjust_hour_past30min)
    
        #"""
        #Rank data based on how far away it is from the hour
        #"""
        data['RK_HR'] = data.index.map(rank_closest_to_hour)
    
        # """
        # Rank data based on Wind Speed Quality
        # """
        data['RK_QC'] = data['QCWindSpeed'].apply(rank_quality)
        
        # """
        # Rank data based on Report Type Quality (merger of reports is better)
        # """
        data['RK_RT'] = data['ReportType'].apply(rank_reporttype)
    
        # """
        # Add all the ranks together
        # """
        
        data['Rank'] = data['RK_HR'] + data['RK_QC'] + data['RK_RT']
    
        
    
        # make a new column that only has the hour to see which hours are repeated
        data['Hour'] = data.index.strftime('%Y-%m-%d-%H')
        tqdm.pandas(desc='\tVC filtering')
        
        
        # """
        # for each group of duplicated hours, apply the function "choose rank"
        # select the hour that has the highest ranking
        # """
        data_cleaned_VC = data.groupby(by=["Hour"]).progress_apply(choose_rank)
    
        # convert the index to datetime and rename it to what it was originally
        data_cleaned_VC.index = pd.to_datetime(data_cleaned_VC.index)
        
        # make index timezone aware (use infer_dst to prevent pytz from trying 
        # to guess which rows are related to daylight savings time (DST). 
        # i.e., Assume all is daylight time (DT)
        # solution here: https://stackoverflow.com/questions/36757981/python-pandas-tz-localize-ambiguoustimeerror-cannot-infer-dst-time-with-non-d)
        infer_dst = np.array([False] * data_cleaned_VC.shape[0])
        data_cleaned_VC.index = data_cleaned_VC.index.tz_localize(tz,ambiguous=infer_dst)
        
        # rename it to LocalTime
        data_cleaned_VC.index.name = 'LocalTime'

        data_cleaned_VC = data_cleaned_VC[(data_cleaned_VC.index.year >= start_year) & (data_cleaned_VC.index.year <= end_year)]

        removed_count = len(input_data)-len(data_cleaned_VC)
        removed_percentage = round(removed_count/len(input_data)*100,1)
        print(f'\tTimestamps removed by VC filter: {removed_count} ({removed_percentage}%)')
        
    except KeyError:
        print('\tVC rank method is only available for NOAA Data.')
        data_cleaned_VC = input_data.copy()

    # make the correct 'UTC' column using the new VC-filtered timezone aware index
    data_cleaned_VC['UTC'] = data_cleaned_VC.index.tz_convert('UTC')
    
    if index_type == 'UTC':
        data_cleaned_VC.set_index('UTC', inplace=True)
    
    
    return data_cleaned_VC

def remove_ranked_rows(data, 
                       hourly = False,
                       report_types = ['S-S-A','SA-AU','SY-AE','SY-AU','SY-MT', 'FM-15', 'FM-16'], 
                       qc_speeds=[0,1,4,5,9]):
    '''
    Removes duplicate data entries from NOAA data frames
    
    Parameters
    ----------
    data : pandas.DataFrame
        The input data frame to be cleaned.
    
    hourly : bool, optional
        If true, only data near the hour will be considered. The default is False.
    
    report_types : list, optional
        List of NOAA data report types. The default is ['S-S-A','SA-AU','SY-AE','SY-AU','SY-MT', 'FM-15', 'FM-16'].
    
    qc_speeds : list, optional
        list of NOAA QC wind speeds. The default is [0,1,4,5,9].

    Returns
    -------
    data_ranked : pandas.DataFrame
        The cleaned dataframe is returned.

    '''
    
    data_raw_ranked = data.copy(deep = True)
    
    if hourly:
        # extract time information
        data_raw_ranked['year'] = data_raw_ranked.index.year
        data_raw_ranked['month'] = data_raw_ranked.index.month
        data_raw_ranked['day'] = data_raw_ranked.index.day
        data_raw_ranked['hour_original'] = data_raw_ranked.index.hour
        data_raw_ranked['minute'] = data_raw_ranked.index.minute
        data_raw_ranked['hour'] = data_raw_ranked['hour_original']

        # """
        # Rounding up or down the hours that are within 10 minutes to the top of the hour
        # """

        # round up the hour
        id_rndup = (data_raw_ranked['minute'] >= 50) & (data_raw_ranked['minute'] <= 60)
        data_raw_ranked.loc[id_rndup,'hour'] = data_raw_ranked['hour'][id_rndup] + 1

        #---for rounding up the 24th hour, bring it back to 0
        id_rndup_24th_hr = (data_raw_ranked['minute'] >= 50) & (data_raw_ranked['minute'] <= 60) & (data_raw_ranked['hour'] == 23)
        data_raw_ranked.loc[id_rndup_24th_hr,'hour'] = 0

        # round down the hour
        id_rnddwn = (data_raw_ranked['minute'] >= 0) & (data_raw_ranked['minute'] <= 10)
        data_raw_ranked.loc[id_rnddwn,'hour'] = data_raw_ranked['hour'][id_rnddwn]

        # """
        # Remove intra hourly data
        # (i.e. hours that are not within 10 minutes to the top)
        # """

        # hours not within ten minutes of the top of the hour
        id_inbtwn = (data_raw_ranked['minute'] > 10) & (data_raw_ranked['minute'] < 50)

        # assign these data points as "half" hours. Hence the 0.5
        # NOTE: this step is probably not needed but useful for tracking
        data_raw_ranked.loc[id_inbtwn,'hour'] = data_raw_ranked['hour_original'][id_inbtwn] + 0.5

        # only keep the data that has hours within 10 minutes to the top of the hour
        data_raw_ranked = data_raw_ranked[~id_inbtwn]
        
        # """
        # Filter out duplicated hours using the ranking system described in:
        # https://www.visualcrossing.com/resources/documentation/weather-data/how-we-process-integrated-surface-database-historical-weather-data/
        # """

        # find duplicated hours
        # d = pd.to_datetime(data_raw_ranked[['year','month','day','hour']])
        # dup = d.duplicated()
    
    try:
        df_rankings = data_raw_ranked[['QCWindSpeed','ReportType']]
        
        high_mask = (df_rankings['ReportType'].isin(report_types)) & (df_rankings['QCWindSpeed'].isin(qc_speeds))
        rank_high = df_rankings[high_mask]
        
        rank_high['Ranks'] = 2
        mid_mask = (df_rankings['ReportType'].isin(report_types) | df_rankings['QCWindSpeed'].isin(qc_speeds)) & ~high_mask
        rank_mid = df_rankings[mid_mask]
        rank_mid['Ranks'] = 1
        
        rank_mask = mid_mask = (df_rankings['ReportType'].isin(report_types) | df_rankings['QCWindSpeed'].isin(qc_speeds))
        
        rank_zero = df_rankings[~rank_mask]
        df_rankings = pd.concat([rank_mid, rank_high])
        
        rankings_sorted = df_rankings.sort_values('Ranks', ascending=True)
        duplicates = rankings_sorted.index.duplicated(keep='last')
        df_rankings = rankings_sorted[~duplicates]
        dup_no = len(duplicates) - len(df_rankings)
        df_rankings = df_rankings.sort_index()
        
        dup_removed = rankings_sorted[duplicates]
        # all_ranked = rankings_sorted.sort_index()
        all_remove = pd.concat([dup_removed,rank_zero])
        statementl1 = f'Unacceptable report and QC WindSpeed type: {len(rank_zero)}.\nDuplicate timestamps removed: {len(dup_removed)} '
        statementl2 = f'Total timestamps removed by remove_ranked_rows method: {len(all_remove)} '
        # print(f'{statementl1}\n{statementl2}')

        # Join input dataframe with values to remove
        data3 = pd.concat([data_raw_ranked, all_remove])
        
        # Duplicate index column in temporary collumn called tstemp
        data3['tstemp'] = data3.index

        # Create mask to remove all duplicates 
        duplicate_mask = data3.duplicated(subset = ['tstemp', 'ReportType', 'QCWindSpeed'], keep = False)

        # Delete extra columns
        del data3['tstemp'], data3['Ranks']
        
        # Create cleaned dataframe and remove timestamps
        data_ranked = data3[~duplicate_mask]
        rank_removed = data3[duplicate_mask].dropna(subset = [data3.columns[0]])

    except IndexError:
        print('VC rank method is only available for NOAA Data.')
        data_ranked = data
        rank_removed = None
    
    return data_ranked, rank_removed


# %% EVA (WORK IN PROGRESS)
def clean_extremes(data, interval=1.5, quantile=0.99, column='WindSpeed'):
    
    """
    
    Finds timestamps exceeding a input percentile quantile and removes automatically
    based on statistics of timestamps within an interval range of exceeding timestamp.
    
    Fail parameters:
        Standard Deviation fail
            Will fail the timestamp if the SD of the interval exceeds 3.5 x SD of all data
        Before / After mean
            Fails the timestamp if the max of the interval minus the mean of before/after
            exceeds a certain value
    Pass parameters:
        Temp/RH gradient condition
            Passes the point if the relative humidity is increasing while the
            dry bulb temperature decreases.
        Gust value aligns with mean
            Passes the point if the maximum gust value exceeds the maximum mean
            wind speed value.
        All parameters within reason
            Passes the point if no fail or pass conditions are met.
        
    ----------
    Parameters
    ----------
    
    input_data : dataframe
        Cleaned BOM weather data
    
    interval : float
        Number of hours before and after the exceeding timestamp that is considered.
        Since storm events typically last 3 hours, 1.5 hours is good.
    quantile : string
        2 decimal place percentage string, such as '99.90%', '90.00%'.
        Determines at what percentile of column values will be selected.
        Default is '99.90%' which chooses the top 0.1% of timestamps to return
    column : string
        Column of data to base percentile values on. Must be numerical data.
        Default is 'WindSpeed'   
    ----------
    Returns
    ----------
    
    updated_df : dataframe
        dataframe with fail timestamps removed.
    date_df : dataframe
        dataframe containing all exceeding timestamps within chosen interval
    stat_df : dataframe
        dataframe containing exceeding timestamps, pass/fail, pass/fail reason 
        and the statistics used to determine pass/fail.
    
    """

    threshold, dates_of_interest = data_quantile(data, quantile, column)

    print(f'Timestamps with a {column} > {round(threshold,2)} (>{int(quantile*100)}th percentile) are collected as possible misreadings')

    if len(dates_of_interest) == 0:
        print(f'No data is available in {column}')
        return data
    
    ## Calculate consecutive DBT and RH Difference
    data['DBT Difference'] = data['DryBulbTemperature'].diff()
    data['DBT Difference'].iloc[0] = 0
    data['RH Difference'] = data['RelativeHumidity'].diff()
    data['RH Difference'].iloc[0] = 0
    
    doi = dates_of_interest
    doi_intervals = {i: {'data': data.loc[i-timedelta(hours=interval):i+timedelta(hours=interval)]} for i in doi.index}

    ## SD value of whole column, all timestamps
    for k, v in doi_intervals.items():
        
        df = v['data']
        df_col = df[column]
        
        # calculating std for the interval and whole dataset
        stdev = df_col.std()  
        stdev_all = data[column].std()
        
        # mean_before_max
        diff_max_meanBefore = np.abs(df_col.max() - df_col[df_col.index < df_col.idxmax()].mean())
        
        # mean_after_max
        diff_max_meanAfter = np.abs(df_col.max() - df_col[df_col.index > df_col.idxmax()].mean())
        
        # difference between max gust and max mean
        diff_max_gust = df['WindGust'].max() - df['WindSpeed'].max()
        
        # calculating the mean gradient of drybulb temperature and relative humidity
        Tgrad_mean = df['DBT Difference'].mean()
        Hgrad_mean = df['RH Difference'].mean()
        
        # mean rain
        mean_rain = df['Rain'].mean()
        
        doi_intervals[k]['STD'] = df_col.std()
        doi_intervals[k]['Mean'] = df_col.mean()
        doi_intervals[k]['Max - mean (before)'] = diff_max_meanBefore
        doi_intervals[k]['Max - mean (after)'] = diff_max_meanAfter
        doi_intervals[k]['Max - max (of gust)'] = diff_max_gust
        doi_intervals[k]['Mean DB Temperature gradient'] = Tgrad_mean
        doi_intervals[k]['Mean Relative Humidity gradient'] = Hgrad_mean
        doi_intervals[k]['Mean Rain'] = mean_rain
        doi_intervals[k]['STD Ratio (all date)'] = stdev/stdev_all
        doi_intervals[k]['No Timestamps'] = df_col.count()
        
        if (column == 'Rain'):
            rain_condition = (mean_rain >= 0)
            max_mean_medium_val = 12
            max_mean_high_val = 20
        elif (column == 'RainIntensity'):
            rain_condition = (mean_rain >= 0)
            max_mean_medium_val = 3
            max_mean_high_val = 4
        else:
            rain_condition = (mean_rain == 0)
            max_mean_medium_val = 12
            max_mean_high_val = 20
        
        ## Initialize and pass or fail a timestamp based on calculated statistics
                
        if ((diff_max_meanBefore >= max_mean_high_val) | (diff_max_meanAfter >= max_mean_high_val)) & (rain_condition):
            reason = f'max-mean >= {max_mean_high_val}'
            PF_val = 'Fail'
        elif (Tgrad_mean < -0.5) & (Hgrad_mean > 5 ):
            reason = 'Temp/RH gradient condition'
            PF_val = 'Pass'
        elif diff_max_gust >= 0:
            reason = 'Gust value aligns with mean'
            PF_val = 'Pass'
        else:   
            if (diff_max_meanBefore >= max_mean_medium_val) & (diff_max_meanAfter >= max_mean_medium_val) & (rain_condition):
                reason = f'max-mean > {max_mean_medium_val}'
                PF_val = 'Fail'
            elif stdev >= 3.5*stdev_all:
                reason = 'SD fail'
                PF_val = 'Fail'
            else:
                reason = 'All parameters within reason'
                PF_val = 'Pass'
        
        doi_intervals[k]['Pass/Fail'] = PF_val
        doi_intervals[k]['Reason'] = reason
    
    fail_df = {k: v for k, v in doi_intervals.items() if v['Pass/Fail']=='Fail' }
    
    failed_timestamps = []
    if fail_df:
        for k,v in fail_df.items():
            print(f'timestamp {k} is removed dur to:\n {v["Reason"]}.')
            failed_timestamps.append(k)
        cleaned_data = data.drop(failed_timestamps)
    else: 
        print('no timestamps are removed from extreme cleaning.')
        cleaned_data = data
    
    print('extreme values are cleaned')   
    
    return cleaned_data, doi_intervals

#%%
def clean_bom_historic_offClock(data):
    '''
    A Special cleaning method for BOM data. Data from before Sep 2017 should
    be reported haf hourly. Readings not on these intervals are removed in
    his method.

    Parameters
    ----------
    data : panda.DataFrame
        Imported weather data

    Returns
    -------
    data_joined : panda.DataFrame
        Original data with off clock time stamps removed

    data_not : panda.DataFrame
        A dataframe of just the time stamps that were removed
    '''

    # extract the historic data
    tansition_date_local = datetime.strptime('2017-09-01 00:00:00', "%Y-%m-%d %H:%M:%S").replace(tzinfo=data.index.tz)
    df = data.loc[:tansition_date_local]
    
    valids_historic = pd.Series(index=df.index, dtype=bool)
    valids_historic &= df.index.minute % 30 == 0
    valids_new = pd.Series(index=data.loc[tansition_date_local:].index, dtype=bool)
    
    valids_local = pd.concat([valids_historic, valids_new])
    
    # find half-hourly timestamps
    data_historic_onClock = df[df.index.minute % 30 == 0]
    
    # join data
    data_joined = pd.concat([data_historic_onClock, data.loc[tansition_date_local:]])
    
    # # the process of finding all timestamps in 30 minute interval from the off-clock 
    # off_clock = df[df.index.minute % 30 != 0].index
    
    # minA = 20
    # minB = 50
    # # Divide off-clock values into two lists based on minute value
    # off_clock_0 = off_clock[(off_clock.minute >= minB) | (off_clock.minute < minA)]
    # off_clock_30 = off_clock[(off_clock.minute >= minA) & (off_clock.minute < minB)]
    
    # # getting all timestamps around off-clock timestamps for 50-20 and 20-50 minutes
    # date_ranges_0 = {off_clock_0[i]: df.loc[off_clock_0[i].replace(minute=minB) : (off_clock_0[i]+timedelta(hours=1)).replace(minute=minA)].index if off_clock_0[i].minute>minB else df.loc[(off_clock_0[i]-timedelta(hours=1)).replace(minute=minB) : off_clock_0[i].replace(minute=minA)].index for i in range(len(off_clock_0))}
    # date_ranges_30 = {off_clock_30[i]: df.loc[off_clock_30[i].replace(minute=minA) : off_clock_30[i].replace(minute=minB)].index for i in range(len(off_clock_30))}
    
    # #%
    # averages_0={}
    # averages_30={}
    # for timestamp, date_range in date_ranges_0.items():
    #     if not any(dt.minute == 0 for dt in date_range):
    #         average_0 = df.loc[date_range, 'WindSpeed'].mean()  # select the rows for the date range and calculate the average
    #         df.loc[timestamp.replace(minute=0),'WindSpeed'] = average_0
    #         # TODO: creating aggregation for all attributes/obeservation types if it is needed.
    
    # for timestamp, date_range in date_ranges_30.items():
    #     if not any(dt.minute == 0 for dt in date_range):
    #         average_30 = df.loc[date_range, 'WindSpeed'].mean()  # select the rows for the date range and calculate the average
    #         df.loc[timestamp.replace(minute=0),'WindSpeed'] = average_30
    #         # TODO: creating aggregation for all attributes/obeservation types if it is needed.
    
    # # finding the probability of off-clock to all data 
    # off_clock_prob = round((len(date_ranges_0.keys())+len(date_ranges_30.keys()))/df.shape[0]*100,1)
    # print(f'% off-clock {off_clock_prob}')
    
    # # finding the probability of off-clock with missing half-hourly data to all data 
    # off_clock_woHalfHourly_prob = round((len(averages_0.keys())+len(averages_30.keys()))/df.shape[0]*100,1)

    # print(f'% off-clock without half-hourly neighbour {off_clock_woHalfHourly_prob}')
    data_not = data[-valids_local]

    return data_joined, data_not
    
# %% Finding missing timestamps
def missing_timestamps(df_index):
    """
    This function returns all missing timestamps based on finding the interval
    between first and second timestamps.
    

    Parameters
    ----------
    df_index : DatetimeIndex
        List of dates in DatetimeIndex format

    Raises
    ------
    ValueError
        It would only work for hourly and 10 minute data.

    Returns
    -------
    missing_timestamps : DatetimeIndex
        Returning the missing timestamps.

    """
    try:
        df_index = df_index.index
    except:
        df_index = df_index
        
    # Assuming utci_df is your DataFrame with a DateTimeIndex
    time_difference = pd.Timedelta(df_index[1] - df_index[0])
    
    # Determine the appropriate frequency based on the time difference
    if time_difference == pd.Timedelta(hours=1):
        freq = 'H'
    elif time_difference == pd.Timedelta(minutes=10):
        freq = '10T'
    else:
        raise ValueError("Unsupported time difference.")
        
    ref_index = pd.date_range(start=df_index.min(), end=df_index.max(), freq=freq)
    ref_df = pd.DataFrame(index=ref_index)  
    
    # Find the missing timestamps
    missing_timestamps = ref_index[~ref_index.isin(df_index)]
    
    return missing_timestamps

# %% Filling missing timestamps
def fill_missing_timestamps(df):
    """
    This function will find and insert missing timestamps by using the nearest
    timestamp/row of the missed timestamp
    

    Parameters
    ----------
    df : pd.DataFrame
        Data with index as DatetimeIndex.

    Raises
    ------
    ValueError
        It would only work for hourly and 10 minute data..

    Returns
    -------
    filled_df : pd.DataFrame
        The return dataframe with all missing timestamps filled.

    """
    # Assuming df_index is your DataFrame's index
    # Assuming df is your DataFrame
    
    # Calculate the start and end dates
    start_date = df.index.min()
    end_date = df.index.max()

    # Determine the appropriate frequency based on the time difference
    time_difference = df.index[1] - df.index[0]
    if time_difference == pd.Timedelta(hours=1):
        freq = 'H'
    elif time_difference == pd.Timedelta(minutes=10):
        freq = '10T'
    else:
        raise ValueError("Unsupported time difference.")

    # Create the complete range of timestamps
    ref_index = pd.date_range(start=start_date, end=end_date, freq=freq)

    # Reindex the DataFrame with the complete range of timestamps
    filled_df = df.reindex(ref_index, method='nearest')

    return filled_df


def remove_leap_day(df):
    
    # remove leap day (Feb 29 ) if it exists:
    return df[~((df.index.month==2) & (df.index.day==29))]



#%% NOAA filters used by MATLAB script / windpy

def remove_WD_notmult(data,multiple=10):
    """
    Removes data points that have wind directions not a multiple of multiple

    Parameters
    ----------
    data : pandas.DataFrame
        windpy NOAA dataframe
   
    multiple : int, optional
        Multiple that we want to keep in the dataset. The default is 10.

    Returns
    -------
    data_filt : pandas.DataFrame
        filtered dataframe that do only have wind directions with the desired 
        multiple.
    data_out : pandas.DataFrame
        filtered dataframe that do not have a wind direction with the desired
        multiple.

    """
    if not isinstance(multiple, int):
        raise TypeError("Please input an integer for the wind direction multiple")
        
    filt = data['WindDirection'] % multiple == 0
    
    data_filt = data.loc[filt]
    if data_filt.empty:
        data_out = pd.DataFrame(columns=data.columns)
    data_out = data.loc[~filt]
    if data_out.empty:
        data_out = pd.DataFrame(columns=data.columns)
        
    return data_filt, data_out, filt
    

    
def remove_hours(data, start_hour=0, end_hour=24, flag_extend_end=False):
    """
    Removes data points not between specified hours (inclusive)

    Parameters
    ----------
    data : pandas.DataFrame
        NOAA dataframe
    
    start_hour : int, optional
        Starting hour. The default is 0.
    
    end_hour : int, optional
        Ending hour. The default is 24.

    Raises
    ------
    TypeError
        Integer only for the hours!!.

    Returns
    -------
    data_filt : pandas.DataFrame
        time-filtered dataframe between start_hour and end_hour.
    
    data_out : pandas.DataFrame
        time-filtered dataframe outside of start_hour and end_hour.

    """
    
    #### this was added to get the data_conditioning closer between MATLAB and
    #### weatherpy for hourly subsets. Viet gave up because it's not important
    #### MATLAB treats all hours the same, regardless of minute.
    #### e.g. 18 hr, 30 min is the same as 18 hr 00 min so it "passes" MATALB filter
    #### debatable if that is correct 
    if flag_extend_end:
        if end_hour > start_hour:
            end_hour+=1
        elif start_hour > end_hour:
            start_hour+=1
    
    if not isinstance(start_hour, int) or not isinstance(end_hour, int):
        raise TypeError("Please input an integer to filter data between hours")
        
    start_time = f'{start_hour}:00:00'
    
    if end_hour >= 24:
        end_time = '23:59:59'
    else:
        end_time = f'{end_hour}:00:00'
        
    data_filt = data.between_time(start_time,end_time)
    data_out = data.between_time(end_time,start_time,inclusive='neither')
    if data_out.empty:
        data_out = None

    return data_filt, data_out
    


def remove_years(data,start_year=1900,end_year=2900):
    """
    Removes data points not between specified hours (inclusive)

    Parameters
    ----------
    data : pandas.DataFrame
        windpy NOAA dataframe
    
    start_year : int, optional
        Starting hour. The default is 1900.
    
    end_year : int, optional
        Ending hour. The default is 2900.

    Raises
    ------
    TypeError
        Integer only for the years!!.

    Returns
    -------
    data_filt : pandas.DataFrame
        time-filtered dataframe between start_year and end_year.
    
    data_out : pandas.DataFrame
        time-filtered dataframe outside of start_year and end_year.

    """
    if not isinstance(start_year, int) or not isinstance(end_year, int):
        raise TypeError("Please input an integer to filter data between years")
        
    filt = (data.index.year >= start_year) & (data.index.year <= end_year)
    
    data_filt = data.loc[filt]
    if data_filt.empty:
        data_out = None
    data_out = data.loc[~filt]
    if data_out.empty:
        data_out = None

    return data_filt, data_out



def clean_threshold_NOAA(data, threshold, remove=True, var='WindSpeed', mode='>'):
    '''
    Removes and return samples with speed (or any other variable)
    larger/smaller than a threshold

    Parameters
    ----------
    data : pandas.DataFrame
        windpy NOAA dataframe
    
    threshold : float
        threshold value
    
    remove : bool, optional
        If True, the value are removed from the dataset inplace.
        The default is True.
    
    var: string, optional
        name of the column containing the variable to be checked. The default
        is ``'WindSpeed'``
    
    mode : string, optional
        One among ``'>', '>=', '=', '<', '<='``. Indicates the condition to
        be used to identify samples to be removed. The default is ``'>'``
        indicating that samples larger than the threshold will be removed.

    Returns
    -------
    pandas.DataFrame
        A dateset containing the samples with variable dir.
    '''

    if mode == '>':
        mask = data[var] > threshold

    elif mode == '>=':
        mask = data[var] >= threshold

    elif mode == '<':
        mask = data[var] < threshold

    elif mode == '<=':
        mask = data[var] <= threshold

    elif mode == '=':
        mask = data[var] >= threshold

    else:
        raise(ValueError(f"Invalid mode {mode}. Valid modes are "
                         "'>', '>=', '=', '<', '<='"))

    # use the mask to filter the data. keep what you want
    return ~mask



def clean_ratio(data, ratio=5, threshold=30, var='WindSpeed', step=1):
    '''
    Removes samples with speed higher than n-times both the previous and the
    following sample(s). Samples bellow a threshold will be ignored.

    Parameters
    ----------
    data : pandas.DataFrame
        windpy NOAA dataframe
    
    ratio : float, optional
         maximum admissible ratio with previous and following sample.
         The default is 5.
    
    var: string, optional
        name of the column containing the variable to be checked. The default
        is ``'WindSpeed'``
    
    threshold : float, optional
        wind speed below which a sample is never considered an outlier.
        The default is 10.
    
    step : int or list of int, optional
        number of samples before and after against which each sample is compared.
        If it is a list, all steps in the list will be tried. Sample marked
        as outliers at least one time will be removed. The default is 1.

    Returns
    -------
    pandas.DataFrame
        A dateset containing the samples with variable dir.

    '''

    if not hasattr(step,'__iter__'):
        step = [step]

    flag = pd.Series(index = data.index, dtype=bool)
    
    flag[:] = False

    for s in step:
        # Check for each element the ratio with the following. True = outlier
        # NOTE: we need to check both for a/b > ratio and b/a > ratio => a/b < 1/ratio
        ratioAfter = (1/ratio > data[var].shift(-s)/data[var]) | (data[var].shift(-s)/data[var] > ratio)

        # Overlap the list to the shifted version of itself. True = possible outlier
        ratioBeforeAfter = ratioAfter & ratioAfter.shift(s)

        # Overlap with previus steps
        flag = flag | ratioBeforeAfter

    # Find all the values bigger than the threshold. False = possible outlier
    protected = data[var]<threshold

    # Inds of elements that failed both tests
    mask = flag & ~protected

    # use the mask to filter the data. keep what you want
    return ~mask



def clean_thunderstorms(data, flags=None, remove=True, obs_var='MW1_0'):
    '''
    Remove samples recorded during thunderstorms based on the manually-filled
    column describing the meteorological weather at the time of the recording.

    Parameters
    ----------
    data : pandas.DataFrame
        windpy NOAA dataframe
    
    flags : TYPE, optional
        Flags to be removed. The default is [13, 17, 19, 29, 80 ... 100].
    
    remove : bool, optional
        If True, the value are removed from the dataset inplace.
        The default is True.
    
    obs_var: string, optional
        name of the column containing the variable to be checked. The default
        is ``'MW1_0'``

    Returns
    -------
    pandas.DataFrame
        A dateset containing the samples occurred during a thunderstorm.

    '''

    defaultFlags = [13, #Lightning visible, no thunder heard
                    17, #Thunderstorm, but no precipitation at the time of observation
                    19, #Funnel cloud(s) (Tornado cloud or waterspout) at or within sight of the station during the preceding hour or at the time of observation
                    29, #Thunderstorm (with or without precipitation)
                    ] + list(range(80,100)) # 80-99 Showery precipitation, or precipitation with current or recent thunderstorm

    flags = flags if flags else defaultFlags

    mask = data[obs_var].isin(flags)

    # returns 
    return data[~mask], data[mask]


def isolate_calms(dataIn, dataType, remove = True):
    '''
    Removes calm wind conditions from a dataframe. Data must be cleaned first.

    Parameters
    ----------
    dataIn : pandas.DataFrame
        A dataframe of imported, unified, and cleaned data.

    dataType : str
        A string representing the source of the data, "BOM" or "NOAA"

    remove : Boolean, optional
        If set to true, the calms will be removed from the input dataframe

    Returns
    -------
    data : pandas.DataFrame
        The new dataframe with the clean wind conditions removed.
    
    data_calm : pandas.DataFrame
        A dataframe of just the removed (calm) data
    '''

    data = dataIn.copy(deep = True)

    # Create dataframe of only calm wind timestamps. For NOAA, add WindType == 'C'
    # VL
    if dataType == 'NOAA':
        data_calm = data[(data['WindDirection'] == 0) | \
                         (data['WindSpeed'] == 0) | \
                             (data['WindType'] == 'C')].copy(deep = True)
    else:
        data_calm = data[(data['WindDirection'] == 0) & (data['WindSpeed'] == 0)].copy(deep = True)
        
        
    # If calm removal is requested
    if remove == True:
        # VL
        if dataType == 'NOAA':
            data = data[(data['WindDirection'] != 0) & (data['WindSpeed'] != 0) & (data['WindType'] != 'C')]
        else:
            data = data[(data['WindDirection'] != 0) | (data['WindSpeed'] != 0)]
        return data, data_calm

    else:
        return data_calm
    



