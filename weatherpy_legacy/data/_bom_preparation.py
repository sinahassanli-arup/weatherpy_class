import os, sys
import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime, date, timedelta, timezone
import pytz
from  collections import OrderedDict
from .stations import station_info
import certifi
import tempfile


# api for new data (post 2017)
API_ENDPOINT_DEV = 'https://cp6f84ey30.execute-api.ap-southeast-2.amazonaws.com/dev/raw-obs'

# define transition date in UTC time between historic and new data, and make a datetime object for when the new data begins on 02/09/2017
transition_date_str =  '2017-09-01 00:00'
transition_date_minus_10_min = '2017-08-31 23:50'
transition_date_UTC = datetime.strptime(transition_date_str, '%Y-%m-%d %H:%M').replace(tzinfo=pytz.utc)
    
#%%
# define convenient global variables for the final column names of some observation types, namely all historic types and the apparent temperature
UNIFIED_OBS_TYPES= OrderedDict({
    'WindDirection_col' : 'WindDirection',
    'WindSpeed_col' : 'WindSpeed',
    'WindGust_col' : 'WindGust',
    'SeaLevelPressure_col' : 'SeaLevelPressure',
    'DryBulbTemperature_col' : 'DryBulbTemperature',
    'WetBulbTemperature_col' : 'WetBulbTemperature',
    'DewPointTemperature_col' : 'DewPointTemperature',
    'RelativeHumidity_col': 'RelativeHumidity',
    'Rain_col' : 'Rain',
    'RainIntensity_col' : 'RainIntensity',
    'RainCumulative_col' : 'RainCumulative',
    'Visibility_col' : 'Visibility',
    'CloudHight_col' : 'CloudHight',
    'CloudOktass_col' : 'CloudOktas'
    })    

# constant lists, they contain all the observation types and the observation types present in historic data (pre-2017)
ALL_OBS_TYPES = ['air_temperature','delta_t','dew_point','pres','rainfall','rel_humidity','wind_dir_deg','wind_spd_kmh','wind_gust_spd','BOM_id','obs_period_id','wind_src','level_index','level_type','apparent_temp','wind_dir','maximum_air_temperature','minimum_air_temperature','maximum_gust_dir','maximum_gust_spd','weather','cloud','visibility','sea_height','swell_height','swell_period','swell_dir','cloud_base','cloud_oktas','cloud_type','cloud_type_id','press_tend','timestep_dur_sec']
HIST_OBS_TYPES = ['air_temperature','delta_t','dew_point','pres','rainfall','rel_humidity','wind_dir_deg','wind_spd_kmh','wind_gust_spd']
UNIFIED_OBS_TYPES_NEWDATA = ['obs_period_time_utc','wind_dir_deg','wind_spd_kmh','wind_gust_spd','pres','air_temperature','dew_point','delta_t','rel_humidity','rainfall','visibility','cloud_oktas']
selected_obs_types = UNIFIED_OBS_TYPES_NEWDATA

#%% define conversion functions
def _knots_to_ms(wind_gust, accuracy=2):
    return round(float(wind_gust)*0.51444, accuracy)

def _kmh_to_ms(spd):
    return float(spd)/3.6

# rainfall intensity
def _rain_intensity(data, transition_date_local):

    # create rain intensity per timestamps
    data['Rain'] = data.loc[:,'RainCumulative'].diff()
    
    # Create apply  mask to reset 9am value
    # mask_9am = data.index.time == pd.Timestamp('9:00:00').time()
    data_historic = data.loc[:transition_date_local,:]
    mask_9am = pd.Series(data_historic.index.time == pd.Timestamp('9:00:00').time(),index=data_historic.index)
    mask_9am_after = mask_9am.shift(1, fill_value=False)
    
    data_historic.loc[mask_9am_after, 'Rain'] = data_historic.loc[mask_9am_after, 'RainCumulative']
    
    data_new = data.loc[transition_date_local:,:]
    mask_9am_new = pd.Series(data_new.index.time == pd.Timestamp('9:00:00').time(),index=data_new.index)
    data_new.loc[mask_9am_new, 'Rain'] = data_new.loc[mask_9am_new, 'RainCumulative']
    
    data_joined = pd.concat([data_historic, data_new],axis=0)
    # create rain intensity per min
    minute_delta = data.index.to_series().diff().dt.total_seconds()/60
    data_joined['RainIntensity'] = data_joined['Rain']/minute_delta
    
    return data_joined

def _which_data_present(transition_date_UTC, start_date_UTC, end_date_UTC):
    
    # determine the presence of historic or new data, specify in flags
    historic_data_present = True
    new_data_present = True
    
    if start_date_UTC >= transition_date_UTC:
        historic_data_present = False
    elif end_date_UTC < transition_date_UTC:
        new_data_present = False

    return historic_data_present, new_data_present

def _daylight_saving_time(data, timezone_local):
    
    # if the datetimes (which are in Local Time) are timezone aware, change them to naive Local Time with tz_localize()
    if data['LocalTime'][0].tzinfo != None:
        data['LocalTime']= data['LocalTime'].dt.tz_localize(tz=None) # note that tz_convert(tz=None) would have removed timezone information AND converted the time to UTC which is not the aim here
    
    # with naive Local Time, the timezone can be set with the tz_localize() method which checks for the validity of Local Times with respect to daylight saving, then, any questionable times can be removed
    try:
        data['LocalTime'] = data['LocalTime'].dt.tz_localize(tz=timezone_local,ambiguous='infer',nonexistent='NaT')
    except:
        # print("Notice: In the historic data set, there is an ambiguous time when the clock goes backward (due to daylight saving) which cannot be inferred. Consequently, due to limitations of this program, all repeated Local Times (even those that can be inferred) are removed from the dataframe.")
        data['LocalTime'] = data['LocalTime'].dt.tz_localize(tz=timezone_local,ambiguous='NaT',nonexistent='NaT')

    # remove all the rows that have the time as NaT from ambiguous or nonexistent checks, the presence of a NaT can be detected with pd.isnull()
    data = data[pd.isnull(data['LocalTime']) == False]
        
    return data
    x = index
    while (df.index[x+1] - df.index[x]).total_seconds() < 10*60-5: # if the next data point is less than 10 minutes away, move to that data point and check again
        x+=1; 
    return df.index[x]

def _get_observation(start, end, station, datatypes):
    """
    The main function for creating an API URL for interfacing with the
    observations application and retrieving live data
    
    Parameters
    ----------
    start : datetime
        A datetime timestamp representing the first timestamps of the import
    
    end : datetime
        A datetime timestamp representing the last timestamps of the import
    
    station : str
        The BOM station code. should be a string of 6 digits e.g. '066037'

    datatypes : list
        List of data codes to be retrieved

    Returns
    -------
    url_address : responses 
        If API is successful, the data retrieved wil be returned
    """

    def format_date(dateIn):
        return dateIn.strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        start = format_date(start)
        end = format_date(end)
    except:
        pass
    
    # Create API URL
    datatypes = ','.join(datatypes)
    api_request_url = '{}?station={}&start={}&end={}&datatypes={}'.format(
        API_ENDPOINT_DEV, station, start, end, datatypes)

    # Attempt API. Terminate attempt after (timeout_timer) seconds
    timeout_timer = 240
    try:
        response = requests.get(api_request_url, stream = True, timeout = timeout_timer)
    except requests.Timeout:
        print(f"Failed (API timeout after {timeout_timer} seconds)")
        return None

    # If API attempt was successful (status code = 200)
    if response.status_code==200:
        url_address = response.content.decode('utf-8')  # get the content of the request
        return url_address

    # If API attempt was unsuccessful
    else:
        print(f"Failed ({response.reason}. Code {response.status_code})")
        return None
    
def _import_data_historic(stationID, start_date_local, end_date_local, timezone_local, unified_obs_types=UNIFIED_OBS_TYPES):
    """
    The main function for accessing historic data stored in AWS S3 bucket
    
    Parameters
    ----------
    station : str
        The BOM station code. should be a string of 6 digits e.g. '066037'

    start_date_local : datetime
        A datetime timestamp representing the first timestamps of the import
    
    end_date_local : datetime
        A datetime timestamp representing the last timestamps of the import
    
    timezone_local : str
        The local timezone of the weather station 

    unified_obs_types : list
        List of data codes to be retrieved

    Returns
    -------
    url_address : responses 
        If API is successful, the data retrieved wil be returned
    """


    df = pd.DataFrame()
    
    weatherFileUrl = f'https://bomhistoric.s3.ap-southeast-2.amazonaws.com/hourly_data_{stationID}.csv.gz'

    df = pd.read_csv(weatherFileUrl, compression='gzip', parse_dates=['Local Time'])  
    
    df['LocalTime'] = pd.to_datetime(df['Local Time'])
        
    df = _daylight_saving_time(df, timezone_local)
  
    # set index to date
    df.set_index('LocalTime', inplace=True)
    
    # sort values wrt time
    df = df.sort_index()
    
    df = df[start_date_local:end_date_local] # explicitly make a copy when creating the new data frame, this avoids a SettingWithCopyWarning
   
    # Changing column names
    historic_obs_types_kept = []
    historic_obs_rename_dict = {'Air Temperature in degrees C': unified_obs_types['DryBulbTemperature_col'], 
                                'Wet bulb temperature in degrees C': unified_obs_types['WetBulbTemperature_col'], 
                                'Dew point temperature in degrees C': unified_obs_types['DewPointTemperature_col'],
                                'Mean sea level pressure in hPa': unified_obs_types['SeaLevelPressure_col'],
                                'Precipitation since 9am local time in mm': unified_obs_types['RainCumulative_col'],
                                'Relative humidity in percentage %': unified_obs_types['RelativeHumidity_col'],
                                'Wind direction in degrees true': unified_obs_types['WindDirection_col'],
                                'Wind speed in m/s': unified_obs_types['WindSpeed_col'],
                                'Speed of maximum windgust in last 10 minutes in m/s': unified_obs_types['WindGust_col'],
                                }
    
    df.rename(columns=historic_obs_rename_dict, inplace=True) 
    
    return df
      
def _import_data_new(stationID, start_date_UTC, end_date_UTC, timezone_local, obs_types=selected_obs_types, unified_obs_types=UNIFIED_OBS_TYPES):
    
    # number of attempt to retrieve the data from repository
    numAttempts = 10
    i=0
    
    while i < numAttempts:
        i += 1
        print(f"\tAPI attempt {i}/{numAttempts} : ", end='\r')
        time_request = time.time()
        url_address = _get_observation(start_date_UTC,end_date_UTC, str(int(stationID)), obs_types)         
        
        try: 
            df_new = pd.read_pickle(url_address)
            print(f'Successful (fetch time: {round(time.time()-time_request,1)} sec)')
            break
        except FileNotFoundError: 
            print("Failed (file not located)")
        except:
            pass

    if not 'df_new' in locals():
        print(f'\tresource could not be fetched after {numAttempts} iterations\n')
        print('--Exiting Script--')
        sys.exit()
        
    # add UTC timezone to new data index if it is not already assigned
    if df_new.index[0].tzinfo == None:
        df_new.index = df_new.index.dt.tz_localize(tz='UTC')

    # sort values wrt time
    df_new = df_new.sort_index()

    # convert new data to local time
    df_new.index = df_new.index.tz_convert(timezone_local)
    df_new.rename_axis('LocalTime', inplace=True)

    # Changing column names
    newData_obs_rename_dict = {'air_temperature':unified_obs_types['DryBulbTemperature_col'],
                               'dew_point':unified_obs_types['DewPointTemperature_col'],
                               'pres':unified_obs_types['SeaLevelPressure_col'],
                               'rainfall':unified_obs_types['RainCumulative_col'],
                               'rel_humidity':unified_obs_types['RelativeHumidity_col'],
                               'wind_dir_deg':unified_obs_types['WindDirection_col'],
                               'cloud_oktas':unified_obs_types['CloudOktass_col'],
                               'visibility':unified_obs_types['Visibility_col'],
                               }
    
    df_new.rename(columns=newData_obs_rename_dict, inplace=True) 
    
    # calculate drybulb and wind speeds to SI units
    
    # wetbulb from drybulb and delta_t
    df_new[unified_obs_types['WetBulbTemperature_col']] = df_new[unified_obs_types['DryBulbTemperature_col']]-df_new['delta_t']
    
    # convert wind speed unit from km/h or m/s 
    df_new[unified_obs_types['WindSpeed_col']] = df_new['wind_spd_kmh'].apply(_kmh_to_ms)
    
    # convert gust speed unit from knots or m/s 
    df_new[unified_obs_types['WindGust_col']] = df_new['wind_gust_spd'].apply(_knots_to_ms)
    
    return df_new

def _bom_date_bounds(ID, yearStart, yearEnd, timeZone):
    '''
    The function is used to generate the upper and lower date bounds for defining a data import.
    The function takes in two years in integers, the stations ID, and the user specified time zone.
    the function returns two timezone aware datetime objects that define the bounds of the data import.

    the result of the function is that data imports will be from 00:00 Jan 1 to 23:59 Dec 31
    regardless of whether the timezone is local or UTC

    Parameters
    ----------
    ID : str
        A weather station ID used to find the station's local timezone
    
    yearStart : int
        The first year of the data import. Will be from 00:00 Jan 1 for local timezone
    
    yearEnd : int
        The last year of the data import. Will be to 23:59 Dec 31 for local timezone
    
    timeZone : str
        The desired timezone, either "LocalTime" or "UTC"

    Returns
    -------
    start_date : datetime.datetime (time zone aware)
        The datetime object denoting the start of the data import
    
    end_date : datetime.datetime (time zone aware)
        The datetime object denoting the end of the data import
    '''
    
    timezone_offset = station_info(ID,printed=False)['Timezone UTC']
    timezone_local = pytz.timezone(station_info(ID, printed=False)['Timezone Name'])
    
    start_date = timezone_local.localize(datetime.strptime(str(yearStart)+' 01 01 00:00','%Y %m %d %H:%M'))
    end_date = timezone_local.localize(datetime.strptime(str(yearEnd)+' 12 31 23:59','%Y %m %d %H:%M'))
    
    if timeZone == 'LocalTime':
        return start_date, end_date
    
    if timeZone == 'UTC':
        tz_hours = int(timezone_offset[5:7])
        tz_mins = int(timezone_offset[8:10])
        tz_delta = timedelta(hours = tz_hours, minutes = tz_mins)
        if timezone_offset[4:5] == '-': tz_delta = -tz_delta
        start_date = start_date + tz_delta
        end_date = end_date + tz_delta
        return start_date, end_date

    else:
        raise ValueError('timeZone argument must be either \"UTC\" or \"LocalTime\"')


# CONSOLIDATED BOM DATA (PRIOR AND AFTER 2017)    
def bom_consolidate(stationID, timeZone = 'LocalTime', yearStart=None, yearEnd=None):
    '''
    The primary function for creating a dataframe of BOM data
    This function creates accesses lve and historic data and joins them

    Parameters
    ----------
    stationID : str
        A weather station ID used to select the station for data retrieval
    
    timeZone : str
        The desired timezone, either "LocalTime" or "UTC"

    yearStart : int
        The first year of the data import. Will be from 00:00 Jan 1 for local timezone
    
    yearEnd : int
        The last year of the data import. Will be to 23:59 Dec 31 for local timezone
    
    Returns
    -------
    data_joined : pandas.DataFrame
        The imported weather dataframe
    '''        
    #%% Get start and end date for historic and new data
   
    # get the timezone of the weather station
    timezone_local = pytz.timezone(station_info(stationID, printed=False)['Timezone Name'])

    # Creates the start and end datetime objects
    start_date_local = timezone_local.localize(datetime.strptime(str(yearStart - 1)+' 12 25 00:00','%Y %m %d %H:%M'))
    end_date_local = timezone_local.localize(datetime.strptime(str(yearEnd + 1)+' 01 05 23:59','%Y %m %d %H:%M'))

    transition_date_local = transition_date_UTC.astimezone(timezone_local)
    
    start_date_UTC = start_date_local.astimezone(pytz.utc)
    end_date_UTC = end_date_local.astimezone(pytz.utc)
    
    historic_data_present, new_data_present = _which_data_present(transition_date_UTC, start_date_UTC, end_date_UTC)
    
    # specify the start date for the new data request to the API, depending on the case
    if start_date_UTC < transition_date_UTC:
        start_date_UTC_new_data = transition_date_UTC
    else:
        start_date_UTC_new_data = start_date_UTC
    
    # specify the end date for the old data retrieval
    if end_date_UTC > transition_date_UTC:
        end_date_old_data = transition_date_local
    else:
        end_date_old_data = end_date_local
    
    # %% Import and unify column names
    # Import Historic data (pre 2017)
    df_historic = pd.DataFrame()
    if historic_data_present:
        df_historic = _import_data_historic(stationID, start_date_local, end_date_old_data, timezone_local, UNIFIED_OBS_TYPES)
    # Import new data (post 2017)
    df_new = pd.DataFrame()
    if new_data_present:
        df_new = _import_data_new(stationID, start_date_UTC_new_data, end_date_UTC, timezone_local, selected_obs_types, UNIFIED_OBS_TYPES)

    #%% Joining data
    # reduce the columsn to the unified columns if they are available
    valid_cols_historic = [col for col in UNIFIED_OBS_TYPES.values() if col in df_historic.columns]
    df_historic_valid = df_historic.loc[:,valid_cols_historic]
    
    valid_cols_new = [col for col in UNIFIED_OBS_TYPES.values() if col in df_new.columns]
    df_new_valid = df_new.loc[:,valid_cols_new]
    
    df_joined = pd.concat([df_historic_valid, df_new_valid],axis=0)

    df_joined = df_joined[~df_joined.index.duplicated(keep='first')]
    df_joined.sort_index(inplace=True)
 
    # if rainfall is present, produce rain intensity in mm/timestamp and mm/min
    if 'rainfall' in selected_obs_types:
        df_joined = _rain_intensity(df_joined, transition_date_local)
    
    df_joined.insert(loc=0, column='UTC', value=df_joined.index.tz_convert(pytz.utc))
    
    return df_joined


# Function to get 1/10/60min BOM historic weather data from 2000 to 2023
def _import_bomhistoric(stationID, interval, timeZone, yearStart, yearEnd):
    
    # api url
    url = "https://rr0yh3ttf5.execute-api.ap-southeast-2.amazonaws.com/Prod/v1/bomhistoric"

    # preparing POST argument for request
    stationFile = f"{stationID}.zip" if interval==1 else f"{stationID}-{interval}minute.zip"

    body = {
    "bucket": f"bomhistoric-{interval}minute",
    "stationID": stationFile
    }
  
    response_url = requests.post(url, json=body)
    signed_url = response_url.json()['body']

    signed_url_statusCode = response_url.json()['statusCode']
    
    if signed_url_statusCode != 200:
        raise ValueError(f'signed_url: {signed_url} Error code: {signed_url_statusCode}')

    response_data = requests.get(signed_url)

    if response_data.status_code == 200:
        
        # Create a temporary file to save the response content
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(response_data.content)
            temp_file_path = temp_file.name
            df = pd.read_pickle(temp_file_path, compression='zip') 
            print("data is imported successfully")
        os.remove(temp_file_path) 
    else:
        print("API request failed")
        
    # Switch UTC and Local time datetime index if needed 
    if timeZone != df.index.name:
        df = df.reset_index()
        df = df.set_index(df.columns[1])
    
    # print(f's3 file years:{df.index.year.min()}-{df.index.year.max()}')
    return df
