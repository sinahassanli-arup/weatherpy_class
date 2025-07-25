import pandas as pd
import numpy as np
import os
from ._bom_preparation import bom_consolidate, _bom_date_bounds, _import_bomhistoric
from ._noaa_preparation import _getNOAA_api, _noaa_date_bounds
from .stations import station_info
import time
import pytz
import inspect 
from datetime import datetime, timedelta
from ..analysis import psychrolib
import platform
import tempfile
from ..analysis.processing import generate_ranges_wind_direction

def _get_weatherpy_temp_folder():
    """
    This is to get the proper temporary folder based on linux or windows system
    """
    current_system = platform.system()

    if current_system == 'Windows':
        temp_path = os.path.join(tempfile.gettempdir(),'weatherpy')
        
    elif current_system == 'Linux':
        temp_path = os.path.join('/tmp','weatherpy')  # Linux temporary folder
    else:
        raise NotImplementedError("This system is not recognized as Windows or Linux.")

    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    
    return temp_path

#%% DEVELOPER FUNCTIONS
def _validate_station_years(stationID, dataType, yearStart, yearEnd):
    """
    This is to check if start year is earlier than the earliest year of the station measurement
    and end year is after the most recent year of station measurement

    Parameters
    ----------
    stationID : str
        BOM station ID.
    
    dataType : str
        Weather station type. Either 'NOAA' or 'BOM'.
    
    yearStart : int
        Start year for data retrieval.
    
    yearEnd : int
        End year for data retrieval.

    Returns
    -------
    yearStart : int
        updated start of the year
    
    yearEnd.
        updated end of the year
    """
    # Checks from database the years of operation of the weather station 
    stn_info = station_info(stationID, printed=False)
    
    if dataType == 'new':
        return None, None
    
    if dataType == 'BOM':
        try:
            yearStart_actual = int(stn_info['Start'])
            yearEnd_actual = int(stn_info['End'])
        except:
            yearStart_actual = yearStart
            yearEnd_actual = yearEnd
    
    elif dataType == 'NOAA':
        yearStart_actual = int(stn_info['Start'].split('/')[-1])
        yearEnd_actual = int(stn_info['End'].split('/')[-1])
    
    else:
        yearStart_actual = yearStart
        yearEnd_actual = yearEnd
    
    # Adjusts start and end year to be within years of operation
    if yearStart_actual > int(yearStart):
        yearStart = str(yearStart_actual)
    if yearEnd_actual < int(yearEnd):
        yearEnd = str(yearEnd_actual)
    
    return yearStart, yearEnd


def _read_custom(local_filepath, idxcol='LocalTime'):
    """
    Reads custom data from a local file.

    Parameters
    ----------
    local_filepath : str
        The desired csv data file location.

    Returns
    -------
    data_imported : pandas.DataFrame
        A dataframe of the imported data
    
    yearStart_updated : int
        Updated staring year by looking at the earliest timestamp.
    
    yearEnd_updated : int
        Updated ending year by looking at the latest timestamp.
    """
    # FIXME
    # A way to map custom column names on to unified column names e.g. WindSpeed
    # check tz

    # # Imports the data from user specified location
    # data_imported = pd.read_csv(local_filepath, parse_dates=['Date'])
    # data_imported.set_index('Date',inplace=True)
    # data_imported = data_imported.apply(pd.to_numeric, errors='coerce')

    """
    TODO: Check with Sydney if this is alright. 
    #VL
    """

    data_imported = pd.read_csv(local_filepath)
    data_imported = data_imported.set_index('UTC', drop=True)
    
    # convert applicable column to numeric
    data_imported = data_imported.apply(pd.to_numeric, errors='ignore')
    
    # Updates the start and end years
    yearStart_updated = data_imported.index.year.min()
    yearEnd_updated = data_imported.index.year.max()
    
    print('\t--Data is imported--\n')
    return data_imported, yearStart_updated, yearEnd_updated


def _read_bom_new(stationID, yearStart = None, yearEnd = None, path = r''):
    # If the address passed is a csv file
    if path[::-1][0:4] == 'vsc.':
        data_import  = pd.read_csv(path)

    # If the address passed is a directory containing parquet files
    else:
        # Create a list of all files in given directory
        files = os.listdir(path)
        year_data = []
        
        # Iterate thorugh files in directory
        for filename in files:
            file_path = os.path.join(path, filename)
            
            if os.path.exists(file_path):
                try:
                    data_df = pd.read_parquet(file_path, engine = 'pyarrow')  # Read data from the file
                    year_data.append(data_df)
                except:
                    pass
        
        # Concatenate each year of data to a single dataframe
        data_import = pd.concat(
            year_data,
            axis = 0,
            ignore_index = True).drop_duplicates(subset = 'UTC')

        # Create deep copy of imported data
        data_raw = data_import.copy(deep = True)
        
        # Localise UTC datetime objects to UTC
        data_raw['UTC'] = data_raw['UTC'].dt.tz_localize(pytz.UTC)
        
        # Find station local timezone
        local_tz = pytz.timezone(
            station_info(stationID, printed = False)['Timezone Name'])
        
        # Place UTC column in index
        data_raw.index = data_raw['UTC']
        
        # Coerce index from UTC to local time zone
        data_raw.index = data_raw.index.tz_convert(local_tz)
        
        # Rename index column
        data_raw.index.name = 'LocalTime'
        
        # Remove Timezone naive column
        del data_raw['Station Local Time']
        
        # Fix order 
        data_raw = data_raw.sort_values(by = 'UTC')
    
    # Slices the entire dataFrame to the correct size
    if isinstance(yearStart, int):
        startDate = local_tz.localize(datetime.strptime(f'{yearStart} 01 01 00:00','%Y %m %d %H:%M'))
        data_raw = data_raw[(data_raw.index >= startDate)]
        
    if isinstance(yearEnd, int):
        endDate = local_tz.localize(datetime.strptime(f'{yearEnd} 12 31 23:59','%Y %m %d %H:%M'))
        data_raw = data_raw[(data_raw.index <= endDate)]
    
    # Start and end years
    year_start = data_raw.index.min().year
    year_end = data_raw.index.max().year
    
    return data_raw, year_start, year_end


def _read_from_tmp(stationID, dataType, yearStart, yearEnd, interval, timeZone, file_fullpath, temp_folder, file_basename):
    """
    This function will read the weather data from a cached file if a data import
    with the same parameters has been made before and was saved to the temp folder.
    Alternatively, if the desired data exists within a larger file in the temp folder,
    this function can slice the desired data out of the larger file.

    Parameters
    ----------
    stationID : str
        A string of either a BOM or NOAA station ID
    
    dataType : str
        A string denoting that data source. "BOM" or "NOAA"
    
    yearStart : int
        Start year for data retrieval.
    
    yearEnd : TYPE
        End year for data retrieval.
    
    timeZone : str
        the timezone for the data timestamps. Either "LocalTime" or "UTC".
    
    file_fullpath : str
        The address of the chached data file.
    
    temp_folder : str
        The address of the temporary data folder.
    
    file_basename : str
        The temp file name.

    Returns
    -------
    data_imported : pandas.DataFrame
        A dataframe of the imported data
    
    yearStart_updated : int
        Updated staring year by looking at the earliest timestamp.
    
    yearEnd_updated : int
        Updated ending year by looking at the latest timestamp.
    """
    # Initialises bounds for temporary folder data import
    if timeZone == 'LocalTime': start_bound, end_bound = _bom_date_bounds(stationID, yearStart, yearEnd, 'LocalTime')
    elif timeZone == 'UTC': start_bound, end_bound = _noaa_date_bounds(stationID, yearStart, yearEnd, 'UTC')
    
    # Attempt direct import from temporary folder
    try:
        data_imported = pd.read_pickle(file_fullpath)
        print('Local File')
    
    # Attempt data slice from temporary folder
    except FileNotFoundError:
        # Generate list of all file names in temp folder
        filenames = os.listdir(temp_folder)
        file_found = False
    
        # Checks if temp files would satisfy requirements for a slice
        for i in filenames:
            import_params = i.replace('-', '_').replace('.', '_').split('_')[:-1]
            if (import_params[0], import_params[1], import_params[4]) == (dataType, stationID, interval) and int(import_params[2]) <= yearStart and int(import_params[3]) >= yearEnd:
                large_file_name = r'{}_{}-{}-{}.zip'.format(file_basename, import_params[2], import_params[3],import_params[4])
                data_filepath = os.path.join(temp_folder , large_file_name)
                data_imported = pd.read_pickle(data_filepath)
                print('Slicing Local File')
                file_found = True
                break
        
        # If no file satisfies requirements for a data slice
        if file_found == False:
            raise FileNotFoundError
    
    # Switch UTC and Local time datetime index if needed
    if timeZone != data_imported.index.name:
        data_imported = data_imported.reset_index()
        data_imported = data_imported.set_index(data_imported.columns[1])
        print('\tSwitching LocalTime and UTC columns')

    # Slices data to size
    data_imported = data_imported[(data_imported.index >= start_bound) & (data_imported.index <= end_bound)]
    
    # Updates the start and end years
    yearStart_updated = data_imported.index.year.min()
    yearEnd_updated = data_imported.index.year.max()
    
    print('\t--Data is imported--\n')
    return data_imported, yearStart_updated, yearEnd_updated
    

def _read_from_server(stationID, dataType, timeZone, yearStart, yearEnd, interval, save_raw, file_fullpath, temp_folder):
    """
    Reads data from either the BOM or the NOAA server.

    Parameters
    ----------
    stationID : str
        A string of either a BOM or NOAA station ID
    
    dataType : str
        A string denoting that data source. "BOM" or "NOAA"
    
    timeZone : str
        the timezone for the data timestamps. Either "LocalTime" or "UTC".
    
    yearStart : int
        Start year for data retrieval.
    
    yearEnd : TYPE
        End year for data retrieval.
    
    save_raw : Boolean
        True or False for whether the user wishes to save the imported data.
    
    file_fullpath : str
        Full file address of the imported data file.
    
    temp_folder : str
        Directory address of the temporary file for clearing old chached files.

    Returns
    -------
    data_imported : pandas.DataFrame
        A dataframe of the imported data
    
    yearStart_updated : int
        Updated staring year by looking at the earliest timestamp.
    
    yearEnd_updated : int
        Updated ending year by looking at the latest timestamp.
    """
    # Import data via API
    if dataType == 'BOM':
        # data_imported = bom_consolidate(stationID, timeZone=timeZone, yearStart=yearStart, yearEnd=yearEnd)    
        data_imported = _import_bomhistoric(stationID, interval=interval, timeZone=timeZone, yearStart=yearStart, yearEnd=yearEnd)
        start_bound, end_bound = _bom_date_bounds(stationID, yearStart, yearEnd, timeZone)

    elif dataType == 'NOAA':
        data_imported = _getNOAA_api(ID=stationID, timeZone=timeZone, yearStart=yearStart, yearEnd=yearEnd, quiet=False)
        start_bound, end_bound = _noaa_date_bounds(stationID, yearStart, yearEnd, timeZone)

    # Switch UTC and Local time datetime index if needed
    if timeZone != data_imported.index.name:
        data_imported = data_imported.reset_index()
        data_imported = data_imported.set_index(data_imported.columns[1])
        
    # Slices to size
    data_sliced = data_imported[(data_imported.index >= start_bound) & (data_imported.index <= end_bound)]
    
    # Update start and end years
    yearStart_updated = data_sliced.index.year.min()
    yearEnd_updated = data_sliced.index.year.max()

    # Saves data locally if requested
    if save_raw:
        data_imported.to_pickle(file_fullpath)
        
        # Remove the oldest file in temporary folder if it is more than 40 files.
        list_of_files = os.listdir(temp_folder)
        full_path = [os.path.join(temp_folder,x) for x in list_of_files]
        
        if len(list_of_files) > 40:
            oldest_file = min(full_path, key=os.path.getctime)
            os.remove(oldest_file)
    
    print('\t--Data is imported--\n')
    return data_sliced, yearStart_updated, yearEnd_updated

def infer_time_interval(idx, dataType='BOM'):

    if dataType=='NOAA':
        return 10
    elif dataType=='BOM':
        interval = idx.to_series().diff().mean()

        # Convert the average time interval to minutes
        interval_minutes = int(interval.total_seconds() / 60)

        return min([1,10,30,60], key=lambda x: abs(x - interval_minutes))


#%% IMPORT DATA
def import_data(
    stationID,
    dataType = 'BOM',
    timeZone = 'LocalTime',
    yearStart = None,
    yearEnd = None,
    interval = 60,
    save_raw = True,
    local_filepath = None):
    """
    Import BOM or NOAA data.
    
    Parameters
    ----------
    stationID : str
        Station Number or ID. For BOM it is 6 digits and for NOAA is 10 digits.
    
    dataType : str, optional
        Which data type to use between BOM, NOAA, and custom for a local data import. The default is 'BOM'.
    
    timeZone : str, optional
        To indicate the timezone of the data import, either "LocalTime" or "UTC".
        If argument is None, "LocalTime" will be chosen for BOM and "UTC" will be chosen for NOAA.
    
    yearStart : int, optional
        First year of data for plotting, if set as None, the first available year will be selected.
    
    yearEnd : int, optional
        last year of data for plotting, if set as None, the last available year will be selected.
    
    save_raw : bool, optional
        Indicates whether the user wishes to save the imported data to a temporary file. Default is True.
    
    local_filepath : str, optional
        If the dataType argument is set to 'custom', data will be read from a local filepath. Default is ''.

    Returns
    -------
    data_imported : pandas.DataFrame
        The imported raw data with all available parameters (columns).
    
    yearStart_updated : int
        The start year of the imported data
    
    yearEnd_updated : int
        The year end of the imported data
    """
    # Perform preliminary checks
    assert isinstance(stationID, str), 'Station ID must be a string (number enclosed in quotation marks. e.g: "066037")'
    assert timeZone in {'LocalTime', 'UTC'}, f'timeZone argument must be either "LocalTime" or "UTC". You have input: {timeZone}'

    if dataType == 'BOM':
        assert len(stationID) == 6, f'BOM station ID must be 6 digits in length. You have input: {stationID}'
    elif dataType == 'NOAA':
        assert len(stationID) <= 11, f'NOAA station ID must be 11 digits in length. You have input: {stationID}'
    elif dataType == 'custom':
        assert local_filepath != None, 'The filepath of a locally stored data file must be defined for a custom data import'
    elif dataType == 'new':
        assert local_filepath != None, 'The filepath of a locally stored data file must be defined for a new data import'
    else:
        assert 0, f'dataType argument must be either \"BOM\", \"NOAA\", or \"custom\". You have input: {dataType}'
    
    # Print station data and validate start and end years
    station_info(stationID)
    yearStart_updated, yearEnd_updated = _validate_station_years(stationID, dataType, yearStart, yearEnd)
    
    # Initialise important file paths
    weatherpy_temp_folder = _get_weatherpy_temp_folder()
    temp_folder = os.path.join(weatherpy_temp_folder,'sourcedata')

    file_basename = f'{dataType}_{stationID}'

    # set NOAA interval to 30
    interval=interval if dataType == 'BOM' else 30
    file_name = r'{}_{}-{}_{}minute.zip'.format(file_basename, yearStart_updated, yearEnd_updated, interval)

    file_fullpath = os.path.join(temp_folder , file_name)
    
    # Creates temporary folder if one does not already exist
    if not os.path.exists(temp_folder): os.mkdir(temp_folder)

    print('Importing Data From: ', end='')

    #TODO: Copy object to S3 and using that for testing Using object in S3 to test. 
    # If arguments match source file parameters
    if (yearStart, yearEnd) == (2018, 2022) and (stationID, timeZone) in {('066037', 'LocalTime'), ('72509014739', 'UTC')}:
        import inspect
        packagefolder = os.path.dirname(inspect.getfile(inspect.currentframe()))
        print('Source File')
        src_fullpath = os.path.join(packagefolder, 'src', f'src_{dataType}_{stationID}_2018-2022.zip')
        data_imported = pd.read_pickle(src_fullpath)
        print('\t--Data Imported--\n')
        return data_imported, 2018, 2022
    
    # If cusom data import requested
    if dataType == 'custom':
        print('Custom Data In Local csv\n')
        return _read_custom(local_filepath)
    
    # If cusom data import requested
    if dataType == 'new':
        print('New Data In Local file\n')
        return _read_bom_new(stationID,  yearStart = yearStart, yearEnd = yearEnd, path = local_filepath)
    
    # If file exists in temporary folder or if a slice is possible
    try:
        return _read_from_tmp(stationID, dataType, yearStart, yearEnd, interval, timeZone, file_fullpath, temp_folder, file_basename)
    except FileNotFoundError:
        pass   

    # Else read from server
    print(f'{dataType} Server')
    return _read_from_server(stationID, dataType, timeZone, yearStart, yearEnd, interval, save_raw, file_fullpath, temp_folder)


#%% UNIFY DATATYPE 
def unify_datatype(data, dataType):
    """
    Extract wind fields and make consistent columns for BOM and NOAA
    
    Parameters
    ----------
    data : pandas.dataframe
        BOM or NOAA weather data.
    dataType : str
        type of weather data(BOM or NOAA).

    Returns
    -------
    data_unified : pandas.dataframe
        wind field columns with consistent columns for BOM and NOAA.

    """

    observation_types = ['WindDirection', 'WindSpeed', 'WindGust', 'SeaLevelPressure', 
                   'DryBulbTemperature', 'WetBulbTemperature', 'DewPointTemperature', 
                   'RelativeHumidity', 'Rain', 'RainIntensity', 'RainCumulative', 'CloudHeight', 'Visibility', 'WindType', 'CloudOktas']
 
         
    if dataType=='NOAA':
        
        # initialize data_unified 
        data_unified = pd.DataFrame(index=data.index)
        # data_unified.index.names=['UTC']

        # placing data in correct column
        data_unified[data.columns[0]] = data.iloc[:,0]
        data_unified[observation_types[0]] = data['WindDir']
        data_unified[observation_types[1]] = data['WindSpeed']
        data_unified[observation_types[2]] = data['OC1_0']
        data_unified[observation_types[3]] = data['SeaLevelPressure']
        data_unified[observation_types[4]] = data['Temperature']
        data_unified[observation_types[5]] = np.nan
        data_unified[observation_types[6]] = data['DewPointTemp']
        data_unified[observation_types[7]] = data['RH1_2']
        data_unified[observation_types[8]] = np.nan
        data_unified[observation_types[9]] = np.nan
        data_unified[observation_types[10]] = data['RainCumulative']
        data_unified[observation_types[11]] = data['CloudHgt']
        data_unified[observation_types[12]] = data['Visibility']
        data_unified[observation_types[13]] = data['WindType']
        data_unified[observation_types[14]] = data['CloudOktas']

        # weather observations and snow depth data (in cm)
        data_unified['MW1_0'] = data['MW1_0']
        data_unified['MW1_1'] = data['MW1_1']
        data_unified['AJ1_0'] = data['AJ1_0']

        # reporting data
        data_unified['ReportType'] = data['ReportType']
        data_unified['QCName'] = data['QCName']
        data_unified['QCName'] = data['QCName']
        data_unified['QCWindSpeed'] = data['QCWindSpeed']
        data_unified['QCWindDir'] = data['QCWindDir']
        
    elif dataType=='BOM':
               
        # initialization
        data_unified = pd.DataFrame()
        
        # placing data in correct column
        data_unified[data.columns[0]] = data.iloc[:,0]
        data_unified[observation_types[0]] = data['WindDirection']
        data_unified[observation_types[1]] = data['WindSpeed']
        data_unified[observation_types[2]] = data['WindGust']
        data_unified[observation_types[3]] = data['SeaLevelPressure']
        data_unified[observation_types[4]] = data['DryBulbTemperature']
        data_unified[observation_types[5]] = data['WetBulbTemperature']
        data_unified[observation_types[6]] = data['DewPointTemperature']
        data_unified[observation_types[7]] = data['RelativeHumidity']
        try:
            data_unified[observation_types[8]] = data['Rain']
            data_unified[observation_types[9]] = data['RainIntensity']
        except:
            data_unified[observation_types[8]] = np.nan
            data_unified[observation_types[9]] = np.nan
        # data_unified[observation_types[10]] = data['RainCumulative']
        data_unified[observation_types[10]] = np.nan
        data_unified[observation_types[11]] = np.nan
        data_unified[observation_types[12]] = np.nan
        data_unified[observation_types[13]] = np.nan
        data_unified[observation_types[14]] = np.nan
        # try:
        #     data_unified[observation_types[14]] = data['CloudOktas']
        # except:
        #     data_unified[observation_types[14]] = np.nan
    
    elif dataType=='new':
        # Hardcoded column unification
        wp_column_dict = {
            'UTC' : 'UTC',
            'Wind Average Direction (°)' : 'WindDirection',
            'Wind Average Speed (m/s)' : 'WindSpeed',
            'Wind Maximum Speed (m/s)' : 'WindGust',
            'Air Pressure (hPa)' : 'SeaLevelPressure',
            'Dry Temperature (℃)' : 'DryBulbTemperature',
            'Wet Temperature (℃)' : 'WetBulbTemperature',
            'Dew Point Temperature (℃)' : 'DewPointTemperature',
            'Relative Humidity (%)' : 'RelativeHumidity',
            'Rain Accumulation (mm)' : 'RainCumulative',
            'Visibility (m)' : 'Visibility',
            'Rain In Interval (mm)' : 'Rain',
            }
        
        missing_cols = list(set(wp_column_dict.keys()) - set(data.columns))
        
        # Create a deep copy of data frame to protect raw data
        data_unified = data.copy(deep = True)
        
        for i in missing_cols:
            data_unified[i] = np.nan
        
        # Rename extra columns
        data_unified.rename(columns = wp_column_dict, inplace = True)
        
        # Add empty extra column
        data_unified['WindType'] = np.nan
        data_unified['RainIntensity'] = np.nan
        data_unified['CloudHeight'] = np.nan
        data_unified['CloudOktas'] = np.nan


        data_unified = data_unified[['UTC', 'WindDirection', 'WindSpeed', 'WindGust',
                'SeaLevelPressure', 'DryBulbTemperature', 'WetBulbTemperature',
                'DewPointTemperature', 'RelativeHumidity', 'Rain', 'RainIntensity',
                'RainCumulative', 'CloudHeight', 'Visibility', 'WindType',
                'CloudOktas']]
    
    elif dataType == 'custom':
        data_unified = data
    
    print('Data Unified\n')
    
    return data_unified

#%% Try to infer nan attributes of unified data from existing data attributes

def safe_GetTWetBulbFromRelHum(tdb, rh, stnLvlPres):
    try:
        return psychrolib.GetTWetBulbFromRelHum(tdb, rh, stnLvlPres)
    except:
        return np.nan

def safe_GetTWetBulbFromTDewPoint(tdb, tdp, stnLvlPres):
    try:
        return psychrolib.GetTWetBulbFromTDewPoint(tdb, tdp, stnLvlPres)
    except:
        return np.nan

def safe_GetWBFromRelHumAndDewPoint(tdb, rh, tdp, stnLvlPres):
    try:
        return psychrolib.GetTWetBulbFromHumRatioAndDewPoint(tdb, rh, tdp, stnLvlPres)
    except:
        return np.nan

def safe_GetRelHumFromTDewPoint(tdb, tdp):
    try:
        return psychrolib.GetRelHumFromTDewPoint(tdb, tdp)
    except:
        return np.nan
    
def safe_GetTDewPointFromRelHum(tdb, rh):
    try:
        return psychrolib.GetTDewPointFromRelHum(tdb, rh)
    except:
        return np.nan


    
def infer_data(data, station_altitude=0):
    """
    It is based on psychrolib
    Infer and add StationLevelPressure (station_altitude is required)
    Infer and add Wet bulb Temperature as well as Dew-Point Temperature or Relative Humidity

    Parameters
    ----------
    data : pandas.DataFrame
        Imported and unified weather data
    
    station_altitude : int, optional
        The elevation of the weather station for modifying the air
        pressure for high elevation weather stations. A stations elevation
        can be found by typing alt = wp.station_info(SID)['Elevation']

    Returns
    -------
    data_inferred : pandas.DataFrame
        Imported and inferred data
    """
    print('Inferring Data')
    
    tdb, tdp, rh, slp = data['DryBulbTemperature'], data['DewPointTemperature'], data['RelativeHumidity'], data['SeaLevelPressure']

    data_inferred = data.copy()
    
    psychrolib.SetUnitSystem(psychrolib.SI)
    
    # Get the station level pressure
    Altitude = station_altitude*np.ones_like(tdb)
    GetStationPressure_array = np.vectorize(psychrolib.GetStationPressure)
    stnLvlPres = GetStationPressure_array(slp*100, Altitude, tdb) 
    stnLvlPres[stnLvlPres > 110000] = 101000
    data_inferred['StationLevelPressure'] = stnLvlPres
    
    # Calculating wet bulb and dew-point temperature or relative humidity
    if (np.any(rh)) & (np.any(tdp)):
        print('\tCalculating WB Temperature from Relative Humidity and DP Temperature')
        safe_GetWBFromRelHumAndDewPoint_array = np.vectorize(safe_GetWBFromRelHumAndDewPoint)
        data_inferred['WetBulbTemperature'] = safe_GetWBFromRelHumAndDewPoint_array(tdb, rh, tdp, stnLvlPres)
        
    elif (not np.any(rh)) & (np.any(tdp)):
        print('\tCalculating WB Temperature from DP Temperature')
        safe_GetTWetBulbFromTDewPoint_array = np.vectorize(safe_GetTWetBulbFromTDewPoint)
        data_inferred['WetBulbTemperature'] = safe_GetTWetBulbFromTDewPoint_array(tdb, tdp, stnLvlPres)
        
        # Adding Relative Humdity to the weather data
        safe_GetRelHumFromTDewPoint_array = np.vectorize(safe_GetRelHumFromTDewPoint)
        data_inferred['RelativeHumidity'] = safe_GetRelHumFromTDewPoint_array(tdb, tdp)
        
    elif (np.any(rh)) & (not np.any(tdp)):
        print('\tCalculating WB Temperature from Relative Humidity')
        safe_GetTWetBulbFromRelHum_array = np.vectorize(safe_GetTWetBulbFromRelHum)
        data_inferred['WetBulbTemperature'] = safe_GetTWetBulbFromRelHum_array(tdb, rh, stnLvlPres)
        
        # Adding Dew point temperature to the weather data
        safe_GetTDewPointFromRelHum_array = np.vectorize(safe_GetTDewPointFromRelHum)
        data_inferred['DewPointTemperature'] = safe_GetTDewPointFromRelHum_array(tdb, rh)
    else:
        print('\tNo data could be inferred')
        print('\t--Data Inference Complete--\n')
        return data_inferred
    
    print('\t--Data Inference Complete--\n')
    return data_inferred

#%% CORRECT TERRAIN
def correct_terrain(data, stationID, dataType='BOM', source='database', correctionFactors=None):
    """
    Correct wind speed based on correction factors. 
    If a path to correctionFactors is not specified, the default location in the data folder of package will be used.

    Parameters
    ----------
    data : pandas.dataframe
        data to be corrected.
    
    stationID : str
        station number or ID to be checked against correction factors in database.
    
    dataType : str, optional
        Type of data. The default is 'BOM'.
    
    siteType : str, optional
        Description of weather station site. Default is 'ReferenceSite'
    
    source : str, optional
        Denotes from where the correction cations will be taken. Default is 'database'.
    
    correctionFactorsManual : str, optional
        Path to an alternative correction factor csv file. The default is None.
    
    Returns
    -------
    data_corrected : pandas.dataframe
        corrected wind speeds of dataframe based on provided correction factors.
    
    isTerrainCorrected : boolean
        A boolean which state whether or not the data has been corrected.
    """
    # Create a deep copy
    data_corrected = data.copy(deep = True)
    print('Correcting Terrain')
    
    if dataType=='BOM':
        
        # check if manual correction factor input is given
        if source=='database': 
            cf = station_info(stationID, printed=False)['Mean Correction Factors']
            if cf !='':
               corrFactors = [float(f) for f in cf.split(',')]
            
               print('\tCorrection factors are applied from {}'.format(source))
               print('\tCorrection Factors:')

               if len(corrFactors)==16:
                   _, windDirLbl = generate_ranges_wind_direction(16)
                   print([f'{i}:{j}' for i, j in zip(windDirLbl[1:], corrFactors)])
               else:
                   print([i for i in corrFactors])
               corrFactors.append(corrFactors[0]) # add the north value at the end for interpolation
               isTerrainCorrected = True
            else:
                print('\tNo correction factors are not found in {}'.format(source))
                isTerrainCorrected = False
                corrFactors = np.ones(17)
        elif source=='manual': 
            corrFactors = correctionFactors
            corrFactors.append(corrFactors[0]) # add the north value at the end for interpolation
            isTerrainCorrected = True
            print('\tCorrection factors are applied from {}'.format(source))
        else:
            corrFactors = np.ones(17)
            isTerrainCorrected = False

        # interpolate correction factors
        # apply correction factors to WindSpeed column
        windDirs = np.linspace(0,360,len(corrFactors))
        corrFactors_interpolated = np.interp(data.loc[:, 'WindDirection'], windDirs, corrFactors)
        data_corrected['WindSpeed'] = data_corrected['WindSpeed']*corrFactors_interpolated
    else:
        isTerrainCorrected = False
        print('\tNo correction factors are applied for NOAA data')
    
    print('\t--Terrain Correction Complete--\n')
    return data_corrected, isTerrainCorrected


#%% BOM SPEED CORRECTION OFFSET
def spdCorrection_bomOffset(data, reductionOffset=0.4, speedThreshold=2):
    """
    (experimental) Apply correction offset to wind speed for specific type anemometer of BOM weather stations. 
    The reductionOffset applies to wind speed greater than speedThreshold (For more information please contact Sina Hassanli)

    Parameters
    ----------
    data : pandas.dataframe
        data to be corrected.
    
    reductionOffset : float
        (default 0.4) reduction offset based on an study by Graeme Wood 
    
    speedThreshold : float
        (default 2) the threshold beyond which the reduction offset is applied.

    Returns
    -------
    data : pandas.dataframe
        corrected data
    """
    # copy data
    data_ = data.copy()
    
    # create and apply mask 
    mask = data_['WindSpeed']>speedThreshold
    data_valid = data_[mask]
    
    # apply reduction offset to masked cells
    data_.loc[mask,'WindSpeed'] = data_valid['WindSpeed']-reductionOffset
    
    return data_


#%% WIND SPEED CONVERSION CORRECTION 
def spdCorrection_10minTo1h(data, factor=1.05):
    """
    (experimental) Apply correction factor to convert 10min to hourly mean wind speed.

    Parameters
    ----------
    data : pandas.dataframe
        data to be corrected.
    
    factor : float
        (default 1.05) the correction factor to convert 10min to hourly mean wind speed.

    Returns
    -------
    data : pandas.dataframe
        corrected data
    """
    # copy data
    data_ = data.copy()
     
    # apply correction factor
    data_['WindSpeed'] = data_['WindSpeed']/factor
    
    return data_

#%% 1. Read in an EPW file to a Pandas dataframe

def import_data_epw(path):
    """
    Reads epw file from location
    
    Parameters
    ----------
    data : str
        A string with the filepath location of the epw file

    returns
    -------
    df : pandas.DataFrame
        pandas dataframe with the epw data
    latitude : float
        latitude coordinate
    longitude : float
        longitude coordinate
    """
    
    # Read the location
    df = pd.read_csv(path, nrows=1, header=None) # Read only the first row
    latitude = df.iloc[0, 6]
    longitude = df.iloc[0, 7]
    timezone = df.iloc[0, 8]

    # Read the hourly data
    df = pd.read_csv(path, skiprows=range(8), header=None) # Read the file, skip the 8 header rows
    # There are no column names in the file. We need to assign them.
    df.columns = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Data Source and Uncertainty Flags',
                  'DryBulbTemperature', 'DewPointTemperature', 'RelativeHumidity', 'StationLevelPressure',
                  'ExtraterrestrialHorizontalRadiation', 'ExtraterrestrialDirectNormalRadiation', 'HorizInfra',
                  'GHI', 'DNI', 'DHI',
                  'GlobalHorizontalIlluminance', 'DirectNormalIlluminance', 'DiffuseHorizontalIlluminance', 'ZenithLuminance',
                  'WindDirection', 'WindSpeed', 'TotalSkyCover', 'OpaqueSkyCover', 'Visibility', 'CeilingHeight',
                  'PresentWeatherObservation', 'PresentWeatherCodes', 'PrecipitableWater', 'AerosolOpticalDepth',
                  'SnowDepth', 'DaysSinceLastSnowfall', 'Albedo', 'LiquidPrecipitationDepth', 'LiquidPrecipitationQuantity']
    
    index_date = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    index_date_tz = index_date.dt.tz_localize(int(timezone * 3600))
    # Set the Datetime column as the index
    df.set_index(index_date_tz, inplace=True)
    df.index.name='LocalTime'
    
    return df, latitude, longitude

#%% LEGACY TEXT DATA IMPORT

def export_txt(data, localpath):
    col_export = {
        'CloudHeight': 'ceiling_height.txt',
        'WindSpeed': 'meanwind.txt',
        'WindDirection': 'directions.txt',
        'WindGust': 'gustwind.txt',
        'SeaLevelPressure': 'pressure.txt',
        'DryBulbTemperature': 'temperature.txt',
        'WetBulbTemperature': 'wbt.txt',
        'MW1_0': 'weather_obs.txt',
        'RelativeHumidity': 'relative_humidity.txt',
        'DewPointTemperature': 'dew_point_temperature.txt',
        'Visibility': 'visibility.txt',
        
        'ReportType': 'report_type.txt',
        'QCWindDir': 'QCWindDir.txt',
        'QCWindSpeed': 'QCWindSpeed.txt',
        'QCWindSpeed': 'QCWindSpeed.txt',
        
        'Date': 'date.txt',
        }

    for col, fileout in col_export.items():
        if col == 'Date':
            date = pd.DataFrame(data.index.strftime('%Y%m%d%H%M'))
            date.to_csv(localpath.joinpath(fileout),index=False,header=False)
        else:
            try:
                data[col].to_csv(localpath.joinpath(fileout),index=False,header=False)
            except:
                print(f"Could not export {col} to {fileout}")
    return    
    
    
def import_txt(path, timezone):

    text_files = []
    for file in os.listdir(path):
        if file.endswith('.txt'):
            text_files.append(file)
    text_files.remove('station_downloaded_data.txt')
    
    df = pd.DataFrame()
    for file in text_files:
        try:
            df = pd.concat(
                [df, pd.read_csv(os.path.join(path,file),header=None)],
                axis=1
                )
        except:
            print(f"Could not import and concatenate {file}")
    
    df.columns = [text[:-4] for text in text_files]
    
    date_time_str = df['date'].apply(
        lambda x: datetime.strptime(str(x),'%Y%m%d%H%M')
        )
    
    df = df.set_index(date_time_str)
    df = df.drop(['date'], axis=1)
    
    data = df[['directions','meanwind','gustwind','weather_obs','pressure','temperature']]
    data = data.rename(columns = {
        'directions':'WindDirection',
        'meanwind':'WindSpeed',
        'gustwind':'WindGust',
        'weather_obs':'MW1_0',
        'pressure':'Pressure',
        'temperature': 'DryBulbTemperature',
        })
    data['WindType'] = 'N'

    yearStart = data.index.year.min()
    yearEnd = data.index.year.max()
    
    
    
    """
    Add timezone to dataframe and local time as a separate column
    """
    data = data.tz_localize('utc')
    
    if isinstance(timezone, str):
        data['Local Time'] = data.index.tz_convert(timezone)
    elif isinstance(timezone, (float, int)):
        data['Local Time'] = data.index + timedelta(hours=timezone)
    else:
        raise Exception('Please provide a timezone either as a region (string format) or float/integer!')
    
    # VL
    data.index.name = 'UTC'
    
    
    return data, yearStart, yearEnd


def slice_data(data, start = None, end = None, bounds = 'both'):
    '''
    slices the dataframe according to a start and end date

    Parameters
    ----------
    data : pandas.DataFrame
        A pandas dataframe with imported weather data.
        
    start : str, optional
        A string of the desired start timestamp which must be in the form:
        "YYYY MM DD hh:mm". If the desired start point is the start of the
        dataframe, this argument can be left as None. The default is None.
        
    end : TYPE, optional
        A string of the desired end timestamp which must be in the form:
        "YYYY MM DD hh:mm". If the desired end point is the end of the
        dataframe, this argument can be left as None. The default is None.
        
    bounds : str, optional
        A string to specify if the bounding timestmaps should be included or
        excluded from the sliced dataframes. The argument could be 'upper'
        which will include the end timestamp, 'lower' which will include the 
        start timestamp, or 'both' which will include both timestamps. The
        default is 'both'.

    Raises
    ------
    ValueError
        A ValueError will be raised if the start or end string arguments are
        not in the correct form.

    Returns
    -------
    data_sliced : pandas.DataFrame
        A dataframe with only the specified timestmaps between the start and 
        end timestamps.

    '''

    # Make a deep copy of the input dataframe to avoid object reference issues
    data_sliced = data.copy(deep = True)
    
    # Obtains the time zone object based on the datetime index column
    if data_sliced.index.name == 'LocalTime':
        local_tz = data_sliced.index.tz
    else:
        local_tz = pytz.utc
    
    # Sets the start and and timestamps if unspecified
    if start == None:
        start = '1900 01 01 12:00'
    if end == None:
        end = '2100 01 01 12:00'
        
    # Raise ValueError if the input timestamp strings are not in the correct format
    try:  
        start_dt = local_tz.localize(datetime.strptime(start,'%Y %m %d %H:%M'))
        end_dt = local_tz.localize(datetime.strptime(end,'%Y %m %d %H:%M'))
    except:
        raise ValueError('start/end input string must be in the format \"YYYY MM DD hh:mm\"')
    
    # Slices data depending on boundaries
    if bounds == 'lower':
        return data_sliced[(data_sliced.index >= start_dt) & (data_sliced.index < end_dt)]

    elif bounds == 'upper':
        return data_sliced[(data_sliced.index > start_dt) & (data_sliced.index <= end_dt)]

    elif bounds == 'both':
        return data_sliced[(data_sliced.index >= start_dt) & (data_sliced.index <= end_dt)]

    # Raises a ValueError bounds argument isn't defined correctly
    raise ValueError('bounds argument must be either \"upper\", \"lower\", or \"both\"')

