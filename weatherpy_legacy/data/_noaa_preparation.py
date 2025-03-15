# from ftplib import FTP
# import gzip
import requests
import sys
import time
import pandas as pd
import numpy as np
# from struct import Struct
# import io
# from tqdm import tqdm
import pytz 
from .stations import station_info
from datetime import datetime, timedelta

API_ENDPOINT = r'https://www.ncei.noaa.gov/access/services/data/v1?'

_names = {
    'DATE': 'UTC',
    'LATITUDE': 'Lat',
    'LONGITUDE': 'Lon',
    'REPORT_TYPE': 'ReportType',
    'ELEVATION':'Elev',
    'CALL_SIGN': 'CallLetters',
    'QUALITY_CONTROL': 'QCName',
    'SOURCE' : 'DataSource',
    }

_mandatorySectionGroups = {
    'WND': ['WindDir', 'QCWindDir', 'WindType', 'WindSpeed', 'QCWindSpeed'],
    'CIG': ['CloudHgt', 'QCCloudHgt', 'CeilingDetCode', 'CavokCode'],
    'VIS': ['Visibility', 'QCVisibility', 'VisibilityVarCode', 'QCVisVar'],
    'TMP': ['Temperature', 'QCTemperature'],
    'DEW': ['DewPointTemp','QCDewPoint'],
    'SLP': ['SeaLevelPressure','QCSeaLevelPressure'],
    'GA1': ['CloudOktas', 'GA2', 'GA3', 'GA4', 'GA5', 'GA6'],
    'AA2': ['AA1', 'RainCumulative', 'AA3', 'AA4']
    }

def _fix_NOAA_dimensions(dataRaw, custom_fields = {}, ignore_missing=True):
    '''
    Converts raw data into engineering units
    The following mandatory fields are maintained:

        * ``WindDir`` - mean wind direction [deg E of N]
        * ``WindSpeed`` - mean wind speed [m/s]
        * ``SeaLevelPressure`` - Sea level pressure [mbar]
        * ``Temperature`` - Temperature [Celsius]
        * ``DewPointTemp`` - Dew point temperature [Celsius]
        * ``CloudHgt`` - Ceiling height above ground level [m]
        * ``CloudOktas`` - Cloud Oktas 1-8
        * ``Visibility`` - Horizontal visibility [m]

    The following additional (optional) fields are maintained if present:

        * ``OC1_0`` - Wind gust [m/s]
        * ``MW1_0`` - Manual weather occurrence identifier
        * ``AJ1_0`` - Snow depth [cm]
        * ``RH1_0`` - Relative humidity [%]


    When missing, these fields will be nan

    Parameters
    ----------
    dataRaw : pandas.DataFrame
        Raw NOAA dataframe as imported from NOAA API server
    
    custom_fields : dict, optional
        Additional fields to be added to the output. The keys of the dictionary
        represent the columns names of the raw dataset to extract, while the
        values represent the scaling factors

    ignore_missing : bool, optional
        If True an error is raised if a field of ``custom_fields`` is missing,
        otherwise a ``KeyError`` is raised. The default is True.

    Returns
    -------
    data : pandas.DataFrame
        Rescaled dataframe

    '''

    data = pd.DataFrame(index=dataRaw.index)
    data['WindDir' ] = dataRaw['WindDir']
    data['WindSpeed'] = dataRaw['WindSpeed']/10        # [m/s]
    data['SeaLevelPressure'] = dataRaw['SeaLevelPressure']/10 # [mbar]
    data['Temperature'] = dataRaw['Temperature']/10      # [Celsius]
    data['DewPointTemp']= dataRaw['DewPointTemp']/10 # [Celsius] # Dew Temperature
    data['CloudHgt'] = dataRaw['CloudHgt'] #[m]
    
    try:
        data['CloudOktas'] = dataRaw['CloudOktas']
    except:
        data['CloudOktas'] = np.nan
    
    data['Visibility'] = dataRaw['Visibility']        # [m]
    
    try:
        data['RainCumulative'] = dataRaw['RainCumulative']/10
    except:
        data['RainCumulative'] = np.nan
    
    # Rainfall
    if 'OC1_0' in dataRaw.columns:
        data['OC1_0'] = dataRaw['OC1_0']/10
    else:
        data['OC1_0'] = np.nan

    if 'MW1_0' in dataRaw.columns:
        data['MW1_0'] = dataRaw['MW1_0']
    else:
        data['MW1_0'] = np.nan

    if 'MW1_1' in dataRaw.columns:
        data['MW1_1'] = dataRaw['MW1_1']
    else:
        data['MW1_1'] = np.nan

    if 'AJ1_0' in dataRaw.columns:
        data['AJ1_0'] = dataRaw['AJ1_0']
    else:
        data['AJ1_0'] = np.nan

    if 'RH1_2' in dataRaw.columns:
        data['RH1_2'] = dataRaw['RH1_2']
    else:
        data['RH1_2'] = np.nan
        
    if 'GA1_0' in dataRaw.columns:
        data['GA1_0'] = dataRaw['GA1_0']
    else:
        data['GA1_0'] = np.nan
    

    for field, scaling_factor in custom_fields.items():
        if field in dataRaw:
            data[field] = dataRaw[field]/scaling_factor
        else:
            if not ignore_missing:
                raise(KeyError(field))
            else:
                data[field] = np.nan
    
    data['WindType'] = dataRaw['WindType']
    data['ReportType']= dataRaw['ReportType']
    data['QCWindSpeed']= dataRaw['QCWindSpeed']
    data['QCName'] = dataRaw['QCName']
    data['QCWindDir'] = dataRaw['QCWindDir']

    return data


def _noaa_date_bounds(ID, yearStart, yearEnd, timeZone):
    '''
    The function is used to generate the upper and lower date bounds for defining a data import
    The function takes in two years in integers, the stations ID, and the user specified time zone.
    The function returns two timezone aware datetime objects that define the bounds of the data import.

    The result of the function is that data imports will be from 00:00 Jan 1 to 23:59 Dec 31
    regardless of whether the timezone is local or UTC

    Parameters
    ----------
    ID : str
        A weather station ID used to find the station's local timezone.
    
    yearStart : int
        The first year of the data import. Will be from 00:00 Jan 1 for UTC.
    
    yearEnd : int
        The last year of the data import. Will be to 23:59 Dec 31 for UTC.
    
    timeZone : str
        The desired timezone, either "LocalTime" or "UTC".

    Raises
    -------
    ValueError
        Raised if a timezone argument other than "UTC" or "LocalTime" is passed.

    Returns
    -------
    start_date : datetime.datetime (time zone aware)
        The datetime object denoting the start of the data import.
    
    end_date : datetime.datetime (time zone aware)
        The datetime object denoting the end of the data import.
    '''
    start_date = pytz.UTC.localize(datetime.strptime(str(yearStart)+' 01 01 00:00','%Y %m %d %H:%M'))
    end_date = pytz.UTC.localize(datetime.strptime(str(yearEnd)+' 12 31 23:59','%Y %m %d %H:%M'))
    
    if timeZone == 'UTC':
        return start_date, end_date

    elif timeZone == 'LocalTime':
        timezone_offset = station_info(ID,printed=False)['Timezone UTC']
        tz_hours = int(timezone_offset[5:7])
        tz_mins = int(timezone_offset[8:10])
        tz_delta = timedelta(hours = tz_hours, minutes = tz_mins)
        if timezone_offset[4:5] == '-': tz_delta = -tz_delta
        start_date = start_date - tz_delta
        end_date = end_date - tz_delta
        return start_date, end_date

    else:
        raise ValueError('timeZone argument must be either \"UTC\" or \"LocalTime\"')


def _getNOAA_api(ID, yearStart, yearEnd, timeZone='UTC', add_fields=[], quiet=False):
    # Prepare fields for API request
    ID = int(ID)
    data_types = 'WND,CIG,VIS,TMP,DEW,SLP,LONGITUDE,LATITUDE,ELEVATION,GA1,AA2'

    if add_fields:
        data_types = data_types + ',' + ','.join(add_fields)

    # Creates the upper and lower date bounds for the import
    start_date_UTC = pytz.UTC.localize(datetime.strptime(str(yearStart - 1)+' 12 25 00:00','%Y %m %d %H:%M'))
    end_date_UTC = pytz.UTC.localize(datetime.strptime(str(yearEnd + 1)+' 01 05 23:59','%Y %m %d %H:%M'))
    
    start_date_str = start_date_UTC.strftime('%Y-%m-%dT%H:%M:00')
    end_date_str = end_date_UTC.strftime('%Y-%m-%dT%H:%M:00')
    
    # Generate API request string
    s = (API_ENDPOINT +
         '&'.join(
             ('dataset=global-hourly',
              f'stations={ID:011d}',
              f'dataTypes={data_types}',
              f'startDate={start_date_str}',
              f'endDate={end_date_str}',
              'format=json'
              )
             )
         )


    """
    Send API request and wait for answer
    """
    numAttempts = 10
    i = 0

    # Bracket for repeat attempts at API
    while i < numAttempts:
        # Increase iteration counter and start timer
        i += 1
        time_request = time.time()
        if not quiet: print(f"\tAPI attempt {i}/{numAttempts} : ", end='')
        
        # Attempts to requests.get the data
        try:
            response = requests.get(s)
            
            # If status code is 200 (successfult API) the loop is exited
            if response.status_code == 200:
                if not quiet: print(f'Successful (fetch time: {round(time.time()-time_request,1)} sec)')
                break
            
            # If status code is not 200, the exception is triggered
            raise KeyError

        except KeyboardInterrupt:
            raise KeyboardInterrupt

        # Can be triggered by line 250 (failed connection), 253 (non-Responses object), or 258 (unsuccessful API)
        except:
            if not quiet:
                # If response object was successfully created, display error code
                if 'response' in locals():
                    print(f"Failed ({response.reason}. Code {response.status_code})")
                    del(response)
                    
                # If response object was not created, display connecion error
                else:
                    print("Failed (https connection error)")

    # Check status code
    try:    
        if response.status_code != 200:
            errors = response.json()['errors']
            errtxt = 'Error fetching data from NOAA API:\n'
            errtxt += f'Status code: {response.status_code}\n'
            errtxt += f'Error message: {response.json()["errorMessage"]}\n'
            for err in errors:
                errtxt += (f'Field "{err["field"]}": {err["message"]} '
                           f'(current value {err["value"]})\n')
            raise(requests.RequestException(errtxt))
    except NameError:
        print(f'\nResource could not be fetched after {numAttempts} iterations')
        print('NOAA server could be temporarily unavailable\n')
        print('--Exiting Script--')
        sys.exit()

    # Parse json data
    data = pd.read_json(response.text)

    # Early return if data is empty
    if len(data) == 0:
        if not quiet: print('WARNING: No data received')
        columns = list(_names.values())
        columns += [name for l in _mandatorySectionGroups.values() for name in l]
        columns.remove('Date')
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name='Date'))

    # Convert date to datetime
    data['DATE'] = pd.to_datetime(data['DATE'])

    # Rename columns according to windpy convention
    data.rename(_names, axis='columns', inplace=True)

    # Split mandatory groups and rename
    for group_name, group_fields in _mandatorySectionGroups.items():
        if group_name in data.columns:
            data[group_fields] = data[group_name].str.split(',', expand=True)
            data.drop(group_name, axis='columns', inplace=True)

    # Split additional groups and rename
    for group_name in add_fields:
        if group_name not in data.columns:
            print(f"Column '{group_name}' not in raw NOAA data. Will be replaced with NaN")
            continue
        else:
            sub = data[group_name].str.split(',', expand=True)
            sub.columns = [f'{group_name}_{i}' for i in range(sub.shape[1])]
            data[sub.columns] = sub
            data.drop(group_name, axis='columns', inplace=True)
        
        #sub = data[group_name].str.split(',', expand=True)
        #sub.columns = [f'{group_name}_{i}' for i in range(sub.shape[1])]
        #data[sub.columns] = sub
        #data.drop(group_name, axis='columns', inplace=True)

    # Set DateTime Index
    data.set_index('UTC', inplace=True)
    data.drop('STATION', axis='columns', inplace=True)

    # Try converting to numeric
    for col in data.columns:
        try:
            data[col] = pd.to_numeric(data[col], errors="raise")
        except:
            pass
    
    data = data.tz_localize('UTC')  # localize the timezone of the index to UTC
    
    # fix and convert to SI units for some of the observation types
    data_fixed = _fix_NOAA_dimensions(data)
    
    timezone_local = pytz.timezone(station_info(ID, printed=False)['Timezone Name'])
    data_fixed.insert(loc=0, column='LocalTime', value=data_fixed.index.tz_convert(timezone_local))

    return data_fixed