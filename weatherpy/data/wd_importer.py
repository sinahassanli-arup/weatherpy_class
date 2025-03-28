"""
Module for importing weather data from various sources.
"""

from typing import Optional, List, Dict, Tuple, Union, Any
import pandas as pd
import numpy as np
import os
import pytz
from datetime import datetime, timedelta
import requests
import time
import sys
import logging
import tempfile
import zipfile
import io
from pathlib import Path
from abc import ABC, abstractmethod
from .wd_stations import WeatherStationDatabase, WeatherStation
from .wd_base import WeatherData

class WeatherDataImporter(ABC):
    """
    Abstract base class for importing weather data.
    
    This class provides common functionality for importing weather data from
    different sources. It handles caching, date bounds, and timezone conversions.
    Subclasses must implement the _import_from_server method.
    """
    
    def __init__(self, 
                 station_id: str,
                 data_type: str = None,
                 time_zone: Optional[str] = None,
                 year_start: Optional[int] = None,
                 year_end: Optional[int] = None,
                 interval: Optional[int] = None,
                 save_raw: bool = False):
        """Initialize the importer."""
        self._station_id = station_id
        self._data_type = data_type
        
        # Set default timezone based on data_type if not provided
        if time_zone is None:
            self._time_zone = 'LocalTime' if data_type == 'BOM' else 'UTC'
        else:
            self._time_zone = time_zone
            
        self._year_start = year_start
        self._year_end = year_end
        
        # Set default interval based on data_type if not provided
        if interval is None:
            self._interval = 60 if data_type == 'BOM' else 30
        else:
            self._interval = interval
            
        self._save_raw = save_raw
        
        # Initialize station
        station_db = WeatherStationDatabase(data_type=self._data_type)
        self.station = station_db.get_station(self._station_id)

        
        # Validate inputs
        self._validate_inputs()
        
        # Validate years
        self._year_start, self._year_end = self._validate_years()
    
    @property
    def station_id(self) -> str:
        """Get the station ID."""
        return self._station_id
    
    @property
    def data_type(self) -> str:
        """Get the data type."""
        return self._data_type
    
    @property
    def time_zone(self) -> str:
        """Get the time zone."""
        return self._time_zone
    
    @property
    def year_start(self) -> Optional[int]:
        """Get the start year."""
        return self._year_start
    
    @property
    def year_end(self) -> Optional[int]:
        """Get the end year."""
        return self._year_end
    
    @property
    def interval(self) -> int:
        """Get the interval in minutes."""
        return self._interval
    
    @property
    def save_raw(self) -> bool:
        """Get the save_raw flag."""
        return self._save_raw
    
    def _validate_inputs(self):
        """Validate inputs."""
        # Validate station ID
        if not isinstance(self._station_id, str):
            raise ValueError("Station ID must be a string")
        
        # Validate data type
        if self._data_type not in ['BOM', 'NOAA']:
            raise ValueError("Data type must be 'BOM' or 'NOAA'")
        
        # Validate time zone
        if self._time_zone not in ['LocalTime', 'UTC']:
            raise ValueError("Time zone must be 'LocalTime' or 'UTC'")
        
        # Validate interval based on data type
        if self._data_type == 'BOM' and self._interval not in [1, 10, 60]:
            raise ValueError("For BOM data, interval must be 1, 10, or 60 minutes")
        elif self._data_type == 'NOAA' and self._interval != 30:
            raise ValueError("For NOAA data, interval must be 30 minutes")
    
    def _validate_years(self) -> Tuple[int, int]:
        """
        Validate station years.
        
        Returns
        -------
        Tuple[int, int]
            Start and end years.
        """
        # Validate years
        if self._year_start is not None and self._year_end is not None:
            if self._year_start > self._year_end:
                raise ValueError("year_start must be less than or equal to year_end")
        
        # Get station start and end years if station is available
        if self.station is not None:
            station_start_year = getattr(self.station, 'start_year', 1900)
            station_end_year = getattr(self.station, 'end_year', datetime.now().year)
            
            # Convert to integers if they are strings
            station_start_year = int(station_start_year)
            station_end_year = int(station_end_year)
            
            # Use user-provided years if they are within the station's range
            start_year = self._year_start if self._year_start is not None else station_start_year
            end_year = self._year_end if self._year_end is not None else station_end_year
            
            # Validate years
            if start_year < station_start_year:
                print(f"Warning: Start year {start_year} is before station start year {station_start_year}. Using station start year.")
                start_year = station_start_year
            
            if end_year > station_end_year:
                print(f"Warning: End year {end_year} is after station end year {station_end_year}. Using station end year.")
                end_year = station_end_year
        else:
            # If no station info, use provided years or defaults
            start_year = self._year_start if self._year_start is not None else 1900
            end_year = self._year_end if self._year_end is not None else datetime.now().year
        
        return start_year, end_year
    
    def _get_cache_path(self) -> Tuple[str, str]:
        """
        Get cache path.
        
        Returns
        -------
        Tuple[str, str]
            Cache directory and file path.
        """
        # Create temporary folder if it doesn't exist
        temp_folder = os.path.join(os.path.expanduser('~'), '.weatherpy', 'temp')
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)
        
        # Create cache folder if it doesn't exist
        cache_dir = os.path.join(temp_folder, self._data_type)
        
        # Create cache filename
        cache_file = f"{self._station_id}_{self._year_start}_{self._year_end}_{self._interval}_{self._time_zone}.pkl"
        
        return cache_dir, os.path.join(cache_dir, cache_file)
    
    def _read_from_cache(self) -> Optional[pd.DataFrame]:
        """
        Read data from cache.
        
        Returns
        -------
        Optional[pandas.DataFrame]
            Cached data, or None if not found.
        """
        # Get cache path
        cache_dir, cache_file = self._get_cache_path()
        
        # Check if cache file exists
        if not os.path.exists(cache_file):
            return None
        
        # Read cache file
        try:
            return pd.read_pickle(cache_file)
        except Exception as e:
            print(f"Error reading cache file: {e}")
            return None
    
    def _save_to_cache(self, data: pd.DataFrame):
        """
        Save data to cache.
        
        Parameters
        ----------
        data : pandas.DataFrame
            Data to save.
        """
        # Get cache path
        cache_dir, cache_file = self._get_cache_path()
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # Save data to cache
        try:
            data.to_pickle(cache_file)
            print(f"Saved data to cache: {cache_file}")
        except Exception as e:
            print(f"Error saving cache file: {e}")
    
    @abstractmethod
    def _import_from_server(self, year_start: int, year_end: int, interval: int, time_zone: str) -> pd.DataFrame:
        """
        Import data from the source.
        
        Parameters
        ----------
        year_start : int
            Start year.
        year_end : int
            End year.
        interval : int
            Interval in minutes.
        time_zone : str
            Time zone.
        
        Returns
        -------
        pandas.DataFrame
            Imported data.
        """
        pass

    def import_data(self, year_start=None, year_end=None, interval=None, time_zone=None, save_raw=None) -> WeatherData:
        """
        Import data from the source.
        
        Parameters
        ----------
        year_start : int, optional
            Start year. If None, uses the value from initialization.
        year_end : int, optional
            End year. If None, uses the value from initialization.
        interval : int, optional
            Interval in minutes. If None, uses the value from initialization.
        time_zone : str, optional
            Time zone. If None, uses the value from initialization.
        save_raw : bool, optional
            Save raw data. If None, uses the value from initialization.
        
        Returns
        -------
        WeatherData
            Imported weather data object
        """
        # Use instance variables if parameters are not provided
        year_start = year_start if year_start is not None else self._year_start
        year_end = year_end if year_end is not None else self._year_end
        interval = interval if interval is not None else self._interval
        time_zone = time_zone if time_zone is not None else self._time_zone
        save_raw = save_raw if save_raw is not None else self._save_raw
        
        # Try to read from cache first
        data = self._read_from_cache()
        if data is not None:
            print(f"Using cached data for station {self._station_id}")
            return WeatherData(data, self.station)
        
        # Import data from source
        data = self._import_from_server(year_start, year_end, interval, time_zone)
        
        # If user wants timezone that is different from the default time_zone, swap them
        if time_zone != data.index.name:
            data.reset_index(inplace=True)
            data.set_index(data.columns[1], inplace=True, drop=True)
            
        data = data[(data.index.year >= year_start) & (data.index.year <= year_end)]
        
        # Save to cache if requested
        if save_raw:
            try:
                self._save_to_cache(data)
            except Exception as e:
                print(f"Error saving to cache: {e}")
        
        # Create and return a WeatherData object with station information
        return WeatherData(data, self.station)

class BOMWeatherDataImporter(WeatherDataImporter):
    """
    Class for importing weather data from the Bureau of Meteorology (BOM).
    
    This class handles the import of weather data from BOM sources, including
    data preprocessing, caching, and timezone conversions.
    """
    
    def __init__(self, station_id: str, **kwargs):
        """
        Initialize the BOM weather data importer.
        
        Parameters
        ----------
        station_id : str
            BOM station ID.
        **kwargs : dict
            Additional parameters to pass to the parent class.
        """
        # Set data_type to 'BOM'
        kwargs['data_type'] = 'BOM'
        
        # Initialize parent class
        super().__init__(station_id, **kwargs)
        
        # Validate BOM station ID format
        if len(station_id) != 6:
            raise ValueError(f"BOM station ID must be 6 digits, got: {station_id}")
    
    def _import_from_server(self, year_start: int, year_end: int, interval: int, time_zone: str) -> pd.DataFrame:
        """
        Import data from BOM source.
        
        Parameters
        ----------
        year_start : int
            Start year.
        year_end : int
            End year.
        interval : int
            Interval in minutes.
        time_zone : str
            Time zone.
        
        Returns
        -------
        pandas.DataFrame
            Imported data.
        """
        print(f"Importing BOM data for station {self._station_id}")

        try:
            # Direct implementation of BOM data import
            # API URL
            url = "https://rr0yh3ttf5.execute-api.ap-southeast-2.amazonaws.com/Prod/v1/bomhistoric"

            # Preparing POST argument for request
            stationFile = f"{self._station_id}.zip" if interval == 1 else f"{self._station_id}-{interval}minute.zip"

            body = {
                "bucket": f"bomhistoric-{interval}minute",
                "stationID": stationFile
            }
            
            import requests
            
            print(f"Making API request to {url} with body: {body}")
            response_url = requests.post(url, json=body)
            signed_url = response_url.json()['body']
            signed_url_statusCode = response_url.json()['statusCode']
            
            if signed_url_statusCode != 200:
                raise ValueError(f'signed_url: {signed_url} Error code: {signed_url_statusCode}')

            print(f"Getting data from signed URL")
            response_data = requests.get(signed_url)

            if response_data.status_code == 200:
                # Read data directly from memory instead of saving to file
                from io import BytesIO
                data = pd.read_pickle(BytesIO(response_data.content), compression='zip')
                print("Data imported successfully")
            else:
                raise ValueError(f"API request failed with status code: {response_data.status_code}")
            
            return data
        except Exception as e:
            print(f"Error importing BOM data: {e}")


class NOAAWeatherDataImporter(WeatherDataImporter):
    """
    Class for importing weather data from NOAA.
    
    This class handles the import of weather data from NOAA sources, including
    API requests, caching, and data preprocessing.
    """
    
    # NOAA API endpoint - using the legacy endpoint that doesn't require a token
    API_ENDPOINT = "https://www.ncei.noaa.gov/access/services/data/v1?"

    # NOAA data types to request
    _data_types = 'WND,CIG,VIS,TMP,DEW,SLP,LONGITUDE,LATITUDE,ELEVATION,GA1,AA2'
    
    def __init__(self, station_id: str, **kwargs):
        """
        Initialize the NOAA weather data importer.
        
        Parameters
        ----------
        station_id : str
            NOAA station ID.
        **kwargs : dict
            Additional parameters to pass to the parent class.
        """
        # Set data_type to 'NOAA'
        kwargs['data_type'] = 'NOAA'
        
        # Initialize parent class
        super().__init__(station_id, **kwargs)
        
        # Validate NOAA station ID format
        if len(station_id) > 11:
            raise ValueError(f"NOAA station ID must be max 11 characters, got: {station_id}")
        
        # Initialize request cache
        self._request_cache = {}
    
    def _get_date_bounds(self, year_start: int, year_end: int) -> Tuple[datetime, datetime]:
        """
        Get UTC date bounds for NOAA API call. It will get 5 extra days before year_start and 5 after the year_end.
        The data should be clipped after import
        
        Parameters
        ----------
        year_start : int
            Start year.
        year_end : int
            End year.
            
        Returns
        -------
        Tuple[datetime, datetime]
            Start and end UTC datetime bounds (with extra days).
        """
        # Creates the upper and lower date bounds for the import
        start_date_UTC = pytz.UTC.localize(datetime.strptime(str(year_start - 1)+' 12 25 00:00','%Y %m %d %H:%M'))
        end_date_UTC = pytz.UTC.localize(datetime.strptime(str(year_end + 1)+' 01 05 23:59','%Y %m %d %H:%M'))
        
        start_date_str = start_date_UTC.strftime('%Y-%m-%dT%H:%M:00')
        end_date_str = end_date_UTC.strftime('%Y-%m-%dT%H:%M:00')
        
        return start_date_str, end_date_str
    
    def _import_from_server(self, year_start: int, year_end: int, interval: int, time_zone: str) -> pd.DataFrame:
        """
        Import data from NOAA source.
        
        Parameters
        ----------
        year_start : int
            Start year.
        year_end : int
            End year.
        interval : int
            Interval in minutes.
        time_zone : str
            Time zone.
        
        Returns
        -------
        pandas.DataFrame
            Imported data.
        """
        print(f"Importing NOAA data for station {self._station_id} from {year_start} to {year_end}")
        
        # Get date bounds for API call
        start_date, end_date = self._get_date_bounds(year_start, year_end)
        
        print(f"API date range (5 extra days either end): {start_date} to {end_date}")
        
        # Generate API request URL
        station_id_int = int(self._station_id)
        api_url = (self.API_ENDPOINT +
                  '&'.join(
                      ('dataset=global-hourly',
                       f'stations={station_id_int:011d}',
                       f'dataTypes={self._data_types}',
                       f'startDate={start_date}',
                       f'endDate={end_date}',
                       'format=json'
                       )
                      )
                  )
        
        # Make API request
        data = self._make_api_request(api_url)
        
        if data is None or len(data) == 0:
            raise ValueError("Failed to retrieve data from NOAA API")
        
        # set index
        data['DATE'] = pd.to_datetime(data['DATE'])
        data.set_index('DATE', inplace=True)
        data.index.name = 'UTC'
        data.index = data.index.tz_localize('UTC')

        # Process data groups
        mandatory_groups = {
            'WND': ['WindDirection', 'QCWindDirection', 'WindType', 'WindSpeed', 'QCWindSpeed'],
            'CIG': ['CloudHgt', 'QCCloudHgt', 'CeilingDetCode', 'CavokCode'],
            'VIS': ['Visibility', 'QCVisibility', 'VisibilityVarCode', 'QCVisVar'],
            'TMP': ['DryBulbTemperature', 'QCTemperature'],
            'DEW': ['DewPointTemperature','QCDewPoint'],
            'SLP': ['SeaLevelPressure','QCSeaLevelPressure'],
            'GA1': ['CloudOktas', 'GA2', 'GA3', 'GA4', 'GA5', 'GA6'],
            'AA2': ['AA1', 'RainCumulative', 'AA3', 'AA4']
        }
        
        # Split mandatory groups and rename
        for group_name, group_fields in mandatory_groups.items():
            if group_name in data.columns:
                data[group_fields] = data[group_name].str.split(',', expand=True)
                data.drop(group_name, axis='columns', inplace=True)
        
        if 'STATION' in data.columns:
            data.drop('STATION', axis='columns', inplace=True)
        
        # Convert numeric columns
        for col in data.columns:
            try:
                data[col] = pd.to_numeric(data[col], errors="raise")
            except:
                pass
        
        # add LocalTime as a first column
        if self.station is not None and hasattr(self.station, 'timezone_name'):
            station_tz = self.station.timezone_name
            timezone_local = pytz.timezone(station_tz)
            data.insert(loc=0, column='LocalTime', value=data.index.tz_convert(timezone_local))
        else:
            print('Local Timezone could not be found')

        
        return data
    
    def _make_api_request(self, url: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """
        Make API request with retries.
        
        Parameters
        ----------
        url : str
            API URL.
        max_retries : int, optional
            Maximum number of retries. The default is 3.
        
        Returns
        -------
        Optional[pandas.DataFrame]
            Response data as DataFrame, or None if the request failed.
        """
        # Create cache key from URL
        cache_key = url
        
        # Check if response is in cache
        if cache_key in self._request_cache:
            return self._request_cache[cache_key]
        
        # Make request with exponential backoff
        for retry in range(max_retries):
            try:
                print(f"API attempt {retry+1}/{max_retries}")
                response = requests.get(url)
                
                # Check if request was successful
                if response.status_code == 200:
                    # Parse JSON data
                    from io import StringIO
                    data = pd.read_json(StringIO(response.text))
                    
                    # Cache response
                    self._request_cache[cache_key] = data
                    return data
                
                # Handle rate limiting
                if response.status_code == 429:
                    wait_time = 2 ** retry
                    print(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                
                # Handle other errors
                print(f"API request failed with status code {response.status_code}")
                if hasattr(response, 'json'):
                    try:
                        error_data = response.json()
                        if 'errors' in error_data:
                            for err in error_data['errors']:
                                print(f"Error: {err}")
                    except:
                        pass
                return None
                
            except Exception as e:
                print(f"Error making API request: {e}")
                wait_time = 2 ** retry
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        
        return None