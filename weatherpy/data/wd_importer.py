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
    Subclasses must implement the _import_from_source method.
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
        try:
            station_db = WeatherStationDatabase(data_type)
            self.station = station_db.get_station(self._station_id)
        except Exception as e:
            print(f"Warning: Could not get station information: {e}")
            self.station = None
        
        # Validate inputs
        self._validate_inputs()
        
        # Validate years
        self._year_start, self._year_end = self.validate_years()
        
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
    
    def validate_years(self) -> Tuple[int, int]:
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
                raise ValueError("yearStart must be less than or equal to yearEnd")
        
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
    def _import_from_source(self, yearStart: int, yearEnd: int, interval: int, timeZone: str) -> pd.DataFrame:
        """
        Import data from the source.
        
        Parameters
        ----------
        yearStart : int
            Start year.
        yearEnd : int
            End year.
        interval : int
            Interval in minutes.
        timeZone : str
            Time zone.
        
        Returns
        -------
        pandas.DataFrame
            Imported data.
        """
        pass

    def import_data(self, yearStart=None, yearEnd=None, interval=None, timeZone=None, save_raw=None) -> WeatherData:
        """
        Import data from the source.
        
        Parameters
        ----------
        yearStart : int, optional
            Start year. If None, uses the value from initialization.
        yearEnd : int, optional
            End year. If None, uses the value from initialization.
        interval : int, optional
            Interval in minutes. If None, uses the value from initialization.
        timeZone : str, optional
            Time zone. If None, uses the value from initialization.
        save_raw : bool, optional
            Save raw data. If None, uses the value from initialization.
        
        Returns
        -------
        WeatherData
            Imported weather data object
        """
        # Use instance variables if parameters are not provided
        yearStart = yearStart if yearStart is not None else self._year_start
        yearEnd = yearEnd if yearEnd is not None else self._year_end
        interval = interval if interval is not None else self._interval
        timeZone = timeZone if timeZone is not None else self._time_zone
        save_raw = save_raw if save_raw is not None else self._save_raw
        
        # Try to read from cache first
        data = self._read_from_cache()
        if data is not None:
            print(f"Using cached data for station {self._station_id}")
            return WeatherData(data, self.station)
        
        # Import data from source
        data = self._import_from_source(yearStart, yearEnd, interval, timeZone)
        
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
    
    def _import_from_source(self, yearStart: int, yearEnd: int, interval: int, timeZone: str) -> pd.DataFrame:
        """
        Import data from BOM source.
        
        Parameters
        ----------
        yearStart : int
            Start year.
        yearEnd : int
            End year.
        interval : int
            Interval in minutes.
        timeZone : str
            Time zone.
        
        Returns
        -------
        pandas.DataFrame
            Imported data.
        """
        print(f"Importing BOM data for station {self._station_id} from {yearStart} to {yearEnd}")
        
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
                import io
                data = pd.read_pickle(io.BytesIO(response_data.content), compression='zip')
                print("Data imported successfully")
            else:
                raise ValueError(f"API request failed with status code: {response_data.status_code}")
            
            # BOM data comes with both UTC and LocalTime columns
            # The index is set to LocalTime by default
            
            # If user wants UTC timezone but data index is LocalTime, swap them
            if timeZone == 'UTC' and data.index.name == 'LocalTime':
                print("Swapping index from LocalTime to UTC")
                # Check if UTC column exists
                if 'UTC' in data.columns:
                    # Save the LocalTime column
                    data['LocalTime'] = data.index
                    # Reset index and set UTC as the new index
                    data = data.reset_index().set_index('UTC')
                    data.index.name = 'UTC'
                else:
                    print("Warning: UTC column not found in data. Cannot swap index.")
            
            # Filter data based on year range
            if data.index.name == 'LocalTime':
                # Get the station timezone
                if self.station is not None and hasattr(self.station, 'timezone_name'):
                    station_tz = self.station.timezone_name

                # Ensure index has timezone info
                if not hasattr(data.index, 'tz') or data.index.tz is None:
                    data.index = pd.DatetimeIndex(data.index).tz_localize(station_tz)
                
                # Filter by year
                data = data[(data.index.year >= yearStart) & (data.index.year <= yearEnd)]
            else:  # UTC
                # Ensure index has timezone info
                if not hasattr(data.index, 'tz') or data.index.tz is None:
                    data.index = pd.DatetimeIndex(data.index).tz_localize('UTC')
                
                # Filter by year
                data = data[(data.index.year >= yearStart) & (data.index.year <= yearEnd)]
            
            print(f"Data filtered successfully. Shape: {data.shape}")
            print(f"Data index type: {type(data.index)}")
            print(f"Data index timezone: {data.index.tz if hasattr(data.index, 'tz') else 'None'}")
            print(f"Data columns: {data.columns.tolist()}")
            
            return data
        except Exception as e:
            print(f"Error importing BOM data: {e}")
            import traceback
            traceback.print_exc()
            raise

class NOAAWeatherDataImporter(WeatherDataImporter):
    """
    Class for importing weather data from NOAA.
    
    This class handles the import of weather data from NOAA sources, including
    API requests, caching, and data preprocessing.
    """
    
    # NOAA API endpoint - using the legacy endpoint that doesn't require a token
    API_ENDPOINT = "https://www.ncei.noaa.gov/access/services/data/v1?"
    
    # NOAA field names mapping to our standardized names
    _noaa_names = {
        'Temperature': 'air_temperature',
        'DewPointTemp': 'dew_point',
        'SeaLevelPressure': 'pres',
        'RainCumulative': 'rainfall',
        'CloudOktas': 'cloud_oktas',
        'Visibility': 'visibility',
        'WindDir': 'wind_dir_deg',
        'WindSpeed': 'wind_spd_kmh',
        'WindGust': 'wind_gust_kmh'
    }
    
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
    
    def _get_date_bounds(self, yearStart: int, yearEnd: int) -> Tuple[datetime, datetime]:
        """
        Get date bounds for NOAA API call.
        
        Parameters
        ----------
        yearStart : int
            Start year.
        yearEnd : int
            End year.
            
        Returns
        -------
        Tuple[datetime, datetime]
            Start and end datetime bounds.
        """
        # For NOAA, we add buffer days before and after
        start_date = datetime(yearStart - 1, 12, 25)
        end_date = datetime(yearEnd + 1, 1, 5, 23, 59, 59)
        
        return start_date, end_date
    
    def _import_from_source(self, yearStart: int, yearEnd: int, interval: int, timeZone: str) -> pd.DataFrame:
        """
        Import data from NOAA source.
        
        Parameters
        ----------
        yearStart : int
            Start year.
        yearEnd : int
            End year.
        interval : int
            Interval in minutes.
        timeZone : str
            Time zone.
        
        Returns
        -------
        pandas.DataFrame
            Imported data.
        """
        print(f"Importing NOAA data for station {self._station_id} from {yearStart} to {yearEnd}")
        
        # Get date bounds for API call
        start_date, end_date = self._get_date_bounds(yearStart, yearEnd)
        
        # Format dates for API request
        start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:00')
        end_date_str = end_date.strftime('%Y-%m-%dT%H:%M:00')
        
        print(f"API date range: {start_date_str} to {end_date_str}")
        
        # Generate API request URL
        station_id_int = int(self._station_id)
        api_url = (self.API_ENDPOINT +
                  '&'.join(
                      ('dataset=global-hourly',
                       f'stations={station_id_int:011d}',
                       f'dataTypes={self._data_types}',
                       f'startDate={start_date_str}',
                       f'endDate={end_date_str}',
                       'format=json'
                       )
                      )
                  )
        
        # Make API request
        data = self._make_api_request(api_url)
        
        if data is None or len(data) == 0:
            raise ValueError("Failed to retrieve data from NOAA API")
        
        # Convert date to datetime
        data['DATE'] = pd.to_datetime(data['DATE'])
        
        # Rename standard columns
        name_mapping = {
            'DATE': 'UTC',
            'LATITUDE': 'lat',
            'LONGITUDE': 'lon',
            'ELEVATION': 'elevation'
        }
        data.rename(name_mapping, axis='columns', inplace=True)
        
        # Process data groups
        mandatory_groups = {
            'WND': ['WindDir', 'QCWindDir', 'WindType', 'WindSpeed', 'QCWindSpeed'],
            'CIG': ['CloudHgt', 'QCCloudHgt', 'CeilingDetCode', 'CavokCode'],
            'VIS': ['Visibility', 'QCVisibility', 'VisibilityVarCode', 'QCVisVar'],
            'TMP': ['Temperature', 'QCTemperature'],
            'DEW': ['DewPointTemp','QCDewPoint'],
            'SLP': ['SeaLevelPressure','QCSeaLevelPressure'],
            'GA1': ['CloudOktas', 'GA2', 'GA3', 'GA4', 'GA5', 'GA6'],
            'AA2': ['AA1', 'RainCumulative', 'AA3', 'AA4']
        }
        
        # Split mandatory groups and rename
        for group_name, group_fields in mandatory_groups.items():
            if group_name in data.columns:
                data[group_fields] = data[group_name].str.split(',', expand=True)
                data.drop(group_name, axis='columns', inplace=True)
        
        # Set DateTime Index
        data.set_index('UTC', inplace=True)
        if 'STATION' in data.columns:
            data.drop('STATION', axis='columns', inplace=True)
        
        # Convert numeric columns
        for col in data.columns:
            try:
                data[col] = pd.to_numeric(data[col], errors="raise")
            except:
                pass
        
        # Add timezone info to index
        data.index = data.index.tz_localize('UTC')
        
        # Convert units and fix dimensions
        data = self._fix_dimensions(data)
        
        # Rename columns using NOAA field mappings
        data.rename(columns=self._noaa_names, inplace=True)
        
        # If user wants LocalTime, convert the index
        if timeZone == 'LocalTime':
            # Get the station timezone
            if self.station is not None and hasattr(self.station, 'timezone_name'):
                station_tz = self.station.timezone_name
            else:
                station_tz = 'Australia/Sydney'  # Default timezone
            
            # Convert index to LocalTime
            data['LocalTime'] = data.index.tz_convert(station_tz)
            data = data.reset_index().set_index('LocalTime')
            data.index.name = 'LocalTime'
        
        # Filter data to requested years
        data = data[(data.index.year >= yearStart) & (data.index.year <= yearEnd)]
        
        print(f"Data filtered successfully. Shape: {data.shape}")
        print(f"Data index type: {type(data.index)}")
        print(f"Data index timezone: {data.index.tz if hasattr(data.index, 'tz') else 'None'}")
        print(f"Data columns: {data.columns.tolist()}")
        
        return data
    
    def _fix_dimensions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fix dimensions and convert units for NOAA data.
        
        Parameters
        ----------
        data : pandas.DataFrame
            Raw NOAA data.
            
        Returns
        -------
        pandas.DataFrame
            Processed data with fixed dimensions and units.
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Convert wind speed from tenths of meters per second to km/h
        if 'WindSpeed' in df.columns:
            df['WindSpeed'] = df['WindSpeed'].astype(float) * 0.1 * 3.6  # 0.1 m/s to km/h
        
        # Convert temperature from tenths of degrees Celsius
        if 'Temperature' in df.columns:
            df['Temperature'] = df['Temperature'].astype(float) * 0.1
        
        # Convert dew point from tenths of degrees Celsius
        if 'DewPointTemp' in df.columns:
            df['DewPointTemp'] = df['DewPointTemp'].astype(float) * 0.1
        
        # Convert sea level pressure from tenths of hPa
        if 'SeaLevelPressure' in df.columns:
            df['SeaLevelPressure'] = df['SeaLevelPressure'].astype(float) * 0.1
        
        # Convert visibility from meters to kilometers
        if 'Visibility' in df.columns:
            df['Visibility'] = df['Visibility'].astype(float) / 1000
        
        return df
    
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
                    data = pd.read_json(response.text)
                    
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