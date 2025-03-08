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
from .wd_stations import WeatherStationDatabase
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
                 data_type: str = 'BOM',
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
        
        # Initialize station database
        self._station_db = WeatherStationDatabase(data_type)
        
        # Get station info
        self._station_info = self.station_info
        
        # Get timezone
        self._timezone = pytz.timezone(self._station_info['Timezone Name'])
        
        # Validate inputs
        self._validate_inputs()
        
        # Validate years
        if self._year_start is not None and self._year_end is not None:
            self._year_start, self._year_end = self._validate_station_years()
        
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
    
    @property
    def station_info(self) -> Dict[str, Any]:
        """
        Get station information.
        
        Returns
        -------
        Dict[str, Any]
            Station information.
        """
        return self._station_db.get_station_info(self._station_id)
    
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
        
        # Validate interval
        if self._interval not in [1, 10, 30, 60]:
            raise ValueError("Interval must be 1, 10, 30, or 60 minutes")
    
    def _validate_station_years(self) -> Tuple[int, int]:
        """
        Validate station years.
        
        Returns
        -------
        Tuple[int, int]
            Validated start and end years.
        """
        # Get station info
        station_info = self.station_info
        
        # Get start and end years
        if self._data_type == 'BOM':
            start_year = int(station_info['Start'])
            end_year = int(station_info['End'])
        elif self._data_type == 'NOAA':
            # NOAA dates are in format 'YYYY-MM-DD'
            start_year = int(station_info['Start'].split('-')[0])
            end_year = int(station_info['End'].split('-')[0])
        else:
            start_year = 1900
            end_year = 2100
        
        # Validate years
        if self._year_start < start_year:
            print(f"Warning: Start year {self._year_start} is earlier than station's first year {start_year}")
            self._year_start = start_year
        
        if self._year_end > end_year:
            print(f"Warning: End year {self._year_end} is later than station's last year {end_year}")
            self._year_end = end_year
        
        return self._year_start, self._year_end
    
    @staticmethod
    def _get_temp_folder() -> str:
        """
        Get temporary folder path.
        
        Returns
        -------
        str
            Path to temporary folder.
        """
        # Create temporary folder if it doesn't exist
        temp_folder = os.path.join(os.path.expanduser('~'), '.weatherpy', 'temp')
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)
        
        return temp_folder
    
    def _get_cache_path(self) -> Tuple[str, str]:
        """
        Get cache path.
        
        Returns
        -------
        Tuple[str, str]
            Cache directory and file path.
        """
        # Create cache folder if it doesn't exist
        cache_dir = os.path.join(self._get_temp_folder(), 'cache', self._data_type)
        
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
    
    def _get_date_bounds(self) -> Tuple[datetime, datetime]:
        """
        Get date bounds for data import.
        
        Returns
        -------
        Tuple[datetime, datetime]
            Start and end datetime bounds.
        """
        # Get station timezone
        station_info = self.station_info
        timezone_local = pytz.timezone(station_info['Timezone Name'])
        
        # Create date bounds in local timezone
        if self._data_type == 'BOM':
            # For BOM, we use the start and end of the year
            start_date = timezone_local.localize(
                datetime.strptime(f"{self._year_start} 01 01 00:00", '%Y %m %d %H:%M'))
            end_date = timezone_local.localize(
                datetime.strptime(f"{self._year_end} 12 31 23:59", '%Y %m %d %H:%M'))
        elif self._data_type == 'NOAA':
            # For NOAA, we add buffer days before and after
            start_date = timezone_local.localize(
                datetime.strptime(f"{self._year_start-1} 12 25 00:00", '%Y %m %d %H:%M'))
            end_date = timezone_local.localize(
                datetime.strptime(f"{self._year_end+1} 01 05 23:59", '%Y %m %d %H:%M'))
        else:
            raise ValueError(f"Unsupported data type: {self._data_type}")
        
        # For UTC, convert to UTC timezone
        if self._time_zone == 'UTC':
            start_date_utc = start_date.astimezone(pytz.UTC)
            end_date_utc = end_date.astimezone(pytz.UTC)
            return start_date_utc, end_date_utc
        else:
            # For LocalTime, return the local timezone dates
            return start_date, end_date
    
    def _standardize_timezone(self, data: pd.DataFrame, to_timezone: str) -> pd.DataFrame:
        """
        Standardize timezone handling for weather data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data with datetime index
        to_timezone : str
            Target timezone ('UTC' or 'LocalTime')
            
        Returns
        -------
        pd.DataFrame
            Data with standardized timezone columns and index
        """
        if not hasattr(data.index, 'tz') or data.index.tz is None:
            raise ValueError("DataFrame index must be timezone-aware")
            
        df = data.copy()
        station_tz = self._timezone
        
        # Standardize timezone conversion
        if to_timezone == 'UTC':
            # Convert index to UTC
            df.index = df.index.tz_convert('UTC')
            df.index.name = 'UTC'
            # Add LocalTime column
            df['LocalTime'] = df.index.tz_convert(station_tz)
        else:  # LocalTime
            # Convert index to station timezone
            df.index = df.index.tz_convert(station_tz)
            df.index.name = 'LocalTime'
            # Add UTC column
            df['UTC'] = df.index.tz_convert('UTC')
            
        return df
    
    def _process_data(self, data: pd.DataFrame, yearStart: int, yearEnd: int, timeZone: str) -> pd.DataFrame:
        """
        Process and filter imported data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Raw imported data
        yearStart : int
            Start year
        yearEnd : int
            End year
        timeZone : str
            Time zone
            
        Returns
        -------
        pd.DataFrame
            Processed and filtered data
        """
        # Convert timezone first
        data = self._standardize_timezone(data, timeZone)
        
        # Get the appropriate date column for filtering
        date_col = data.index.name
        
        # Create date bounds in the appropriate timezone
        tz = pytz.UTC if timeZone == 'UTC' else self._timezone
        start_date = pd.Timestamp(f"{yearStart}-01-01", tz=tz)
        end_date = pd.Timestamp(f"{yearEnd}-12-31 23:59:59", tz=tz)
        
        # Filter data
        if self._data_type == 'BOM' and timeZone == 'UTC':
            # Special case for BOM UTC data
            data = data[(data.index.year >= yearStart) & (data.index.year <= yearEnd)]
        else:
            data = data[(data.index >= start_date) & (data.index <= end_date)]
        
        # Log import status
        if len(data) > 0:
            actual_start_year = data.index.min().year
            actual_end_year = data.index.max().year
            print(f"Data imported successfully for years {actual_start_year} to {actual_end_year}")
        else:
            print("No data found for the specified date range")
        
        return data
    
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

class BOMWeatherDataImporter(WeatherDataImporter):
    """
    Class for importing weather data from the Bureau of Meteorology (BOM).
    
    This class handles the import of weather data from BOM sources, including
    data preprocessing, caching, and timezone conversions.
    """
    
    # BOM field names mapping to our standardized names
    _bom_names = {
        'Precipitation': 'rainfall',
        'Air Temperature': 'air_temperature',
        'Dew Point': 'dew_point',
        'Relative Humidity': 'rel_humidity',
        'Wind Speed': 'wind_spd_kmh',
        'Wind Direction': 'wind_dir_deg',
        'Wind Gust': 'wind_gust_kmh',
        'Station Level Pressure': 'pres',
        'CloudOktass_col': 'cloud_oktas',
        'Visibility': 'visibility',
        'Delta-T': 'delta_t'
    }
    
    # Unified observation types for new data
    UNIFIED_OBS_TYPES_NEWDATA = [
        'obs_period_time_utc', 'wind_dir_deg', 'wind_spd_kmh', 'wind_gust_kmh',
        'pres', 'air_temperature', 'dew_point', 'delta_t', 'rel_humidity',
        'rainfall', 'visibility', 'cloud_oktas'
    ]
    
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
        # Try to read from cache first
        cached_data = self._read_from_cache()
        if cached_data is not None:
            print(f"Using cached data for BOM station {self._station_id}")
            return cached_data
        
        # Get date bounds
        start_date, end_date = self._get_date_bounds()
        
        # Import data from BOM
        print(f"Importing BOM data for station {self._station_id} from {yearStart} to {yearEnd}")
        
        # Import data using legacy function
        from weatherpy_legacy.data._bom_preparation import _import_bomhistoric
        
        # Import data - note the parameter order matches the legacy function
        data = _import_bomhistoric(
            stationID=self._station_id,
            interval=interval,
            timeZone=timeZone,
            yearStart=yearStart,
            yearEnd=yearEnd
        )
        
        # Process data
        data = self._process_bom_data(data, timeZone)
        
        # Save to cache
        if self._save_raw:
            self._save_to_cache(data)
        
        return data
    
    def _process_bom_data(self, data: pd.DataFrame, timeZone: str) -> pd.DataFrame:
        """
        Process BOM data.
        
        Parameters
        ----------
        data : pandas.DataFrame
            Raw BOM data.
        timeZone : str
            Time zone.
        
        Returns
        -------
        pandas.DataFrame
            Processed data.
        """
        df = data.copy()
        
        # Rename columns to standardized names
        for bom_name, std_name in self._bom_names.items():
            if bom_name in df.columns:
                df.rename(columns={bom_name: std_name}, inplace=True)
        
        # Ensure all required columns exist
        for col in self.UNIFIED_OBS_TYPES_NEWDATA:
            if col not in df.columns and col not in ['obs_period_time_utc']:
                df[col] = np.nan
        
        # Standardize timezone
        return self._standardize_timezone(df, timeZone)

class NOAAWeatherDataImporter(WeatherDataImporter):
    """
    Class for importing weather data from NOAA.
    
    This class handles the import of weather data from NOAA sources, including
    API requests, caching, and data preprocessing.
    """
    
    # NOAA API endpoint
    API_ENDPOINT = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data?"
    
    # NOAA field names mapping to our standardized names
    _noaa_names = {
        'TEMP': 'air_temperature',
        'DEWP': 'dew_point',
        'SLP': 'pres',
        'PRCP': 'rainfall',
        'RHUM': 'rel_humidity',
        'WDIR': 'wind_dir_deg',
        'CLDC': 'cloud_oktas',
        'VISIB': 'visibility',
        'WDSP': 'wind_spd_kmh',
        'GUST': 'wind_gust_kmh'
    }
    
    # Move _names and _mandatory_section_groups here
    _names = {
        'STATION': 'Station',
        'DATE': 'UTC',
        'LATITUDE': 'Latitude',
        'LONGITUDE': 'Longitude',
        'ELEVATION': 'Elevation',
        'NAME': 'Name',
        'REPORT_TYPE': 'ReportType',
        'SOURCE': 'Source',
        'HourlyDewPointTemperature': 'DewPointTemp',
        'HourlyDryBulbTemperature': 'Temperature',
        'HourlyPrecipitation': 'RainCumulative',
        'HourlyPresentWeatherType': 'PresentWeather',
        'HourlyPressureChange': 'PressureChange',
        'HourlyPressureTendency': 'PressureTendency',
        'HourlyRelativeHumidity': 'RelativeHumidity',
        'HourlySeaLevelPressure': 'SeaLevelPressure',
        'HourlyStationPressure': 'StationPressure',
        'HourlyVisibility': 'Visibility',
        'HourlyWetBulbTemperature': 'WetBulbTemp',
        'HourlyWindDirection': 'WindDir',
        'HourlyWindSpeed': 'WindSpeed',
        'Sunrise': 'Sunrise',
        'Sunset': 'Sunset',
        'CloudLayerHeight': 'CloudHgt',
        'CloudLayerOktas': 'CloudOktas',
        'WindType': 'WindType',
        'QualityControlWindSpeed': 'QCWindSpeed',
        'QualityControlName': 'QCName',
        'QualityControlWindDirection': 'QCWindDir'
    }
    
    _mandatory_section_groups = {
        'WND': ['WindDir', 'WindSpeed', 'WindType', 'QCWindSpeed', 'QCWindDir'],
        'CIG': ['CloudHgt', 'QCName'],
        'VIS': ['Visibility', 'QCName'],
        'TMP': ['Temperature', 'QCName'],
        'DEW': ['DewPointTemp', 'QCName'],
        'SLP': ['SeaLevelPressure', 'QCName']
    }
    
    def __init__(self, station_id: str, api_token: Optional[str] = None, **kwargs):
        """
        Initialize the NOAA weather data importer.
        
        Parameters
        ----------
        station_id : str
            NOAA station ID.
        api_token : str, optional
            NOAA API token. If None, uses the default token.
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
        
        # Set API token
        self._api_token = api_token or "YourDefaultTokenHere"  # Replace with your default token
        
        # Initialize request cache
        self._request_cache = {}
    
    @property
    def api_token(self) -> str:
        """Get the API token."""
        return self._api_token
    
    @api_token.setter
    def api_token(self, value: str):
        """Set the API token."""
        if not isinstance(value, str):
            raise ValueError("API token must be a string")
        self._api_token = value
    
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
        cached_data = self._read_from_cache()
        if cached_data is not None:
            print(f"Using cached data for NOAA station {self._station_id}")
            return cached_data

        start_date, end_date = self._get_date_bounds()
        print(f"Importing NOAA data for station {self._station_id} from {yearStart} to {yearEnd}")

        # Directly implement the logic from _getNOAA_api
        data = self._make_api_request(
            url=self.API_ENDPOINT,
            params={
                'datasetid': 'GHCND',
                'stationid': f'GHCND:{self._station_id}',
                'startdate': f'{yearStart}-01-01',
                'enddate': f'{yearEnd}-12-31',
                'units': 'metric',
                'limit': 1000
            }
        )

        if data is None:
            raise ValueError("Failed to retrieve data from NOAA API")

        df = pd.DataFrame(data['results'])
        df.rename(columns=self._noaa_names, inplace=True)

        if timeZone == 'LocalTime':
            local_time_col = 'LocalTime'
            if local_time_col in df.columns:
                start_date_local = pd.Timestamp(f"{yearStart}-01-01", tz=self._timezone)
                end_date_local = pd.Timestamp(f"{yearEnd}-12-31 23:59:59", tz=self._timezone)
                df = df[(df[local_time_col] >= start_date_local) & (df[local_time_col] <= end_date_local)]

        if self._save_raw:
            self._save_to_cache(df)

        return df
    
    def _process_noaa_data(self, data: pd.DataFrame, timeZone: str) -> pd.DataFrame:
        """
        Process NOAA data.
        
        Parameters
        ----------
        data : pandas.DataFrame
            Data to process.
        timeZone : str
            Time zone.
        
        Returns
        -------
        pandas.DataFrame
            Processed data.
        """
        df = data.copy()
        
        # Rename columns using NOAA field mappings
        df.rename(columns=self._noaa_names, inplace=True)
        
        # Standardize timezone
        return self._standardize_timezone(df, timeZone)
    
    def _make_api_request(self, url: str, params: Dict[str, Any], max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """
        Make an API request with exponential backoff retry.
        
        Parameters
        ----------
        url : str
            API URL.
        params : Dict[str, Any]
            Request parameters.
        max_retries : int, optional
            Maximum number of retries. The default is 3.
        
        Returns
        -------
        Optional[Dict[str, Any]]
            Response JSON, or None if the request failed.
        """
        # Create cache key from URL and parameters
        cache_key = f"{url}_{str(params)}"
        
        # Check if response is in cache
        if cache_key in self._request_cache:
            return self._request_cache[cache_key]
        
        # Set up headers
        headers = {
            'token': self._api_token
        }
        
        # Make request with exponential backoff
        for retry in range(max_retries):
            try:
                response = requests.get(url, params=params, headers=headers)
                
                # Check if request was successful
                if response.status_code == 200:
                    # Cache response
                    self._request_cache[cache_key] = response.json()
                    return response.json()
                
                # Handle rate limiting
                if response.status_code == 429:
                    wait_time = 2 ** retry
                    print(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                
                # Handle other errors
                print(f"API request failed with status code {response.status_code}")
                return None
                
            except Exception as e:
                print(f"Error making API request: {e}")
                wait_time = 2 ** retry
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        
        print(f"Failed to make API request after {max_retries} retries")
        return None

    def get_station_info(self) -> Dict[str, Any]:
        """
        Get station information.
        
        Returns
        -------
        Dict[str, Any]
            Station information.
        """
        return self._station_db.get_station_info(self.station_id)