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

# Import preparation modules
# from ._noaa_preparation import _fix_NOAA_dimensions as fix_noaa_dimensions
# from ._noaa_preparation import _noaa_date_bounds
# from ._bom_preparation import _import_bomhistoric, _bom_date_bounds

class WeatherDataImporter(WeatherData, ABC):
    """
    Abstract base class for importing weather data.
    
    This class provides common functionality for importing weather data from
    different sources. It handles caching, date bounds, and timezone conversions.
    Subclasses must implement the _import_from_source method.
    """
    
    API_ENDPOINT = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data?"
    
    def __init__(self, 
                 station_id: str,
                 data_type: str = 'BOM',
                 time_zone: Optional[str] = None,
                 year_start: Optional[int] = None,
                 year_end: Optional[int] = None,
                 interval: Optional[int] = None,
                 save_raw: bool = False):
        """
        Initialize the importer.
        
        Parameters
        ----------
        station_id : str
            Station ID.
        data_type : str, optional
            Type of data to import. The default is 'BOM'.
        time_zone : str, optional
            Time zone. The default is None.
        year_start : int, optional
            Start year. The default is None.
        year_end : int, optional
            End year. The default is None.
        interval : int, optional
            Interval in minutes. The default is None.
        save_raw : bool, optional
            Save raw data. The default is False.
        """
        # Initialize with empty DataFrame
        super().__init__(pd.DataFrame())
        
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
        
        # Mapping of NOAA field names to our names
        self._names = {
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
        
        # Groups of fields that must be processed together
        self._mandatory_section_groups = {
            'WND': ['WindDir', 'WindSpeed', 'WindType', 'QCWindSpeed', 'QCWindDir'],
            'CIG': ['CloudHgt', 'QCName'],
            'VIS': ['Visibility', 'QCName'],
            'TMP': ['Temperature', 'QCName'],
            'DEW': ['DewPointTemp', 'QCName'],
            'SLP': ['SeaLevelPressure', 'QCName']
        }
        
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
    
    @staticmethod
    def convert_timezone(data: pd.DataFrame, from_tz: str, to_tz: str) -> pd.DataFrame:
        """
        Convert DataFrame timezone.
        
        Parameters
        ----------
        data : pandas.DataFrame
            Data to convert.
        from_tz : str
            Source timezone.
        to_tz : str
            Target timezone.
        
        Returns
        -------
        pandas.DataFrame
            Converted data.
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Check if the index is timezone-aware
        if not hasattr(df.index, 'tz') or df.index.tz is None:
            raise ValueError("DataFrame index must be timezone-aware")
        
        # Convert index timezone
        if to_tz == 'UTC':
            # Convert to UTC
            df.index = df.index.tz_convert('UTC')
            # Rename index to 'UTC'
            df.index.name = 'UTC'
            # Add LocalTime column if it doesn't exist
            if 'LocalTime' not in df.columns:
                df['LocalTime'] = df.index.tz_convert(from_tz)
        else:
            # Convert to local timezone
            df.index = df.index.tz_convert(to_tz)
            # Rename index to 'LocalTime'
            df.index.name = 'LocalTime'
            # Add UTC column if it doesn't exist
            if 'UTC' not in df.columns:
                df['UTC'] = df.index.tz_convert('UTC')
        
        return df
    
    def import_data(self, yearStart=None, yearEnd=None, interval=None, timeZone=None, save_raw=None) -> Tuple[pd.DataFrame, int, int]:
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
        Tuple[pandas.DataFrame, int, int]
            (data, yearStart, yearEnd)
        """
        # Use instance variables if parameters are not provided
        yearStart = yearStart if yearStart is not None else self._year_start
        yearEnd = yearEnd if yearEnd is not None else self._year_end
        interval = interval if interval is not None else self._interval
        timeZone = timeZone if timeZone is not None else self._time_zone
        save_raw = save_raw if save_raw is not None else self._save_raw
        
        # Validate years
        if yearStart > yearEnd:
            raise ValueError("yearStart must be less than or equal to yearEnd")
        
        # Try to read from cache if save_raw is enabled
        if save_raw:
            try:
                data = self._read_from_cache()
                if data is not None:
                    return data, yearStart, yearEnd
            except Exception as e:
                print(f"Error reading from cache: {e}")
        
        # Import data from source
        data = self._import_from_source(yearStart, yearEnd, interval, timeZone)
        
        # Determine date bounds for filtering
        if timeZone == 'LocalTime':
            # Get the station timezone
            station_info = self.station_info
            station_timezone = pytz.timezone(station_info['Timezone Name'])
            
            start_date = pd.Timestamp(f"{yearStart}-01-01", tz=station_timezone)
            end_date = pd.Timestamp(f"{yearEnd}-12-31 23:59:59", tz=station_timezone)
            date_col = 'LocalTime'
        else:  # UTC
            start_date = pd.Timestamp(f"{yearStart}-01-01", tz='UTC')
            end_date = pd.Timestamp(f"{yearEnd}-12-31 23:59:59", tz='UTC')
            date_col = 'UTC'
        
        # Special handling for BOM data with UTC timezone to match legacy implementation exactly
        if self._data_type == 'BOM' and timeZone == 'UTC':
            # For BOM with UTC, we need to match the legacy implementation exactly
            # The legacy implementation includes all data from the start and end years
            # without filtering by the exact UTC date bounds
            print(f"Filtering data by year range: {yearStart} to {yearEnd}")
            data = data[(data.index.year >= yearStart) & (data.index.year <= yearEnd)]
        else:
            # Filter data by date range if the date_col exists in the dataframe
            if date_col in data.columns:
                data = data[(data[date_col] >= start_date) & (data[date_col] <= end_date)]
            elif date_col == data.index.name:
                data = data[(data.index >= start_date) & (data.index <= end_date)]
            else:
                print(f"Warning: {date_col} not found in data. Skipping date filtering.")
        
        # Save to cache if requested
        if save_raw:
            try:
                self._save_to_cache(data)
            except Exception as e:
                print(f"Error saving to cache: {e}")
        
        # Determine actual start and end years from the filtered data
        if len(data) > 0:
            if date_col in data.columns:
                actual_start_year = data[date_col].min().year
                actual_end_year = data[date_col].max().year
            elif date_col == data.index.name:
                actual_start_year = data.index.min().year
                actual_end_year = data.index.max().year
            else:
                actual_start_year = yearStart
                actual_end_year = yearEnd
            print(f"Data imported successfully for years {actual_start_year} to {actual_end_year}")
        else:
            print("No data found for the specified date range")
            actual_start_year = yearStart
            actual_end_year = yearEnd
        
        return data, actual_start_year, actual_end_year
    
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
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Rename columns to standardized names
        for bom_name, std_name in self._bom_names.items():
            if bom_name in df.columns:
                df.rename(columns={bom_name: std_name}, inplace=True)
        
        # Ensure all required columns exist
        for col in self.UNIFIED_OBS_TYPES_NEWDATA:
            if col not in df.columns and col not in ['obs_period_time_utc']:
                df[col] = np.nan
        
        # Set index name based on timezone
        if timeZone == 'UTC':
            df.index.name = 'UTC'
        else:
            df.index.name = 'LocalTime'
        
        return df

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
        # Try to read from cache first
        cached_data = self._read_from_cache()
        if cached_data is not None:
            print(f"Using cached data for NOAA station {self._station_id}")
            return cached_data
        
        # Get date bounds
        start_date, end_date = self._get_date_bounds()
        
        # Import data from NOAA
        print(f"Importing NOAA data for station {self._station_id} from {yearStart} to {yearEnd}")
        
        # Import data using legacy function
        from weatherpy_legacy.data._noaa_preparation import _getNOAA_api
        
        # Import data - note the parameter order matches the legacy function
        data = _getNOAA_api(
            ID=self._station_id,
            yearStart=yearStart,
            yearEnd=yearEnd,
            timeZone=timeZone
        )
        
        # Switch UTC and Local time datetime index if needed
        # This matches the behavior in the legacy _read_from_server function
        if timeZone != data.index.name:
            data = data.reset_index()
            data = data.set_index(data.columns[1])
        
        # Save to cache
        if self._save_raw:
            self._save_to_cache(data)
        
        return data
    
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
        # Rename the time column based on the requested time zone
        if timeZone == 'UTC':
            # If UTC is requested, rename the index to LocalTime (to match legacy behavior)
            data = data.rename_axis('LocalTime')
        else:
            # If local time is requested, rename the index to LocalTime
            data = data.rename_axis('LocalTime')
        
        return data
    
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