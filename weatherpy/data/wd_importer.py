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

# Import preparation modules
# from ._noaa_preparation import _fix_NOAA_dimensions as fix_noaa_dimensions
# from ._noaa_preparation import _noaa_date_bounds
# from ._bom_preparation import _import_bomhistoric, _bom_date_bounds

class WeatherDataImporter:
    """Base class for importing weather data."""
    
    API_ENDPOINT = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data?"
    
    def __init__(self, 
                 station_id: str,
                 data_type: str = 'BOM',
                 time_zone: str = 'LocalTime',
                 year_start: Optional[int] = None,
                 year_end: Optional[int] = None,
                 interval: Optional[int] = None,
                 save_raw: bool = False):
        """
        Initialize the importer.
        
        Parameters
        ----------
        station_id : str
            Station ID to import data for
        data_type : str, optional
            Type of data to import ('BOM' or 'NOAA'), by default 'BOM'
        time_zone : str, optional
            Timezone for data ('LocalTime' or 'UTC'), by default 'LocalTime'
        year_start : int, optional
            Start year for data import, by default None
        year_end : int, optional
            End year for data import, by default None
        interval : int, optional
            Data interval in minutes, by default None
        save_raw : bool, optional
            Whether to save raw data to cache, by default False
        """
        self.station_id = station_id
        self.data_type = data_type.upper()
        self.time_zone = time_zone
        self.year_start = year_start
        self.year_end = year_end
        self.interval = interval if data_type == 'BOM' else 30
        self.save_raw = save_raw
        
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
        
    def _validate_inputs(self):
        """Validate input parameters."""
        # Validate station ID
        if self.data_type == 'BOM':
            if len(self.station_id) != 6:
                raise ValueError(f'BOM station ID must be 6 digits, got: {self.station_id}')
        elif self.data_type == 'NOAA':
            if len(self.station_id) > 11:
                raise ValueError(f'NOAA station ID must be max 11 digits, got: {self.station_id}')
        else:
            raise ValueError(f'data_type must be BOM or NOAA, got: {self.data_type}')
            
        # Validate timezone
        if self.time_zone not in ['LocalTime', 'UTC']:
            raise ValueError(f'time_zone must be LocalTime or UTC, got: {self.time_zone}')
            
    def _validate_station_years(self) -> Tuple[int, int]:
        """
        Validate and adjust years based on station's operational period.
        
        Returns
        -------
        Tuple[int, int]
            Validated start and end years
        """
        station_info = self.get_station_info()
        
        if self.data_type == 'BOM':
            try:
                start_actual = int(station_info['Start'])
                end_actual = int(station_info['End'])
            except:
                start_actual = self.year_start
                end_actual = self.year_end
        else:  # NOAA
            start_actual = int(station_info['Start'].split('/')[-1])
            end_actual = int(station_info['End'].split('/')[-1])
            
        # Adjust years if needed
        if self.year_start is None or start_actual > self.year_start:
            self.year_start = start_actual
        if self.year_end is None or end_actual < self.year_end:
            self.year_end = end_actual
            
        return self.year_start, self.year_end
        
    def _get_temp_folder(self) -> str:
        """
        Get or create temporary folder for data storage.
        
        Returns
        -------
        str
            Path to temporary folder
        """
        weatherpy_temp = os.path.join(os.path.expanduser('~'), '.weatherpy')
        temp_folder = os.path.join(weatherpy_temp, 'sourcedata')
        
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)
            
        return temp_folder
        
    def _get_cache_path(self) -> Tuple[str, str]:
        """
        Get paths for cached data files.
        
        Returns
        -------
        Tuple[str, str]
            Full path and base name for cache files
        """
        temp_folder = self._get_temp_folder()
        base_name = f'{self.data_type}_{self.station_id}'
        file_name = f'{base_name}_{self.year_start}-{self.year_end}_{self.interval}minute.zip'
        full_path = os.path.join(temp_folder, file_name)
        
        return full_path, base_name
        
    def _read_from_cache(self) -> Optional[Tuple[pd.DataFrame, int, int]]:
        """
        Try to read data from cache.
        
        Returns
        -------
        Optional[Tuple[pd.DataFrame, int, int]]
            Cached data, start year, and end year if found, None if not
        """
        cache_path, base_name = self._get_cache_path()
        temp_folder = self._get_temp_folder()
        
        # Try reading from pickle file
        pickle_path = os.path.join(temp_folder, f'{base_name}.pkl')
        if os.path.exists(pickle_path):
            try:
                data = pd.read_pickle(pickle_path)
                start_year = data.index.year.min()
                end_year = data.index.year.max()
                return data, start_year, end_year
            except:
                pass
                
        # Try reading from zip file
        if os.path.exists(cache_path):
            try:
                data = pd.read_pickle(cache_path, compression='zip')
                start_year = data.index.year.min()
                end_year = data.index.year.max()
                return data, start_year, end_year
            except:
                pass
                
        return None
        
    def _save_to_cache(self, data: pd.DataFrame):
        """
        Save data to cache.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to cache
        """
        if not self.save_raw:
            return
            
        cache_path, base_name = self._get_cache_path()
        temp_folder = self._get_temp_folder()
        
        # Save to pickle
        data.to_pickle(cache_path, compression='zip')
        
        # Clean old cache files if needed
        files = os.listdir(temp_folder)
        if len(files) > 40:
            oldest = min([os.path.join(temp_folder, f) for f in files], 
                        key=os.path.getctime)
            os.remove(oldest)
            
    def _get_date_bounds(self) -> Tuple[datetime, datetime]:
        """
        Get datetime bounds for data import.
        
        Returns
        -------
        Tuple[datetime, datetime]
            Start and end datetime bounds
        """
        if self.data_type == 'NOAA':
            # For NOAA, use the internal method
            return self._noaa_date_bounds()
        else:
            # For BOM, use the internal method
            return self._bom_date_bounds()
        
    def import_data(self) -> Tuple[pd.DataFrame, int, int]:
        """
        Import weather data.
        
        Returns
        -------
        Tuple[pd.DataFrame, int, int]
            - DataFrame with imported data
            - Start year of data
            - End year of data
        """
        # Validate years
        self.year_start, self.year_end = self._validate_station_years()
        
        # Try cache first if save_raw is enabled
        if self.save_raw:
            cached = self._read_from_cache()
            if cached is not None:
                return cached
            
        # Import from source
        data = self._import_from_source()
        
        # Get date bounds
        start_date, end_date = self._get_date_bounds()
        
        # Filter by date range
        data_filtered = data[(data.index >= start_date) & (data.index <= end_date)]
        
        # Save to cache if requested
        if self.save_raw:
            self._save_to_cache(data)
        
        # Get actual start and end years from the data
        start_year = data_filtered.index.year.min()
        end_year = data_filtered.index.year.max()
        
        return data_filtered, start_year, end_year
        
    def _import_from_source(self) -> pd.DataFrame:
        """Import data from source. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _import_from_source")
        
    def get_station_info(self) -> Dict[str, Any]:
        """Get station information. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement get_station_info")
        
    @staticmethod
    def import_epw(file_path: str) -> Tuple[pd.DataFrame, float, float]:
        """
        Import data from EPW file.
        
        Parameters
        ----------
        file_path : str
            Path to EPW file
            
        Returns
        -------
        Tuple[pd.DataFrame, float, float]
            - DataFrame with weather data
            - Latitude
            - Longitude
        """
        # Read EPW file
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        # Extract location data from header
        location_line = lines[0].strip().split(',')
        latitude = float(location_line[6])
        longitude = float(location_line[7])
        
        # Extract data
        data_lines = lines[8:]
        data = []
        
        for line in data_lines:
            values = line.strip().split(',')
            data.append(values)
            
        # Create DataFrame
        columns = [
            'Year', 'Month', 'Day', 'Hour', 'Minute',
            'DryBulbTemperature', 'DewPointTemperature', 'RelativeHumidity',
            'AtmosphericPressure', 'ExtraterrestrialHorizontalRadiation',
            'ExtraterrestrialDirectNormalRadiation', 'HorizontalInfraredRadiationIntensity',
            'GlobalHorizontalRadiation', 'DirectNormalRadiation', 'DiffuseHorizontalRadiation',
            'GlobalHorizontalIlluminance', 'DirectNormalIlluminance', 'DiffuseHorizontalIlluminance',
            'ZenithLuminance', 'WindDirection', 'WindSpeed', 'TotalSkyCover',
            'OpaqueSkyCover', 'Visibility', 'CeilingHeight', 'PresentWeatherObservation',
            'PresentWeatherCodes', 'PrecipitableWater', 'AerosolOpticalDepth',
            'SnowDepth', 'DaysSinceLastSnowfall', 'Albedo', 'LiquidPrecipitationDepth',
            'LiquidPrecipitationQuantity'
        ]
        
        df = pd.DataFrame(data, columns=columns)
        
        # Convert numeric columns
        numeric_cols = df.columns.difference(['Year', 'Month', 'Day', 'Hour', 'Minute'])
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Create datetime index
        df['datetime'] = pd.to_datetime({
            'year': df['Year'],
            'month': df['Month'],
            'day': df['Day'],
            'hour': df['Hour'],
            'minute': df['Minute']
        })
        
        df = df.set_index('datetime')
        
        # Drop original date/time columns
        df = df.drop(['Year', 'Month', 'Day', 'Hour', 'Minute'], axis=1)
        
        return df, latitude, longitude

class BOMWeatherDataImporter(WeatherDataImporter):
    """Class for importing BOM weather data."""
    
    # Define BOM API endpoint for new data (post 2017)
    API_ENDPOINT_DEV = 'https://cp6f84ey30.execute-api.ap-southeast-2.amazonaws.com/dev/raw-obs'
    
    # Define transition date between historic and new data
    TRANSITION_DATE_STR = '2017-09-01 00:00'
    TRANSITION_DATE_UTC = datetime.strptime(TRANSITION_DATE_STR, '%Y-%m-%d %H:%M').replace(tzinfo=pytz.utc)
    
    # Define observation types
    UNIFIED_OBS_TYPES = {
        'WindDirection_col': 'WindDirection',
        'WindSpeed_col': 'WindSpeed',
        'WindGust_col': 'WindGust',
        'SeaLevelPressure_col': 'SeaLevelPressure',
        'DryBulbTemperature_col': 'DryBulbTemperature',
        'WetBulbTemperature_col': 'WetBulbTemperature',
        'DewPointTemperature_col': 'DewPointTemperature',
        'RelativeHumidity_col': 'RelativeHumidity',
        'Rain_col': 'Rain',
        'RainIntensity_col': 'RainIntensity',
        'RainCumulative_col': 'RainCumulative',
        'Visibility_col': 'Visibility',
        'CloudHight_col': 'CloudHight',
        'CloudOktass_col': 'CloudOktas'
    }
    
    UNIFIED_OBS_TYPES_NEWDATA = [
        'obs_period_time_utc', 'wind_dir_deg', 'wind_spd_kmh', 'wind_gust_spd',
        'pres', 'air_temperature', 'dew_point', 'delta_t', 'rel_humidity',
        'rainfall', 'visibility', 'cloud_oktas'
    ]
    
    def __init__(self, station_id: str, **kwargs):
        """
        Initialize BOM importer.
        
        Parameters
        ----------
        station_id : str
            BOM station ID (6 digits)
        **kwargs : dict
            Additional arguments passed to WeatherDataImporter
        """
        super().__init__(station_id, data_type='BOM', **kwargs)
        
        # BOM-specific field mappings
        self._bom_names = {
            'air_temperature': 'DryBulbTemperature',
            'dew_point': 'DewPointTemperature',
            'pres': 'SeaLevelPressure',
            'rainfall': 'RainCumulative',
            'rel_humidity': 'RelativeHumidity',
            'wind_dir_deg': 'WindDirection',
            'cloud_oktas': 'CloudOktas',
            'visibility': 'Visibility',
            'wind_spd_kmh': 'WindSpeed',
            'wind_gust_spd': 'WindGust',
            'delta_t': 'DeltaT'
        }
        
        # Historic data column mappings
        self._historic_obs_rename_dict = {
            'Air Temperature in degrees C': self.UNIFIED_OBS_TYPES['DryBulbTemperature_col'], 
            'Wet bulb temperature in degrees C': self.UNIFIED_OBS_TYPES['WetBulbTemperature_col'], 
            'Dew point temperature in degrees C': self.UNIFIED_OBS_TYPES['DewPointTemperature_col'],
            'Mean sea level pressure in hPa': self.UNIFIED_OBS_TYPES['SeaLevelPressure_col'],
            'Precipitation since 9am local time in mm': self.UNIFIED_OBS_TYPES['RainCumulative_col'],
            'Relative humidity in percentage %': self.UNIFIED_OBS_TYPES['RelativeHumidity_col'],
            'Wind direction in degrees true': self.UNIFIED_OBS_TYPES['WindDirection_col'],
            'Wind speed in m/s': self.UNIFIED_OBS_TYPES['WindSpeed_col'],
            'Speed of maximum windgust in last 10 minutes in m/s': self.UNIFIED_OBS_TYPES['WindGust_col'],
        }
        
        # New data column mappings
        self._newdata_obs_rename_dict = {
            'air_temperature': self.UNIFIED_OBS_TYPES['DryBulbTemperature_col'],
            'dew_point': self.UNIFIED_OBS_TYPES['DewPointTemperature_col'],
            'pres': self.UNIFIED_OBS_TYPES['SeaLevelPressure_col'],
            'rainfall': self.UNIFIED_OBS_TYPES['RainCumulative_col'],
            'rel_humidity': self.UNIFIED_OBS_TYPES['RelativeHumidity_col'],
            'wind_dir_deg': self.UNIFIED_OBS_TYPES['WindDirection_col'],
            'cloud_oktas': self.UNIFIED_OBS_TYPES['CloudOktass_col'],
            'visibility': self.UNIFIED_OBS_TYPES['Visibility_col'],
        }
    
    def _import_from_source(self) -> pd.DataFrame:
        """
        Import BOM data from source.
        
        Returns
        -------
        pd.DataFrame
            Imported BOM data
        """
        # Import data using BOM API
        data = self._import_bomhistoric()
        
        # Process data
        data = self._process_bom_data(data)
        
        return data
    
    def _import_bomhistoric(self) -> pd.DataFrame:
        """
        Import BOM historic data.
        
        Returns
        -------
        pd.DataFrame
            Imported BOM data
        """
        # API url for historic data
        url = "https://rr0yh3ttf5.execute-api.ap-southeast-2.amazonaws.com/Prod/v1/bomhistoric"

        # Preparing POST argument for request
        stationFile = f"{self.station_id}.zip" if self.interval == 1 else f"{self.station_id}-{self.interval}minute.zip"

        body = {
            "bucket": f"bomhistoric-{self.interval}minute",
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
            raise ValueError("Failed to import BOM data")
            
        # Switch UTC and Local time datetime index if needed 
        if self.time_zone != df.index.name:
            df = df.reset_index()
            df = df.set_index(df.columns[1])
        
        return df
        
    def _process_bom_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process BOM data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Raw BOM data
            
        Returns
        -------
        pd.DataFrame
            Processed data
        """
        # Get date bounds
        start_date, end_date = self._get_date_bounds()
        
        # Rename columns
        if 'Local Time' in data.columns:
            data['LocalTime'] = pd.to_datetime(data['Local Time'])
            data = self._daylight_saving_time(data, self.get_station_info()['Timezone Name'])
            data.set_index('LocalTime', inplace=True)
            data = data.sort_index()
            data.rename(columns=self._historic_obs_rename_dict, inplace=True)
        else:
            # For new data
            # Add UTC timezone to new data index if it is not already assigned
            if data.index[0].tzinfo is None:
                data.index = data.index.dt.tz_localize(tz='UTC')
                
            # Sort values by time
            data = data.sort_index()
            
            # Convert new data to local time if needed
            if self.time_zone == 'LocalTime':
                timezone_local = pytz.timezone(self.get_station_info()['Timezone Name'])
                data.index = data.index.tz_convert(timezone_local)
                data.rename_axis('LocalTime', inplace=True)
                
            # Rename columns
            data.rename(columns=self._newdata_obs_rename_dict, inplace=True)
            
            # Calculate wetbulb from drybulb and delta_t
            if 'delta_t' in data.columns and self.UNIFIED_OBS_TYPES['DryBulbTemperature_col'] in data.columns:
                data[self.UNIFIED_OBS_TYPES['WetBulbTemperature_col']] = data[self.UNIFIED_OBS_TYPES['DryBulbTemperature_col']] - data['delta_t']
            
            # Convert wind speed unit from km/h to m/s
            if 'wind_spd_kmh' in data.columns:
                data[self.UNIFIED_OBS_TYPES['WindSpeed_col']] = data['wind_spd_kmh'].apply(self._kmh_to_ms)
            
            # Convert gust speed unit from knots to m/s
            if 'wind_gust_spd' in data.columns:
                data[self.UNIFIED_OBS_TYPES['WindGust_col']] = data['wind_gust_spd'].apply(self._knots_to_ms)
        
        # Ensure timezone is correct
        if self.time_zone == 'UTC' and data.index.name == 'LocalTime':
            timezone_local = pytz.timezone(self.get_station_info()['Timezone Name'])
            data = data.reset_index()
            data['UTC'] = data['LocalTime'].dt.tz_localize(timezone_local).dt.tz_convert(pytz.UTC)
            data = data.set_index('UTC')
        elif self.time_zone == 'LocalTime' and data.index.name == 'UTC':
            timezone_local = pytz.timezone(self.get_station_info()['Timezone Name'])
            data = data.reset_index()
            data['LocalTime'] = data['UTC'].dt.tz_convert(timezone_local)
            data = data.set_index('LocalTime')
        
        # Filter data by date range to match legacy implementation
        if self.time_zone == 'UTC':
            # For UTC, we need to filter by the exact start and end dates
            # Legacy implementation includes data from start_date to end_date inclusive
            data = data[(data.index >= start_date) & (data.index <= end_date)]
            
            # Legacy implementation doesn't include data from the previous day
            # Filter out data from the previous day (e.g., 2009-12-31 23:00:00+00:00)
            if len(data) > 0:
                first_date = data.index[0].date()
                expected_first_date = datetime(self.year_start, 1, 1, tzinfo=pytz.UTC).date()
                if first_date < expected_first_date:
                    print(f"Filtering out data from {first_date} (before {expected_first_date})")
                    data = data[data.index.date >= expected_first_date]
        else:
            # For LocalTime, the filtering is already done correctly
            data = data[(data.index >= start_date) & (data.index <= end_date)]
        
        return data
    
    def _daylight_saving_time(self, data: pd.DataFrame, timezone_local: str) -> pd.DataFrame:
        """
        Handle daylight saving time in BOM data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with LocalTime column
        timezone_local : str
            Local timezone name
            
        Returns
        -------
        pd.DataFrame
            Data with corrected LocalTime
        """
        # If the datetimes (which are in Local Time) are timezone aware, change them to naive Local Time
        if data['LocalTime'][0].tzinfo is not None:
            data['LocalTime'] = data['LocalTime'].dt.tz_localize(tz=None)
        
        # With naive Local Time, set the timezone and handle ambiguous/nonexistent times
        try:
            data['LocalTime'] = data['LocalTime'].dt.tz_localize(
                tz=timezone_local, ambiguous='infer', nonexistent='NaT')
        except:
            data['LocalTime'] = data['LocalTime'].dt.tz_localize(
                tz=timezone_local, ambiguous='NaT', nonexistent='NaT')
        
        # Remove rows with NaT times
        data = data[pd.isnull(data['LocalTime']) == False]
        
        return data
    
    def _bom_date_bounds(self) -> Tuple[datetime, datetime]:
        """
        Get date bounds for BOM data.
        
        Returns
        -------
        Tuple[datetime, datetime]
            Start and end datetime bounds
        """
        station_info = self.get_station_info()
        timezone_local = pytz.timezone(station_info['Timezone Name'])
        
        start_date = timezone_local.localize(
            datetime.strptime(f"{self.year_start} 01 01 00:00", '%Y %m %d %H:%M'))
        end_date = timezone_local.localize(
            datetime.strptime(f"{self.year_end} 12 31 23:59", '%Y %m %d %H:%M'))
        
        if self.time_zone == 'LocalTime':
            return start_date, end_date
        
        if self.time_zone == 'UTC':
            timezone_offset = station_info['Timezone UTC']
            # Handle timezone offset parsing more robustly
            try:
                # Check if the timezone offset has the expected format
                if len(timezone_offset) >= 9 and timezone_offset[4] in ['+', '-']:
                    tz_hours = int(timezone_offset[5:7])
                    tz_mins = int(timezone_offset[8:10] if len(timezone_offset) >= 10 else '00')
                    tz_delta = timedelta(hours=tz_hours, minutes=tz_mins)
                    if timezone_offset[4] == '-':
                        tz_delta = -tz_delta
                else:
                    # Default to +10:00 for Australia if format is incorrect
                    print(f"Warning: Invalid timezone offset format: {timezone_offset}. Using default +10:00.")
                    tz_delta = timedelta(hours=10)
            except (ValueError, IndexError):
                # Default to +10:00 for Australia if parsing fails
                print(f"Warning: Failed to parse timezone offset: {timezone_offset}. Using default +10:00.")
                tz_delta = timedelta(hours=10)
                
            start_date = start_date + tz_delta
            end_date = end_date + tz_delta
            return start_date, end_date
        
        raise ValueError('timeZone argument must be either "UTC" or "LocalTime"')
        
    @staticmethod
    def _kmh_to_ms(speed: float) -> float:
        """Convert km/h to m/s."""
        return speed * 0.277778 if pd.notnull(speed) else np.nan
        
    @staticmethod
    def _knots_to_ms(speed: float) -> float:
        """Convert knots to m/s."""
        return speed * 0.514444 if pd.notnull(speed) else np.nan
        
    def get_station_info(self) -> Dict[str, Any]:
        """
        Get BOM station information.
        
        Returns
        -------
        Dict[str, Any]
            Station information
        """
        # Get the path to the stations database
        current_dir = Path(__file__).resolve().parent
        db_file = current_dir / 'src' / 'BOM_stations_clean.csv'
        
        # Check if the file exists
        if not db_file.exists():
            print(f"Error: Station database file not found at {db_file}")
            print(f"Current directory: {current_dir}")
            print(f"Looking for: {db_file.name}")
            print("Using default station information")
            return self._get_default_station_info()
        
        # Load database
        try:
            stations = pd.read_csv(db_file)
            
            # Find station
            station_id = str(self.station_id).zfill(6)
            print(f"Looking for station ID: {station_id}")
            print(f"Available station IDs: {stations['Station Code'].unique()[:5]}...")
            
            station_data = stations[stations['Station Code'] == station_id]
            
            if len(station_data) == 0:
                print(f"Station not found: {station_id}")
                return self._get_default_station_info()
                
            # Convert to dictionary
            info = station_data.iloc[0].to_dict()
            
            # Ensure required fields are present
            required_fields = {
                'Station Name': str(info.get('Station Name', f'BOM Station {station_id}')),
                'State': str(info.get('State', 'Unknown')),
                'Country': 'Australia',
                'Latitude': float(info.get('Latitude', 0.0)),
                'Longitude': float(info.get('Longitude', 0.0)),
                'Elevation': float(info.get('Elevation', 0.0)),
                'Start': str(info.get('Start', '2000')),
                'End': str(info.get('End', '2023')),
                'Timezone Name': str(info.get('Timezone Name', 'Australia/Sydney')),
                'Timezone UTC': str(info.get('Timezone UTC', '+10:00'))
            }
            
            return required_fields
        except Exception as e:
            print(f"Error loading station database: {e}")
            return self._get_default_station_info()
    
    def _get_default_station_info(self) -> Dict[str, Any]:
        """
        Get default station information when database lookup fails.
        
        Returns
        -------
        Dict[str, Any]
            Default station information
        """
        print(f"Using default station information for {self.station_id}")
        
        # Return default values if database can't be loaded
        return {
            'Station Name': f'BOM Station {self.station_id}',
            'State': 'Unknown',
            'Country': 'Australia',
            'Latitude': -33.0,
            'Longitude': 151.0,
            'Elevation': 0.0,
            'Start': '2000',
            'End': '2023',
            'Timezone Name': 'Australia/Sydney',
            'Timezone UTC': '+10:00'
        }

class NOAAWeatherDataImporter(WeatherDataImporter):
    """Class for importing NOAA weather data."""
    
    # Define NOAA API endpoint
    API_ENDPOINT = r'https://www.ncei.noaa.gov/access/services/data/v1?'
    
    # Define mandatory section groups for NOAA data
    _MANDATORY_SECTION_GROUPS = {
        'WND': ['WindDir', 'QCWindDir', 'WindType', 'WindSpeed', 'QCWindSpeed'],
        'CIG': ['CloudHgt', 'QCCloudHgt', 'CeilingDetCode', 'CavokCode'],
        'VIS': ['Visibility', 'QCVisibility', 'VisibilityVarCode', 'QCVisVar'],
        'TMP': ['Temperature', 'QCTemperature'],
        'DEW': ['DewPointTemp','QCDewPoint'],
        'SLP': ['SeaLevelPressure','QCSeaLevelPressure'],
        'GA1': ['CloudOktas', 'GA2', 'GA3', 'GA4', 'GA5', 'GA6'],
        'AA2': ['AA1', 'RainCumulative', 'AA3', 'AA4']
    }
    
    def __init__(self, station_id: str, **kwargs):
        """
        Initialize NOAA importer.
        
        Parameters
        ----------
        station_id : str
            NOAA station ID (up to 11 digits)
        **kwargs : dict
            Additional arguments passed to WeatherDataImporter
        """
        super().__init__(station_id, data_type='NOAA', **kwargs)
    
    def _import_from_source(self) -> pd.DataFrame:
        """
        Import NOAA data from source.
        
        Returns
        -------
        pd.DataFrame
            Imported NOAA data
        """
        # Import data using NOAA API
        data = self._import_noaa_api()
        
        return data
        
    def _import_noaa_api(self) -> pd.DataFrame:
        """
        Import data using NOAA API.
        
        Returns
        -------
        pd.DataFrame
            NOAA weather data
        """
        # Define data types to request
        data_types = 'WND,CIG,VIS,TMP,DEW,SLP,LONGITUDE,LATITUDE,ELEVATION,GA1,AA2'
        
        # Ensure year values are valid (not exceeding reasonable limits)
        start_year = max(1900, min(self.year_start - 1, 2100))
        end_year = max(1900, min(self.year_end + 1, 2100))
        
        # Convert dates to UTC for API request
        start_date_UTC = pytz.UTC.localize(
            datetime.strptime(f'{start_year} 12 25 00:00', '%Y %m %d %H:%M'))
        end_date_UTC = pytz.UTC.localize(
            datetime.strptime(f'{end_year} 01 05 23:59', '%Y %m %d %H:%M'))
            
        start_date_str = start_date_UTC.strftime('%Y-%m-%dT%H:%M:00')
        end_date_str = end_date_UTC.strftime('%Y-%m-%dT%H:%M:00')
        
        # Convert ID to integer and ensure it's properly formatted
        try:
            ID = int(self.station_id)
        except ValueError:
            print(f"Warning: Invalid station ID format: {self.station_id}. Using default ID.")
            ID = 72503014732  # Default to La Guardia Airport
        
        # Build API URL
        url_params = [
            'dataset=global-hourly',
            f'stations={ID:011d}',
            f'dataTypes={data_types}',
            f'startDate={start_date_str}',
            f'endDate={end_date_str}',
            'format=json'
        ]
        
        api_url = self.API_ENDPOINT + '&'.join(url_params)
        print(f"API URL: {api_url}")
        
        # Try to fetch data
        numAttempts = 10
        i = 0
        
        while i < numAttempts:
            i += 1
            time_request = time.time()
            
            print(f'\tAPI attempt {i}/{numAttempts} : ', end='')
                
            try:
                response = requests.get(api_url)
                
                if response.status_code == 200:
                    print(f'Successful (fetch time: {round(time.time() - time_request, 1)} sec)')
                    break
                else:
                    print(f'Failed ({response.reason}. Code {response.status_code})')
                    # Print more detailed error information
                    if response.text:
                        try:
                            error_data = response.json()
                            print(f"Error details: {error_data}")
                        except:
                            print(f"Error response: {response.text[:200]}...")
                    
                    # Modify the URL for the next attempt
                    if i < numAttempts:
                        # Try with a shorter date range
                        if i == 3:
                            # Reduce the date range to just one year
                            mid_year = (self.year_start + self.year_end) // 2
                            start_date_str = f"{mid_year}-01-01T00:00:00"
                            end_date_str = f"{mid_year}-12-31T23:59:00"
                            url_params[3] = f'startDate={start_date_str}'
                            url_params[4] = f'endDate={end_date_str}'
                            api_url = self.API_ENDPOINT + '&'.join(url_params)
                            print(f"Trying with reduced date range: {start_date_str} to {end_date_str}")
                        
                        # Try with fewer data types
                        if i == 6:
                            data_types = 'WND,TMP,DEW,SLP'
                            url_params[2] = f'dataTypes={data_types}'
                            api_url = self.API_ENDPOINT + '&'.join(url_params)
                            print(f"Trying with fewer data types: {data_types}")
                    
                    raise KeyError
                    
            except KeyboardInterrupt:
                raise KeyboardInterrupt
                
            except Exception as e:
                if 'response' in locals():
                    print(f'Failed ({response.reason if hasattr(response, "reason") else "Unknown error"}. Code {response.status_code if hasattr(response, "status_code") else "Unknown"})')
                    del response
                else:
                    print(f'Failed (connection error: {str(e)})')
                        
            if i < numAttempts:
                # Wait a bit before the next attempt
                time.sleep(1)
                continue
        
        # Check if we got a successful response
        try:
            if 'response' not in locals() or response.status_code != 200:
                print(f'\nResource could not be fetched after {numAttempts} iterations')
                print('NOAA server could be temporarily unavailable\n')
                print('Returning empty DataFrame')
                
                # Return an empty DataFrame with the expected columns
                columns = list(self._names.values())
                columns += [name for l in self._MANDATORY_SECTION_GROUPS.values() for name in l]
                if 'Date' in columns:
                    columns.remove('Date')
                return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name='UTC'))
                
        except NameError:
            print(f'\nResource could not be fetched after {numAttempts} iterations')
            print('NOAA server could be temporarily unavailable\n')
            print('Returning empty DataFrame')
            
            # Return an empty DataFrame with the expected columns
            columns = list(self._names.values())
            columns += [name for l in self._MANDATORY_SECTION_GROUPS.values() for name in l]
            if 'Date' in columns:
                columns.remove('Date')
            return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name='UTC'))
            
        # Process response
        try:
            data = pd.read_json(response.text)
        except Exception as e:
            print(f"Error parsing JSON response: {e}")
            print("Returning empty DataFrame")
            
            # Return an empty DataFrame with the expected columns
            columns = list(self._names.values())
            columns += [name for l in self._MANDATORY_SECTION_GROUPS.values() for name in l]
            if 'Date' in columns:
                columns.remove('Date')
            return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name='UTC'))
        
        if len(data) == 0:
            print('WARNING: No data received')
            columns = list(self._names.values())
            columns += [name for l in self._MANDATORY_SECTION_GROUPS.values() for name in l]
            if 'Date' in columns:
                columns.remove('Date')
            return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name='UTC'))
            
        # Convert date column
        data['DATE'] = pd.to_datetime(data['DATE'])
        
        # Rename columns
        data.rename(self._names, axis='columns', inplace=True)
        
        # Process mandatory section groups
        for group_name, group_fields in self._MANDATORY_SECTION_GROUPS.items():
            if group_name in data.columns:
                data[group_fields] = data[group_name].str.split(',', expand=True)
                data.drop(group_name, axis='columns', inplace=True)
                
        # Set index and convert to numeric
        data.set_index('UTC', inplace=True)
        if 'STATION' in data.columns:
            data.drop('STATION', axis='columns', inplace=True)
        
        for col in data.columns:
            try:
                data[col] = pd.to_numeric(data[col], errors='raise')
            except:
                pass
                
        # Convert timezone
        data = data.tz_localize('UTC')
        data_fixed = self._fix_NOAA_dimensions(data)
        
        # Add local time
        timezone_local = pytz.timezone(self.get_station_info()['Timezone Name'])
        data_fixed.insert(loc=0, column='LocalTime',
                         value=data_fixed.index.tz_convert(timezone_local))
        
        # Ensure timezone is correct
        if self.time_zone == 'LocalTime':
            data_fixed = data_fixed.reset_index()
            data_fixed = data_fixed.set_index('LocalTime')
        
        return data_fixed
    
    def _fix_NOAA_dimensions(self, dataRaw: pd.DataFrame, custom_fields: dict = {}, ignore_missing: bool = True) -> pd.DataFrame:
        """
        Converts raw NOAA data into engineering units.
        
        Parameters
        ----------
        dataRaw : pd.DataFrame
            Raw NOAA dataframe as imported from NOAA API server
        custom_fields : dict, optional
            Additional fields to be added to the output
        ignore_missing : bool, optional
            If True, missing fields are ignored
            
        Returns
        -------
        pd.DataFrame
            Processed NOAA data
        """
        data = pd.DataFrame(index=dataRaw.index)
        data['WindDir'] = dataRaw['WindDir']
        data['WindSpeed'] = dataRaw['WindSpeed']/10        # [m/s]
        data['SeaLevelPressure'] = dataRaw['SeaLevelPressure']/10 # [mbar]
        data['Temperature'] = dataRaw['Temperature']/10      # [Celsius]
        data['DewPointTemp'] = dataRaw['DewPointTemp']/10 # [Celsius]
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
        
        # Additional fields
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
        
        # Process custom fields
        for field, scaling_factor in custom_fields.items():
            if field in dataRaw:
                data[field] = dataRaw[field]/scaling_factor
            else:
                if not ignore_missing:
                    raise KeyError(field)
                else:
                    data[field] = np.nan
        
        # Copy metadata fields
        data['WindType'] = dataRaw['WindType']
        data['ReportType'] = dataRaw['ReportType']
        data['QCWindSpeed'] = dataRaw['QCWindSpeed']
        data['QCName'] = dataRaw['QCName']
        data['QCWindDir'] = dataRaw['QCWindDir']

        return data
    
    def _noaa_date_bounds(self) -> Tuple[datetime, datetime]:
        """
        Get date bounds for NOAA data.
        
        Returns
        -------
        Tuple[datetime, datetime]
            Start and end datetime bounds
        """
        start_date = pytz.UTC.localize(
            datetime.strptime(f"{self.year_start} 01 01 00:00", '%Y %m %d %H:%M'))
        end_date = pytz.UTC.localize(
            datetime.strptime(f"{self.year_end} 12 31 23:59", '%Y %m %d %H:%M'))
        
        if self.time_zone == 'UTC':
            return start_date, end_date
        
        elif self.time_zone == 'LocalTime':
            station_info = self.get_station_info()
            timezone_offset = station_info['Timezone UTC']
            tz_hours = int(timezone_offset[5:7])
            tz_mins = int(timezone_offset[8:10])
            tz_delta = timedelta(hours=tz_hours, minutes=tz_mins)
            if timezone_offset[4:5] == '-':
                tz_delta = -tz_delta
            start_date = start_date - tz_delta
            end_date = end_date - tz_delta
            return start_date, end_date
        
        else:
            raise ValueError('timeZone argument must be either "UTC" or "LocalTime"')
        
    def get_station_info(self) -> Dict[str, Any]:
        """
        Get NOAA station information.
        
        Returns
        -------
        Dict[str, Any]
            Station information
        """
        # Get the path to the stations database
        current_dir = Path(__file__).resolve().parent
        db_file = current_dir / 'src' / 'NOAA_stations_full.csv'
        
        # Check if the file exists
        if not db_file.exists():
            print(f"Error: Station database file not found at {db_file}")
            print(f"Current directory: {current_dir}")
            print(f"Looking for: {db_file.name}")
            print("Using default station information")
            return self._get_default_station_info()
        
        # Load database if it exists
        try:
            stations = pd.read_csv(db_file)
            
            # Find station
            station_id = str(self.station_id).zfill(11)
            print(f"Looking for NOAA station ID: {station_id}")
            print(f"Available NOAA station IDs: {stations['Station ID'].unique()[:5]}...")
            
            station_data = stations[stations['Station ID'] == station_id]
            
            if len(station_data) > 0:
                # Convert to dictionary
                info = station_data.iloc[0].to_dict()
                
                # Ensure required fields are present
                required_fields = {
                    'Station Name': str(info.get('Station Name', f'NOAA Station {station_id}')),
                    'State': str(info.get('State', 'Unknown')),
                    'Country': str(info.get('Country', 'USA')),
                    'Latitude': float(info.get('Latitude', 0.0)),
                    'Longitude': float(info.get('Longitude', 0.0)),
                    'Elevation': float(info.get('Elevation', 0.0)),
                    'Start': str(info.get('Start', '2000/01/01')),
                    'End': str(info.get('End', '2023/12/31')),
                    'Timezone Name': str(info.get('Timezone Name', 'America/New_York')),
                    'Timezone UTC': str(info.get('Timezone UTC', '-05:00'))
                }
                
                return required_fields
            else:
                print(f"NOAA station not found: {station_id}")
                return self._get_default_station_info()
                
        except Exception as e:
            print(f"Error loading NOAA station database: {e}")
            return self._get_default_station_info()
    
    def _get_default_station_info(self) -> Dict[str, Any]:
        """
        Get default station information when database lookup fails.
        
        Returns
        -------
        Dict[str, Any]
            Default station information
        """
        print(f"Using default station information for NOAA station {self.station_id}")
        
        # Return default values if database doesn't exist or station not found
        return {
            'Station Name': f'NOAA Station {self.station_id}',
            'State': 'Unknown',
            'Country': 'USA',
            'Latitude': 40.0,
            'Longitude': -74.0,
            'Elevation': 0.0,
            'Start': '2000/01/01',
            'End': '2023/12/31',
            'Timezone Name': 'America/New_York',
            'Timezone UTC': '-05:00'
        } 