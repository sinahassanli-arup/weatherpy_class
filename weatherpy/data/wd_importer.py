"""
Module for importing weather data from various sources.
"""

from typing import Optional, List, Dict, Tuple, Union, Any
import pandas as pd
import numpy as np
import os
import pytz
from datetime import datetime
import requests
import time
import sys
import logging
import tempfile
import zipfile
import io
from pathlib import Path

# Import preparation modules
from ._noaa_preparation import _fix_NOAA_dimensions as fix_noaa_dimensions
from ._noaa_preparation import _noaa_date_bounds
from ._bom_preparation import _import_bomhistoric, _bom_date_bounds

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
            # For NOAA, use the date bounds from _noaa_preparation
            return _noaa_date_bounds(self.station_id, self.year_start, self.year_end, self.time_zone)
        else:
            # For BOM, use the date bounds from _bom_preparation
            return _bom_date_bounds(self.station_id, self.year_start, self.year_end, self.time_zone)
        
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
    
    def _import_from_source(self) -> pd.DataFrame:
        """
        Import BOM data from source.
        
        Returns
        -------
        pd.DataFrame
            Imported BOM data
        """
        # Import data using BOM API
        data = _import_bomhistoric(
            self.station_id, 
            interval=self.interval, 
            timeZone=self.time_zone, 
            yearStart=self.year_start, 
            yearEnd=self.year_end
        )
        
        # Process data
        data = self._process_bom_data(data)
        
        return data
        
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
        # Rename columns
        data.rename(columns=self._bom_names, inplace=True)
        
        # Convert units
        if 'WindSpeed' in data.columns:
            data['WindSpeed'] = data['WindSpeed'].apply(self._kmh_to_ms)
        if 'WindGust' in data.columns:
            data['WindGust'] = data['WindGust'].apply(self._knots_to_ms)
            
        # Calculate wet bulb temperature if possible
        if all(col in data.columns for col in ['DryBulbTemperature', 'DeltaT']):
            data['WetBulbTemperature'] = data['DryBulbTemperature'] - data['DeltaT']
            
        # Ensure timezone is correct
        if self.time_zone == 'UTC' and data.index.name == 'LocalTime':
            data = data.reset_index()
            data = data.set_index('UTC')
        elif self.time_zone == 'LocalTime' and data.index.name == 'UTC':
            data = data.reset_index()
            data = data.set_index('LocalTime')
        
        return data
        
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
        
        # Load database
        stations = pd.read_csv(db_file)
        
        # Find station
        station_id = str(self.station_id).zfill(6)
        station_data = stations[stations['Station Code'] == station_id]
        
        if len(station_data) == 0:
            raise ValueError(f"Station not found: {station_id}")
            
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

class NOAAWeatherDataImporter(WeatherDataImporter):
    """Class for importing NOAA weather data."""
    
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
        
        # Convert dates to UTC for API request
        start_date_UTC = pytz.UTC.localize(
            datetime.strptime(f'{self.year_start-1} 12 25 00:00', '%Y %m %d %H:%M'))
        end_date_UTC = pytz.UTC.localize(
            datetime.strptime(f'{self.year_end+1} 01 05 23:59', '%Y %m %d %H:%M'))
            
        start_date_str = start_date_UTC.strftime('%Y-%m-%dT%H:%M:00')
        end_date_str = end_date_UTC.strftime('%Y-%m-%dT%H:%M:00')
        
        # Convert ID to integer
        ID = int(self.station_id)
        
        # Build API URL
        s = self.API_ENDPOINT + '&'.join([
            'dataset=global-hourly',
            f'stations={ID:011d}',
            f'dataTypes={data_types}',
            f'startDate={start_date_str}',
            f'endDate={end_date_str}',
            'format=json'
        ])
        
        # Try to fetch data
        numAttempts = 10
        i = 0
        
        while i < numAttempts:
            i += 1
            time_request = time.time()
            
            print(f'\tAPI attempt {i}/{numAttempts} : ', end='')
                
            try:
                response = requests.get(s)
                
                if response.status_code == 200:
                    print(f'Successful (fetch time: {round(time.time() - time_request, 1)} sec)')
                    break
                else:
                    raise KeyError
                    
            except KeyboardInterrupt:
                raise KeyboardInterrupt
                
            except:
                if 'response' in locals():
                    print(f'Failed ({response.reason}. Code {response.status_code})')
                    del response
                else:
                    print('Failed (https connection error)')
                        
            if i < numAttempts:
                continue
                
        try:
            if response.status_code != 200:
                errors = response.json()['errors']
                errtxt = 'Error fetching data from NOAA API:\n'
                errtxt += f'Status code: {response.status_code}\n'
                errtxt += f'Error message: {response.json()["errorMessage"]}\n'
                for err in errors:
                    errtxt += f'Field "{err["field"]}": {err["message"]} (current value {err["value"]})\n'
                raise requests.RequestException(errtxt)
                
        except NameError:
            print(f'\nResource could not be fetched after {numAttempts} iterations')
            print('NOAA server could be temporarily unavailable\n')
            print('--Exiting Script--')
            sys.exit()
            
        # Process response
        data = pd.read_json(response.text)
        
        if len(data) == 0:
            print('WARNING: No data received')
            columns = list(self._names.values())
            columns += [name for l in self._mandatory_section_groups.values() for name in l]
            columns.remove('Date')
            return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name='Date'))
            
        # Convert date column
        data['DATE'] = pd.to_datetime(data['DATE'])
        
        # Rename columns
        data.rename(self._names, axis='columns', inplace=True)
        
        # Process mandatory section groups
        for group_name, group_fields in self._mandatory_section_groups.items():
            if group_name in data.columns:
                data[group_fields] = data[group_name].str.split(',', expand=True)
                data.drop(group_name, axis='columns', inplace=True)
                
        # Set index and convert to numeric
        data.set_index('UTC', inplace=True)
        data.drop('STATION', axis='columns', inplace=True)
        
        for col in data.columns:
            try:
                data[col] = pd.to_numeric(data[col], errors='raise')
            except:
                pass
                
        # Convert timezone
        data = data.tz_localize('UTC')
        data_fixed = fix_noaa_dimensions(data)
        
        # Add local time
        timezone_local = pytz.timezone(self.get_station_info()['Timezone Name'])
        data_fixed.insert(loc=0, column='LocalTime',
                         value=data_fixed.index.tz_convert(timezone_local))
        
        # Ensure timezone is correct
        if self.time_zone == 'LocalTime':
            data_fixed = data_fixed.reset_index()
            data_fixed = data_fixed.set_index('LocalTime')
        
        return data_fixed
        
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
        db_file = current_dir / 'src' / 'NOAA_stations_clean.csv'
        
        # Load database if it exists
        if db_file.exists():
            stations = pd.read_csv(db_file)
            
            # Find station
            station_id = str(self.station_id).zfill(11)
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
        
        # If database doesn't exist or station not found, return default values
        return {
            'Station Name': f'NOAA Station {self.station_id}',
            'State': 'Unknown',
            'Country': 'USA',
            'Latitude': 0.0,
            'Longitude': 0.0,
            'Elevation': 0.0,
            'Start': '2000/01/01',
            'End': '2023/12/31',
            'Timezone Name': 'America/New_York',
            'Timezone UTC': '-05:00'
        } 