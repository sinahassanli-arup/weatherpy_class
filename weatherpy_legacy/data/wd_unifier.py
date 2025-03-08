"""
Module for unifying weather data from different sources.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union
from .wd_base import WeatherData

class WeatherDataUnifier(WeatherData):
    """Base class for unifying weather data from different sources."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with weather data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Weather data DataFrame
        """
        super().__init__(data)
        self.unified_columns = [
            'UTC',
            'WindDirection',
            'WindSpeed',
            'WindGust',
            'SeaLevelPressure',
            'DryBulbTemperature',
            'WetBulbTemperature',
            'DewPointTemperature',
            'RelativeHumidity',
            'Rain',
            'RainIntensity',
            'RainCumulative',
            'CloudHeight',
            'Visibility'
        ]
        
    def unify_data(self) -> pd.DataFrame:
        """
        Unify data columns to standard format.
        
        Returns
        -------
        pd.DataFrame
            Unified data
        """
        raise NotImplementedError("Subclasses must implement unify_data")
        
    def validate_unified_data(self) -> bool:
        """
        Validate unified data.
        
        Returns
        -------
        bool
            True if data is valid
        """
        missing_cols = set(self.unified_columns) - set(self.data.columns)
        if missing_cols:
            logging.warning(f"Missing unified columns: {missing_cols}")
            return False
        return True
        
    def get_unified_columns(self) -> List[str]:
        """
        Get unified column names.
        
        Returns
        -------
        List[str]
            List of unified column names
        """
        return self.unified_columns
        
    def add_unified_column(self, column: str):
        """
        Add a new unified column.
        
        Parameters
        ----------
        column : str
            Column name to add
        """
        if column not in self.unified_columns:
            self.unified_columns.append(column)
            
    def remove_unified_column(self, column: str):
        """
        Remove a unified column.
        
        Parameters
        ----------
        column : str
            Column name to remove
        """
        if column in self.unified_columns:
            self.unified_columns.remove(column)
            
    def rename_columns(self, mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Rename columns using mapping.
        
        Parameters
        ----------
        mapping : Dict[str, str]
            Column name mapping
            
        Returns
        -------
        pd.DataFrame
            Data with renamed columns
        """
        self.data = self.data.rename(columns=mapping)
        return self.data
        
    def drop_unused_columns(self) -> pd.DataFrame:
        """
        Drop columns not in unified format.
        
        Returns
        -------
        pd.DataFrame
            Data with only unified columns
        """
        cols_to_keep = [col for col in self.unified_columns if col in self.data.columns]
        self.data = self.data[cols_to_keep]
        return self.data
        
    def convert_units(self, conversions: Dict[str, callable]) -> pd.DataFrame:
        """
        Convert units using conversion functions.
        
        Parameters
        ----------
        conversions : Dict[str, callable]
            Column name to conversion function mapping
            
        Returns
        -------
        pd.DataFrame
            Data with converted units
        """
        for col, func in conversions.items():
            if col in self.data.columns:
                self.data[col] = self.data[col].apply(func)
        return self.data
        
    def standardize_datetime(self, column: str = 'UTC') -> pd.DataFrame:
        """
        Standardize datetime column.
        
        Parameters
        ----------
        column : str, optional
            Datetime column name, by default 'UTC'
            
        Returns
        -------
        pd.DataFrame
            Data with standardized datetime
        """
        if column in self.data.columns:
            self.data[column] = pd.to_datetime(self.data[column])
        return self.data
        
    def sort_by_datetime(self, column: str = 'UTC') -> pd.DataFrame:
        """
        Sort data by datetime.
        
        Parameters
        ----------
        column : str, optional
            Datetime column name, by default 'UTC'
            
        Returns
        -------
        pd.DataFrame
            Sorted data
        """
        if column in self.data.columns:
            self.data = self.data.sort_values(column)
        return self.data
        
    def validate_column_types(self) -> bool:
        """
        Validate column data types.
        
        Returns
        -------
        bool
            True if all column types are valid
        """
        expected_types = {
            'UTC': 'datetime64[ns]',
            'WindDirection': 'float64',
            'WindSpeed': 'float64',
            'WindGust': 'float64',
            'SeaLevelPressure': 'float64',
            'DryBulbTemperature': 'float64',
            'WetBulbTemperature': 'float64',
            'DewPointTemperature': 'float64',
            'RelativeHumidity': 'float64',
            'Rain': 'float64',
            'RainIntensity': 'float64',
            'RainCumulative': 'float64',
            'CloudHeight': 'float64',
            'Visibility': 'float64'
        }
        
        for col, dtype in expected_types.items():
            if col in self.data.columns:
                if str(self.data[col].dtype) != dtype:
                    logging.warning(f"Invalid type for {col}: {self.data[col].dtype} != {dtype}")
                    return False
        return True
        
    def validate_value_ranges(self) -> bool:
        """
        Validate value ranges.
        
        Returns
        -------
        bool
            True if all values are within valid ranges
        """
        valid_ranges = {
            'WindDirection': (0, 360),
            'WindSpeed': (0, 100),
            'WindGust': (0, 150),
            'SeaLevelPressure': (800, 1200),
            'DryBulbTemperature': (-80, 60),
            'WetBulbTemperature': (-80, 60),
            'DewPointTemperature': (-80, 60),
            'RelativeHumidity': (0, 100),
            'Rain': (0, 500),
            'RainIntensity': (0, 1000),
            'RainCumulative': (0, 10000),
            'CloudHeight': (0, 30000),
            'Visibility': (0, 100000)
        }
        
        for col, (min_val, max_val) in valid_ranges.items():
            if col in self.data.columns:
                if self.data[col].min() < min_val or self.data[col].max() > max_val:
                    logging.warning(f"Values out of range for {col}")
                    return False
        return True
        
class BOMDataUnifier(WeatherDataUnifier):
    """Class for unifying BOM weather data."""
    
    def unify_data(self) -> pd.DataFrame:
        """
        Unify BOM data columns to standard format.
        
        Returns
        -------
        pd.DataFrame
            Unified data
        """
        # Column mapping for BOM data
        mapping = {
            'UTC': 'UTC',
            'WindDir': 'WindDirection',
            'WindSpeed': 'WindSpeed',
            'SeaLevelPressure': 'SeaLevelPressure',
            'Temperature': 'DryBulbTemperature',
            'DewPointTemp': 'DewPointTemperature',
            'CloudHgt': 'CloudHeight',
            'CloudOktas': 'CloudCover',
            'Visibility': 'Visibility',
            'RainCumulative': 'RainCumulative'
        }
        
        # Rename columns
        self.rename_columns(mapping)
        
        # Convert units if needed
        conversions = {
            'WindSpeed': lambda x: x * 0.514444,  # knots to m/s
            'SeaLevelPressure': lambda x: x / 10,  # hPa to kPa
            'CloudHeight': lambda x: x * 100  # hundreds of feet to feet
        }
        self.convert_units(conversions)
        
        # Standardize datetime
        self.standardize_datetime()
        
        # Sort by datetime
        self.sort_by_datetime()
        
        # Drop unused columns
        self.drop_unused_columns()
        
        return self.data
        
class NOAADataUnifier(WeatherDataUnifier):
    """Class for unifying NOAA weather data."""
    
    def unify_data(self) -> pd.DataFrame:
        """
        Unify NOAA data columns to standard format.
        
        Returns
        -------
        pd.DataFrame
            Unified data
        """
        # Column mapping for NOAA data
        mapping = {
            'DATE': 'UTC',
            'WND': 'WindSpeed',
            'DIR': 'WindDirection',
            'SLP': 'SeaLevelPressure',
            'TMP': 'DryBulbTemperature',
            'DEW': 'DewPointTemperature',
            'VIS': 'Visibility',
            'PCP01': 'Rain'
        }
        
        # Rename columns
        self.rename_columns(mapping)
        
        # Convert units if needed
        conversions = {
            'WindSpeed': lambda x: x * 0.514444,  # knots to m/s
            'DryBulbTemperature': lambda x: (x - 32) * 5/9,  # F to C
            'DewPointTemperature': lambda x: (x - 32) * 5/9,  # F to C
            'Visibility': lambda x: x * 1.609344  # miles to km
        }
        self.convert_units(conversions)
        
        # Standardize datetime
        self.standardize_datetime()
        
        # Sort by datetime
        self.sort_by_datetime()
        
        # Drop unused columns
        self.drop_unused_columns()
        
        return self.data 