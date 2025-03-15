"""
Module for cleaning weather data.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from .wd_base import WeatherData

class WeatherDataCleaner(WeatherData):
    """Base class for cleaning weather data."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with weather data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Weather data DataFrame
        """
        super().__init__(data)
        
    def clean_data(self) -> 'WeatherDataCleaner':
        """
        Clean data using all available methods.
        
        Returns
        -------
        WeatherDataCleaner
            Self for method chaining
        """
        raise NotImplementedError("Subclasses must implement clean_data")
        
    def clean_invalid(self, columns: List[str]) -> 'WeatherDataCleaner':
        """
        Clean invalid values in specified columns.
        
        Parameters
        ----------
        columns : List[str]
            List of columns to clean
            
        Returns
        -------
        WeatherDataCleaner
            Self for method chaining
        """
        data_copy = self.data.copy()
        for col in columns:
            if col in data_copy.columns:
                data_copy[col] = pd.to_numeric(data_copy[col], errors='coerce')
        
        # Remove rows with NaN values in specified columns (to match legacy behavior)
        mask = data_copy[columns].isna().any(axis=1)
        if mask.any():
            data_copy = data_copy[~mask]
            logging.info(f"Removed {mask.sum()} rows with invalid values in columns: {columns}")
        
        self.data = data_copy
        return self
        
    def clean_threshold(self, thresholds: Dict[str, Tuple[float, float]]) -> 'WeatherDataCleaner':
        """
        Clean values outside thresholds.
        
        Parameters
        ----------
        thresholds : Dict[str, Tuple[float, float]]
            Column name to (min, max) threshold mapping
            
        Returns
        -------
        WeatherDataCleaner
            Self for method chaining
        """
        data_copy = self.data.copy()
        rows_to_remove = pd.Series(False, index=data_copy.index)
        
        for col, (min_val, max_val) in thresholds.items():
            if col in data_copy.columns:
                mask = (data_copy[col] < min_val) | (data_copy[col] > max_val)
                rows_to_remove = rows_to_remove | mask
        
        # Remove rows outside thresholds (to match legacy behavior)
        if rows_to_remove.any():
            data_copy = data_copy[~rows_to_remove]
            logging.info(f"Removed {rows_to_remove.sum()} rows with values outside thresholds")
        
        self.data = data_copy
        return self
        
    def clean_outliers(self, columns: List[str], method: str = 'zscore', threshold: float = 3) -> 'WeatherDataCleaner':
        """
        Clean outliers in specified columns.
        
        Parameters
        ----------
        columns : List[str]
            List of columns to clean
        method : str, optional
            Outlier detection method ('zscore' or 'iqr'), by default 'zscore'
        threshold : float, optional
            Outlier threshold, by default 3
            
        Returns
        -------
        WeatherDataCleaner
            Self for method chaining
        """
        data_copy = self.data.copy()
        rows_to_remove = pd.Series(False, index=data_copy.index)
        
        for col in columns:
            if col in data_copy.columns:
                if method == 'zscore':
                    z_scores = np.abs((data_copy[col] - data_copy[col].mean()) / data_copy[col].std())
                    mask = z_scores > threshold
                    rows_to_remove = rows_to_remove | mask
                elif method == 'iqr':
                    Q1 = data_copy[col].quantile(0.25)
                    Q3 = data_copy[col].quantile(0.75)
                    IQR = Q3 - Q1
                    mask = (data_copy[col] < Q1 - threshold * IQR) | (data_copy[col] > Q3 + threshold * IQR)
                    rows_to_remove = rows_to_remove | mask
        
        # Remove outlier rows (to match legacy behavior)
        if rows_to_remove.any():
            data_copy = data_copy[~rows_to_remove]
            logging.info(f"Removed {rows_to_remove.sum()} rows with outlier values")
        
        self.data = data_copy
        return self
        
    def clean_duplicates(self) -> 'WeatherDataCleaner':
        """
        Clean duplicate rows.
        
        Returns
        -------
        WeatherDataCleaner
            Self for method chaining
        """
        self.data = self.data.drop_duplicates()
        return self
        
    def clean_missing(self, threshold: float = 0.5) -> 'WeatherDataCleaner':
        """
        Clean columns with too many missing values.
        
        Parameters
        ----------
        threshold : float, optional
            Missing value threshold, by default 0.5
            
        Returns
        -------
        WeatherDataCleaner
            Self for method chaining
        """
        data_copy = self.data.copy()
        missing_ratio = data_copy.isnull().sum() / len(data_copy)
        cols_to_drop = missing_ratio[missing_ratio > threshold].index
        data_copy = data_copy.drop(columns=cols_to_drop)
        self.data = data_copy
        return self
        
    def interpolate_missing(self, method: str = 'linear') -> 'WeatherDataCleaner':
        """
        Interpolate missing values.
        
        Parameters
        ----------
        method : str, optional
            Interpolation method, by default 'linear'
            
        Returns
        -------
        WeatherDataCleaner
            Self for method chaining
        """
        data_copy = self.data.copy()
        data_copy = data_copy.interpolate(method=method)
        self.data = data_copy
        return self
        
    def fill_missing(self, value: Any = 0) -> 'WeatherDataCleaner':
        """
        Fill missing values.
        
        Parameters
        ----------
        value : Any, optional
            Fill value, by default 0
            
        Returns
        -------
        WeatherDataCleaner
            Self for method chaining
        """
        data_copy = self.data.copy()
        data_copy = data_copy.fillna(value)
        self.data = data_copy
        return self
        
class BOMDataCleaner(WeatherDataCleaner):
    """Class for cleaning BOM weather data."""
    
    def clean_data(self) -> 'WeatherDataCleaner':
        """
        Clean BOM data using all available methods.
        
        Returns
        -------
        WeatherDataCleaner
            Self for method chaining
        """
        # Fix wind directions between 0° and 5° to 360°
        self.fix_wind_direction_0_to_5()
        
        # Round wind direction to nearest 10°
        self.round_wind_direction()
        
        # Set wind direction to 0° for calm conditions
        self.zero_calm_direction()
        
        # Clean invalid values
        self.clean_invalid(['WindSpeed', 'WindDirection'])
        
        # Clean values outside thresholds
        thresholds = {
            'WindSpeed': (0, 50),
            'PrePostRatio': (5, 30)
        }
        self.clean_threshold(thresholds)
        
        # Clean duplicates
        self.clean_duplicates()
        
        # Clean missing values
        self.clean_missing()
        
        # Interpolate remaining missing values
        self.interpolate_missing()
        
        return self
        
    def clean_calms(self) -> 'WeatherDataCleaner':
        """
        Clean calm wind conditions.
        
        Returns
        -------
        WeatherDataCleaner
            Self for method chaining
        """
        data_copy = self.data.copy()
        if 'WindSpeed' in data_copy.columns:
            calm_mask = data_copy['WindSpeed'] < 0.5
            data_copy.loc[calm_mask, 'WindSpeed'] = 0
            if 'WindDirection' in data_copy.columns:
                data_copy.loc[calm_mask, 'WindDirection'] = np.nan
        self.data = data_copy
        return self
        
    def clean_off_clock(self) -> 'WeatherDataCleaner':
        """
        Clean off-clock wind directions.
        
        Returns
        -------
        WeatherDataCleaner
            Self for method chaining
        """
        data_copy = self.data.copy()
        if 'WindDirection' in data_copy.columns:
            # Round to nearest 10 degrees
            data_copy['WindDirection'] = data_copy['WindDirection'].round(-1)
            # Set values outside 0-360 to NaN
            mask = (data_copy['WindDirection'] < 0) | (data_copy['WindDirection'] > 360)
            data_copy.loc[mask, 'WindDirection'] = np.nan
        self.data = data_copy
        return self
        
    def round_wind_direction(self) -> 'WeatherDataCleaner':
        """
        Round wind directions to nearest 10 degrees.
        
        Returns
        -------
        WeatherDataCleaner
            Self for method chaining
        """
        data_copy = self.data.copy()
        if 'WindDirection' in data_copy.columns:
            data_copy['WindDirection'] = data_copy['WindDirection'].round(-1)
            # Handle special case of 0 degrees
            data_copy.loc[data_copy['WindDirection'] == 0, 'WindDirection'] = 360
        self.data = data_copy
        return self
        
    def fix_wind_direction_0_to_5(self) -> 'WeatherDataCleaner':
        """
        Fix wind directions between 0° and 5° to 360°.
        
        Returns
        -------
        WeatherDataCleaner
            Self for method chaining
        """
        data_copy = self.data.copy()
        if 'WindDirection' in data_copy.columns:
            mask = (data_copy['WindDirection'] >= 0) & (data_copy['WindDirection'] <= 5)
            data_copy.loc[mask, 'WindDirection'] = 360
        self.data = data_copy
        return self
        
    def adjust_low_wind_direction(self) -> 'WeatherDataCleaner':
        """
        Adjust wind directions for low wind speeds.
        
        Returns
        -------
        WeatherDataCleaner
            Self for method chaining
        """
        data_copy = self.data.copy()
        if 'WindSpeed' in data_copy.columns and 'WindDirection' in data_copy.columns:
            low_wind_mask = data_copy['WindSpeed'] < 0.5
            data_copy.loc[low_wind_mask, 'WindDirection'] = np.nan
        self.data = data_copy
        return self
        
    def zero_calm_direction(self) -> 'WeatherDataCleaner':
        """
        Set wind direction to zero for calm conditions.
        
        Returns
        -------
        WeatherDataCleaner
            Self for method chaining
        """
        data_copy = self.data.copy()
        if 'WindSpeed' in data_copy.columns and 'WindDirection' in data_copy.columns:
            calm_mask = data_copy['WindSpeed'] == 0
            data_copy.loc[calm_mask, 'WindDirection'] = 0
        self.data = data_copy
        return self
        
class NOAADataCleaner(WeatherDataCleaner):
    """Class for cleaning NOAA weather data."""
    
    def clean_data(self) -> 'WeatherDataCleaner':
        """
        Clean NOAA data using all available methods.
        
        Returns
        -------
        WeatherDataCleaner
            Self for method chaining
        """
        # Clean invalid values
        self.clean_invalid(['WindSpeed', 'WindDirection'])
        
        # Clean values outside thresholds
        thresholds = {
            'WindSpeed': (0, 50),
            'PrePostRatio': (5, 30)
        }
        self.clean_threshold(thresholds)
        
        # Clean ranked rows
        self.clean_ranked_rows()
        
        # Clean VC filter
        self.clean_VC_filter()
        
        # Clean wind direction
        self.clean_direction()
        
        # Clean storm data
        self.clean_storms()
        
        # Clean duplicates
        self.clean_duplicates()
        
        # Clean missing values
        self.clean_missing()
        
        # Interpolate remaining missing values
        self.interpolate_missing()
        
        return self
        
    def clean_ranked_rows(self) -> 'WeatherDataCleaner':
        """
        Clean ranked rows.
        
        Returns
        -------
        WeatherDataCleaner
            Self for method chaining
        """
        data_copy = self.data.copy()
        if 'Rank' in data_copy.columns:
            data_copy = data_copy[data_copy['Rank'] == 1]
        self.data = data_copy
        return self
        
    def clean_VC_filter(self) -> 'WeatherDataCleaner':
        """
        Clean variable/changeable weather codes.
        
        Returns
        -------
        WeatherDataCleaner
            Self for method chaining
        """
        data_copy = self.data.copy()
        if 'WeatherCode' in data_copy.columns:
            vc_mask = data_copy['WeatherCode'].str.contains('VC', na=False)
            data_copy.loc[vc_mask, 'WeatherCode'] = np.nan
        self.data = data_copy
        return self
        
    def clean_direction(self) -> 'WeatherDataCleaner':
        """
        Clean wind direction data.
        
        Returns
        -------
        WeatherDataCleaner
            Self for method chaining
        """
        data_copy = self.data.copy()
        if 'WindDirection' in data_copy.columns:
            # Convert VRB to NaN
            vrb_mask = data_copy['WindDirection'].astype(str).str.contains('VRB', na=False)
            data_copy.loc[vrb_mask, 'WindDirection'] = np.nan
            
            # Convert to numeric
            data_copy['WindDirection'] = pd.to_numeric(data_copy['WindDirection'], errors='coerce')
            
            # Set values outside 0-360 to NaN
            mask = (data_copy['WindDirection'] < 0) | (data_copy['WindDirection'] > 360)
            data_copy.loc[mask, 'WindDirection'] = np.nan
        self.data = data_copy
        return self
        
    def clean_storms(self) -> 'WeatherDataCleaner':
        """
        Clean storm data.
        
        Returns
        -------
        WeatherDataCleaner
            Self for method chaining
        """
        data_copy = self.data.copy()
        if 'WeatherCode' in data_copy.columns:
            storm_codes = ['TS', 'SQ', 'FC']
            storm_mask = data_copy['WeatherCode'].str.contains('|'.join(storm_codes), na=False)
            data_copy.loc[storm_mask, ['WindSpeed', 'WindDirection']] = np.nan
        self.data = data_copy
        return self
        
    def zero_calm_direction(self) -> 'WeatherDataCleaner':
        """
        Set wind direction to zero for calm conditions.
        
        Returns
        -------
        WeatherDataCleaner
            Self for method chaining
        """
        data_copy = self.data.copy()
        if 'WindSpeed' in data_copy.columns and 'WindDirection' in data_copy.columns:
            calm_mask = data_copy['WindSpeed'] == 0
            data_copy.loc[calm_mask, 'WindDirection'] = 0
        self.data = data_copy
        return self 