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
        
    def clean_data(self) -> pd.DataFrame:
        """
        Clean data using all available methods.
        
        Returns
        -------
        pd.DataFrame
            Cleaned data
        """
        raise NotImplementedError("Subclasses must implement clean_data")
        
    def clean_invalid(self, columns: List[str]) -> pd.DataFrame:
        """
        Clean invalid values in specified columns.
        
        Parameters
        ----------
        columns : List[str]
            List of columns to clean
            
        Returns
        -------
        pd.DataFrame
            Data with invalid values cleaned
        """
        for col in columns:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        return self.data
        
    def clean_threshold(self, thresholds: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
        """
        Clean values outside thresholds.
        
        Parameters
        ----------
        thresholds : Dict[str, Tuple[float, float]]
            Column name to (min, max) threshold mapping
            
        Returns
        -------
        pd.DataFrame
            Data with values outside thresholds cleaned
        """
        for col, (min_val, max_val) in thresholds.items():
            if col in self.data.columns:
                mask = (self.data[col] < min_val) | (self.data[col] > max_val)
                self.data.loc[mask, col] = np.nan
        return self.data
        
    def clean_outliers(self, columns: List[str], method: str = 'zscore', threshold: float = 3) -> pd.DataFrame:
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
        pd.DataFrame
            Data with outliers cleaned
        """
        for col in columns:
            if col in self.data.columns:
                if method == 'zscore':
                    z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
                    self.data.loc[z_scores > threshold, col] = np.nan
                elif method == 'iqr':
                    Q1 = self.data[col].quantile(0.25)
                    Q3 = self.data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    mask = (self.data[col] < Q1 - threshold * IQR) | (self.data[col] > Q3 + threshold * IQR)
                    self.data.loc[mask, col] = np.nan
        return self.data
        
    def clean_duplicates(self) -> pd.DataFrame:
        """
        Clean duplicate rows.
        
        Returns
        -------
        pd.DataFrame
            Data with duplicates cleaned
        """
        self.data = self.data.drop_duplicates()
        return self.data
        
    def clean_missing(self, threshold: float = 0.5) -> pd.DataFrame:
        """
        Clean columns with too many missing values.
        
        Parameters
        ----------
        threshold : float, optional
            Missing value threshold, by default 0.5
            
        Returns
        -------
        pd.DataFrame
            Data with high-missing columns cleaned
        """
        missing_ratio = self.data.isnull().sum() / len(self.data)
        cols_to_drop = missing_ratio[missing_ratio > threshold].index
        self.data = self.data.drop(columns=cols_to_drop)
        return self.data
        
    def interpolate_missing(self, method: str = 'linear') -> pd.DataFrame:
        """
        Interpolate missing values.
        
        Parameters
        ----------
        method : str, optional
            Interpolation method, by default 'linear'
            
        Returns
        -------
        pd.DataFrame
            Data with interpolated values
        """
        self.data = self.data.interpolate(method=method)
        return self.data
        
    def fill_missing(self, value: Any = 0) -> pd.DataFrame:
        """
        Fill missing values.
        
        Parameters
        ----------
        value : Any, optional
            Fill value, by default 0
            
        Returns
        -------
        pd.DataFrame
            Data with filled values
        """
        self.data = self.data.fillna(value)
        return self.data
        
class BOMDataCleaner(WeatherDataCleaner):
    """Class for cleaning BOM weather data."""
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean BOM data using all available methods.
        
        Returns
        -------
        pd.DataFrame
            Cleaned data
        """
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
        
        return self.data
        
    def clean_calms(self) -> pd.DataFrame:
        """
        Clean calm wind conditions.
        
        Returns
        -------
        pd.DataFrame
            Data with calm conditions cleaned
        """
        if 'WindSpeed' in self.data.columns:
            calm_mask = self.data['WindSpeed'] < 0.5
            self.data.loc[calm_mask, 'WindSpeed'] = 0
            if 'WindDirection' in self.data.columns:
                self.data.loc[calm_mask, 'WindDirection'] = np.nan
        return self.data
        
    def clean_off_clock(self) -> pd.DataFrame:
        """
        Clean off-clock wind directions.
        
        Returns
        -------
        pd.DataFrame
            Data with off-clock directions cleaned
        """
        if 'WindDirection' in self.data.columns:
            # Round to nearest 10 degrees
            self.data['WindDirection'] = self.data['WindDirection'].round(-1)
            # Set values outside 0-360 to NaN
            mask = (self.data['WindDirection'] < 0) | (self.data['WindDirection'] > 360)
            self.data.loc[mask, 'WindDirection'] = np.nan
        return self.data
        
    def round_wind_direction(self) -> pd.DataFrame:
        """
        Round wind directions to nearest 10 degrees.
        
        Returns
        -------
        pd.DataFrame
            Data with rounded wind directions
        """
        if 'WindDirection' in self.data.columns:
            self.data['WindDirection'] = self.data['WindDirection'].round(-1)
        return self.data
        
    def adjust_low_wind_direction(self) -> pd.DataFrame:
        """
        Adjust wind directions for low wind speeds.
        
        Returns
        -------
        pd.DataFrame
            Data with adjusted wind directions
        """
        if 'WindSpeed' in self.data.columns and 'WindDirection' in self.data.columns:
            low_wind_mask = self.data['WindSpeed'] < 0.5
            self.data.loc[low_wind_mask, 'WindDirection'] = np.nan
        return self.data
        
    def zero_calm_direction(self) -> pd.DataFrame:
        """
        Set wind direction to zero for calm conditions.
        
        Returns
        -------
        pd.DataFrame
            Data with zeroed calm directions
        """
        if 'WindSpeed' in self.data.columns and 'WindDirection' in self.data.columns:
            calm_mask = self.data['WindSpeed'] == 0
            self.data.loc[calm_mask, 'WindDirection'] = 0
        return self.data
        
class NOAADataCleaner(WeatherDataCleaner):
    """Class for cleaning NOAA weather data."""
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean NOAA data using all available methods.
        
        Returns
        -------
        pd.DataFrame
            Cleaned data
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
        
        return self.data
        
    def clean_ranked_rows(self) -> pd.DataFrame:
        """
        Clean ranked rows.
        
        Returns
        -------
        pd.DataFrame
            Data with ranked rows cleaned
        """
        if 'Rank' in self.data.columns:
            self.data = self.data[self.data['Rank'] == 1]
        return self.data
        
    def clean_VC_filter(self) -> pd.DataFrame:
        """
        Clean variable/changeable weather codes.
        
        Returns
        -------
        pd.DataFrame
            Data with VC codes cleaned
        """
        if 'WeatherCode' in self.data.columns:
            vc_mask = self.data['WeatherCode'].str.contains('VC', na=False)
            self.data.loc[vc_mask, 'WeatherCode'] = np.nan
        return self.data
        
    def clean_direction(self) -> pd.DataFrame:
        """
        Clean wind direction data.
        
        Returns
        -------
        pd.DataFrame
            Data with wind direction cleaned
        """
        if 'WindDirection' in self.data.columns:
            # Convert VRB to NaN
            vrb_mask = self.data['WindDirection'].astype(str).str.contains('VRB', na=False)
            self.data.loc[vrb_mask, 'WindDirection'] = np.nan
            
            # Convert to numeric
            self.data['WindDirection'] = pd.to_numeric(self.data['WindDirection'], errors='coerce')
            
            # Set values outside 0-360 to NaN
            mask = (self.data['WindDirection'] < 0) | (self.data['WindDirection'] > 360)
            self.data.loc[mask, 'WindDirection'] = np.nan
        return self.data
        
    def clean_storms(self) -> pd.DataFrame:
        """
        Clean storm data.
        
        Returns
        -------
        pd.DataFrame
            Data with storm data cleaned
        """
        if 'WeatherCode' in self.data.columns:
            storm_codes = ['TS', 'SQ', 'FC']
            storm_mask = self.data['WeatherCode'].str.contains('|'.join(storm_codes), na=False)
            self.data.loc[storm_mask, ['WindSpeed', 'WindDirection']] = np.nan
        return self.data
        
    def zero_calm_direction(self) -> pd.DataFrame:
        """
        Set wind direction to zero for calm conditions.
        
        Returns
        -------
        pd.DataFrame
            Data with zeroed calm directions
        """
        if 'WindSpeed' in self.data.columns and 'WindDirection' in self.data.columns:
            calm_mask = self.data['WindSpeed'] == 0
            self.data.loc[calm_mask, 'WindDirection'] = 0
        return self.data 