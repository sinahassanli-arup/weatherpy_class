"""
Module for cleaning weather data.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from .wd_base import WeatherData

class WeatherDataCleaner:
    """Base class for cleaning weather data."""
    
    def __init__(self, weather_data: WeatherData):
        """
        Initialize with weather data.
        
        Parameters
        ----------
        weather_data : WeatherData
            Weather data object to clean
        """
        self.weather_data = weather_data
        self.data = weather_data.data.copy()  # Work on a copy of the data
        
    def clean_data(self) -> WeatherData:
        """
        Clean data using all available methods.
        
        Returns
        -------
        WeatherData
            cleaned weather data
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
        # Log the operation with new format
        self.weather_data._log_operation(
            operation_class="Cleaner",
            operation_method="clean_invalid",
            inputs={"columns": columns},
            outputs={"dataChanged": bool(~self.data.equals(self.weather_data.data)),
                    "shape": self.data.shape}
        )
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
        # Log the operation with new format
        self.weather_data._log_operation(
            operation_class="Cleaner",
            operation_method="clean_threshold",
            inputs={"thresholds": thresholds},
            outputs={"removed_count": rows_to_remove.sum(), 
                    "dataChanged": bool(~self.data.equals(self.weather_data.data)),
                    "shape": self.data.shape}
        )
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
        # Log the operation with new format
        self.weather_data._log_operation(
            operation_class="Cleaner", 
            operation_method="clean_outliers",
            inputs={"columns": columns, "method": method, "threshold": threshold},
            outputs={"removed_count": rows_to_remove.sum(), 
                    "dataChanged": bool(~self.data.equals(self.weather_data.data)),
                    "shape": self.data.shape}
        )
        return self
        
    def clean_duplicates(self) -> 'WeatherDataCleaner':
        """
        Clean duplicate rows.
        
        Returns
        -------
        WeatherDataCleaner
            Self for method chaining
        """
        original_count = len(self.data)
        self.data = self.data.drop_duplicates()
        removed_count = original_count - len(self.data)
        
        # Log the operation with new format
        self.weather_data._log_operation(
            operation_class="Cleaner",
            operation_method="clean_duplicates",
            inputs={},
            outputs={"removed_count": removed_count, 
                    "dataChanged": bool(original_count != len(self.data)),
                    "shape": self.data.shape}
        )
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
        na_count_before = data_copy.isna().sum().sum()
        data_copy = data_copy.interpolate(method=method)
        na_count_after = data_copy.isna().sum().sum()
        self.data = data_copy
        
        # Log the operation with new format
        self.weather_data._log_operation(
            operation_class="Cleaner",
            operation_method="interpolate_missing",
            inputs={"method": method},
            outputs={"filled_count": na_count_before - na_count_after, 
                     "dataChanged": bool(~self.data.equals(self.weather_data.data)),
                     "shape": self.data.shape}
        )
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
        na_count_before = data_copy.isna().sum().sum()
        data_copy = data_copy.fillna(value)
        na_count_after = data_copy.isna().sum().sum()
        self.data = data_copy
        
        # Log the operation with new format
        self.weather_data._log_operation(
            operation_class="Cleaner",
            operation_method="fill_missing",
            inputs={"value": value},
            outputs={"filled_count": na_count_before - na_count_after, 
                     "dataChanged": bool(~self.data.equals(self.weather_data.data)),
                     "shape": self.data.shape}
        )
        return self
    
    def _update_weather_data(self, inplace: bool = True) -> WeatherData:
        """
        Update the WeatherData object with the cleaned data.
        
        Parameters
        ----------
        inplace : bool, optional
            If True, modify the original WeatherData object. Otherwise, create a new one.
            
        Returns
        -------
        WeatherData
            Updated WeatherData object
        """
        if inplace:
            # Update the existing WeatherData object's data
            self.weather_data.data = self.data
            return self.weather_data
        else:
            # Create a new WeatherData object with the cleaned data
            new_weather_data = WeatherData(
                data=self.data.copy(),
                station=self.weather_data.station,
                data_type=self.weather_data.data_type,
                interval=self.weather_data.interval
            )
            # Copy operations log
            new_weather_data._operations_log = self.weather_data.operations_log.copy()
            # Add the cleaning operations that were performed
            for op in self.weather_data.operations_log:
                if op not in new_weather_data._operations_log:
                    new_weather_data._operations_log.append(op)
            return new_weather_data
        
class BOMDataCleaner(WeatherDataCleaner):
    """Class for cleaning BOM weather data."""
    
    def clean_data(
        self,
        clean_invalid: bool = True,
        invalid_columns: Optional[List[str]] = None,
        clean_threshold: bool = True,
        thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
        clean_duplicates: bool = True,
        interpolate_missing: bool = False,
        inplace: bool = True
    ) -> WeatherData:
        """
        Clean BOM data using specified methods.
        
        Parameters
        ----------
        clean_invalid : bool, optional
            Whether to clean invalid values, by default True
        invalid_columns : List[str], optional
            Columns to clean for invalid values, by default ['WindSpeed', 'WindDirection']
        clean_threshold : bool, optional
            Whether to clean values outside thresholds, by default includes common weather variables
        thresholds : Dict[str, Tuple[float, float]], optional
            Thresholds for cleaning, by default includes common weather variables
        clean_duplicates : bool, optional
            Whether to clean duplicate rows, by default True
        interpolate_missing : bool, optional
            Whether to interpolate remaining missing values, by default False
        inplace : bool, optional
            If True, modify the original WeatherData object. Otherwise, create a new one.
            
        Returns
        -------
        WeatherData
            Cleaned weather data object
        """
        # Set default values if None
        if invalid_columns is None:
            invalid_columns = ['WindSpeed', 'WindDirection']
            
        if thresholds is None:
            thresholds = {
                'WindSpeed': (0, 50),
                'DryBulbTemperature': (-25, 55),
                'RelativeHumidity': (0, 100),
                'Pressure': (900, 1100)
            }
            
        if clean_invalid:
            self.clean_invalid(invalid_columns)
        
        if clean_threshold:
            self.clean_threshold(thresholds)
        
        if clean_duplicates:
            self.clean_duplicates()
    
        if interpolate_missing:
            self.interpolate_missing()
        
        return self._update_weather_data(inplace=inplace)
        
    def clean_calms(self) -> 'WeatherDataCleaner':
        """
        Clean calm wind conditions.
        
        Returns
        -------
        WeatherDataCleaner
            Self for method chaining
        """
        data_copy = self.data.copy()
        modified_count = 0
        if 'WindSpeed' in data_copy.columns:
            calm_mask = data_copy['WindSpeed'] < 0.5
            modified_count = calm_mask.sum()
            data_copy.loc[calm_mask, 'WindSpeed'] = 0
            if 'WindDirection' in data_copy.columns:
                data_copy.loc[calm_mask, 'WindDirection'] = np.nan
        self.data = data_copy
        
        # Log the operation with new format
        self.weather_data._log_operation(
            operation_class="BOMCleaner",
            operation_method="clean_calms",
            inputs={},
            outputs={"modified_count": modified_count, 
                     "dataChanged": bool(modified_count > 0),
                     "shape": self.data.shape}
        )
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
        modified_count = 0
        if 'WindDirection' in data_copy.columns:
            # Round to nearest 10 degrees
            original_values = data_copy['WindDirection'].copy()
            data_copy['WindDirection'] = data_copy['WindDirection'].round(-1)
            # Set values outside 0-360 to NaN
            mask = (data_copy['WindDirection'] < 0) | (data_copy['WindDirection'] > 360)
            data_copy.loc[mask, 'WindDirection'] = np.nan
            # Count modified values
            modified_count = (original_values != data_copy['WindDirection']).sum()
        self.data = data_copy
        
        # Log the operation with new format
        self.weather_data._log_operation(
            operation_class="BOMCleaner",
            operation_method="clean_off_clock",
            inputs={},
            outputs={"modified_count": modified_count,
                     "dataChanged": bool(modified_count > 0),
                     "shape": self.data.shape}
        )
        return self
        
class NOAADataCleaner(WeatherDataCleaner):
    """Class for cleaning NOAA weather data."""
    
    def clean_data(
        self,
        clean_invalid: bool = True,
        invalid_columns: Optional[List[str]] = None,
        clean_threshold: bool = True,
        thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
        clean_duplicates: bool = True,
        interpolate_missing: bool = False,
        clean_ranked_rows: bool = True,
        clean_VC_filter: bool = True,
        clean_storms: bool = True,
        inplace: bool = True
    ) -> WeatherData:
        """
        Clean NOAA data using specified methods.
        
        Parameters
        ----------
        clean_invalid : bool, optional
            Whether to clean invalid values, by default True
        invalid_columns : List[str], optional
            Columns to clean for invalid values, by default ['WindSpeed', 'WindDirection']
        clean_threshold : bool, optional
            Whether to clean values outside thresholds, by default True
        thresholds : Dict[str, Tuple[float, float]], optional
            Thresholds for cleaning, by default includes common weather variables
        clean_duplicates : bool, optional
            Whether to clean duplicate rows, by default True
        interpolate_missing : bool, optional
            Whether to interpolate remaining missing values, by default False
        clean_ranked_rows : bool, optional
            Whether to clean ranked rows, by default True
        clean_VC_filter : bool, optional
            Whether to clean variable/changeable weather codes, by default True
        clean_storms : bool, optional
            Whether to clean storm data, by default True
        inplace : bool, optional
            If True, modify the original WeatherData object. Otherwise, create a new one.
            
        Returns
        -------
        WeatherData
            Cleaned weather data object
        """
        # Set default values if None
        if invalid_columns is None:
            invalid_columns = ['WindSpeed', 'WindDirection']
            
        if thresholds is None:
            thresholds = {
                'WindSpeed': (0, 50),
                'DryBulbTemperature': (-25, 55),
                'RelativeHumidity': (0, 100),
                'Pressure': (900, 1100)
            }
            
        if clean_invalid:
            self.clean_invalid(invalid_columns)
        
        if clean_threshold:
            self.clean_threshold(thresholds)
        
        if clean_ranked_rows:
            self.clean_ranked_rows()
        
        if clean_VC_filter:
            self.clean_VC_filter()
        
        if clean_storms:
            self.clean_storms()
        
        if clean_duplicates:
            self.clean_duplicates()
        
        if interpolate_missing:
            self.interpolate_missing()
        
        # Update and return the WeatherData object
        return self._update_weather_data(inplace=inplace)
        
    def clean_ranked_rows(self) -> 'WeatherDataCleaner':
        """
        Clean ranked rows.
        
        Returns
        -------
        WeatherDataCleaner
            Self for method chaining
        """
        data_copy = self.data.copy()
        original_count = len(data_copy)
        if 'Rank' in data_copy.columns:
            data_copy = data_copy[data_copy['Rank'] == 1]
        self.data = data_copy
        removed_count = original_count - len(data_copy)
        
        # Log the operation with new format
        self.weather_data._log_operation(
            operation_class="NOAACleaner",
            operation_method="clean_ranked_rows",
            inputs={},
            outputs={"removed_count": removed_count,
                    "dataChanged": bool(removed_count > 0),
                    "shape": self.data.shape}
        )
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
        modified_count = 0
        if 'WeatherCode' in data_copy.columns:
            vc_mask = data_copy['WeatherCode'].str.contains('VC', na=False)
            modified_count = vc_mask.sum()
            data_copy.loc[vc_mask, 'WeatherCode'] = np.nan
        self.data = data_copy
        
        # Log the operation with new format
        self.weather_data._log_operation(
            operation_class="NOAACleaner",
            operation_method="clean_VC_filter",
            inputs={},
            outputs={"modified_count": modified_count,
                    "dataChanged": bool(modified_count > 0),
                    "shape": self.data.shape}
        )
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
        modified_count = 0
        if 'WeatherCode' in data_copy.columns:
            storm_codes = ['TS', 'SQ', 'FC']
            storm_mask = data_copy['WeatherCode'].str.contains('|'.join(storm_codes), na=False)
            modified_count = storm_mask.sum()
            data_copy.loc[storm_mask, ['WindSpeed', 'WindDirection']] = np.nan
        self.data = data_copy
        
        # Log the operation with new format
        self.weather_data._log_operation(
            operation_class="NOAACleaner",
            operation_method="clean_storms",
            inputs={"storm_codes": ['TS', 'SQ', 'FC']},
            outputs={"modified_count": modified_count,
                    "dataChanged": bool(modified_count > 0),
                    "shape": self.data.shape}
        )
        return self 