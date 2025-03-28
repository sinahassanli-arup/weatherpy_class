"""
Module for unifying weather data from different sources.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from .wd_base import WeatherData

class WeatherDataUnifier:
    """Class for selecting specific columns from weather data."""
        
    def __init__(self, columns: Optional[List[str]] = None, additional_columns: Optional[List[str]] = None):
        """
        Initialize the weather data unifier.
        
        Parameters
        ----------
        columns : Optional[List[str]], optional
            List of columns to select from the data, by default None
            If None, a default list of standard weather columns will be used
        additional_columns : Optional[List[str]], optional
            Additional columns to always include beyond the base columns, by default None
        """
        # Default unified columns if none provided
        base_columns = columns or [
            'UTC',
            'LocalTime',
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
            'CloudOktas',
            'Visibility',
            'WindType'
        ]
        
        # Create the complete list of unified columns
        self.unified_columns = base_columns.copy()
        
        # Add any additional columns that aren't already in the list
        if additional_columns:
            for col in additional_columns:
                if col not in self.unified_columns:
                    self.unified_columns.append(col)
    
    def unify_data(self, weather_data: WeatherData, additional_columns: Optional[List[str]] = None, inplace: bool = False) -> WeatherData:
        """
        Select specific columns from weather data based on unified_columns list and apply unification methods.
        
        Parameters
        ----------
        weather_data : WeatherData
            Weather data object to unify
        additional_columns : Optional[List[str]], optional
            Additional columns to include for this specific operation, by default None
        inplace : bool, optional
            If True, modify the data in place. Otherwise, return a new WeatherData object.
            
        Returns
        -------
        WeatherData
            Weather data object with only the selected columns and unified data
        """
        # Create a copy to avoid modifying the original
        data_copy = weather_data.data.copy()
        
        # Create the complete list of columns to include
        columns_to_include = self.unified_columns.copy()
        
        # Add any operation-specific additional columns
        if additional_columns:
            for col in additional_columns:
                if col not in columns_to_include:
                    columns_to_include.append(col)
        
        # Create a new DataFrame with only the columns that exist in the columns_to_include
        result = pd.DataFrame(index=data_copy.index)
        
        # Get the index name to avoid duplicate columns
        index_name = data_copy.index.name
        
        # Add columns that exist in both data and columns_to_include
        # Skip any column that has the same name as the index
        for col in columns_to_include:
            if col in data_copy.columns and col != index_name:
                result[col] = data_copy[col]
            elif col not in data_copy.columns and col != index_name:
                # Add missing columns with NaN values to match legacy behavior
                result[col] = np.nan
        
        # Preserve the index name
        result.index.name = index_name
        
        if inplace:
            # Update the existing WeatherData object's data
            weather_data.data = result
            unified_weather_data = weather_data
        else:
            # Create a new WeatherData object with the unified data
            unified_weather_data = WeatherData(
                data=result,
                station=weather_data.station,
                data_type=weather_data.data_type,
                interval=weather_data.interval
            )
        
        # Log the operation with the new format
        unified_weather_data._log_operation(
            operation_class="Unifier",
            operation_method="unify_data",
            inputs={
                "additional_columns": additional_columns,
                "inplace": inplace
            },
            outputs={
                "columns_before": list(data_copy.columns),
                "columns_after": list(result.columns)
            }
        )
        
        # Apply unification methods based on data type
        if isinstance(self, BOMWeatherDataUnifier):
            self.adjust_wind_direction(unified_weather_data)
        elif isinstance(self, NOAAWeatherDataUnifier):
            self.fix_dimensions(unified_weather_data)
            self.adjust_wind_direction(unified_weather_data)
        
        # Apply main class methods
        self.round_wind_direction(unified_weather_data)
        self.zero_calm_direction(unified_weather_data)
        
        return unified_weather_data

    def round_wind_direction(self, weather_data: WeatherData) -> WeatherData:
        """
        Round wind direction to the nearest 10 degrees.
        
        Parameters
        ----------
        weather_data : WeatherData
            Weather data object to adjust
            
        Returns
        -------
        WeatherData
            Weather data object with rounded wind direction
        """
        data_copy = weather_data.data.copy()
        modified_count = 0
        if 'WindDirection' in data_copy.columns:
            original_values = data_copy['WindDirection'].copy()
            data_copy['WindDirection'] = data_copy['WindDirection'].round(-1)
            modified_count = (original_values != data_copy['WindDirection']).sum()
        
        weather_data.data = data_copy
        # Update the log operation call
        weather_data._log_operation(
            operation_class="Unifier",
            operation_method="round_wind_direction",
            inputs={},
            outputs={"modified_count": modified_count,
                    "dataChanged": bool(modified_count > 0),
                    "shape": data_copy.shape}
        )
        return weather_data

    def zero_calm_direction(self, weather_data: WeatherData) -> WeatherData:
        """
        Set wind direction to zero for calm conditions.
        
        Parameters
        ----------
        weather_data : WeatherData
            Weather data object to adjust
            
        Returns
        -------
        WeatherData
            Weather data object with calm conditions assigned
        """
        data_copy = weather_data.data.copy()
        modified_count = 0
        if 'WindSpeed' in data_copy.columns and 'WindDirection' in data_copy.columns:
            calm_mask = data_copy['WindSpeed'] == 0
            modified_count = calm_mask.sum()
            data_copy.loc[calm_mask, 'WindDirection'] = 0
        
        weather_data.data = data_copy
        # Update the log operation call
        weather_data._log_operation(
            operation_class="Unifier",
            operation_method="zero_calm_direction",
            inputs={},
            outputs={"modified_count": modified_count,
                    "dataChanged": bool(modified_count > 0),
                    "shape": data_copy.shape}
        )
        return weather_data

class BOMWeatherDataUnifier(WeatherDataUnifier):
    """Class for unifying BOM weather data."""
    
    def adjust_wind_direction(self, weather_data: WeatherData) -> WeatherData:
        """
        Adjust wind directions by fixing directions between 0° and 5° to 360° and rounding to the nearest 10°.
        
        Parameters
        ----------
        weather_data : WeatherData
            Weather data object to adjust
            
        Returns
        -------
        WeatherData
            Weather data object with adjusted wind directions
        """
        data_copy = weather_data.data.copy()
        modified_count = 0
        if 'WindDirection' in data_copy.columns:
            # Fix wind directions between 0° and 5° to 360°
            original_values = data_copy['WindDirection'].copy()
            mask = (data_copy['WindDirection'] >= 0) & (data_copy['WindDirection'] <= 5)
            data_copy.loc[mask, 'WindDirection'] = 360
            modified_count = (original_values != data_copy['WindDirection']).sum()
        
        weather_data.data = data_copy
        # Update the log operation call
        weather_data._log_operation(
            operation_class="BOMUnifier",
            operation_method="adjust_wind_direction",
            inputs={},
            outputs={"modified_count": modified_count,
                    "dataChanged": bool(modified_count > 0),
                    "shape": data_copy.shape}
        )
        return weather_data

class NOAAWeatherDataUnifier(WeatherDataUnifier):
    """Class for unifying NOAA weather data."""
    
    def fix_dimensions(self, weather_data: WeatherData) -> WeatherData:
        """
        Fix dimensions and convert units for NOAA data.
        
        Parameters
        ----------
        weather_data : WeatherData
            Weather data object to adjust
            
        Returns
        -------
        WeatherData
            Weather data object with fixed dimensions and units
        """
        data_copy = weather_data.data.copy()
        
        # Define fields that need to be scaled by 0.1
        scale_by_tenth = [
            'WindSpeed',              # 0.1 m/s to m/s
            'DryBulbTemperature',     # 0.1 °C to °C
            'DewPointTemperature',    # 0.1 °C to °C
            'SeaLevelPressure',       # 0.1 hPa to hPa
            'RainCumulative'          # 0.1 mm to mm
        ]
        
        # Apply scaling to all fields that exist in the dataframe
        modified_fields = []
        for field in scale_by_tenth:
            if field in data_copy.columns:
                data_copy[field] = round(data_copy[field].astype(float) * 0.1, 2)
                modified_fields.append(field)
        
        # Set default values for fields that might be missing
        default_nan_fields = [
            'CloudOktas', 'RainCumulative', 'OC1_0', 'MW1_0', 
            'MW1_1', 'AJ1_0', 'RH1_2', 'GA1_0'
        ]
        
        added_fields = []
        for field in default_nan_fields:
            if field not in data_copy.columns:
                data_copy[field] = np.nan
                added_fields.append(field)
        
        weather_data.data = data_copy
        # Update the log operation call
        weather_data._log_operation(
            operation_class="NOAAUnifier",
            operation_method="fix_dimensions",
            inputs={"scale_by_tenth": scale_by_tenth},
            outputs={"modified_fields": modified_fields,
                   "added_fields": added_fields,
                   "shape": data_copy.shape}
        )
        return weather_data

    def adjust_wind_direction(self, weather_data: WeatherData) -> WeatherData:
        """
        Adjust wind direction data by converting VRB to NaN and ensuring values are within 0-360.
        
        Parameters
        ----------
        weather_data : WeatherData
            Weather data object to adjust
            
        Returns
        -------
        WeatherData
            Weather data object with adjusted wind direction
        """
        data_copy = weather_data.data.copy()
        vrb_count = 0
        outside_range_count = 0
        
        if 'WindDirection' in data_copy.columns:
            # Convert VRB to NaN
            vrb_mask = data_copy['WindDirection'].astype(str).str.contains('VRB', na=False)
            vrb_count = vrb_mask.sum()
            data_copy.loc[vrb_mask, 'WindDirection'] = np.nan
            
            # Convert to numeric
            data_copy['WindDirection'] = pd.to_numeric(data_copy['WindDirection'], errors='coerce')
            
            # Set values outside 0-360 to NaN
            mask = (data_copy['WindDirection'] < 0) | (data_copy['WindDirection'] > 360)
            outside_range_count = mask.sum()
            data_copy.loc[mask, 'WindDirection'] = np.nan
        
        weather_data.data = data_copy
        # Update the log operation call
        weather_data._log_operation(
            operation_class="NOAAUnifier",
            operation_method="adjust_wind_direction",
            inputs={},
            outputs={"vrb_count": vrb_count,
                   "outside_range_count": outside_range_count,
                   "dataChanged": bool((vrb_count + outside_range_count) > 0),
                   "shape": data_copy.shape}
        )
        return weather_data