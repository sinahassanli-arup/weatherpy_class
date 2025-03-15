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
    
    def unify(self, weather_data: WeatherData, additional_columns: Optional[List[str]] = None) -> WeatherData:
        """
        Select specific columns from weather data based on unified_columns list.
        
        Parameters
        ----------
        weather_data : WeatherData
            Weather data object to unify
        additional_columns : Optional[List[str]], optional
            Additional columns to include for this specific operation, by default None
            
        Returns
        -------
        WeatherData
            Weather data object with only the selected columns
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
        # and in the same order as columns_to_include
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
        
        # Update the existing WeatherData object's data
        weather_data.data = result
        
        return weather_data