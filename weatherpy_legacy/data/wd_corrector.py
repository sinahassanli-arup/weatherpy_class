"""
Module for correcting weather data.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple
from .wd_base import WeatherData

class WeatherDataCorrector(WeatherData):
    """Base class for correcting weather data."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with weather data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Weather data DataFrame
        """
        super().__init__(data)
        
    def correct_data(self) -> pd.DataFrame:
        """
        Correct data using all available methods.
        
        Returns
        -------
        pd.DataFrame
            Corrected data
        """
        raise NotImplementedError("Subclasses must implement correct_data")
        
    def correct_terrain(self, terrain_params: Dict[str, float]) -> pd.DataFrame:
        """
        Apply terrain corrections.
        
        Parameters
        ----------
        terrain_params : Dict[str, float]
            Terrain correction parameters
            
        Returns
        -------
        pd.DataFrame
            Data with terrain corrections applied
        """
        raise NotImplementedError("Subclasses must implement correct_terrain")
        
class BOMDataCorrector(WeatherDataCorrector):
    """Class for correcting BOM weather data."""
    
    def correct_data(self) -> pd.DataFrame:
        """
        Correct BOM data using all available methods.
        
        Returns
        -------
        pd.DataFrame
            Corrected data
        """
        # Apply terrain corrections
        terrain_params = {
            'offset': 0.0,
            'scale': 1.0
        }
        self.correct_terrain(terrain_params)
        
        # Correct 10-min to 1-hour data
        self._correct_10min_to_1h()
        
        return self.data
        
    def correct_terrain(self, terrain_params: Dict[str, float]) -> pd.DataFrame:
        """
        Apply terrain corrections to BOM data.
        
        Parameters
        ----------
        terrain_params : Dict[str, float]
            Terrain correction parameters:
            - offset: Wind speed offset
            - scale: Wind speed scaling factor
            
        Returns
        -------
        pd.DataFrame
            Data with terrain corrections applied
        """
        if 'WindSpeed' in self.data.columns:
            offset = terrain_params.get('offset', 0.0)
            scale = terrain_params.get('scale', 1.0)
            self.data['WindSpeed'] = (self.data['WindSpeed'] + offset) * scale
        return self.data
        
    def _correct_10min_to_1h(self) -> pd.DataFrame:
        """
        Convert 10-minute data to hourly data.
        
        Returns
        -------
        pd.DataFrame
            Data converted to hourly intervals
        """
        if 'UTC' not in self.data.columns:
            logging.warning("UTC column not found for time conversion")
            return self.data
            
        # Ensure datetime type
        self.data['UTC'] = pd.to_datetime(self.data['UTC'])
        
        # Set UTC as index
        self.data.set_index('UTC', inplace=True)
        
        # Resample to hourly
        agg_dict = {
            'WindSpeed': 'mean',
            'WindDirection': 'mean',
            'SeaLevelPressure': 'mean',
            'DryBulbTemperature': 'mean',
            'WetBulbTemperature': 'mean',
            'DewPointTemperature': 'mean',
            'RelativeHumidity': 'mean',
            'Rain': 'sum',
            'RainIntensity': 'mean',
            'RainCumulative': 'last',
            'CloudHeight': 'mean',
            'Visibility': 'mean'
        }
        
        # Only use columns that exist in the data
        agg_dict = {k: v for k, v in agg_dict.items() if k in self.data.columns}
        
        # Resample
        self.data = self.data.resample('1H').agg(agg_dict)
        
        # Reset index
        self.data.reset_index(inplace=True)
        
        return self.data
        
class NOAADataCorrector(WeatherDataCorrector):
    """Class for correcting NOAA weather data."""
    
    def correct_data(self) -> pd.DataFrame:
        """
        Correct NOAA data using all available methods.
        
        Returns
        -------
        pd.DataFrame
            Corrected data
        """
        # Apply terrain corrections
        terrain_params = {
            'offset': 0.0,
            'scale': 1.0
        }
        self.correct_terrain(terrain_params)
        
        return self.data
        
    def correct_terrain(self, terrain_params: Dict[str, float]) -> pd.DataFrame:
        """
        Apply terrain corrections to NOAA data.
        
        Parameters
        ----------
        terrain_params : Dict[str, float]
            Terrain correction parameters:
            - offset: Wind speed offset
            - scale: Wind speed scaling factor
            
        Returns
        -------
        pd.DataFrame
            Data with terrain corrections applied
        """
        if 'WindSpeed' in self.data.columns:
            offset = terrain_params.get('offset', 0.0)
            scale = terrain_params.get('scale', 1.0)
            self.data['WindSpeed'] = (self.data['WindSpeed'] + offset) * scale
        return self.data 