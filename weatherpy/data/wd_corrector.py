"""
Module for correcting weather data.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from .wd_base import WeatherData

class WeatherDataCorrector:
    """Base class for correcting weather data."""
    
    def __init__(self, weather_data: WeatherData):
        """
        Initialize with weather data.
        
        Parameters
        ----------
        weather_data : WeatherData
            Weather data object to correct
        """
        self.weather_data = weather_data
        self.data = weather_data.data.copy()  # Work on a copy of the data
        
    def correct_data(self) -> WeatherData:
        """
        Correct data using all available methods.
        
        Returns
        -------
        WeatherData
            corrected weather data
        """
        raise NotImplementedError("Subclasses must implement correct_data")
        
    def correct_terrain(self, correction_params: Dict[str, Any]) -> 'WeatherDataCorrector':
        """
        Apply terrain corrections to wind speed.
        
        Parameters
        ----------
        correction_params : Dict[str, Any]
            Terrain correction parameters:
            - factors: List of directional correction factors
            
        Returns
        -------
        WeatherDataCorrector
            Self for method chaining
        """
        data_copy = self.data.copy()
        modified_count = 0
        
        if 'WindDirection' in data_copy.columns and 'WindSpeed' in data_copy.columns:
            # Get correction factors
            factors = correction_params.get('factors', [])
            
            if factors:
                # Append the first factor to the end to complete the circle (0° = 360°)
                factors_circle = factors.copy()
                factors_circle.append(factors_circle[0])
                
                # Create directional bins for interpolation
                directions = np.linspace(0, 360, len(factors_circle))
                
                # Interpolate correction factors based on wind direction
                correction_values = np.interp(data_copy['WindDirection'], directions, factors_circle)
                
                # Apply correction factors to wind speed
                original_speeds = data_copy['WindSpeed'].copy()
                data_copy['WindSpeed'] = data_copy['WindSpeed'] * correction_values
                modified_count = (original_speeds != data_copy['WindSpeed']).sum()
                
                # Log the operation
                self.weather_data._log_operation(
                    operation_class="Corrector",
                    operation_method="correct_terrain",
                    inputs={"correction_factors": factors},
                    outputs={"modified_count": modified_count,
                            "dataChanged": bool(modified_count > 0),
                            "shape": data_copy.shape}
                )
                
                logging.info(f"Applied terrain correction with {len(factors)} directional factors")
        
        self.data = data_copy
        return self
        
    def _update_weather_data(self, inplace: bool = True) -> WeatherData:
        """
        Update the WeatherData object with the corrected data.
        
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
            # Create a new WeatherData object with the corrected data
            new_weather_data = WeatherData(
                data=self.data.copy(),
                station=self.weather_data.station,
                data_type=self.weather_data.data_type,
                interval=self.weather_data.interval
            )
            # Copy operations log
            new_weather_data._operations_log = self.weather_data.operations_log.copy()
            # Add the correction operations that were performed
            for op in self.weather_data.operations_log:
                if op not in new_weather_data._operations_log:
                    new_weather_data._operations_log.append(op)
            return new_weather_data
        
class BOMDataCorrector(WeatherDataCorrector):
    """Class for correcting BOM weather data."""
    
    def correct_data(
        self,
        correct_terrain: bool = False,
        correction_factors: Optional[List[float]] = None,
        correct_speed_offset: bool = True,
        speed_offset: float = 0.4,
        speed_threshold: float = 2.0,
        correct_10min_to_1h: bool = False,
        conversion_factor: float = 1.05,
        inplace: bool = True
    ) -> WeatherData:
        """
        Correct BOM data using specified methods.
        
        Parameters
        ----------
        correct_terrain : bool, optional
            Whether to apply terrain corrections, by default False
        correction_factors : Optional[List[float]], optional
            Direction-specific terrain correction factors, by default None
        correct_speed_offset : bool, optional
            Whether to apply speed offset correction, by default False
        speed_offset : float, optional
            Reduction offset for speed correction, by default 0.4
        speed_threshold : float, optional
            Speed threshold for offset correction, by default 2.0
        correct_10min_to_1h : bool, optional
            Whether to convert 10-minute data to hourly, by default False
        conversion_factor : float, optional
            Conversion factor for 10min to 1h correction, by default 1.05
        inplace : bool, optional
            If True, modify the original WeatherData object. Otherwise, create a new one.
            
        Returns
        -------
        WeatherData
            Corrected weather data object
        """
        if correct_terrain and correction_factors is not None:
            self.correct_terrain({"factors": correction_factors})
        
        if correct_speed_offset:
            self.correct_speed_offset(speed_offset, speed_threshold)
        
        if correct_10min_to_1h:
            self.correct_10min_to_1h(conversion_factor)
            
        # Update and return the WeatherData object
        return self._update_weather_data(inplace=inplace)
    
    def correct_speed_offset(self, speed_offset: float = 0.4, speed_threshold: float = 2.0) -> 'WeatherDataCorrector':
        """
        Apply correction offset to wind speed for specific type anemometer of BOM weather stations.
        
        Parameters
        ----------
        speed_offset : float, optional
            Reduction offset to apply, by default 0.4 m/s
        speed_threshold : float, optional
            Speed threshold above which to apply offset, by default 2.0 m/s
            
        Returns
        -------
        WeatherDataCorrector
            Self for method chaining
        """
        data_copy = self.data.copy()
        modified_count = 0
        
        if 'WindSpeed' in data_copy.columns:
            # Create mask for wind speeds above threshold
            mask = data_copy['WindSpeed'] > speed_threshold
            modified_count = mask.sum()
            
            # Apply offset only to wind speeds above threshold
            original_speeds = data_copy.loc[mask, 'WindSpeed'].copy()
            data_copy.loc[mask, 'WindSpeed'] = data_copy.loc[mask, 'WindSpeed'] - speed_offset
            
            # Log the operation
            self.weather_data._log_operation(
                operation_class="BOMCorrector",
                operation_method="correct_speed_offset",
                inputs={"speed_offset": speed_offset, 
                       "speed_threshold": speed_threshold},
                outputs={"modified_count": modified_count,
                        "dataChanged": bool(modified_count > 0),
                        "shape": data_copy.shape}
            )
            
            logging.info(f"Applied wind speed offset correction: {speed_offset} m/s for speeds above {speed_threshold} m/s")
        
        self.data = data_copy
        return self
    
    def correct_10min_to_1h(self, conversion_factor: float = 1.05) -> 'WeatherDataCorrector':
        """
        Convert 10-minute wind speeds to hourly mean wind speeds.
        
        Parameters
        ----------
        conversion_factor : float, optional
            Conversion factor for 10min to 1h, by default 1.05
            
        Returns
        -------
        WeatherDataCorrector
            Self for method chaining
        """
        data_copy = self.data.copy()
        modified_count = 0
        
        if 'WindSpeed' in data_copy.columns:
            # Apply conversion factor
            original_speeds = data_copy['WindSpeed'].copy()
            data_copy['WindSpeed'] = data_copy['WindSpeed'] / conversion_factor
            modified_count = (original_speeds != data_copy['WindSpeed']).sum()
            
            # Log the operation
            self.weather_data._log_operation(
                operation_class="BOMCorrector",
                operation_method="correct_10min_to_1h",
                inputs={"conversion_factor": conversion_factor},
                outputs={"modified_count": modified_count,
                        "dataChanged": bool(modified_count > 0),
                        "shape": data_copy.shape}
            )
            
            logging.info(f"Converted 10-min to 1-hour wind speeds using factor: {conversion_factor}")
        
        self.data = data_copy
        return self
        
        
class NOAADataCorrector(WeatherDataCorrector):
    """Class for correcting NOAA weather data."""
    
    def correct_data(
        self,
        correct_terrain: bool = False,
        correction_factors: Optional[List[float]] = None,
        inplace: bool = True
    ) -> WeatherData:
        """
        Correct NOAA data using specified methods.
        
        Parameters
        ----------
        correct_terrain : bool, optional
            Whether to apply terrain corrections, by default False
        correction_factors : Optional[List[float]], optional
            Direction-specific terrain correction factors, by default None
        inplace : bool, optional
            If True, modify the original WeatherData object. Otherwise, create a new one.
            
        Returns
        -------
        WeatherData
            Corrected weather data object
        """
        if correct_terrain and correction_factors is not None:
            self.correct_terrain({"factors": correction_factors})
            
        # Update and return the WeatherData object
        return self._update_weather_data(inplace=inplace) 