"""
Module for inferring weather data properties using psychrometric calculations.
"""

import pandas as pd
import numpy as np
import logging
from ..analysis import psychrolib
from typing import Callable, Union, Optional

class WeatherDataInferer:
    """Class for inferring weather data properties using psychrometric calculations."""
    
    def __init__(self, data: pd.DataFrame, station_altitude: float = 0):
        """
        Initialize the inferer with weather data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Weather data DataFrame with required columns:
            - DryBulbTemperature
            - SeaLevelPressure
            - DewPointTemperature (optional)
            - RelativeHumidity (optional)
        station_altitude : float, optional
            The elevation of the weather station in meters, by default 0
        """
        self.data = data.copy()
        self.station_altitude = station_altitude
        self.weather_data = None  # This will be set when used with a WeatherData object
        psychrolib.SetUnitSystem(psychrolib.SI)
        
    @staticmethod
    def _safe_psychro_calc(func: Callable, *args) -> float:
        """
        Safely execute psychrolib calculations handling exceptions.
        
        Parameters
        ----------
        func : Callable
            The psychrolib function to execute
        *args
            Arguments to pass to the function
            
        Returns
        -------
        float
            Result of the calculation or np.nan if calculation fails
        """
        try:
            return func(*args)
        except:
            return np.nan

    @staticmethod
    def GetTWetBulbFromRelHum(tdb: Union[float, np.ndarray], 
                             rh: Union[float, np.ndarray], 
                             stnLvlPres: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Safely calculate wet bulb temperature from relative humidity.
        
        Parameters
        ----------
        tdb : float or np.ndarray
            Dry bulb temperature [°C]
        rh : float or np.ndarray
            Relative humidity [%]
        stnLvlPres : float or np.ndarray
            Station level pressure [Pa]
            
        Returns
        -------
        float or np.ndarray
            Wet bulb temperature [°C]
        """
        try:
            return psychrolib.GetTWetBulbFromRelHum(tdb, rh, stnLvlPres)
        except:
            return np.nan

    @staticmethod
    def GetTWetBulbFromTDewPoint(tdb: Union[float, np.ndarray], 
                                tdp: Union[float, np.ndarray], 
                                stnLvlPres: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Safely calculate wet bulb temperature from dew point temperature.
        
        Parameters
        ----------
        tdb : float or np.ndarray
            Dry bulb temperature [°C]
        tdp : float or np.ndarray
            Dew point temperature [°C]
        stnLvlPres : float or np.ndarray
            Station level pressure [Pa]
            
        Returns
        -------
        float or np.ndarray
            Wet bulb temperature [°C]
        """
        try:
            return psychrolib.GetTWetBulbFromTDewPoint(tdb, tdp, stnLvlPres)
        except:
            return np.nan

    @staticmethod
    def GetWBFromRelHumAndDewPoint(tdb: Union[float, np.ndarray], 
                                  rh: Union[float, np.ndarray], 
                                  tdp: Union[float, np.ndarray], 
                                  stnLvlPres: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Safely calculate wet bulb temperature from both relative humidity and dew point.
        
        Parameters
        ----------
        tdb : float or np.ndarray
            Dry bulb temperature [°C]
        rh : float or np.ndarray
            Relative humidity [%]
        tdp : float or np.ndarray
            Dew point temperature [°C]
        stnLvlPres : float or np.ndarray
            Station level pressure [Pa]
            
        Returns
        -------
        float or np.ndarray
            Wet bulb temperature [°C]
        """
        try:
            return psychrolib.GetTWetBulbFromHumRatioAndDewPoint(tdb, rh, tdp, stnLvlPres)
        except:
            return np.nan

    @staticmethod
    def GetRelHumFromTDewPoint(tdb: Union[float, np.ndarray], 
                              tdp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Safely calculate relative humidity from dew point temperature.
        
        Parameters
        ----------
        tdb : float or np.ndarray
            Dry bulb temperature [°C]
        tdp : float or np.ndarray
            Dew point temperature [°C]
            
        Returns
        -------
        float or np.ndarray
            Relative humidity [%]
        """
        try:
            return psychrolib.GetRelHumFromTDewPoint(tdb, tdp)
        except:
            return np.nan
    
    @staticmethod
    def GetTDewPointFromRelHum(tdb: Union[float, np.ndarray], 
                              rh: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Safely calculate dew point temperature from relative humidity.
        
        Parameters
        ----------
        tdb : float or np.ndarray
            Dry bulb temperature [°C]
        rh : float or np.ndarray
            Relative humidity [%]
            
        Returns
        -------
        float or np.ndarray
            Dew point temperature [°C]
        """
        try:
            return psychrolib.GetTDewPointFromRelHum(tdb, rh)
        except:
            return np.nan
            
    def infer_station_pressure(self) -> pd.DataFrame:
        """
        Calculate station level pressure using altitude and temperature.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with added StationLevelPressure column
        """
        tdb = self.data['DryBulbTemperature']
        slp = self.data['SeaLevelPressure']
        
        # Create array of station altitude values
        altitude = np.full_like(tdb, self.station_altitude)
        
        # Vectorize the GetStationPressure function
        get_station_pressure = np.vectorize(psychrolib.GetStationPressure)
        
        # Calculate station level pressure (convert sea level pressure to Pa)
        stn_lvl_pres = get_station_pressure(slp * 100, altitude, tdb)
        
        # Cap unrealistic values and convert back to hPa
        stn_lvl_pres[stn_lvl_pres > 110000] = 101000
        self.data['StationLevelPressure'] = stn_lvl_pres / 100
        
        # Log the operation if weather_data is available
        if hasattr(self, 'weather_data') and self.weather_data is not None:
            self.weather_data._log_operation(
                operation_class="Inferer",
                operation_method="infer_station_pressure",
                inputs={"station_altitude": self.station_altitude},
                outputs={"added_columns": ["StationLevelPressure"],
                         "shape": self.data.shape}
            )
        
        return self.data
        
    def infer_psychro_properties(self) -> pd.DataFrame:
        """
        Infer wet bulb temperature and either dew point temperature or relative humidity
        based on available data.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with inferred psychrometric properties:
            - WetBulbTemperature
            - DewPointTemperature (if not present and RH available)
            - RelativeHumidity (if not present and DewPoint available)
        """
        tdb = self.data['DryBulbTemperature']
        tdp = self.data['DewPointTemperature']
        rh = self.data['RelativeHumidity']
        stn_lvl_pres = self.data['StationLevelPressure'] * 100  # Convert to Pa
        
        # Vectorize the methods
        get_wb_from_rh_dp = np.vectorize(self.GetWBFromRelHumAndDewPoint)
        get_wb_from_dp = np.vectorize(self.GetTWetBulbFromTDewPoint)
        get_wb_from_rh = np.vectorize(self.GetTWetBulbFromRelHum)
        get_rh_from_dp = np.vectorize(self.GetRelHumFromTDewPoint)
        get_dp_from_rh = np.vectorize(self.GetTDewPointFromRelHum)
        
        # Track added columns for logging
        added_columns = []
        
        # Case 1: Both RH and DP available
        if np.any(rh) and np.any(tdp):
            logging.info('\tCalculating WB Temperature from Relative Humidity and DP Temperature')
            self.data['WetBulbTemperature'] = get_wb_from_rh_dp(tdb, rh, tdp, stn_lvl_pres)
            added_columns.append('WetBulbTemperature')
            
        # Case 2: Only DP available
        elif not np.any(rh) and np.any(tdp):
            logging.info('\tCalculating WB Temperature from DP Temperature')
            self.data['WetBulbTemperature'] = get_wb_from_dp(tdb, tdp, stn_lvl_pres)
            self.data['RelativeHumidity'] = get_rh_from_dp(tdb, tdp)
            added_columns.extend(['WetBulbTemperature', 'RelativeHumidity'])
            
        # Case 3: Only RH available
        elif np.any(rh) and not np.any(tdp):
            logging.info('\tCalculating WB Temperature from Relative Humidity')
            self.data['WetBulbTemperature'] = get_wb_from_rh(tdb, rh, stn_lvl_pres)
            self.data['DewPointTemperature'] = get_dp_from_rh(tdb, rh)
            added_columns.extend(['WetBulbTemperature', 'DewPointTemperature'])
            
        else:
            logging.info('\tNo data could be inferred')
            
        # Log the operation if weather_data is available
        if hasattr(self, 'weather_data') and self.weather_data is not None and added_columns:
            self.weather_data._log_operation(
                operation_class="Inferer",
                operation_method="infer_psychro_properties",
                inputs={
                    "has_relative_humidity": np.any(rh),
                    "has_dew_point": np.any(tdp)
                },
                outputs={"added_columns": added_columns,
                         "shape": self.data.shape}
            )
            
        return self.data
        
    def infer_all(self) -> pd.DataFrame:
        """
        Perform all available inference calculations.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with all inferred properties:
            - StationLevelPressure
            - WetBulbTemperature
            - DewPointTemperature (if inferred)
            - RelativeHumidity (if inferred)
        """
        initial_columns = set(self.data.columns)
        self.infer_station_pressure()
        self.infer_psychro_properties()
        
        # Log the operation if weather_data is available
        if hasattr(self, 'weather_data') and self.weather_data is not None:
            added_columns = list(set(self.data.columns) - initial_columns)
            self.weather_data._log_operation(
                operation_class="Inferer",
                operation_method="infer_all",
                inputs={"station_altitude": self.station_altitude},
                outputs={"added_columns": added_columns,
                         "shape": self.data.shape}
            )
        
        return self.data 