"""
Module for managing weather station data.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

class WeatherStation:
    """Class representing a single weather station."""
    
    def __init__(self, data: pd.Series):
        """Initialize station with data."""
        self.data = data
        
    @property
    def code(self) -> str:
        """Station code."""
        return str(self.data['Station Code'])
    
    @property
    def name(self) -> str:
        """Station name."""
        return str(self.data['Station Name'])
    
    @property
    def latitude(self) -> float:
        """Station latitude."""
        return float(self.data['Latitude'])
    
    @property
    def longitude(self) -> float:
        """Station longitude."""
        return float(self.data['Longitude'])
    
    @property
    def elevation(self) -> float:
        """Station elevation."""
        return float(self.data['Elevation'])
    
    @property
    def start_year(self) -> str:
        """Start year."""
        return str(self.data['Start'])
    
    @property
    def end_year(self) -> str:
        """End year."""
        return str(self.data['End'])
    
    @property
    def available_measurements(self) -> Dict[str, str]:
        """Available measurements."""
        measurements = {}
        for col in self.data.index:
            if col in ['Wind Direction', 'Wind Speed', 'Wind Gust',
                      'Sea Level Pressure', 'Dry Bulb Temperature',
                      'Wet Bulb Temperature', 'Relative Humidity',
                      'Rain', 'Rain Intensity', 'Cloud Oktas']:
                if str(self.data[col]).lower() == 'true':
                    measurements[col] = 'True'
        return measurements

class WeatherStationDatabase:
    """Class for managing weather station databases."""
    
    def __init__(self, data_type: str = 'BOM'):
        """
        Initialize the database.
        
        Parameters
        ----------
        data_type : str, optional
            Type of database. The default is 'BOM'.
        """
        self.data_type = data_type
        self._data = self._load_database()
        
    def _load_database(self):
        """
        Load the station database.
        
        Returns
        -------
        pandas.DataFrame
            Station database.
        """
        # Get the path to the stations database
        current_dir = Path(__file__).resolve().parent
        
        if self.data_type == 'BOM':
            db_file = current_dir / 'src' / 'BOM_stations_clean.csv'
        elif self.data_type == 'NOAA':
            db_file = current_dir / 'src' / 'NOAA_stations.csv'
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")
        
        # Check if the file exists
        if not db_file.exists():
            print(f"Error: Station database file not found at {db_file}")
            return pd.DataFrame()
        
        # Load database with string dtypes to preserve leading zeros
        try:
            return pd.read_csv(db_file, dtype=str)
        except Exception as e:
            print(f"Error loading station database: {e}")
            return pd.DataFrame()
    
    def get_station(self, station_id: str) -> WeatherStation:
        """
        Get a station by ID.
        
        Parameters
        ----------
        station_id : str
            Station ID.
        
        Returns
        -------
        WeatherStation
            Station object.
        
        Raises
        ------
        ValueError
            If the station is not found.
        """
        if self.data_type == 'BOM':
            station_id = str(station_id).zfill(6)
            station_data = self._data[self._data['Station Code'] == station_id]
        elif self.data_type == 'NOAA':
            station_id = str(station_id)
            station_data = self._data[self._data['Station ID'] == station_id]
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")
        
        if len(station_data) == 0:
            raise ValueError(f"Station not found: {station_id}")
            
        return WeatherStation(station_data.iloc[0])
    
    def get_station_info(self, station_id: str, debug: bool = False) -> Dict[str, Any]:
        """
        Get station information in a dictionary format.
        
        Parameters
        ----------
        station_id : str
            Station ID.
        debug : bool, optional
            Whether to print debug information. Default is False.
        
        Returns
        -------
        Dict[str, Any]
            Station information.
        
        Raises
        ------
        ValueError
            If the station is not found.
        """
        try:
            # Print available station IDs for debugging
            if debug:
                print(f"Looking for station ID: {station_id}")
                if not self._data.empty:
                    if self.data_type == 'BOM':
                        id_column = 'Station Code'
                    elif self.data_type == 'NOAA':
                        id_column = 'Station ID'
                    else:
                        id_column = None
                    
                    if id_column and id_column in self._data.columns:
                        print(f"Available station IDs: {self._data[id_column].unique()[:5]}...")
            
            # Get station
            station = self.get_station(station_id)
            
            # Return station info
            if self.data_type == 'BOM':
                return {
                    'Station ID': station.code,
                    'Station Name': station.name,
                    'Country': str(station.data.get('Country', 'Australia')),
                    'State': str(station.data.get('State', 'Unknown')),
                    'Timezone': str(station.data.get('Timezone Name', 'Australia/Sydney')),
                    'Timezone Ofset': str(station.data.get('Timezone UTC', '+10:00')),
                    'Coordinate': (station.latitude, station.longitude),
                    'Altitude': station.elevation,
                    'Years Active': f"{station.start_year} - {station.end_year}",
                    'Source': 'BOM',
                    'Data Available': station.available_measurements
                }
            elif self.data_type == 'NOAA':
                return {
                    'Station ID': station.code,
                    'Station Name': station.name,
                    'Country': str(station.data.get('Country', 'United States')),
                    'State': str(station.data.get('State', 'Unknown')),
                    'Coordinate': (station.latitude, station.longitude),
                    'Altitude': station.elevation,
                    'Years Active': f"{station.start_year} - {station.end_year}",
                    'Source': 'NOAA',
                    'Timezone': station.data.get('Timezone Name', 'America/New_York'),
                    'Timezone Ofset': station.data.get('Timezone UTC', '-05:00')
                }
            else:
                raise ValueError(f"Unsupported data type: {self.data_type}")
        except Exception as e:
            # Re-raise the exception to propagate it to the caller
            raise ValueError(f"Error getting station info: {e}")
    
    def find_stations(self,
                     city: Optional[str] = None,
                     coordinates: Optional[Tuple[float, float]] = None,
                     nearest: Optional[int] = None,
                     radius: Optional[float] = None) -> List[WeatherStation]:
        """Find stations based on criteria."""
        if city is not None:
            # Load cities database
            cities_file = Path(__file__).resolve().parent / 'src' / 'Cities_database.csv'
            cities = pd.read_csv(cities_file)
            
            # Find city
            city_data = cities[cities['city'] == city]
            if len(city_data) == 0:
                raise ValueError(f"City not found: {city}")
            
            # Get coordinates
            lat = float(city_data.iloc[0]['lat'])
            lon = float(city_data.iloc[0]['lng'])
            coordinates = (lat, lon)
        
        if coordinates is None:
            raise ValueError("Must provide either city or coordinates")
        
        # Calculate distances
        lat1, lon1 = coordinates
        distances = self._haversine_distance(
            lat1, self._data['Latitude'].astype(float),
            lon1, self._data['Longitude'].astype(float)
        )
        
        # Add distances to DataFrame
        self._data = self._data.copy()
        self._data['Distance (km)'] = distances
        
        # Sort by distance
        self._data = self._data.sort_values('Distance (km)')
        
        # Filter results
        if nearest is not None:
            results = self._data.head(nearest)
        elif radius is not None:
            results = self._data[self._data['Distance (km)'] <= radius]
        else:
            results = self._data
        
        # Convert to station objects
        return [WeatherStation(row) for _, row in results.iterrows()]
    
    def _haversine_distance(self, lat1: float, lat2: np.ndarray, lon1: float, lon2: np.ndarray) -> np.ndarray:
        """Calculate distances using Haversine formula."""
        R = 6371  # Earth radius in kilometers
        
        # Convert to radians
        lat1, lat2 = np.radians(lat1), np.radians(lat2)
        lon1, lon2 = np.radians(lon1), np.radians(lon2)
        
        # Differences
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        # Haversine formula
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c 